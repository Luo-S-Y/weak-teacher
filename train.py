#!/usr/bin/env python3
"""Step 1: 在线蒸馏训练 (v2: 学生 on-policy rollout + 教师逐 token logits 加权 reverse-KL)

训练循环 (与 ESR 一致, 差异在可信度加权):
  ① 采样 16 题 -> ② 学生 πθ rollout 轨迹 ŷ (temp=0.7, N=100) -> ③ 教师对前缀给逐 token 分布 q_t
  -> ④ 估计可信度 c (E0-E6) -> ⑤ 加权 loss L = Σ c·KL(πθ‖πT) 更新学生 (LoRA)

42 组合 = 7 估计器 × 6 机制, 每组 STEPS 步 (阶段 A). 训练侧无判题, 纯 logits 监督.

用法: python train.py --all | python train.py E3_W1
"""
import os
import sys
import json
import math
import random
import argparse

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from utils import log

POOL_PATH = os.path.join(C.RAW_DIR, "pool.json")
PROMPT_SYS = "You are a helpful math assistant. Solve the problem step by step."


# ==================== 估计器 (输出 token 级 c_t, 全部基于 logits) ====================
def estimate_c(e, p_t, log_p_t, q_t, log_q_t, log_q_sampled, V):
    """p_t/q_t: (B,M,V) 学生/教师 softmax 分布; 返回 c_t (B,M) ∈ [0,1]"""
    B, M = log_q_sampled.shape
    if e == "E0":
        return torch.ones(B, M, device=q_t.device)
    if e == "E1":  # 教师自报置信度: 采样 token 概率
        return log_q_sampled.exp().clamp(0, 1)
    K = C.TOP_K
    qt = torch.topk(q_t, K, dim=-1)               # 教师 top-k
    if e == "E2":  # 教师 top-k 分布熵 (越低越可信)
        H = -(qt.values * qt.values.clamp(min=1e-12).log()).sum(-1)
        return (1 - H / math.log(K)).clamp(0, 1)
    if e == "E3":  # 教师尖锐度 top1-top2
        return (qt.values[..., 0] - qt.values[..., 1]).clamp(0, 1)
    if e == "E4":  # 师生 top-k 重叠率 (Rethinking OPD 指标)
        ps = torch.topk(p_t.detach(), K, dim=-1)
        return _overlap(ps.indices, qt.indices)
    if e == "E5":  # 双教师 top-k 重叠率 (分布一致=可信, 需 teacher_extra)
        raise NotImplementedError("E5 需由 train_step 传入双教师分布")
    if e == "E6":  # 组合
        e1 = log_q_sampled.exp().clamp(0, 1)
        H = -(qt.values * qt.values.clamp(min=1e-12).log()).sum(-1)
        e2 = (1 - H / math.log(K)).clamp(0, 1)
        return (C.E6_MIX["E2"] * e2 + C.E6_MIX["E1"] * e1).clamp(0, 1)
    raise ValueError(e)


def _overlap(ids_a, ids_b):
    """两 top-k 索引集的重叠率 (B,M,K) -> (B,M) ∈ [0,1]"""
    K = ids_a.shape[-1]
    inter = (ids_a.unsqueeze(-2) == ids_b.unsqueeze(-1)).any(-1).sum(-1)
    return (inter / K).clamp(0, 1)


# ==================== 加权机制 (作用在 KL 项) ====================
def weighted_loss(w, c_t, p_t, log_p_t, log_q_t, step, mask):
    """c_t: (B,M) 可信度; p_t/log_q_t: (B,M,V); mask: (B,M) 有效 token"""
    if w == "W4":  # 分布插值: 目标 = c·q + (1-c)·uniform (样本级 c 广播)
        c = c_t.mean(-1)                           # (B,)
        q_t = log_q_t.exp()
        target = c.view(-1, 1, 1) * q_t + (1 - c.view(-1, 1, 1)) / log_q_t.shape[-1]
        kl_t = (p_t * (log_p_t - target.clamp(min=1e-12).log())).sum(-1)
    else:  # reverse-KL per token
        kl_t = (p_t * (log_p_t - log_q_t)).sum(-1)
    kl_t = kl_t * mask
    c_s = (c_t * mask).sum(-1) / mask.sum(-1).clamp(min=1)   # 样本级 c
    kl_s = kl_t.sum(-1) / mask.sum(-1).clamp(min=1)          # 样本级 KL
    if w == "W0":
        return kl_t.sum() / mask.sum().clamp(min=1)
    if w == "W4":  # 分布插值 (无加权, 仅目标分布被 c 插值)
        return kl_t.sum() / mask.sum().clamp(min=1)
    if w == "W1":
        return (c_s * kl_s).sum() / c_s.sum().clamp(min=1e-8)
    if w == "W2":
        keep = c_s > C.W2_TAU
        return (kl_s[keep]).mean() if keep.any() else kl_t.sum() / mask.sum().clamp(min=1)
    if w == "W3":
        return (c_t * kl_t).sum() / (c_t * mask).sum().clamp(min=1e-8)
    if w == "W5":
        tau = C.W5_TAU_START * (1 - step / C.STEPS)
        keep = c_s > tau
        return (kl_s[keep]).mean() if keep.any() else kl_t.sum() / mask.sum().clamp(min=1)
    raise ValueError(w)


# ==================== 训练一步 ====================
def make_prompt(problem):
    return (f"<|im_start|>system\n{PROMPT_SYS}<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n")


@torch.no_grad()
def teacher_logits(teacher, seq, attn):
    return teacher(input_ids=seq, attention_mask=attn).logits


def train_step(student, teacher, teacher_extra, tok, problems, step, e, w):
    B = len(problems)
    prompts = [make_prompt(p) for p in problems]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=C.MAX_PROBLEM_LEN).to("cuda")
    prompt_len = enc.input_ids.shape[1]

    # ② 学生 rollout (on-policy, no_grad)
    with torch.no_grad():
        seq = student.generate(**enc, max_new_tokens=C.ROLLOUT_MAX_NEW,
                               temperature=C.ROLLOUT_TEMP, do_sample=True, top_p=0.9,
                               pad_token_id=tok.pad_token_id).to("cuda")
    gen_len = seq.shape[1] - prompt_len
    if gen_len <= 0:
        return None

    # 有效 token mask (排除 eos 之后的 padding)
    new_toks = seq[:, prompt_len:]                          # (B, gen_len)
    valid = torch.ones_like(new_toks, dtype=torch.bool)
    for b in range(B):
        pos = (new_toks[b] == tok.eos_token_id).nonzero()
        if len(pos):
            valid[b, pos[0, 0]:] = False
    if not valid.any():
        return None

    attn = (seq != tok.pad_token_id).long()
    # ③ 学生训练前向 (带梯度) + 教师前向 (no_grad), 取生成 token 的预测分布
    st_logits = student(input_ids=seq, attention_mask=attn).logits
    pred_s = st_logits[:, prompt_len - 1: prompt_len - 1 + gen_len]   # (B,gen,V) grad
    log_p_t_ = F.log_softmax(pred_s, dim=-1)
    p_t = log_p_t_.exp()

    with torch.no_grad():
        pred_t = teacher_logits(teacher, seq, attn)[:, prompt_len - 1: prompt_len - 1 + gen_len]
        log_q_t = F.log_softmax(pred_t, dim=-1)
        q_t = log_q_t.exp()
        if e == "E5":
            pred_e = teacher_logits(teacher_extra, seq, attn)[:, prompt_len - 1: prompt_len - 1 + gen_len]
            q_e = F.log_softmax(pred_e, dim=-1).exp()
            qt = torch.topk(q_t, C.TOP_K, dim=-1)
            qe = torch.topk(q_e, C.TOP_K, dim=-1)
            c_t = _overlap(qt.indices, qe.indices)
        else:
            log_q_sampled = log_q_t.gather(-1, new_toks.unsqueeze(-1)).squeeze(-1)
            c_t = estimate_c(e, p_t, log_p_t_, q_t, log_q_t, log_q_sampled,
                             log_q_t.shape[-1])

    loss = weighted_loss(w, c_t, p_t, log_p_t_, log_q_t, step, valid)
    # 日志指标 (no_grad, 不额外占梯度)
    with torch.no_grad():
        kl_raw = (p_t.detach() * (log_p_t_.detach() - log_q_t)).sum(-1)
        kl_mean = float((kl_raw * valid).sum() / valid.sum())
        gen_len = int(valid.sum().item() // B)
    # rollout 真实输出 (取第一条有效轨迹, 用于日志展示/排查)
    rollout_text, rollout_problem = "", ""
    for b in range(B):
        if valid[b].any():
            rollout_text = tok.decode(new_toks[b][valid[b]], skip_special_tokens=True)
            rollout_problem = problems[b]
            break
    return {"loss": loss, "loss_val": float(loss.item()), "conf": float(c_t.detach().mean()),
            "conf_min": float(c_t.detach().min()), "conf_max": float(c_t.detach().max()),
            "kl": kl_mean, "gen_len": gen_len, "valid_tokens": int(valid.sum().item()),
            "rollout": rollout_text, "problem": rollout_problem}


# ==================== 训练入口 ====================
def load_teacher():
    """教师模型全局只加载一次 (42 组共享, 不随组卸载)"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(C.STUDENT_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = __import__("time").time()
    teacher = AutoModelForCausalLM.from_pretrained(C.TEACHER_MAIN, torch_dtype=dtype,
                                                   trust_remote_code=True,
                                                   low_cpu_mem_usage=True).to("cuda").eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    log(f"教师 {C.TEACHER_MAIN} 加载完成 ({__import__('time').time()-t0:.0f}s, 全程共享)")
    return teacher, tok


def load_student():
    """学生每组从预训练权重重新加载 (每组独立实验)"""
    from transformers import AutoModelForCausalLM
    t0 = __import__("time").time()
    student = AutoModelForCausalLM.from_pretrained(
        C.STUDENT_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True).to("cuda")
    trainable = sum(p.numel() for p in student.parameters())
    log(f"学生 {C.STUDENT_MODEL} 加载完成 ({__import__('time').time()-t0:.0f}s), 全参训练 {trainable/1e6:.0f}M")
    return student


def load_teacher_extra():
    """E5 双教师: 辅助教师 (仅 E5 组需要)"""
    from transformers import AutoModelForCausalLM
    t0 = __import__("time").time()
    te = AutoModelForCausalLM.from_pretrained(
        C.TEACHER_EXTRA, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True).to("cuda").eval()
    for p in te.parameters():
        p.requires_grad_(False)
    log(f"辅助教师 {C.TEACHER_EXTRA} 加载完成 ({__import__('time').time()-t0:.0f}s)")
    return te


def train_one(run_name, teacher, teacher_extra, tok):
    e, w = run_name.split("_")
    ckpt_dir = os.path.join(C.CKPT_DIR, run_name)
    if os.path.exists(os.path.join(ckpt_dir, "config.json")):
        log(f"{run_name} 已存在, 跳过"); return

    pool = json.load(open(POOL_PATH))[:C.POOL_USE]   # 训练阶段仅用 POOL_USE 条
    log(f"训练 {run_name} (E={e}, W={w}), 问题池 {len(pool)} 题 (共 {C.POOL_SIZE}, 用 {C.POOL_USE})")
    student = load_student()
    if e == "E5" and teacher_extra is None:
        teacher_extra = load_teacher_extra()

    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=C.LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / C.WARMUP_STEPS))
    log_path = os.path.join(C.LOG_DIR, f"{run_name}.jsonl")
    lf = open(log_path, "w", buffering=1)          # 行缓冲, 崩溃时日志不丢

    import time as _t
    t_start = _t.time()
    cum_tokens, cum_steps, none_count = 0, 0, 0
    log(f"[{run_name}] 开始训练: {C.STEPS} 步 × batch {C.BATCH}, 每 10 步打印进度")

    for step in range(C.STEPS):
        t0 = _t.time()
        problems = [random.choice(pool)["problem"] for _ in range(C.BATCH)]
        opt.zero_grad()
        try:
            r = train_step(student, teacher, teacher_extra, tok, problems, step, e, w)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log(f"  step {step} OOM, 建议调小 BATCH"); raise
        if r is None:   # rollout 无有效轨迹 (全 eos/空)
            none_count += 1
            if none_count in (1, 10, 50) or none_count % 100 == 0:
                log(f"  WARNING: 累计 {none_count} 步无有效轨迹 (检查 rollout/生成为空)")
            continue
        none_count = 0
        r["loss"].backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), C.MAX_GRAD_NORM)
        opt.step()
        sched.step()

        dt = _t.time() - t0
        cum_tokens += r["valid_tokens"]
        cum_steps += 1
        tok_s = cum_tokens / (_t.time() - t_start)
        elapsed = _t.time() - t_start
        eta = elapsed / cum_steps * (C.STEPS - cum_steps) if cum_steps else 0
        # 指标写日志; 每 10 步额外附带 rollout 真实输出 (截断 400 字符)
        entry = {"step": step, "loss": round(r["loss_val"], 4),
                 "conf": round(r["conf"], 4), "kl": round(r["kl"], 4),
                 "gen_len": r["gen_len"], "tok_s": round(tok_s, 1)}
        if (step + 1) % 10 == 0 or step == C.STEPS - 1:
            entry["rollout"] = r["rollout"][:400]
            entry["problem"] = r["problem"][:200]
        lf.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if step == 0 or (step + 1) % 10 == 0 or step == C.STEPS - 1:
            pct = (step + 1) / C.STEPS * 100
            log(f"[{run_name}] step {step + 1}/{C.STEPS} ({pct:5.1f}%) | "
                f"loss={r['loss_val']:.4f} | KL={r['kl']:.4f} | "
                f"c={r['conf']:.3f}({r['conf_min']:.2f}~{r['conf_max']:.2f}) | "
                f"len={r['gen_len']} | {tok_s:.0f} tok/s | "
                f"步时={dt:.1f}s | 已用={elapsed/60:.1f}m | 剩余≈{eta/60:.1f}m")
            log(f"    └ rollout: {r['rollout'][:220] or '(空)'}")
    lf.close()
    log(f"[{run_name}] 训练完成, 总耗时 {(_t.time()-t_start)/60:.1f}m, 平均 {cum_tokens/(_t.time()-t_start):.0f} tok/s")

    os.makedirs(ckpt_dir, exist_ok=True)
    student.save_pretrained(ckpt_dir)
    tok.save_pretrained(ckpt_dir)
    del student                      # 只释放学生, 教师全程共享不卸载
    torch.cuda.empty_cache()
    log(f"{run_name} 完成 -> {ckpt_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="--all", help="run 名 或 --all")
    args = ap.parse_args()
    # 教师 + tokenizer 全局只加载一次, 42 组共享 (显著减少模型加载开销)
    teacher, tok = load_teacher()
    teacher_extra = None
    if args.run == "--all":
        for run_name in C.all_runs():
            train_one(run_name, teacher, teacher_extra, tok)
    else:
        train_one(args.run, teacher, teacher_extra, tok)


if __name__ == "__main__":
    main()
