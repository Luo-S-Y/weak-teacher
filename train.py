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
    return loss, float(loss.item()), float(c_t.detach().mean())


# ==================== 训练入口 ====================
def load_models():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(C.STUDENT_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    student = AutoModelForCausalLM.from_pretrained(C.STUDENT_MODEL, torch_dtype=dtype,
                                                   trust_remote_code=True)
    lora = LoraConfig(r=C.LORA_R, lora_alpha=C.LORA_ALPHA, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    student = get_peft_model(student, lora)
    student.print_trainable_parameters()
    teacher = AutoModelForCausalLM.from_pretrained(C.TEACHER_MAIN, torch_dtype=dtype,
                                                   trust_remote_code=True).to("cuda").eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher_extra = None
    return student, teacher, teacher_extra, tok


def train_one(run_name):
    e, w = run_name.split("_")
    ckpt_dir = os.path.join(C.CKPT_DIR, run_name)
    if os.path.exists(os.path.join(ckpt_dir, "adapter_config.json")):
        log(f"{run_name} 已存在, 跳过"); return

    pool = json.load(open(POOL_PATH))
    log(f"训练 {run_name} (E={e}, W={w}), 问题池 {len(pool)} 题")
    student, teacher, teacher_extra, tok = load_models()
    if e == "E5":
        from transformers import AutoModelForCausalLM
        teacher_extra = AutoModelForCausalLM.from_pretrained(
            C.TEACHER_EXTRA, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
        for p in teacher_extra.parameters():
            p.requires_grad_(False)

    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=C.LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / C.WARMUP_STEPS))
    log_path = os.path.join(C.LOG_DIR, f"{run_name}.jsonl")
    lf = open(log_path, "w")

    for step in range(C.STEPS):
        problems = [random.choice(pool)["problem"] for _ in range(C.BATCH)]
        opt.zero_grad()
        try:
            r = train_step(student, teacher, teacher_extra, tok, problems, step, e, w)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log(f"  step {step} OOM, 建议调小 BATCH"); raise
        if r is None:
            continue
        loss, lv, cv = r
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), C.MAX_GRAD_NORM)
        opt.step()
        sched.step()
        lf.write(json.dumps({"step": step, "loss": round(lv, 4), "conf": round(cv, 4)}) + "\n")
        if step % 20 == 0:
            log(f"  [{run_name}] step {step}/{C.STEPS} loss={lv:.4f} c={cv:.4f}")
    lf.close()

    os.makedirs(ckpt_dir, exist_ok=True)
    student.save_pretrained(ckpt_dir)
    tok.save_pretrained(ckpt_dir)
    del student, teacher, teacher_extra
    torch.cuda.empty_cache()
    log(f"{run_name} 完成 -> {ckpt_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="--all", help="run 名 或 --all")
    args = ap.parse_args()
    if args.run == "--all":
        for run_name in C.all_runs():
            train_one(run_name)
    else:
        train_one(args.run)


if __name__ == "__main__":
    main()
