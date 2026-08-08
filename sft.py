#!/usr/bin/env python3
"""Qwen3-1.7B-Base SFT (1000 题, 全参) — 能力 sanity check

目的: 先让 Qwen3-1.7B-Base 学会"解题 + \\boxed 答案"格式, 再评估 AIME24,
判断 1.7B 级别的强基座能否撑起当前评测 (决定评测集/教师选择)。
全参微调 (非 LoRA), 显存优化: 8bit AdamW + gradient checkpointing。

用法: python sft.py            # 训练 -> checkpoints/qwen3-1.7b-sft (完整权重)
依赖: transformers>=4.51 (Qwen3 支持); bitsandbytes 可选 (8bit AdamW 省显存)
数据: data/raw/sft1000.jsonl (prepare_data.py 生成)
"""
import os
import sys
import json
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from utils import log

MODEL = "Qwen/Qwen3-1.7B-Base"
SFT_PATH = os.path.join(C.RAW_DIR, "sft1000.jsonl")
OUT_DIR = os.path.join(C.CKPT_DIR, "qwen3-1.7b-sft")

LR = 1e-5               # 全参 SFT 常用 lr (LoRA 才用 2e-4 级)
BATCH = 2
GRAD_ACCUM = 2          # 梯度累积 2 步, 等价 batch 4 的梯度质量
EPOCHS = 1              # 1 epoch: 数据有噪音, 多轮会过拟合坏样本 (复读/时间戳残渣)
MAX_STEPS = 250         # micro 步上限 (1 epoch=500 步; 250 步=覆盖 500 条, 0=不限制)
MAX_LEN = 1024
SEED = 42


def check_env():
    import transformers
    major, minor = map(int, transformers.__version__.split(".")[:2])
    if major < 4 or (major == 4 and minor < 51):
        raise SystemExit(
            f"Qwen3 需要 transformers>=4.51 (当前 {transformers.__version__}), "
            f"请执行: pip install 'transformers>=4.51'")


def make_prompt(problem):
    return (f"Question: {problem}\n\n"
            "Please reason step by step, and put your final answer within \\boxed{}.\nAnswer:")


def build(tok):
    rows = [json.loads(l) for l in open(SFT_PATH)]
    data, skipped = [], 0
    for r in rows:
        p_ids = tok(make_prompt(r["problem"]))["input_ids"]
        if len(p_ids) >= MAX_LEN:          # prompt 超长则跳过, 防止截断后 labels 错位
            skipped += 1
            continue
        full = (p_ids + tok(r["solution"])["input_ids"])[:MAX_LEN]
        labels = [-100] * len(p_ids) + full[len(p_ids):]
        data.append((full, labels))
    if skipped:
        log(f"  跳过 prompt 超长样本 {skipped} 条")
    return data


def collate_fn(tok, batch):
    pad = tok.pad_token_id
    L = max(len(a) for a, _ in batch)
    x = torch.full((len(batch), L), pad, dtype=torch.long)
    y = torch.full((len(batch), L), -100, dtype=torch.long)
    m = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (a, b) in enumerate(batch):
        x[i, :len(a)] = torch.tensor(a)
        y[i, :len(b)] = torch.tensor(b)
        m[i, :len(a)] = 1
    return x, y, m


def main():
    check_env()
    random_state = torch.Generator().manual_seed(SEED)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    data = build(tok)
    log(f"SFT 数据: {len(data)} 条, 平均长度 {sum(len(a) for a, _ in data)/len(data):.0f} token")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})   # 省激活显存, 显式 use_reentrant=False
    model.train()   # 全参: 所有参数参与训练
    n_train = sum(p.numel() for p in model.parameters())
    log(f"模型加载完成 ({time.time()-t0:.0f}s), 全参训练 {n_train/1e9:.2f}B 参数")

    # optimizer: 优先 8bit AdamW 省显存 (全参 1.7B: fp32 Adam 状态约 14GB, 8bit 约 1.7GB)
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit([p for p in model.parameters()], lr=LR)
        log("optimizer: bitsandbytes AdamW8bit")
    except ImportError:
        opt = torch.optim.AdamW([p for p in model.parameters()], lr=LR)
        log("WARNING: bitsandbytes 未安装, 用 torch AdamW (fp32 状态, 显存紧张时改 BATCH=1)")

    total = len(data)
    n_steps = (total + BATCH - 1) // BATCH
    global_step = 0
    accum_steps = 0
    t_start = time.time()
    for ep in range(EPOCHS):
        perm = torch.randperm(total, generator=random_state)
        ep_loss, ep_tokens = 0.0, 0
        for i in range(0, total, BATCH):
            idx = perm[i:i + BATCH].tolist()
            batch = [data[j] for j in idx]
            x, y, m = collate_fn(tok, batch)
            x, y, m = x.to("cuda"), y.to("cuda"), m.to("cuda")
            out = model(input_ids=x, attention_mask=m, use_cache=False).logits
            shift_l = out[..., :-1, :].contiguous()
            shift_y = y[..., 1:].contiguous()
            losses = torch.nn.functional.cross_entropy(
                shift_l.view(-1, shift_l.size(-1)), shift_y.view(-1), reduction="none")
            losses = losses.view(shift_l.size(0), -1)
            valid = (shift_y != -100)
            loss = losses[valid].mean()
            (loss / GRAD_ACCUM).backward()       # 梯度累积 (BATCH 小但梯度等价 batch*GRAD_ACCUM)
            accum_steps += 1
            if accum_steps % GRAD_ACCUM == 0:
                opt.step()
                opt.zero_grad()
            global_step += 1
            ep_loss += loss.item()
            ep_tokens += int(valid.sum().item())
            if global_step % 25 == 0:
                sps = ep_tokens / (time.time() - t_start)
                log(f"  epoch {ep+1}/{EPOCHS} step {global_step} "
                    f"loss={loss.item():.4f} | {sps:.0f} tok/s")
            if MAX_STEPS and global_step >= MAX_STEPS:
                break
        if accum_steps % GRAD_ACCUM != 0:        # epoch 末尾补一次更新
            opt.step()
            opt.zero_grad()
        log(f"epoch {ep+1}/{EPOCHS} 完成, mean loss={ep_loss/n_steps:.4f}")
        if MAX_STEPS and global_step >= MAX_STEPS:
            log(f"达到 MAX_STEPS={MAX_STEPS}, 提前结束训练")
            break

    # 全参模型直接保存完整权重, 评测直接加载
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    log(f"SFT 完成 ({(time.time()-t_start)/60:.1f}m) -> {OUT_DIR}")


if __name__ == "__main__":
    main()
