#!/usr/bin/env python3
"""Step 3: TRL SFTTrainer 加权训练 (阶段 A 粗筛: 7 估计器 x 6 机制 = 42 组)
- 复用共享 tokenize 缓存 base.npz, 每组合仅加载索引 + 权重
- WeightedSFTTrainer: 覆盖 compute_loss 实现 W1 样本级 / W3 token 级 / W4 软标签插值
- W2 硬阈值过滤 / W5 课程排序已在 estimate.py 完成
用法:
  python train.py --all                 # 训练全部 42 组 (缺什么训什么)
  python train.py E3_W1                 # 训练单组
依赖: trl==0.15.1 (SFTTrainer 支持已 tokenize 数据集: dataset_text_field=None)
"""
import os
import sys
import math
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from utils import log

BASE_NPZ = os.path.join(C.TOKEN_DIR, "base.npz")
RUNS_META = os.path.join(C.WEIGHT_DIR, "runs.json")


class WeightedCollator:
    """padding 并透传 sample_weights / token_weights"""

    def __init__(self, tok, use_token_w):
        self.tok = tok
        self.use_token_w = use_token_w

    def __call__(self, features):
        import torch
        max_len = max(len(f["input_ids"]) for f in features)
        n = len(features)
        pad_id = self.tok.pad_token_id
        input_ids = torch.full((n, max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((n, max_len), dtype=torch.long)
        labels = torch.full((n, max_len), -100, dtype=torch.long)
        sample_w = torch.zeros(n, dtype=torch.float32)
        batch = {}
        for i, f in enumerate(features):
            L = len(f["input_ids"])
            input_ids[i, :L] = torch.tensor(f["input_ids"])
            attn[i, :L] = 1
            labels[i, :L] = torch.tensor(f["labels"])
            sample_w[i] = f["sample_weight"]
        batch.update(input_ids=input_ids, attention_mask=attn, labels=labels,
                     sample_weights=sample_w)
        if self.use_token_w:
            tw = torch.zeros((n, max_len), dtype=torch.float32)
            for i, f in enumerate(features):
                L = len(f["token_weight"])
                tw[i, :L] = torch.tensor(f["token_weight"])
            batch["token_weights"] = tw
        return batch


def build_trainer(model, tok, dataset, run_name, ckpt_dir):
    import torch
    import torch.nn as nn
    from trl import SFTTrainer, SFTConfig

    run = json.load(open(RUNS_META))[run_name]
    mechanism = run["mechanism"]

    class WeightedSFT(SFTTrainer):
        def __init__(self, *a, mech, logv, **k):
            super().__init__(*a, **k)
            self.mech = mech
            self.logv = logv

        def compute_loss(self, model, inputs, return_outputs=False):
            sample_w = inputs.pop("sample_weights", None)
            token_w = inputs.pop("token_weights", None)
            labels = inputs["labels"]
            outputs = model(**inputs)
            logits = outputs.logits
            shift_l = logits[..., :-1, :].contiguous()
            shift_y = labels[..., 1:].contiguous()
            ce = nn.CrossEntropyLoss(reduction="none")
            losses = ce(shift_l.view(-1, shift_l.size(-1)), shift_y.view(-1))
            losses = losses.view(shift_l.size(0), -1)
            valid = (shift_y != -100).sum(-1).clamp(min=1).float()
            if self.mech == "W4":
                # 软标签插值: c*CE + (1-c)*logV
                c = sample_w.clamp(0, 1)
                per_sample = losses.sum(-1) / valid
                loss = (c * per_sample + (1 - c) * self.logv).mean()
            elif token_w is not None:
                # W3 token 级 (token_weight 已在非答案位为 0)
                tw = token_w[..., 1:]
                loss = (losses * tw).sum() / tw.sum().clamp(min=1e-6)
            elif sample_w is not None:
                # W1/W2/W5 样本级
                per_sample = losses.sum(-1) / valid
                loss = (per_sample * sample_w).sum() / sample_w.sum().clamp(min=1e-6)
            else:
                loss = losses.mean()
            return (loss, outputs) if return_outputs else loss

    args = SFTConfig(
        output_dir=ckpt_dir,
        per_device_train_batch_size=C.TRAIN_BATCH,
        gradient_accumulation_steps=C.GRAD_ACCUM,
        learning_rate=C.LR,
        lr_scheduler_type=C.LR_SCHEDULE,
        warmup_ratio=C.WARMUP_RATIO,
        num_train_epochs=C.EPOCHS,
        bf16=C.BF16,
        fp16=False,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        max_seq_length=C.MAX_LEN,
        dataset_text_field=None,   # 关键: 数据集已 tokenize, 禁止 TRL 二次处理
        remove_unused_columns=False,  # 关键: 保留 sample_weight 列
        seed=42,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )
    collator = WeightedCollator(tok, use_token_w=(mechanism == "W3"))
    trainer = WeightedSFT(
        model=model, args=args, train_dataset=dataset,
        tokenizer=tok, data_collator=collator,
        mech=mechanism, logv=math.log(tok.vocab_size),
    )
    return trainer


def load_dataset(run_name):
    b = np.load(BASE_NPZ)
    r = np.load(os.path.join(C.WEIGHT_DIR, f"{run_name}.npz"))
    idx = r["indices"]
    sw = r["sample_weights"]
    data = {
        "input_ids": b["input_ids"][idx],
        "attention_mask": b["attention_mask"][idx],
        "labels": b["labels"][idx],
        "sample_weight": sw,
    }
    run = json.load(open(RUNS_META))[run_name]
    if run["mechanism"] == "W3":
        data["token_weight"] = b["token_weights"][idx]
    from datasets import Dataset
    return Dataset.from_dict(data), int(len(idx))


def train_one(run_name):
    ckpt_dir = os.path.join(C.CKPT_DIR, run_name)
    if os.path.isfile(os.path.join(ckpt_dir, "config.json")):
        log(f"{run_name} 已存在, 跳过"); return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"训练 {run_name} ({C.STUDENT_MODEL} 全参 SFT)")
    dtype = torch.bfloat16 if C.BF16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        C.STUDENT_MODEL, torch_dtype=dtype, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(C.STUDENT_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dataset, n = load_dataset(run_name)
    log(f"  样本数: {n}, 步数≈{n // (C.TRAIN_BATCH * C.GRAD_ACCUM)}")
    trainer = build_trainer(model, tok, dataset, run_name, ckpt_dir)
    trainer.train()
    os.makedirs(ckpt_dir, exist_ok=True)
    trainer.save_model(ckpt_dir)
    tok.save_pretrained(ckpt_dir)
    del trainer, model
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
