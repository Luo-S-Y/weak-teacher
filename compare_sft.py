#!/usr/bin/env python3
"""对比 Qwen3-1.7B-Base vs Qwen3-1.7B-SFT: 在 sft1000 中抽 N 条的回答效果

用法: python compare_sft.py [--num 10] [--max-new 1024]
输出: 每题打印 题目 + Base 输出 + SFT 输出 + 与标准解答的答案匹配
"""
import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from utils import log, extract_answer, answers_match
from sft import make_prompt

BASE = "Qwen/Qwen3-1.7B-Base"
SFT = os.path.join(C.CKPT_DIR, "qwen3-1.7b-sft")
SFT_PATH = os.path.join(C.RAW_DIR, "sft1000.jsonl")


def load(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16).to("cuda").eval()
    return m, tok


def gen(m, tok, prompt, max_new):
    import torch
    enc = tok([prompt], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=1024)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(SFT_PATH)]
    random.seed(42)
    sample = random.sample(rows, min(args.num, len(rows)))
    log(f"抽取 {len(sample)} 条 (seed=42): {BASE} vs {SFT}")

    mb, tb = load(BASE)
    ms, ts = load(SFT)
    nb = ns = 0
    for i, r in enumerate(sample):
        p = make_prompt(r["problem"])
        out_b = gen(mb, tb, p, args.max_new)
        out_s = gen(ms, ts, p, args.max_new)
        pred_b, pred_s = extract_answer(out_b), extract_answer(out_s)
        gold = extract_answer(r["solution"])
        cb, cs = answers_match(pred_b, gold), answers_match(pred_s, gold)
        nb += cb
        ns += cs
        print(f"\n{'='*80}\n#{i} 题目: {r['problem']}\n")
        print(f"--- Base ({'OK' if cb else 'XX'}, pred={pred_b!r}, gold~{gold!r}) ---\n{out_b[:1000]}")
        print(f"\n--- SFT  ({'OK' if cs else 'XX'}, pred={pred_s!r}, gold~{gold!r}) ---\n{out_s[:1000]}")
    log(f"答案匹配: Base {nb}/{len(sample)} | SFT {ns}/{len(sample)}")


if __name__ == "__main__":
    main()
