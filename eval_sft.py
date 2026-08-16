#!/usr/bin/env python3
"""评估 Qwen3-1.7B-SFT (checkpoints/qwen3-1.7b-sft) 在 AIME24 上的准确率
用法: python eval_sft.py [--model DIR]     # 默认 checkpoints/qwen3-1.7b-sft
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from utils import log, extract_answer, answers_match
from sft import build_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(C.CKPT_DIR, "qwen3-1.7b-sft"))
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = json.load(open(os.path.join(C.RAW_DIR, "aime24.json")))
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to("cuda").eval()

    correct = 0
    for it in items:
        p = build_prompt(tok, it["problem"])
        enc = tok([p], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=1024, do_sample=False,
                                 repetition_penalty=1.1,   # 防 LoRA 微调后复读循环
                                 pad_token_id=tok.pad_token_id)
        txt = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_answer(txt)
        gold = extract_answer(it["answer"])
        c = answers_match(pred, gold)
        correct += c
        print(f"{'OK' if c else 'XX'} pred={pred!r:10} gold={gold!r:10}")
    log(f"Qwen3-1.7B-SFT: {correct}/{len(items)} = {correct/len(items)*100:.1f}%")


if __name__ == "__main__":
    main()
