#!/usr/bin/env python3
"""Step 4: AIME24 评测 (阶段 A 筛选指标)
- vLLM 批量评测 (4090 推荐, vllm>=0.8.5), 失败自动回退 transformers generate
- 学生评测关闭 thinking (enable_thinking=False), 与蒸馏数据分布一致
用法: python eval.py --all | python eval.py E3_W1
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from utils import log, extract_answer, answers_match


def load_bench():
    """阶段 A 评测集 = AIME24 (data/raw/aime24.json)"""
    path = os.path.join(C.RAW_DIR, "aime24.json")
    if not os.path.exists(path):
        log("未找到 aime24.json, 先运行: python prepare_data.py")
        sys.exit(1)
    items = []
    for row in json.load(open(path)):
        gold = extract_answer(row["answer"])
        items.append({"problem": row["problem"], "gold": gold,
                      "gold_raw": row["answer"]})
    return items


def build_prompts(items):
    prompts = []
    for it in items:
        prompts.append(f"<|im_start|>system\nYou are a helpful math assistant. "
                       f"Solve the problem and provide the final answer in \\boxed{{}}.<|im_end|>\n"
                       f"<|im_start|>user\n{it['problem']}<|im_end|>\n"
                       f"<|im_start|>assistant\n")
    return prompts


def eval_vllm(ckpt_dir, items):
    from vllm import LLM, SamplingParams
    prompts = build_prompts(items)
    llm = LLM(model=ckpt_dir, dtype="bfloat16", trust_remote_code=True,
              max_model_len=C.MAX_LEN, gpu_memory_utilization=0.85,
              enforce_eager=True)
    sp = SamplingParams(max_tokens=1024, temperature=0.0,
                        chat_template_kwargs={"enable_thinking": False})
    outs = llm.generate(prompts, sp, use_tqdm=False)
    texts = [o.outputs[0].text for o in outs]
    del llm
    return texts


def eval_transformers(ckpt_dir, items):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    tok = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = build_prompts(items)
    texts, batch = [], 8
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=C.MAX_PROBLEM_LEN).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=1024, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for b in range(len(prompts[i:i + batch])):
            ids = gen[b][enc.input_ids.shape[1]:]
            texts.append(tok.decode(ids, skip_special_tokens=True))
    return texts


def eval_one(run_name):
    out_path = os.path.join(C.EVAL_DIR, f"eval_{run_name}.json")
    if os.path.exists(out_path):
        log(f"{run_name} 已评测, 跳过"); return None
    ckpt_dir = os.path.join(C.CKPT_DIR, run_name)
    if not os.path.isfile(os.path.join(ckpt_dir, "config.json")):
        log(f"{run_name} checkpoint 不存在, 跳过"); return None
    items = load_bench()
    try:
        texts = eval_vllm(ckpt_dir, items)
        backend = "vllm"
    except Exception as e:
        log(f"vLLM 失败({e}), 回退 transformers")
        texts = eval_transformers(ckpt_dir, items)
        backend = "transformers"
    results = []
    for it, t in zip(items, texts):
        pred = extract_answer(t)
        results.append({"problem": it["problem"], "gold": it["gold"],
                        "gold_raw": it["gold_raw"], "output": t,
                        "pred": pred,
                        "correct": answers_match(pred, it["gold"])})
    correct = sum(r["correct"] for r in results)
    summary = {"run": run_name, "backend": backend, "accuracy": correct / len(results),
               "correct": correct, "total": len(results), "results": results}
    json.dump(summary, open(out_path, "w"), ensure_ascii=False, indent=2)
    log(f"{run_name}: {correct}/{len(results)} = {correct/len(results)*100:.1f}% ({backend})")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="--all")
    args = ap.parse_args()
    runs = C.all_runs() if args.run == "--all" else [args.run]
    for r in runs:
        eval_one(r)


if __name__ == "__main__":
    main()
