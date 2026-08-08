#!/usr/bin/env python3
"""Step 0: 数据集准备 (v2)
- 问题池: 默认 GSM8K train 7473 题 (rollout 输入), 可选 --deepscaler 用 DeepScaleR 8000 题
- 评测: AIME24 (30 题, 内置兜底), 可选 --math500/--gsm8k_test/--code
输出: data/raw/pool.json + data/raw/aime24.json

用法: python prepare_data.py [--deepscaler] [--math500] [--gsm8k_test] [--code] [--all]
"""
import os
import sys
import json
import random
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from utils import log

ASSET_AIME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "aime24.json")


def save(name, rows):
    path = os.path.join(C.RAW_DIR, name)
    json.dump(rows, open(path, "w"), ensure_ascii=False, indent=1)
    log(f"{name}: {len(rows)} 条 -> {path}")


def load_raw(name):
    path = os.path.join(C.RAW_DIR, name)
    return json.load(open(path)) if os.path.exists(path) else None


# ---------- 问题池 ----------
def prep_pool(deepscaler=False):
    """rollout 输入问题池 (只提供题目, 无监督标签)"""
    if load_raw("pool.json"):
        log("问题池已存在, 跳过"); return
    if deepscaler:
        rows = _pool_deepscaler()
    else:
        rows = _pool_gsm8k()
    if not rows:
        log("ERROR: 问题池下载失败")
        raise SystemExit(1)
    random.seed(C.TRAIN_SEED)
    random.shuffle(rows)
    save("pool.json", rows)


def _pool_gsm8k():
    from datasets import load_dataset
    log("下载 GSM8K train (openai/gsm8k)")
    ds = load_dataset("openai/gsm8k", "main", split="train", trust_remote_code=True)
    return [{"problem": r["question"]} for r in ds]


def _pool_deepscaler():
    from datasets import load_dataset
    rows = None
    for ds_id in ("pe-nlp/DeepScaleR-40k-Prompt", "lime-nlp/DeepScaleR_Difficulty"):
        try:
            log(f"下载 DeepScaleR ({ds_id})")
            ds = load_dataset(ds_id, trust_remote_code=True)
            split = "train" if "train" in ds else list(ds.keys())[0]
            rows = [{"problem": r.get("problem", "")} for r in ds[split]]
            rows = [r for r in rows if r["problem"]]
            log(f"  成功: {ds_id} 共 {len(rows)} 题")
            break
        except Exception as e:
            log(f"  {ds_id} 失败: {str(e)[:100]}")
    if rows:
        random.seed(C.TRAIN_SEED)
        random.shuffle(rows)
        rows = rows[:C.DEEPSCALER_NUM]
    return rows


# ---------- 评测 ----------
def prep_aime24():
    if load_raw("aime24.json"):
        log("AIME24 已存在, 跳过"); return
    from datasets import load_dataset
    rows = None
    for ds_id in ("zwhe99/AIME90", "HuggingFaceH4/aime_2024", "MathArena/aime_2024_I",
                  "Hothan/AIME-2024", "di-zhang-fdu/AIME_2024", "Maxwell-Jia/AIME_2024"):
        try:
            log(f"下载 AIME24 ({ds_id})")
            ds = load_dataset(ds_id, trust_remote_code=True)
            if "2024" in ds:
                split = "2024"
            elif "test" in ds:
                split = "test"
            else:
                split = list(ds.keys())[0]
            rows = [{"problem": r.get("problem") or r.get("question") or r.get("text", ""),
                     "answer": str(r.get("answer") if r.get("answer") is not None
                                   else r.get("expected_answer", ""))}
                    for r in ds[split]]
            rows = [r for r in rows if r["problem"]]
            log(f"  HF 成功: {ds_id} (split={split})")
            break
        except Exception as e:
            log(f"  HF {ds_id} 失败: {str(e)[:100]}")
    if not rows and os.path.exists(ASSET_AIME):
        log(f"使用内置数据: {ASSET_AIME}")
        rows = json.load(open(ASSET_AIME))
    if not rows:
        log("ERROR: AIME24 所有数据源均失败")
        raise SystemExit(1)
    save("aime24.json", rows)


def prep_math500():
    if load_raw("math500.json"):
        log("MATH500 已存在, 跳过"); return
    from datasets import load_dataset
    log(f"下载 MATH500 ({C.MATH500_DATASET})")
    ds = load_dataset(C.MATH500_DATASET, split=C.MATH500_SPLIT, trust_remote_code=True)
    rows = [{"problem": r["problem"], "answer": r["answer"]} for r in ds]
    save("math500.json", rows)


def main():
    flags = sys.argv[1:]
    prep_pool(deepscaler=("--deepscaler" in flags or "--all" in flags))
    prep_aime24()
    if "--math500" in flags or "--all" in flags:
        prep_math500()
    log("数据集准备完成: 问题池 + AIME24 评测")


if __name__ == "__main__":
    main()
