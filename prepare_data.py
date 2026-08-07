"""Step 0: 数据集准备 (方案 2.3)
- 默认: GSM8K (train/test) + MATH500  -> data/raw/*.json (归一化, 幂等)
- 可选: --code HumanEval/MBPP | --aime AIME24 | --all 全部 (供阶段 B 使用)

用法: python prepare_data.py [--code] [--aime] [--all]
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from utils import log, gsm8k_gold_answer

RAWS = {
    "gsm8k_train": "gsm8k_train.json",
    "gsm8k_test": "gsm8k_test.json",
    "math500": "math500.json",
    "aime24": "aime24.json",
    "humaneval": "humaneval.json",
    "mbpp": "mbpp.json",
}


def save(name, rows):
    path = os.path.join(C.RAW_DIR, RAWS[name])
    json.dump(rows, open(path, "w"), ensure_ascii=False, indent=1)
    log(f"{name}: {len(rows)} 条 -> {path}")


def load_raw(name):
    path = os.path.join(C.RAW_DIR, RAWS[name])
    return json.load(open(path)) if os.path.exists(path) else None


def prep_gsm8k():
    if load_raw("gsm8k_train") and load_raw("gsm8k_test"):
        log("GSM8K 已存在, 跳过"); return
    from datasets import load_dataset
    log("下载 GSM8K (openai/gsm8k)")
    ds = load_dataset("openai/gsm8k", "main", trust_remote_code=True)
    for split, name in [("train", "gsm8k_train"), ("test", "gsm8k_test")]:
        rows = [{"problem": r["question"], "answer": r["answer"],
                 "gold": gsm8k_gold_answer(r["answer"])} for r in ds[split]]
        save(name, rows)


def prep_math500():
    if load_raw("math500"):
        log("MATH500 已存在, 跳过"); return
    from datasets import load_dataset
    log(f"下载 MATH500 ({C.MATH500_DATASET})")
    ds = load_dataset(C.MATH500_DATASET, split=C.MATH500_SPLIT, trust_remote_code=True)
    rows = [{"problem": r["problem"], "answer": r["answer"]} for r in ds]
    save("math500", rows)


def prep_aime24():
    if load_raw("aime24"):
        log("AIME24 已存在, 跳过"); return
    from datasets import load_dataset
    log("下载 AIME24 (Hothan/AIME-2024)")
    ds = load_dataset("Hothan/AIME-2024", trust_remote_code=True)
    split = "test" if "test" in ds else list(ds.keys())[0]
    rows = [{"problem": r["problem"], "answer": r["answer"]} for r in ds[split]]
    save("aime24", rows)


def prep_code():
    if load_raw("humaneval"):
        log("HumanEval 已存在, 跳过")
    else:
        from datasets import load_dataset
        log("下载 HumanEval (openai_humaneval)")
        ds = load_dataset("openai_humaneval", trust_remote_code=True)
        rows = [{"prompt": r["prompt"], "test": r["test"],
                 "entry_point": r["entry_point"]} for r in ds["test"]]
        save("humaneval", rows)
    if load_raw("mbpp"):
        log("MBPP 已存在, 跳过")
    else:
        from datasets import load_dataset
        log("下载 MBPP (mbpp, sanitized)")
        try:
            ds = load_dataset("mbpp", "sanitized", trust_remote_code=True)
        except Exception:
            ds = load_dataset("google-research-datasets/mbpp", "sanitized",
                              trust_remote_code=True)
        rows = []
        for r in ds["test"]:
            rows.append({"prompt": r.get("prompt") or r.get("text", ""),
                         "test": r.get("test_list") or r.get("test", ""),
                         "entry_point": r.get("entry_point", "")})
        save("mbpp", rows)


def main():
    flags = sys.argv[1:]
    prep_gsm8k()
    prep_math500()
    if "--all" in flags or "--aime" in flags:
        prep_aime24()
    if "--all" in flags or "--code" in flags:
        prep_code()
    log("数据集准备完成: " + ", ".join(RAWS.values()))
    print("默认已就绪: GSM8K train/test + MATH500 (阶段 A 用)", flush=True)


if __name__ == "__main__":
    main()
