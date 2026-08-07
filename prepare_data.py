"""Step 0: 数据集准备
- 默认: DeepScaleR (训练, 40.3k 抽样 8k) + AIME24 (评测, 内置兜底) -> data/raw/*.json
- 可选: --gsm8k 备用 | --math500 | --code | --all

用法: python prepare_data.py [--gsm8k] [--math500] [--code] [--all]
"""
import os
import sys
import json
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from utils import log, gsm8k_gold_answer

RAWS = {
    "deepscaler": "deepscaler.json",
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


def prep_deepscaler():
    """DeepScaleR 40.3k 竞赛级数学题 -> 抽样 TRAIN_NUM 作为训练蒸馏数据源"""
    if load_raw("deepscaler"):
        log("DeepScaleR 已存在, 跳过"); return
    from datasets import load_dataset
    rows = None
    for ds_id in C.TRAIN_DATASETS:
        try:
            log(f"下载 DeepScaleR ({ds_id})")
            ds = load_dataset(ds_id, trust_remote_code=True)
            split = "train" if "train" in ds else list(ds.keys())[0]
            rows = [{"problem": r.get("problem", ""),
                     "answer": str(r.get("ground_truth_answer") or r.get("ground_truth") or "")}
                    for r in ds[split]]
            rows = [r for r in rows if r["problem"] and r["answer"]]
            log(f"  成功: {ds_id} 共 {len(rows)} 题")
            break
        except Exception as e:
            log(f"  {ds_id} 失败: {str(e)[:100]}")
    if not rows:
        log("ERROR: DeepScaleR 下载失败, 所有候选数据集均不可用")
        raise SystemExit(1)
    random.seed(C.TRAIN_SEED)
    random.shuffle(rows)
    rows = rows[:C.TRAIN_NUM]
    save("deepscaler", rows)


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


def _norm_aime(row):
    """兼容不同数据集字段名 (problem/question/text + answer/expected_answer)"""
    ans = row.get("answer")
    if ans is None:
        ans = row.get("expected_answer", "")
    return {"problem": (row.get("problem") or row.get("question") or row.get("text")
                        or row.get("prompt") or ""),
            "answer": str(ans)}


ASSET_AIME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "aime24.json")


def prep_aime24():
    if load_raw("aime24"):
        log("AIME24 已存在, 跳过"); return
    from datasets import load_dataset
    rows = None
    # 1) HF 多候选 (zwhe99/AIME90 为 AIME 2022-2024 官方镜像, 含 2024 split)
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
            rows = [_norm_aime(r) for r in ds[split]]
            log(f"  HF 成功: {ds_id} (split={split})")
            break
        except Exception as e:
            log(f"  HF {ds_id} 失败: {str(e)[:100]}")
    # 2) ModelScope 兜底 (国内速度快, git clone 免登录)
    if not rows:
        rows = _aime_from_modelscope()
    # 3) 仓库内置 30 题 (100% 可用, 无网络依赖)
    if not rows and os.path.exists(ASSET_AIME):
        log(f"使用内置数据: {ASSET_AIME}")
        rows = json.load(open(ASSET_AIME))
    if not rows:
        log("ERROR: AIME24 所有数据源均失败 (HF 镜像 + ModelScope + 内置文件)")
        raise SystemExit(1)
    save("aime24", rows)


def _aime_from_modelscope():
    import subprocess, tempfile, glob
    tmp = tempfile.mkdtemp(prefix="aime24_ms_")
    cmd = ["git", "clone", "--depth", "1",
           "https://www.modelscope.cn/datasets/HuggingFaceH4/aime_2024.git", tmp]
    log("ModelScope 兜底: git clone HuggingFaceH4/aime_2024 ...")
    if subprocess.run(cmd, capture_output=True).returncode != 0:
        return None
    rows = []
    for fp in glob.glob(os.path.join(tmp, "**", "*.jsonl"), recursive=True):
        for line in open(fp):
            line = line.strip()
            if line:
                rows.append(_norm_aime(json.loads(line)))
    return rows


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
    prep_deepscaler()
    prep_aime24()
    if "--all" in flags or "--gsm8k" in flags:
        prep_gsm8k()
    if "--all" in flags or "--math500" in flags:
        prep_math500()
    if "--all" in flags or "--code" in flags:
        prep_code()
    log("数据集准备完成: " + ", ".join(RAWS.values()))
    print(f"默认已就绪: DeepScaleR(抽样{C.TRAIN_NUM}) 训练 + AIME24 评测", flush=True)


if __name__ == "__main__":
    main()
