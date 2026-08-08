#!/usr/bin/env python3
"""Step 0: 数据集准备 + 模型预下载
- 训练问题池: DeepScaleR 下载 POOL_SIZE=8000 题 (排除与 AIME24 重复的题, 防验证集穿越)
- 评测: AIME24 (30 题, 内置兜底)
- 模型: 学生/教师/辅助教师预下载到本地缓存 (snapshot_download, 训练加载不再走网络)
输出: data/raw/pool.json + data/raw/aime24.json + HF 模型缓存

用法: python prepare_data.py [--no-models]   (--no-models 跳过模型下载)
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


# ---------- 问题池 (DeepScaleR 8000 题, 去 AIME24 防穿越) ----------
def prep_pool():
    if load_raw("pool.json"):
        log("问题池已存在, 跳过"); return
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
    if not rows:
        log("ERROR: DeepScaleR 下载失败, 所有候选数据集均不可用")
        raise SystemExit(1)

    # 防验证集穿越: 排除与 AIME24 评测集重复的题
    aime_probs = _aime24_problems()
    if aime_probs:
        before = len(rows)
        rows = [r for r in rows if not _is_aime_dup(r["problem"], aime_probs)]
        log(f"  排除 AIME24 重复题: {before} -> {len(rows)}")

    random.seed(C.TRAIN_SEED)
    random.shuffle(rows)
    rows = rows[:C.POOL_SIZE]
    save("pool.json", rows)


def _aime24_problems():
    """AIME24 评测题问题文本集合 (归一化), 用于训练抽样排除, 防评测泄漏"""
    paths = [os.path.join(C.RAW_DIR, "aime24.json"), ASSET_AIME]
    for p in paths:
        if os.path.exists(p):
            return {re.sub(r"\s+", " ", r["problem"]).strip() for r in json.load(open(p))}
    return set()


def _is_aime_dup(problem, aime_probs):
    """problem 与任一 AIME24 题文本相同/互相包含则视为重复"""
    p = re.sub(r"\s+", " ", problem).strip()
    for a in aime_probs:
        if p == a or (len(p) > 50 and (p in a or a in p)):
            return True
    return False


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


# ---------- SFT 数据 (Qwen3-1.7B-Base sanity check: SFT_NUM 题带解答) ----------
def prep_sft():
    out = os.path.join(C.RAW_DIR, "sft1000.jsonl")
    if os.path.exists(out):
        log("sft1000.jsonl 已存在, 跳过"); return
    import itertools
    from datasets import load_dataset
    rows = None
    for ds_id in ("open-r1/OpenR1-Math-220k", "Hothan/MATH",
                  "pe-nlp/DeepScaleR-40k-Dedup", "lime-nlp/DeepScaleR_Dedup_40k"):
        try:
            log(f"下载 SFT 数据源 ({ds_id}, 流式只取前 {C.SFT_NUM*3} 条)")
            ds = load_dataset(ds_id, trust_remote_code=True, streaming=True)
            split = "train" if "train" in ds else list(ds.keys())[0]
            cand = []
            for r in itertools.islice(iter(ds[split]), C.SFT_NUM * 3):
                if r.get("problem") and (r.get("solution") or r.get("answer")):
                    cand.append(r)
            if len(cand) >= C.SFT_NUM:
                rows = cand
                log(f"  成功: {ds_id} 流式取 {len(cand)} 条候选")
                break
            log(f"  {ds_id} 候选不足 ({len(cand)})")
        except Exception as e:
            log(f"  {ds_id} 失败: {str(e)[:100]}")
    if not rows:
        log("ERROR: SFT 数据源全部失败 (需要含 solution 的数据做 SFT)")
        return
    random.seed(C.TRAIN_SEED)
    random.shuffle(rows)
    cand = rows[:C.SFT_NUM]
    with open(out, "w") as f:
        for r in cand:
            sol = r.get("solution") or r.get("answer")
            f.write(json.dumps({"problem": r["problem"], "solution": sol,
                                "answer": r.get("answer", "")},
                               ensure_ascii=False) + "\n")
    log(f"sft: {len(cand)} 条 -> {out}")


# ---------- 模型预下载 (训练时 from_pretrained 直接命中本地缓存) ----------
def prep_models():
    from huggingface_hub import snapshot_download
    import time
    mids = (C.STUDENT_MODEL, C.TEACHER_MAIN, C.TEACHER_EXTRA, *C.EXTRA_MODELS)
    for mid in mids:
        t0 = time.time()
        log(f"预下载模型 {mid} ...")
        snapshot_download(mid)
        log(f"  {mid} 就绪 ({time.time()-t0:.0f}s)")


def main():
    flags = sys.argv[1:]
    prep_aime24()          # 先就绪 AIME24 (pool 去重需要)
    prep_pool()
    if "--skip-sft" not in flags:
        prep_sft()
    if "--no-models" not in flags:
        prep_models()
    log(f"数据集准备完成: 问题池 {C.POOL_SIZE} 题 (训练用 {C.POOL_USE}) + AIME24 评测 + sft{C.SFT_NUM} + 模型缓存")


if __name__ == "__main__":
    main()
