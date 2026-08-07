"""Step 2: 可信度估计 E0-E6 + 42 组合训练数据组装
- 对每条样本计算 7 种估计器置信度 c
- tokenize 一次生成共享 base (input_ids/labels/token_weights, Qwen3 学生格式)
- 每 (E,W) 组合产出样本索引 + 样本级权重 (data/weights/{run}.npz)
用法: python estimate.py
"""
import os
import sys
import math
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from utils import log, load_jsonl

OUT_JSONL = os.path.join(C.GEN_DIR, "gsm8k.jsonl")
BASE_NPZ = os.path.join(C.TOKEN_DIR, "base.npz")
RUNS_META = os.path.join(C.WEIGHT_DIR, "runs.json")


def compute_confidence(rows):
    """7 种估计器, 返回 {E: [c...]}"""
    cs = {e: [] for e in C.ESTIMATORS}
    for r in rows:
        e1 = max(0.0, min(1.0, math.exp(r["avg_logprob"])))      # 教师自报置信度
        e2 = r["sc_agree"]                                       # 自一致性
        e3 = 1.0 if r["correct"] else 0.0                        # 规则验证器
        e4 = 1.0 if r["student_match"] else 0.0                  # 师生一致性
        e5 = 1.0 if r["extra_match"] else 0.5                    # 双教师多数占比
        e6 = C.E6_MIX["E3"] * e3 + C.E6_MIX["E1"] * e1           # 混合
        for e, c in zip(C.ESTIMATORS, [1.0, e1, e2, e3, e4, e5, e6]):
            cs[e].append(c)
    return cs


def teacher_token_spans(tok, text, logprobs):
    """教师输出逐 token 字符跨度 (与 logprobs 对齐); 失败返回 None"""
    ids = tok(text).input_ids
    if len(ids) != len(logprobs):
        return None
    spans, start = [], 0
    for tid in ids:
        s = tok.convert_tokens_to_string([tok.convert_ids_to_tokens(tid)])
        idx = text.find(s, start)
        if idx < 0:
            return None
        spans.append((idx, idx + len(s)))
        start = idx + len(s)
    return spans


def build_base(rows, stu_tok, tea_tok):
    """共享 tokenize: input_ids/attention_mask/labels + token_weights(W3 用)"""
    N = len(rows)
    input_ids = np.zeros((N, C.MAX_LEN), dtype=np.int64)
    attn = np.zeros((N, C.MAX_LEN), dtype=np.int64)
    labels = np.full((N, C.MAX_LEN), -100, dtype=np.int64)
    tok_w = np.zeros((N, C.MAX_LEN), dtype=np.float32)
    n_fallback = 0
    for i, r in enumerate(rows):
        out = r["teacher_output"]
        text = f"<|im_start|>system\nYou are a helpful math assistant.<|im_end|>\n" \
               f"<|im_start|>user\n{r['problem']}<|im_end|>\n" \
               f"<|im_start|>assistant\n{out}<|im_end|>"
        enc = stu_tok(text, return_offsets_mapping=True, truncation=True,
                      max_length=C.MAX_LEN)
        ids = enc["input_ids"]
        offs = enc["offset_mapping"]
        start = text.find(out)
        span = teacher_token_spans(tea_tok, out, r["token_logprobs"])
        t_w = [math.exp(lp) for lp in r["token_logprobs"]] if span else None
        fb_w = math.exp(r["avg_logprob"]) if span is None else 0.0  # 对齐失败: 样本级广播
        if span is None:
            n_fallback += 1
        L = len(ids)
        input_ids[i, :L] = ids
        attn[i, :L] = 1
        for j, (a, b) in enumerate(offs):
            if a == b:  # special token
                continue
            if start <= a and b <= start + len(out):
                labels[i, j] = ids[j]
                if t_w:
                    mid = (a - start + b - start) / 2
                    k = next((k for k, (s0, s1) in enumerate(span)
                              if s0 <= mid < s1), None)
                    tok_w[i, j] = t_w[k] if k is not None else t_w[-1]
                elif fb_w:
                    tok_w[i, j] = fb_w
    np.savez_compressed(BASE_NPZ, input_ids=input_ids, attention_mask=attn,
                        labels=labels, token_weights=tok_w)
    log(f"base 已保存 -> {BASE_NPZ} (token 对齐失败 {n_fallback}/{N}, 回退样本级)")
    return input_ids.shape


def main():
    rows = load_jsonl(OUT_JSONL)
    log(f"加载生成数据: {len(rows)} 条")

    cs = compute_confidence(rows)
    log("估计器置信度分布:")
    for e in C.ESTIMATORS:
        c = cs[e]
        log(f"  {e}: mean={np.mean(c):.3f} | 中位={np.median(c):.3f} | >0.7 占比={np.mean(np.array(c) > 0.7):.2%}")

    # 共享 tokenize
    if not os.path.exists(BASE_NPZ):
        from transformers import AutoTokenizer
        stu_tok = AutoTokenizer.from_pretrained(C.STUDENT_MODEL, trust_remote_code=True)
        tea_tok = AutoTokenizer.from_pretrained(C.TEACHER_MAIN, trust_remote_code=True)
        build_base(rows, stu_tok, tea_tok)
    else:
        log(f"base 已存在: {BASE_NPZ}")

    # 每 (E,W) 组合: indices + sample_weights
    runs = {}
    for e in C.ESTIMATORS:
        c = np.array(cs[e], dtype=np.float32)
        for w in C.MECHANISMS:
            if w == "W2":
                idx = np.where(c > C.W2_TAU)[0]
                sw = np.ones(len(idx), dtype=np.float32)
            elif w == "W5":  # 课程: 按 c 降序, 权重=c
                idx = np.argsort(-c)
                sw = c[idx]
            elif w == "W0":
                idx = np.arange(len(rows))
                sw = np.ones(len(rows), dtype=np.float32)
            else:  # W1/W3/W4: 样本级权重 c (W3 实际用 token_weights, 样本级仅备用)
                idx = np.arange(len(rows))
                sw = c
            name = C.run_name(e, w)
            np.savez_compressed(os.path.join(C.WEIGHT_DIR, f"{name}.npz"),
                                indices=idx, sample_weights=sw)
            runs[name] = {"estimator": e, "mechanism": w,
                          "n_samples": int(len(idx)),
                          "est_mean": float(np.mean(c)), "est_median": float(np.median(c))}
            log(f"  {name}: n={len(idx)} (全量 {len(rows)})")
    json.dump(runs, open(RUNS_META, "w"), indent=2, ensure_ascii=False)
    log(f"42 组权重已生成 -> {C.WEIGHT_DIR}")


if __name__ == "__main__":
    main()
