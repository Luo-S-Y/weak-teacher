#!/usr/bin/env python3
"""Step 3: 结果汇总报告 (阶段 A 粗筛分析)
- 对比 42 组 AIME24 准确率, 基线 E0_W0 (均匀 reverse-KL), 相对增益
- 每估计器最佳机制 / 每机制最佳估计器
用法: python report.py [--cleanup]   (--cleanup: 删除 adapter 省磁盘, KEEP_MODEL=1 时跳过)
"""
import os
import sys
import json
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from utils import log

SUMMARY_PATH = os.path.join(C.EVAL_DIR, "summary.json")
REPORT_PATH = os.path.join(C.EVAL_DIR, "report.md")


def _steps(run_name):
    """从训练日志读取完成的步数"""
    p = os.path.join(C.LOG_DIR, f"{run_name}.jsonl")
    return sum(1 for _ in open(p)) if os.path.exists(p) else 0


def main():
    cleanup = "--cleanup" in sys.argv
    runs = {}
    for r in C.all_runs():
        p = os.path.join(C.EVAL_DIR, f"eval_{r}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            runs[r] = {"accuracy": d["accuracy"], "correct": d["correct"],
                       "total": d["total"], "backend": d["backend"],
                       "steps": _steps(r)}

    base = runs.get("E0_W0", {}).get("accuracy", 0)
    import time
    lines = ["# 弱教师可信度加权 (v2 纯 logits 蒸馏) - 阶段 A 粗筛报告 (AIME24)",
             f"\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"\n基线 E0_W0 (ESR 式均匀 reverse-KL): {base*100:.1f}%",
             f"\n| run | 估计器 | 机制 | 步数 | 准确率 | 相对增益 |",
             f"|-----|--------|------|------|--------|----------|"]
    rows = []
    for r in C.all_runs():
        if r not in runs:
            continue
        d = runs[r]
        gain = (d["accuracy"] - base) / base if base > 0 else 0
        rows.append((d["accuracy"], r, d))
        lines.append(f"| {r} | {r.split('_')[0]} | {r.split('_')[1]} | {d['steps']} | "
                     f"{d['accuracy']*100:.1f}% | {gain*100:+.1f}% |")
    rows.sort(key=lambda x: -x[0])
    lines.append("\n## 排序 Top 10")
    lines.append("| run | 准确率 |")
    lines.append("|-----|--------|")
    for acc, r, _ in rows[:10]:
        lines.append(f"| {r} | {acc*100:.1f}% |")

    lines.append("\n## 每估计器最佳机制")
    for e in C.ESTIMATORS:
        cand = [(runs[r]["accuracy"], r) for r in C.all_runs()
                if r in runs and r.startswith(e + "_")]
        if cand:
            best = max(cand)
            lines.append(f"- {e}: 最佳 {best[1]} = {best[0]*100:.1f}%")

    lines.append("\n## 每机制最佳估计器")
    for w in C.MECHANISMS:
        cand = [(runs[r]["accuracy"], r) for r in C.all_runs()
                if r in runs and r.endswith("_" + w)]
        if cand:
            best = max(cand)
            lines.append(f"- {w}: 最佳 {best[1]} = {best[0]*100:.1f}%")

    json.dump({"baseline_E0W0": base, "runs": runs},
              open(SUMMARY_PATH, "w"), indent=2, ensure_ascii=False)
    open(REPORT_PATH, "w").write("\n".join(lines) + "\n")
    log(f"报告 -> {REPORT_PATH}")
    print("\n".join(lines), flush=True)

    if cleanup and not C.KEEP_MODEL:
        n = 0
        for r in C.all_runs():
            p = os.path.join(C.CKPT_DIR, r)
            if os.path.isdir(p):
                shutil.rmtree(p)
                n += 1
        log(f"已删除 {n} 个 checkpoint (KEEP_MODEL=1 可保留)")


if __name__ == "__main__":
    main()
