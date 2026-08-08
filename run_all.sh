#!/bin/bash
# v2 全流程: 数据准备 -> 在线蒸馏训练(42组) -> AIME24 评测 -> 报告
set -e
cd "$(dirname "$0")"
PY="${PY:-python3}"   # 默认 python3, 可用 PY=/path/to/python 覆盖
echo "========== Step 0/3 数据集准备 (问题池 + AIME24) =========="
"$PY" prepare_data.py
echo "========== Step 1/3 在线蒸馏训练 (42 组) =========="
"$PY" train.py --all
echo "========== Step 2/3 AIME24 评测 =========="
"$PY" eval.py --all
echo "========== Step 3/3 报告 =========="
"$PY" report.py
echo "========== 完成: results/report.md =========="
