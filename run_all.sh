#!/bin/bash
# 一键全流程: 数据准备 -> 教师生成 -> 估计 -> 训练(42组) -> 评测 -> 报告 -> 清理
set -e
cd "$(dirname "$0")"
PY="${PY:-python3}"   # 默认 python3, 可用 PY=/path/to/python 覆盖 (与 setup.sh 实际安装环境保持一致)
echo "========== Step 0/6 数据集准备 =========="
"$PY" prepare_data.py
echo "========== Step 1/6 教师生成 =========="
"$PY" generate_data.py
echo "========== Step 2/6 可信度估计 =========="
"$PY" estimate.py
echo "========== Step 3/6 加权训练 (42 组) =========="
"$PY" train.py --all
echo "========== Step 4/6 AIME24 评测 =========="
"$PY" eval.py --all
echo "========== Step 5/6 报告 =========="
"$PY" report.py --cleanup
echo "========== 完成: results/report.md =========="
