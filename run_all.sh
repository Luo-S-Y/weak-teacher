#!/bin/bash
# 一键全流程: 生成 -> 估计 -> 训练(42组) -> 评测 -> 报告 -> 清理
set -e
cd "$(dirname "$0")"
echo "========== Step 0/6 数据集准备 =========="
python prepare_data.py
echo "========== Step 1/6 教师生成 =========="
python generate_data.py
echo "========== Step 2/6 可信度估计 =========="
python estimate.py
echo "========== Step 3/6 加权训练 (42 组) =========="
python train.py --all
echo "========== Step 4/6 MATH500 评测 =========="
python eval.py --all
echo "========== Step 5/6 报告 =========="
python report.py --cleanup
echo "========== 完成: results/report.md =========="
