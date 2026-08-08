#!/usr/bin/env python3
"""弱教师可信度加权蒸馏实验 - 全局配置 (v2: 纯 logits 蒸馏, on-policy rollout)
方案: weak-teacher-credibility-experiment_副本2.md
"""
import os

BASE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# CUDA 显存: 减少碎片化 (24GB 卡跑 42 组连续训练, 尤其全参 + 大 vocab logits)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------- 路径 ----------
DATA_DIR = os.path.join(BASE, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")            # 问题池 / 评测数据
CKPT_DIR = os.path.join(BASE, "checkpoints")       # LoRA adapter 输出
EVAL_DIR = os.path.join(BASE, "results")           # 评测结果与报告
LOG_DIR = os.path.join(EVAL_DIR, "logs")           # 训练 loss 日志

for d in (DATA_DIR, RAW_DIR, CKPT_DIR, EVAL_DIR, LOG_DIR):
    # 防御: 失效软链或误建的普通文件会导致 os.makedirs(exist_ok=True) 报 FileExistsError
    if os.path.isdir(d):
        continue
    if os.path.exists(d) or os.path.islink(d):
        os.remove(d)
    os.makedirs(d, exist_ok=True)

# ---------- 模型 ----------
STUDENT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"      # 学生 (全参微调)
TEACHER_MAIN = "Qwen/Qwen2.5-1.5B-Instruct"       # 主教师 (提供逐 token logits)
TEACHER_EXTRA = "Qwen/Qwen2.5-0.5B-Instruct"      # 辅助教师 (E5 双师一致性)

# ---------- 问题池 (rollout 输入) ----------
POOL = "deepscaler"                                # 训练问题池: DeepScaleR
POOL_SIZE = 8000                                   # 下载/抽样题数 (排除 AIME24 重复后)
POOL_USE = 500                                     # 训练阶段实际使用的题数 (可配置, 默认 500)
TRAIN_SEED = 42
MATH500_DATASET = "Hothan/MATH500"                 # 备用评测 (镜像可能无缓存)
MATH500_SPLIT = "test"

# ---------- 在线蒸馏 (on-policy rollout) ----------
BATCH = 16                                         # 每步采样题数 (OOM 时调小)
STEPS = 200                                        # 阶段 A: 42 组合 × 200 步
MAX_LEN = 2048                                     # 评测序列截断 (eval 用)
MAX_PROBLEM_LEN = 512                              # 训练问题最大长度 (训练序列 = 问题 + 轨迹, 改小显著省显存)
ROLLOUT_MAX_NEW = 100                              # 学生轨迹长度 N
ROLLOUT_TEMP = 0.7                                 # 学生采样温度
TOP_K = 16                                         # E2/E4/E5 分布 top-k

# ---------- 训练 (全参) ----------
LR = 1e-5
WARMUP_STEPS = 10
MAX_GRAD_NORM = 1.0

# ---------- 估计器与机制 ----------
ESTIMATORS = ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]
MECHANISMS = ["W0", "W1", "W2", "W3", "W4", "W5"]
W2_TAU = 0.7                                      # W2 硬阈值
W5_TAU_START = 0.8                                # W5 课程起始阈值 (线性降到 0)
E6_MIX = {"E2": 0.7, "E1": 0.3}                   # E6 = 归一化组合

# 训练完是否保留权重 (42 组全参 0.5B 约 42GB, 默认保留; 磁盘不足可 KEEP_MODEL=0)
KEEP_MODEL = os.environ.get("KEEP_MODEL", "1") == "1"


def run_name(e, w):
    return f"{e}_{w}"


def all_runs():
    return [run_name(e, w) for e in ESTIMATORS for w in MECHANISMS]
