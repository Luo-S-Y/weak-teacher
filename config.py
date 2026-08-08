#!/usr/bin/env python3
"""弱教师可信度加权蒸馏实验 - 全局配置 (v2: 纯 logits 蒸馏, on-policy rollout)
方案: weak-teacher-credibility-experiment_副本2.md
"""
import os

BASE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

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
STUDENT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"      # 学生 (主实验, LoRA r=32 α=64)
TEACHER_MAIN = "Qwen/Qwen2.5-0.5B-Instruct"       # 主弱教师 (提供逐 token logits)
TEACHER_EXTRA = "Qwen/Qwen2.5-1.5B-Instruct"      # 辅助教师 (E5 双师一致性, 复用学生底座? 独立加载)

# ---------- 问题池 (rollout 输入) ----------
POOL = "gsm8k"                                     # 问题池: gsm8k | deepscaler
GSM8K_POOL_SIZE = 7473                             # GSM8K train 全量
DEEPSCALER_NUM = 8000                              # DeepScaleR 抽样数 (若用)
TRAIN_SEED = 42
MATH500_DATASET = "Hothan/MATH500"                 # 备用评测 (镜像可能无缓存)
MATH500_SPLIT = "test"

# ---------- 在线蒸馏 (on-policy rollout) ----------
BATCH = 16                                         # 每步采样题数 (OOM 时调小)
STEPS = 200                                        # 阶段 A: 42 组合 × 200 步
MAX_LEN = 2048                                     # 序列截断 (问题 + 轨迹)
MAX_PROBLEM_LEN = 1024                             # 问题最大长度
ROLLOUT_MAX_NEW = 100                              # 学生轨迹长度 N
ROLLOUT_TEMP = 0.7                                 # 学生采样温度
TOP_K = 16                                         # E2/E4/E5 分布 top-k

# ---------- LoRA (学生) ----------
LORA_R = 32
LORA_ALPHA = 64
LR = 1e-5
WARMUP_STEPS = 10
MAX_GRAD_NORM = 1.0

# ---------- 估计器与机制 ----------
ESTIMATORS = ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]
MECHANISMS = ["W0", "W1", "W2", "W3", "W4", "W5"]
W2_TAU = 0.7                                      # W2 硬阈值
W5_TAU_START = 0.8                                # W5 课程起始阈值 (线性降到 0)
E6_MIX = {"E2": 0.7, "E1": 0.3}                   # E6 = 归一化组合

# 训练完是否保留 adapter (42 组 LoRA 很小, 默认保留)
KEEP_MODEL = os.environ.get("KEEP_MODEL", "1") == "1"


def run_name(e, w):
    return f"{e}_{w}"


def all_runs():
    return [run_name(e, w) for e in ESTIMATORS for w in MECHANISMS]
