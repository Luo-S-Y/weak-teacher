"""弱教师可信度加权蒸馏实验 - 全局配置 (AutoDL 4090)"""
import os

BASE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ---------- 路径 ----------
DATA_DIR = os.path.join(BASE, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")            # GSM8K / MATH500 原始数据
GEN_DIR = os.path.join(DATA_DIR, "generated")      # 教师生成结果 (jsonl)
TOKEN_DIR = os.path.join(DATA_DIR, "tokenized")    # 共享 tokenize 缓存
WEIGHT_DIR = os.path.join(DATA_DIR, "weights")     # 每组合的权重/索引
CKPT_DIR = os.path.join(BASE, "checkpoints")       # 训练 checkpoint
EVAL_DIR = os.path.join(BASE, "results")           # 评测结果与报告
LOG_DIR = os.path.join(EVAL_DIR, "logs")           # 训练 loss 日志

for d in (DATA_DIR, RAW_DIR, GEN_DIR, TOKEN_DIR, WEIGHT_DIR, CKPT_DIR, EVAL_DIR, LOG_DIR):
    # 防御: 失效软链或误建的普通文件会导致 os.makedirs(exist_ok=True) 报 FileExistsError
    if os.path.isdir(d):
        continue
    if os.path.exists(d) or os.path.islink(d):
        os.remove(d)
    os.makedirs(d, exist_ok=True)

# ---------- 模型 ----------
STUDENT_MODEL = "Qwen/Qwen3-0.7B-Instruct"        # 学生 (主实验)
TEACHER_MAIN = "Qwen/Qwen2.5-0.5B-Instruct"       # 主弱教师 (极端弱)
TEACHER_EXTRA = "Qwen/Qwen2.5-1.5B-Instruct"      # 辅助教师 (E5 投票 / 强度对照)

# ---------- 数据 ----------
# 评测数据集: 阶段 A 用 AIME24 (MATH500 部分镜像需认证, 备用 --math500 可下载)
AIME24_DATASET = "Hothan/AIME-2024"
AIME24_SPLIT = "test"
MATH500_DATASET = "Hothan/MATH500"
MATH500_SPLIT = "test"
MAX_LEN = 2048                                    # 输入截断 (方案 max_len 2048)
MAX_PROBLEM_LEN = 1024                            # 问题最大长度 (过滤超长)

# ---------- 教师生成 ----------
GEN_MAX_NEW = 1024                                # 教师生成最大新 token
GEN_TEMP = 0.7
GEN_TOP_P = 0.9
SELF_CONSISTENCY_K = 8                            # E2 自一致性采样次数 (可调小加速)
SELF_CONSISTENCY_TEMP = 0.8
STUDENT_ANS_TEMP = 0.0                            # E4 学生基线生成温度 (greedy)
EXTRA_TEACHER_TEMP = 0.0                          # E5 辅助教师投票生成温度

# ---------- 估计器与机制 ----------
ESTIMATORS = ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]
MECHANISMS = ["W0", "W1", "W2", "W3", "W4", "W5"]
W2_TAU = 0.7                                      # W2 硬阈值
W5_TAU = 0.8                                      # W5 课程高置信阈值
E6_MIX = {"E3": 0.7, "E1": 0.3}                   # E6 混合估计器权重

# ---------- 训练 (TRL SFTConfig) ----------
TRAIN_BATCH = 8
GRAD_ACCUM = 1
LR = 2e-5
LR_SCHEDULE = "cosine"
WARMUP_RATIO = 0.03
EPOCHS = 1
BF16 = True

# 训练完是否保留完整 checkpoint (42 组全参约 120GB, 默认删除只留 loss/eval)
KEEP_MODEL = os.environ.get("KEEP_MODEL", "0") == "1"


def run_name(e, w):
    return f"{e}_{w}"


def all_runs():
    return [run_name(e, w) for e in ESTIMATORS for w in MECHANISMS]
