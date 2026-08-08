# 弱教师信号的可信度加权蒸馏 — 阶段 A 复现代码（v2 纯 logits 蒸馏）

> 方案: `weak-teacher-credibility-experiment_副本2.md` | 目标环境: **AutoDL 4090 (24GB, CUDA>=12.1)**
> v2 核心: 训练侧**无判题**，学生 **on-policy rollout** 产生轨迹，教师对轨迹前缀给**逐 token logits**，按可信度加权做 **reverse-KL** 蒸馏（ESR 同款循环 + 加权层）。

## 训练循环（train.py）

```
① 采样 16 题 ──> ② 学生 πθ rollout 轨迹 ŷ (temp=0.7, N=100) ──> ③ 教师(1.5B) 对前缀给 q_t
──> ④ 估计可信度 c (E0-E6, 全基于 logits) ──> ⑤ L = Σ c·KL(πθ‖πT) 更新学生 (0.5B 全参)
```

- 学生 **Qwen2.5-0.5B-Instruct 全参微调**（无 LoRA）；教师 **Qwen2.5-1.5B-Instruct** 提供 logits
- 每步轨迹来自**当前学生**（on-policy），数据不复用
- 学生前向算 `p_t`（带梯度），教师前向算 `q_t`（no_grad），二者取生成 token 位置
- `Loss = Σ c·KL(p_t‖q_t)`：full-vocab reverse-KL，采样轨迹上的 MC 形式

## 42 组合 = 7 估计器 × 6 机制

**估计器 E0-E6**（输出 token 级 c∈[0,1]，全部只依赖 logits，不用判题）：

| 估计器 | 计算方式 | 信息源 |
|---|---|---|
| E0 | 恒 1.0（基线，ESR 式均匀 KL） | — |
| E1 | 教师自报置信度 = 采样 token 概率 exp(log q(t)) | 教师分布 |
| E2 | 教师 top-k 分布熵（熵越低越可信） | 教师分布 |
| E3 | 教师尖锐度（top1−top2） | 教师分布 |
| E4 | 师生 top-k 重叠率（Rethinking OPD 指标） | 师生双方 |
| E5 | 双教师 top-k 重叠率（辅助教师 0.5B） | 教师池 |
| E6 | 0.7×E2 + 0.3×E1 组合 | 组合 |

**机制 W0-W5**（作用在 KL 项）：

| 机制 | 行为 |
|---|---|
| W0 | 无加权（标准 reverse-KL） |
| W1 | 样本级重加权：c_s·KL |
| W2 | 硬阈值过滤：保留 c>0.7 的样本 |
| W3 | token 级重加权：c_t·KL_t |
| W4 | 分布插值：目标分布 = c·q + (1−c)·uniform |
| W5 | 课程加权：阈值从 0.8 线性降到 0 |

## 数据安排

| 角色 | 数据 | 数量 |
|---|---|---|
| 训练问题池 | DeepScaleR（**先排除 AIME24 30 题，防验证集穿越**） | 下载 8000，训练阶段实际用 **500**（`POOL_USE` 可配） |
| 验证/测试 | AIME24 | 30 题（内置兜底） |

## 快速开始（AutoDL）

```bash
cd /root/autodl-tmp/opd
bash setup.sh                    # 环境 + 依赖 (torch 2.6.0 / vllm 0.8.5)
python3 prepare_data.py          # 数据 + 预下载模型 (首次 ~5-15 分钟, 一次性)
python3 train.py E0_W0           # 先验证单组
bash run_all.sh                  # 全量: train(42组) -> eval -> report
```

## 单步执行

| 命令 | 说明 |
|---|---|
| `python prepare_data.py` | DeepScaleR 8000 题（去 AIME24 重复）+ AIME24 30 题 + **预下载学生/教师模型**（`--no-models` 跳过） |
| `python train.py --all` | 42 组 on-policy 蒸馏，每组 200 步（断点续跑）。单组 `python train.py E3_W1` |
| `python eval.py --all` | AIME24 评测（vLLM，回退 transformers） |
| `python report.py` | 报告 → `results/report.md`（基线 E0_W0，相对增益） |

数据流: `prepare_data.py` → `data/raw/pool.json` + `aime24.json` → `train.py`（在线 rollout 蒸馏）→ `checkpoints/` → `eval.py` → `results/`。

## 配置（config.py）

- `STUDENT_MODEL`（Qwen2.5-0.5B-Instruct，**全参**）/ `TEACHER_MAIN`（1.5B）/ `TEACHER_EXTRA`（0.5B，仅 E5）
- `POOL_SIZE=8000`（DeepScaleR 下载量）/ **`POOL_USE=500`（训练实际使用题数）**
- `BATCH=16`（OOM 调小）/ `STEPS=200` / `ROLLOUT_MAX_NEW=100` / `ROLLOUT_TEMP=0.7`
- `W2_TAU` / `W5_TAU_START` / `E6_MIX`

## 训练日志（排查用）

每 10 步打印一行 + rollout 真实输出，写入 `results/logs/{run}.jsonl`：

```
[E3_W1] step 100/200 ( 50.0%) | loss=0.8123 | KL=0.4512 | c=0.320(0.05~0.88) | len=78 | 1876 tok/s | 步时=2.3s | 已用=3.2m | 剩余≈3.2m
    └ rollout: 学生真实生成文本 (每 10 步记录到日志, 截断 400 字符)
```

## 关键实现说明

1. **sampled-token reverse-KL**：loss 用学生采样轨迹上的 KL 估计 `L ≈ Σ c·[Σ_v p_t(v)(log p_t(v) − log q_t(v))]`（full-vocab，严格对应方案公式）
2. **有效 token mask**：rollout 中 eos 之后的 padding 位置不参与 loss
3. **模型加载**：教师 + tokenizer 全程只加载一次（42 组共享），学生每组从预训练加载；`prepare_data.py` 已预下载全部模型
4. **评测与训练分离**：训练侧零判题，评测侧标准基准报准确率

## 已知限制

- 全 vocab KL 计算量大（每步 16×100×15 万 vocab），DeepScaleR 长题可能 OOM，调小 `BATCH`
- 42 组全参 0.5B 权重约 42GB（`KEEP_MODEL=0` 可在 report 后删除）
- E5 双教师额外占用 ~1GB 显存
