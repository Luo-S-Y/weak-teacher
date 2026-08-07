# 弱教师信号的可信度加权蒸馏 — 阶段 A 复现代码

> 方案: `weak-teacher-credibility-experiment.md` | 目标环境: **AutoDL 4090 (24GB, CUDA>=12.1)**
> 与方案的差异（按已确认决策调整）: 学生用 **Qwen3-0.7B-Instruct**（方案写 1.5B）; 弱教师池首期用**小规模教师**（Qwen2.5-0.5B 主 / 1.5B 辅助投票），截断教师后置; 数学域 GSM8K→MATH500。

## 实验设计

```
教师池(0.5B/1.5B) ──生成 CoT+logprob──> GSM8K 训练集 7473 条
    │                                      │
    │  E0 恒1.0  E1 自报置信度  E2 自一致性(K=8)     │
    │  E3 规则验证器  E4 师生一致  E5 双师投票  E6 混合 │
    ▼                                      ▼
7 估计器 × 6 机制 = 42 组 (W0 无加权/W1 样本级/W2 阈值τ=0.7/
                           W3 token级/W4 软标签插值/W5 课程排序)
    ▼
TRL SFTTrainer 加权训练 (Qwen3-0.7B 全参, 1 epoch)
    ▼
MATH500 评测 → report.md (基线 E0_W0, 相对增益, top 组合)
```

## 快速开始 (AutoDL)

```bash
cd /root/autodl-tmp/opd          # 上传本目录后
bash setup.sh                    # 权限配置(数据盘软链) + 清华/HF 镜像 + 依赖 + vllm 评测加速 (同环境)
bash run_all.sh                  # 全流程 (预计 ~10h)
# 或分步: 见下方"单步执行"
```

> 部署脚本自动把 `checkpoints/ results/` 软链到数据盘 `/root/autodl-tmp`（42 组全参 checkpoint 约 120GB，系统盘放不下）；`AUTODL_TMP=0` 可禁用。若 torch 已可用（AutoDL base 自带）不会重装，避免破坏 CUDA 匹配；vllm 直接装同一环境（若与旧 torch 冲突 pip 会自动升级 torch）。

## 单步执行

| 命令 | 说明 |
|---|---|
| `python prepare_data.py` | 下载并缓存 GSM8K(train/test) + MATH500 → `data/raw/*.json`（幂等）。`--code`/`--aime`/`--all` 下载阶段 B 用的代码域/AIME24 |
| `python generate_data.py` | 教师生成 CoT + 每 token logprob + 自一致性(K=8) + 学生基线 + 辅助教师投票。`SKIP_SC=1` 跳过自一致性省 ~3h |
| `python estimate.py` | 计算 E0–E6 置信度, 组装 42 组训练权重 |
| `python train.py --all` | 训练 42 组 (缺什么训什么, 可断点续跑)。单组 `python train.py E3_W1` |
| `python eval.py --all` | MATH500 评测 (vLLM, 失败自动回退 transformers) |
| `python report.py` | 汇总报告 → `results/report.md` |

数据流: `prepare_data.py` → `data/raw/*.json` → `generate_data.py` → `data/generated/gsm8k.jsonl` → `estimate.py` → `data/tokenized/base.npz` + `data/weights/*.npz` → `train.py` → `checkpoints/` → `eval.py` → `results/`。

## 配置 (config.py)

- `TEACHER_MAIN` 主弱教师 (默认 0.5B, 极端弱); 换 1.5B 可做教师强度对照
- `SELF_CONSISTENCY_K` 自一致性采样数 (默认 8, 赶时间改 4)
- `W2_TAU` / `W5_TAU` 阈值
- `KEEP_MODEL=1` 保留 checkpoint (默认跑完删除, 42 组全参约 120GB)

## 关键实现说明

1. **加权 loss**: `train.py` 继承 TRL `SFTTrainer` 覆盖 `compute_loss`。W1/W2/W5 样本级权重, W3 用共享的 token 级权重 (教师 logprob 经字符对齐映射到学生 token, 失败回退样本级), W4 软标签 = `c·CE + (1-c)·logV`。
2. **TRL 已 tokenize 数据集**: 数据在 `estimate.py` 一次 tokenize 成 `data/tokenized/base.npz`, 训练时传 `dataset_text_field=None` + `remove_unused_columns=False` 让 TRL 不做二次处理, 并保留权重列。**请勿升级 trl** (接口变动会破坏此路径)。
3. **Qwen3 thinking**: 训练文本与评测 prompt 均手工拼 `enable_thinking=False` 格式, 保证蒸馏目标与评测一致 (不输出 thinking 块)。
4. **E4 on-policy 近似**: 阶段 A 用学生**基座** (未微调) 生成答案与教师比对; 精测阶段 (B) 需按方案做多轮迭代。

## 结果解读

- 相对增益 = (加权 − E0_W0)/E0_W0, 验证 H1 (加权是否优于不加权)
- 每估计器最佳机制对比验证 H2 (规则验证器是否最优)
- 预计 0.5B 教师 GSM8K 正确率 ~15–25%, 大量低置信样本 → 正是加权要检验的场景

## 已知限制

- `eval.py` 每组合独立加载 vLLM 模型 (~30s/次), 42 组评测 overhead ~20min
- 自一致性 + 辅助教师 + 学生基线生成共 ~3h (0.5B/1.5B/0.7B 各 8K 条)
- 若 TRL 0.15.1 处理 tokenize 数据集报错, 反馈后回退方案: 改用 transformers `Trainer` 子类 (loss 逻辑不变)
