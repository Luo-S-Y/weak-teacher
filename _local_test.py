#!/usr/bin/env python3
"""本地逻辑验证 (CPU, 不依赖 GPU): v2 估计器 E0-E6 + 加权机制 W0-W5
运行: /path/with/torch/python _local_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
import config as C
from train import estimate_c, weighted_loss

B, M, V = 3, 6, 64
logits_s = torch.randn(B, M, V, requires_grad=True)
logits_t = torch.randn(B, M, V)
log_p = F.log_softmax(logits_s, dim=-1); p = log_p.exp()
log_q = F.log_softmax(logits_t, dim=-1); q = log_q.exp()
sampled = torch.randint(0, V, (B, M))
log_q_sampled = log_q.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
mask = torch.ones(B, M, dtype=torch.bool)

print("== 估计器 ==")
for e in ["E0", "E1", "E2", "E3", "E4", "E6"]:
    c = estimate_c(e, p, log_p, q, log_q, log_q_sampled, V)
    assert c.shape == (B, M) and c.min() >= 0 and c.max() <= 1.01, (e, c.shape, c.min(), c.max())
    print(f"  {e}: mean={c.mean():.3f} min={c.min():.3f} max={c.max():.3f}")

# E1 单调性: log_q_sampled 高 -> c 高
c1 = estimate_c("E1", p, log_p, q, log_q, log_q_sampled, V)
assert torch.allclose(c1, log_q_sampled.exp().clamp(0, 1))
# E2 尖锐 vs 平坦: 构造一个尖锐教师分布
q_sharp = torch.zeros_like(q)
q_sharp[0, 0, 0] = 1.0
c_sharp = estimate_c("E2", p, log_p, q_sharp, q_sharp.log().clamp(min=-30), torch.zeros(B, M), V)
assert c_sharp[0, 0] > 0.99
print("== 估计器逻辑 OK ==")

print("== 加权机制 (梯度回传) ==")
for w in ["W0", "W1", "W2", "W3", "W4", "W5"]:
    logits_s.grad = None
    log_p = F.log_softmax(logits_s, dim=-1); p = log_p.exp()   # 每轮重建图 (对应真实训练每步前向)
    c = estimate_c("E2", p, log_p, q, log_q, log_q_sampled, V)
    loss = weighted_loss(w, c, p, log_p, log_q, step=0, mask=mask)
    assert loss.dim() == 0 and torch.isfinite(loss), (w, loss)
    loss.backward()
    assert logits_s.grad is not None and torch.isfinite(logits_s.grad).all(), w
    print(f"  {w}: loss={loss.item():.3f} grad_norm={logits_s.grad.norm():.3f}")

# W4 分布插值: c=0 时目标=uniform, loss 与均匀分布 KL 一致
log_p = F.log_softmax(logits_s.detach(), dim=-1); p = log_p.exp()
c0 = torch.zeros(B, M)
l4 = weighted_loss("W4", c0, p, log_p, log_q, 0, mask)
target_u = torch.full_like(q, 1.0 / V)
kl_uniform = (p * (log_p - target_u.clamp(min=1e-12).log())).sum(-1).mean()
assert abs(l4.item() - kl_uniform.item()) < 1e-3, (l4.item(), kl_uniform.item())
print("== 加权机制 OK ==")

# 42 组合
assert len(C.all_runs()) == 42
print("ALL LOCAL TESTS PASSED")
