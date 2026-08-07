#!/bin/bash
# ============================================================
# AutoDL 4090 一键部署脚本 (opd 实验)
# 功能: 权限配置 + 清华 pip 镜像 + HF 镜像 + 依赖安装 + 环境验证
# 说明: 不初始化 conda, 直接装到当前 python (AutoDL 默认 python 即可)
#
# 用法:
#   bash setup.sh                 # 默认: 训练环境 (trl==0.15.1) + vllm 评测加速
#   PY=/path/to/python bash setup.sh  # 指定解释器
#
# 环境变量:
#   AUTODL_TMP=0  # 禁用数据盘软链 (默认自动把 checkpoints/results 链到 /root/autodl-tmp)
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 自动探测 python (python -> python3), 可用 PY=/path/to/python 覆盖
if [ -n "${PY:-}" ]; then
  :
elif command -v python >/dev/null 2>&1; then
  PY=python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "ERROR: 未找到 python/python3, 请设置 PY=/path/to/python 后重试"
  exit 1
fi

echo "==================== AutoDL 4090 部署 ===================="
echo "工作目录: $SCRIPT_DIR"

# ---------- [0/5] 权限配置 ----------
echo ""
echo "[0/5] 权限配置..."
# 脚本可执行
chmod +x "$SCRIPT_DIR"/*.sh 2>/dev/null || true
echo "  已 chmod +x *.sh"

# 数据目录可写性检查
mkdir -p data/raw data/generated data/tokenized data/weights checkpoints results/logs
touch checkpoints/.write_test && rm checkpoints/.write_test
echo "  数据目录 OK (data/ checkpoints/ results/)"

# 数据盘软链 (42 组 checkpoint 全参约 120GB, 必须放数据盘)
if [ -d /root/autodl-tmp ] && [ "${AUTODL_TMP:-1}" != "0" ]; then
  for d in checkpoints results; do
    target="/root/autodl-tmp/opd_$d"
    if [ ! -e "$target" ]; then mkdir -p "$target"; fi
    if [ ! -L "$d" ]; then
      mv "$d" "$d.bak" 2>/dev/null || true
      ln -s "$target" "$d"
      echo "  $d -> $target (数据盘)"
    fi
  done
  df -h /root/autodl-tmp | tail -1 | awk '{print "  数据盘剩余空间: "$4}'
else
  echo "  未检测到 /root/autodl-tmp, 使用本机磁盘 (注意容量)"
fi

# GPU 检测
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  GPU: /'
else
  echo "  WARNING: 未检测到 nvidia-smi, 请确认在 AutoDL 4090 实例上运行"
fi

# ---------- [1/5] 清华 pip 镜像 ----------
echo ""
echo "[1/5] 配置清华 pip 镜像 (~/.pip/pip.conf)..."
mkdir -p ~/.pip
cat > ~/.pip/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
echo "  已写入 ~/.pip/pip.conf"

# ---------- [2/5] 确认 python + HF 镜像 ----------
echo ""
echo "[2/5] 确认 python + HF 镜像..."
echo "  Python: $("$PY" --version 2>/dev/null || echo '版本信息不可用')"
PY="$(command -v "$PY" 2>/dev/null || echo "$PY")"
echo "  使用: $PY"
grep -q "HF_ENDPOINT" ~/.bashrc 2>/dev/null || echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
export HF_ENDPOINT=https://hf-mirror.com
echo "  HF_ENDPOINT=https://hf-mirror.com"

# ---------- [3/5] 依赖安装 ----------
echo ""
echo "[3/5] 安装依赖 (锁定 torch==2.6.0)..."
# 已装 torch 2.6.0 则跳过, 否则统一装 2.6.0 (PyPI 默认 CUDA 12.4 wheel, 4090 兼容)
if "$PY" -c "import torch; assert torch.__version__ == '2.6.0'" 2>/dev/null; then
  echo "  torch==2.6.0 已就绪, 跳过重装 (仅补装其余依赖)"
  "$PY" -m pip install "transformers>=4.49" datasets accelerate peft "trl==0.15.1" sympy sentencepiece
else
  echo "  安装 torch==2.6.0 + 其余依赖..."
  "$PY" -m pip install torch==2.6.0 "transformers>=4.49" datasets accelerate peft "trl==0.15.1" sympy sentencepiece
fi

# ---------- [4/5] vLLM 评测加速 (锁定 0.8.5, 与训练同环境) ----------
echo ""
echo "[4/5] 安装 vllm==0.8.5 (评测加速, 与训练同环境)..."
if "$PY" -c "import vllm; assert vllm.__version__ == '0.8.5'" 2>/dev/null; then
  echo "  vllm==0.8.5 已安装"
else
  if "$PY" -m pip install "vllm==0.8.5" > /tmp/opd_vllm_install.log 2>&1; then
    echo "  vllm==0.8.5 OK"
  else
    echo "  WARNING: vllm==0.8.5 安装失败, 日志: /tmp/opd_vllm_install.log"
    echo "          评测将自动回退 transformers (eval.py 内置), 不影响训练。"
    echo "          若因 torch 冲突, 请先: pip install torch==2.6.0 再重跑本脚本。"
  fi
fi

# ---------- [5/5] 验证 ----------
echo ""
echo "[5/5] 验证环境..."
"$PY" - <<'PYEOF'
import importlib
mods = ["torch", "transformers", "trl", "datasets", "accelerate", "peft", "sympy"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:<15} {getattr(mod, '__version__', '?')}")
    except ImportError:
        print(f"  {m:<15} 未安装 (!!)")
try:
    import torch
    print(f"  {'torch.cuda':<15} {torch.cuda.is_available()} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
except Exception:
    print("  torch.cuda 不可用 (!!)")
try:
    import vllm
    print(f"  {'vllm':<15} {vllm.__version__}")
except ImportError:
    print(f"  {'vllm':<15} 未安装 (评测走 transformers 回退)")
PYEOF

echo ""
echo "==================== 部署完成 ===================="
echo "下一步: bash run_all.sh   (或分步跑, 见 README.md)"
