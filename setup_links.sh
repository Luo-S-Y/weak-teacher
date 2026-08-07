#!/bin/bash
# 数据盘软链脚本: 把 checkpoints/results 链接到数据盘 (42 组全参约 120GB, 系统盘放不下)
# 用法:
#   bash setup_links.sh                     # 默认数据盘 /root/autodl-tmp (AutoDL 标准路径)
#   bash setup_links.sh /path/to/data_disk  # 指定数据盘路径
#   AUTODL_TMP=0 bash setup_links.sh        # 跳过 (保留本机目录)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ "${AUTODL_TMP:-1}" = "0" ]; then
  echo "AUTODL_TMP=0, 跳过软链, 使用本机目录"
  exit 0
fi

DATA_DISK="${1:-/root/autodl-tmp}"
if [ ! -d "$DATA_DISK" ]; then
  echo "ERROR: 数据盘 $DATA_DISK 不存在"
  echo "       请在 AutoDL 控制台确认数据盘挂载, 或手动指定: bash setup_links.sh /path/to/disk"
  exit 1
fi

echo "==================== 数据盘软链 ===================="
for d in checkpoints results; do
  target="$DATA_DISK/opd_$d"

  # 已有真实目录: 迁移旧数据到数据盘 (cp 成功才删除原目录)
  if [ -d "$d" ] && [ ! -L "$d" ]; then
    echo "  迁移 $d/ 到 $target/ ..."
    mkdir -p "$target"
    if cp -rn "$d"/. "$target"/ 2>/dev/null; then
      rm -rf "$d"
    else
      echo "  WARNING: 迁移 $d 失败, 保留原目录, 跳过软链"
      continue
    fi
  fi

  mkdir -p "$target"
  rm -rf "$d"          # 清除失效/旧软链 (真实目录已在迁移中处理)
  ln -s "$target" "$d"
  echo "  $d -> $target"
done

df -h "$DATA_DISK" | tail -1 | awk '{print "数据盘剩余: "$4}'
echo "完成: checkpoints/ 与 results/ 已链接到 $DATA_DISK"
