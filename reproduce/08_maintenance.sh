#!/usr/bin/env bash
# 数据维护: 类别平衡 / 规范化(默认 dry-run) — 可选
set -e
cd "$(dirname "$0")/.."
# python3 tools/balance_dataset.py   --dataset <dir> --target 2000 --apply
# python3 tools/normalize_dataset.py --dataset <dir> --dedup --fix-broken --apply
echo "编辑本脚本填入目标数据集后取消注释运行"
