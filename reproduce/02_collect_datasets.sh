#!/usr/bin/env bash
# 数据集构建: 全盘收集已标注数据集 -> 裁剪到全局 2000 张(多目标+清晰度优先) -> 体检
# 依赖: 各源数据集在本机的原始位置(见 tools/collect_test_datasets.py 配置区)
set -e
cd "$(dirname "$0")/.."
python3 tools/collect_test_datasets.py            # 收集, 写入 testenv/datasets/
python3 tools/curate_testenv.py --total 2000      # 裁剪: 每大类合计限额
python3 tools/check_testenv.py                    # 体检: 配对/标签/重复/manifest
