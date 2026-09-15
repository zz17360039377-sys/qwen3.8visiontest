#!/usr/bin/env bash
# 分类复核矩阵: VLM 当裁判复核"文件夹=类别"数据集
#   baseline  = ×6 放大
#   best      = ×6 + few-shot 参考图 + 自动对比度 (一致率 99.7%)
set -e
cd "$(dirname "$0")/.."
DS="$HOME/Desktop/模型训练/CNN/datasets/train_2000"
REFS="$HOME/Desktop/模型训练/CNN/参考原始分类"
TS=$(date +%Y%m%d_%H%M%S)
python3 tools/recheck_classify.py --dataset "$DS" --sample 50 --upscale 6 \
  --out "recheck_baseline_$TS"
python3 tools/recheck_classify.py --dataset "$DS" --sample 50 --upscale 6 --autocontrast \
  --refs "$REFS" --out "recheck_best_$TS"
# 生成矫正副本(高置信分歧自动改档):
# python3 tools/recheck_classify.py --dataset "$DS" --sample 0 --upscale 6 --autocontrast \
#   --refs "$REFS" --conf-high 0.95 --corrected corrected_full --out recheck_full_$TS
