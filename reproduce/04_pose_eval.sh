#!/usr/bin/env bash
# pose 关键点精度: VLM vs 参考来源(YOLO-pose 模型), 整帧 + --crop 裁剪放大两种模式
set -e
cd "$(dirname "$0")/.."
VIDEO=$(find "$HOME/Desktop/模型训练/CNN/converted_mp4" -name '*.mp4' | head -1)
python3 tools/eval_pose.py --video "$VIDEO" --max-images 12 \
  --ref-model "$HOME/Desktop/模型训练/CNN/yolo_pose_model/armor_pose.pt" \
  --ref-device cpu --out "eval_pose_$(date +%Y%m%d_%H%M%S)"
