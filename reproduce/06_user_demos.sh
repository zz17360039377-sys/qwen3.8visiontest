#!/usr/bin/env bash
# 用户自写提示词演示: 8 张图 6 个任务 (检测×2 / few-shot pose / 描述×3 / 分割×2)
set -e
cd "$(dirname "$0")/.."
P=reproduce/prompts
RUN="user_prompts_$(date +%Y%m%d_%H%M%S)"

python3 tools/demo_annotate.py --dataset detect_light_luminous \
  --stem light_s0_e18_frame_000751 --mode detect --run "$RUN" --show \
  --hint "The dataset class name for this object is LUMINOUS." \
  --prompt-file "$P/01_luminous_detect.txt"

python3 tools/demo_annotate.py --dataset detect_rm2025_car \
  --stem 0497 --mode detect --run "$RUN" --show \
  --hint "The dataset class name for these robots is car." \
  --prompt-file "$P/02_car_detect.txt"

python3 tools/demo_annotate.py --dataset pose_okbuff2 \
  --stem "hik_20260520_171526_000315+RGB" --mode pose --fewshot 2 --run "$RUN" --show \
  --prompt-file "$P/03_buff_fewshot_pose.txt"

python3 tools/demo_annotate.py --dataset testenv/datasets_extra/demo_caption \
  --stem caption_kitchen --mode caption --run "$RUN" \
  --prompt-file "$P/04_caption_kitchen.txt"

python3 tools/demo_annotate.py --dataset testenv/datasets_extra/demo_caption \
  --stem caption_motorcycle --mode caption --run "$RUN" \
  --prompt-file "$P/05_caption_motorcycle.txt"

python3 tools/demo_annotate.py --dataset testenv/datasets_extra/demo_caption \
  --stem caption_horse --mode caption --run "$RUN" \
  --prompt-file "$P/06_caption_horse.txt"

python3 tools/demo_annotate.py --dataset testenv/datasets_extra/demo_seg \
  --stem seg_motorcycle --mode seg --run "$RUN" \
  --prompt-file "$P/07_seg_motorcycle.txt"

python3 tools/demo_annotate.py --dataset testenv/datasets_extra/demo_seg \
  --stem seg_kitchen --mode seg --run "$RUN" \
  --prompt-file "$P/08_seg_kitchen_person.txt"

echo "评测: python3 tools/eval_testenv.py --run $RUN --vis 12"
