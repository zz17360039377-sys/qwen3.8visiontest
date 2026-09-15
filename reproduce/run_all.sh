#!/usr/bin/env bash
# 全链路复现: 00 -> 07 (06 为交互式委托, 需在 ZCode 中人工发起)
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
bash "$DIR/00_env.sh"
bash "$DIR/01_bench.sh"
bash "$DIR/02_collect_datasets.sh"
bash "$DIR/03_recheck_classify.sh"
bash "$DIR/04_pose_eval.sh"
bash "$DIR/05_delegate_test.sh"
bash "$DIR/06_user_demos.sh"
for d in testenv/results/*/; do
  n=$(basename "$d")
  [ "$n" = "vis" ] && continue
  bash "$DIR/07_evaluate.sh" "$n" || true
done
