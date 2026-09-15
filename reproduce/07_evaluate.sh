#!/usr/bin/env bash
# 评测任意运行 + 可视化: 指标报告 / 逐匹配明细 / 随机抽样与全部失败样例图
set -e
cd "$(dirname "$0")/.."
RUN=${1:?"用法: 07_evaluate.sh <运行名>   (如 vlm_probe / user_prompts_...)"}
python3 tools/eval_testenv.py --run "$RUN" --vis 12
python3 tools/render_showcase.py                 # 标注效果组合大图
python3 tools/render_report_figures.py           # 报告全部图表
