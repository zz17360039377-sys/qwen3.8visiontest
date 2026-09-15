#!/usr/bin/env bash
# 委托独立 AI 测试代理: 在 ZCode 中新建会话, 粘贴下方提示词原文。
# 代理会在 testenv/ 沙箱内自己找环境/写 harness/跑推理/自统计;
# 完成后用 08_evaluate.sh 对其运行评分。
set -e
cd "$(dirname "$0")/.."
echo "================ 委托提示词 (发给测试代理) ================"
cat reproduce/prompts/delegation.txt
echo "=========================================================="
echo "代理完成后执行: python3 tools/eval_testenv.py --run <代理的测试名> --vis 12"
echo "参考结果: testenv/results/vlm_probe/  全程记录: docs/委托测试记录.md"
