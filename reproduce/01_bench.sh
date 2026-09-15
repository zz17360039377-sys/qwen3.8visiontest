#!/usr/bin/env bash
# 性能基准: 纯文本/多分辨率/三类任务/thinking 对比 -> bench_*/bench_report.md
set -e
cd "$(dirname "$0")/.."
python3 tools/bench_vlm.py --rounds 5 --outdir "bench_$(date +%Y%m%d_%H%M%S)"
