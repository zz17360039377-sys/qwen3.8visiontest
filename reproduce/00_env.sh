#!/usr/bin/env bash
# 启动本地 VLM 服务（Qwen3.8-27B + mmproj 视觉头, llama.cpp CUDA）
# 前置: RTX 4090 (24G), ~/bigmodel/start-qwen.sh 已配置
set -e
bash "$HOME/bigmodel/start-qwen.sh"
curl -s -m 5 http://127.0.0.1:8080/health && echo " <- VLM ready"
