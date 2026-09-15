# VLM 性能基准报告

- 时间：2026-09-07 20:40:52
- 服务：http://127.0.0.1:8080/v1  模型：/home/dreamchaser/bigmodel/models/Qwen3.8-27B-UD-Q4_K_XL.gguf
- GPU 显存占用：20409 MiB
- 轮数：5（thinking 场景固定 2 轮）
- 切片来源：/home/dreamchaser/Desktop/模型训练/CNN/datasets/train_2000（7 张，每类 1 张，seed=0）
- 整帧来源：无（5 帧，max_side=1280）

| 场景 | 轮数 | 请求成功 | JSON解析 | 平均耗时(s) | p50(s) | p95(s) | 生成 tok/s | prompt tok/s | prompt_n(≈文本+视觉token) | 标签一致 |
|---|---|---|---|---|---|---|---|---|---|---|
| text_baseline | 5 | 5 | 5 | 0.37 | 0.35 | 0.42 | 40.3 | 231.3 | 27 | - |
| recheck_x3 | 5 | 5 | 5 | 0.82 | 0.84 | 0.85 | 42.0 | 817.6 | 235 | 4/5 |
| recheck_x6 | 5 | 5 | 5 | 0.83 | 0.86 | 0.89 | 42.3 | 793.6 | 238 | 4/5 |
| recheck_x9 | 5 | 5 | 5 | 0.92 | 0.92 | 0.93 | 42.1 | 1133.7 | 385 | 4/5 |
| detect_frame | 5 | 5 | 5 | 1.31 | 1.31 | 1.33 | 40.7 | 1666.4 | 1514 | - |
| pose_frame | 5 | 5 | 5 | 2.68 | 3.49 | 3.62 | 41.8 | 1756.5 | 1692 | - |
| recheck_x6_thinking | 2 | 2 | 0 | 2.83 | 2.79 | 2.86 | 42.1 | 1098.0 | 363 | - |
| detect_frame_thinking | 2 | 2 | 2 | 13.97 | 13.61 | 14.33 | 42.5 | 1715.7 | 1586 | - |

## 吞吐结论

- `recheck_x3`：73.2 张/分钟；1.4 万张 ≈ 3.2 小时
- `recheck_x6`：72.1 张/分钟；1.4 万张 ≈ 3.2 小时
- `recheck_x9`：65.3 张/分钟；1.4 万张 ≈ 3.6 小时
- `detect_frame`：45.9 张/分钟
- `pose_frame`：22.3 张/分钟

## 说明

- `prompt_n` 含文本 prompt 与视觉 token；与 `text_baseline` 的差值 ≈ 该分辨率下的视觉 token 数。
- llama.cpp 对相同前缀有 KV 缓存，第 1 轮偏慢属正常；不同切片/帧之间不共享图像 token。
- 原始逐轮数据见 `bench_results.csv`。
