# VLM 性能基准报告

- 时间：2026-09-07 21:33:07
- 服务：http://127.0.0.1:8080/v1  模型：/home/dreamchaser/bigmodel/models/Qwen3.8-27B-UD-Q4_K_XL.gguf
- GPU 显存占用：20937 MiB
- 轮数：5（thinking 场景固定 2 轮）
- 切片来源：/home/dreamchaser/Desktop/模型训练/CNN/datasets/train_2000（7 张，每类 1 张，seed=0）
- 整帧来源：无（5 帧，max_side=1280）

| 场景 | 轮数 | 请求成功 | JSON解析 | 平均耗时(s) | p50(s) | p95(s) | 生成 tok/s | prompt tok/s | prompt_n(≈文本+视觉token) | 标签一致 |
|---|---|---|---|---|---|---|---|---|---|---|
| text_baseline | 5 | 5 | 5 | 0.40 | 0.36 | 0.59 | 43.6 | 233.6 | 27 | - |
| recheck_x3 | 5 | 5 | 5 | 0.81 | 0.84 | 0.86 | 45.3 | 838.9 | 235 | 4/5 |
| recheck_x6 | 5 | 5 | 5 | 0.86 | 0.86 | 0.88 | 45.4 | 987.2 | 297 | 4/5 |
| recheck_x9 | 5 | 5 | 5 | 0.91 | 0.91 | 0.92 | 45.2 | 1156.1 | 385 | 4/5 |
| detect_frame | 5 | 5 | 5 | 1.24 | 1.25 | 1.27 | 43.9 | 1763.9 | 1514 | - |
| pose_frame | 5 | 5 | 5 | 2.51 | 3.27 | 3.35 | 44.7 | 1875.5 | 1692 | - |
| recheck_x6_thinking | 2 | 2 | 0 | 2.64 | 2.61 | 2.66 | 46.0 | 1141.4 | 363 | - |
| detect_frame_thinking | 2 | 2 | 0 | 17.79 | 17.77 | 17.81 | 46.0 | 1807.6 | 1586 | - |

## 吞吐结论

- `recheck_x3`：73.8 张/分钟；1.4 万张 ≈ 3.2 小时
- `recheck_x6`：69.8 张/分钟；1.4 万张 ≈ 3.3 小时
- `recheck_x9`：66.2 张/分钟；1.4 万张 ≈ 3.5 小时
- `detect_frame`：48.2 张/分钟
- `pose_frame`：23.9 张/分钟

## 说明

- `prompt_n` 含文本 prompt 与视觉 token；与 `text_baseline` 的差值 ≈ 该分辨率下的视觉 token 数。
- llama.cpp 对相同前缀有 KV 缓存，第 1 轮偏慢属正常；不同切片/帧之间不共享图像 token。
- 原始逐轮数据见 `bench_results.csv`。
