# 复现指南

按编号顺序执行，每步产物如下。全部脚本可在本机直接重跑；提示词原文在 `prompts/`。

| 脚本 | 作用 | 产物 |
|---|---|---|
| `00_env.sh` | 启动本地 VLM 服务（llama.cpp + Qwen3.8-27B + mmproj） | :8080 OpenAI 兼容 API |
| `01_bench.sh` | 性能基准（文本/多分辨率/三任务/thinking 对比） | `bench_*/bench_report.md` |
| `02_collect_datasets.sh` | 全盘收集已标注数据集 → 裁剪到 2000 张 → 体检 | `testenv/datasets/` 22 套 + `dataset_check_report.md` |
| `03_recheck_classify.sh` | 分类复核矩阵（baseline vs 最优配置）+ 可选全量矫正 | `recheck_*/recheck_report.md` + csv + 分歧拼图 |
| `04_pose_eval.sh` | pose 关键点精度（VLM vs YOLO-pose 模型） | `eval_pose_*/eval_pose_report.md` |
| `05_delegate_test.sh` | 委托独立 AI 测试代理（打印提示词；在 ZCode 新会话中发给代理） | `testenv/results/vlm_probe/`（代理自建 harness 与报告） |
| `06_user_demos.sh` | 人工提示词演示 ×8（检测 2 / few-shot pose / 描述 3 / 分割 2） | `testenv/results/user_prompts_*/` + vis/ 对比图 |
| `07_evaluate.sh <run>` | 评测任意运行：指标 + 失败可视化 + 全部图表 | `results/<run>/eval_report.md` + `vis/` |
| `08_maintenance.sh` | （可选）类别平衡 / 规范化 | 平衡报告 / normalize_report |
| `run_all.sh` | 顺序执行 00–07（06 除外，交互式） | 全部以上 |

## 前置条件

- RTX 4090 (24G) + CUDA；`~/bigmodel/start-qwen.sh`（Qwen3.8-27B-UD-Q4_K_XL.gguf + mmproj-F16.gguf）
- Python 3.10：`pip install -r requirements.txt`（+ `ultralytics` 用于 pose 参考模型）
- 数据源：`02` 读取各数据集在本机的原始位置（清单见 `tools/collect_test_datasets.py` 配置区）——
  在其他机器复现时需按注释改路径，或直接使用已裁剪的 `testenv/datasets/`

## 提示词存档

`prompts/01–08` 为人工撰写的演示提示词原文（与报告第 11 节一一对应）；
`prompts/delegation.txt` 为委托独立 AI 测试代理的完整提示词（全程记录见 `docs/委托测试记录.md`）。
