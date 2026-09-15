# selflabel — 本地 Qwen3.8-27B VLM 全自动标注 + 数据集复核矫正

用本地部署的多模态大模型（**Qwen3.8-27B**，llama.cpp CUDA，`~/bigmodel/start-qwen.sh` 一键启动）做：

1. **分类数据复核矫正（主线）**：对「文件夹=类别」的既有数据集做 VLM 逐张复核，输出一致率/混淆矩阵/分歧清单，经网页审查或自动矫正生成修正副本（**原数据集只读**）；
2. **pose 标注精度测试**：VLM 关键点 vs 参考来源（YOLO-pose 模型 / 人工标注）的量化对比；
3. **通用自动标注**：detect / pose / classify 三种任务，YOLO 与 COCO 两种导出格式。

完整测试过程、图片与数据分析见 **[docs/测试报告.md](docs/测试报告.md)**；给 VLM 做提示词测试的封闭环境见 **[testenv/限制.md](testenv/限制.md)**。

**应用报告（ModelScope 分享）**：[docs/应用报告.md](docs/应用报告.md) —— pose 关键点 + 批量分类的自主实测，配图在 [docs/pic/](docs/pic/)；委托独立测试代理的全过程记录见 [docs/委托测试记录.md](docs/委托测试记录.md)。

## 复现

全部实验按编号脚本一键复现（环境 → 基准 → 数据集构建 → 复核矩阵 → pose 评测 → 委托测试 → 演示 → 评测出图），
提示词原文存档在 `reproduce/prompts/`。详见 **[reproduce/README.md](reproduce/README.md)**。

```bash
bash reproduce/run_all.sh        # 全链路（约 8–10 小时，含全量推理）
bash reproduce/00_env.sh         # 或分步执行
bash reproduce/01_bench.sh
...
```

## 目录结构

```
selflabel/
├── core/                # 通用核心：vlm_client（llama.cpp OpenAI 兼容客户端）、prompts、
│                        #   parse_geometry（防御解析+NMS）、draw（叠加图）、io_utils
├── export/              # 导出层：exporters 注册表 + fmt_yolo / fmt_coco（新格式=加一个文件注册一行）
├── apps/                # 入口应用：selflabel.py（标注 CLI）、webapp.py（Web UI，stdlib）
├── tools/               # 工具 CLI：recheck_classify（复核矫正）、eval_pose（pose 精度）、
│                        #   bench_vlm（性能基准）、balance_dataset、normalize_dataset、
│                        #   collect_test_datasets（收集成品数据集）、make_radar_detect（雷达权重+录播自制帧）
├── static/              # Web 前端（vanilla JS）
├── docs/                # 测试报告 + 测试图片归档
├── testenv/             # ★ VLM 测试环境：datasets（只读数据集）+ 限制.md（规则）+ results/
├── config.example.yaml  # 任务配置模板（classes/keypoints/skeleton，全 per-task）
├── README.md / requirements.txt / .gitignore
```

## 快速开始

```bash
~/bigmodel/start-qwen.sh                     # 1. 拉起 VLM 服务 (:8080)

# 2a. 复核既有分类数据集（数字类别名自动注入装甲领域提示词）
python3 tools/recheck_classify.py \
  --dataset ~/Desktop/模型训练/CNN/datasets/train_2000 \
  --sample 50 --upscale 6 --autocontrast \
  --refs ~/Desktop/模型训练/CNN/参考原始分类 \
  --out recheck_run1          # 加 --corrected <dir> 生成矫正副本

# 2b. 网页审查 / 矫正
python3 apps/webapp.py --port 8089 --root ~/Desktop/模型训练/CNN/datasets
#   「审查」页：缩略图网格 + VLM 建议角标 + 批量移动/删除/采纳 + 后台复核任务

# 3. pose 标注精度测试
python3 tools/eval_pose.py --video <整帧视频> --max-images 12 \
  --ref-model ~/Desktop/模型训练/CNN/yolo_pose_model/armor_pose.pt   # 或 --crop / --ref-labels

# 4. 通用标注 CLI
python3 apps/selflabel.py --mode detect --images <dir> --classes "red_armor=...;blue_armor=..." --format coco
python3 apps/selflabel.py --mode pose   --config config.example.yaml --video <mp4>
python3 apps/selflabel.py --mode classify --images <切片dir> --upscale 6 --conf-min 0.3

# 5. 数据集平衡 / 规范化（默认 dry-run，--apply 才写盘）
python3 tools/balance_dataset.py  --dataset <dir> --target 2000
python3 tools/normalize_dataset.py --dataset <dir> --dedup --fix-broken --resize 40x56 --grayscale

# 6. 性能基准 / 测试集收集 / 雷达自制帧
python3 tools/bench_vlm.py --rounds 5
python3 tools/collect_test_datasets.py     # 按脚本顶部配置区收集，写入 testenv/datasets/
python3 tools/make_radar_detect.py         # 自动切换 radar-2027 环境，TensorRT 权重+录播 → 伪标签帧
```

## 复核矫正工作流（对应 CNN 流水线）

| 环节 | CNN 原流程 | selflabel 对应 |
|---|---|---|
| 自动归类 | `classify_dataset.py`（旧 onnx 模型） | `apps/selflabel.py --mode classify`（VLM） |
| 一致率复核 | `eval_predictions.py`（新模型当裁判） | `tools/recheck_classify.py`（VLM 当裁判） |
| 人工审查 | `dataset_reviewer.py` 网页 | `apps/webapp.py`「审查」页（VLM 建议角标 + 一键采纳 + 后台复核） |
| 类别平衡 | `supplement_dataset.py` | `tools/balance_dataset.py`（抽稀/overflow） |
| 规范化 | 各脚本内 letterbox 铁律 | `tools/normalize_dataset.py`（去重/损坏图/等比缩放） |

### 复核配置矩阵实测（详细过程与分析见 [docs/测试报告.md](docs/测试报告.md)）

| 配置 | 一致率 | 速度 | 1.4 万张全量 |
|---|---|---|---|
| ×6 放大 | 93.7% | 0.90 s/张 | ≈3.5 h |
| ×3 放大 | 94.3% | 0.89 s/张 | ≈3.5 h |
| ×3 + 自动对比度 | 95.7% | 0.85 s/张 | ≈3.3 h |
| ×6 + 参考图 few-shot | 99.4% | 1.49 s/张 | ≈5.8 h |
| **×6 + 参考图 + 自动对比度** | **99.7%** | 1.42 s/张 | ≈5.5 h |
| 最优配置 @700 张换种子复测 | 99.0% | 1.42 s/张 | 稳定 ✅ |
| 最优配置 @未清洗数据集 | 97.7% | 1.42 s/张 | 发现系统性错标 |

要点：数字类 1–4 在所有配置下都 ≈100%；难点全在图标类 0（哨兵）/6（前哨）/7（基地）的暗光切片——**注入 `参考原始分类/*.png` 作 few-shot 参考图后混淆几乎清零（0 类 68%→98%，6/7 类→100%）**。55 张分歧样本逐张人工核验的结论：**自动改档阈值必须 ≥0.95（或图标类一律人工确认）**——0.8 会被图标类高置信错误分歧穿透；对未清洗数据集复核收益最大。55 张分歧原图与增亮拼图归档在 `docs/test_assets/`。

## 输出目录结构

```
out_detect_*/  out_pose_*/        # selflabel.py detect/pose
  labels/*.txt   classes.txt      #   YOLO（或 annotations/instances_*.json COCO）
  images/                         #   原图副本（训练可直接用）
  previews/*_annot.jpg            #   叠加预览
out_classify_*/                   # selflabel.py classify
  <类别名>/*.jpg  _lowconf/  classify_summary.csv
recheck_*/  recheck_web_*/        # 复核输出
  recheck_report.md  recheck_results.csv  disagreement_N.jpg
eval_pose_*/                      # pose 精度评估
  eval_pose_report.md  eval_pose_details.csv  compare_N.jpg
```

## 扩展指南

* **加导出格式**：在 `export/` 新建 `fmt_xxx.py`，实现 `write(out_dir, records, classes, keypoints, skeleton, copy_images) -> manifest`，在 `export/exporters.py` 底部注册一行（参考 `fmt_coco.py`）。
* **换类别/任务**：全部由 `--config` YAML（classes/keypoints/skeleton，每项带喂给模型的 description）或 `--classes` 内联参数决定，代码零改动。
* **换 VLM 后端**：任何 OpenAI 兼容接口（`--api`）；多图 few-shot 需后端支持。

## 依赖

`requests`、`PyYAML`、`opencv-python`、`Pillow`、`numpy`（系统 Python 3.10 已满足）；`tools/eval_pose.py --ref-model` 另需 `ultralytics`；`tools/make_radar_detect.py` 自动借用 radar-2027 的环境。Web 为纯标准库实现，无 fastapi/uvicorn。

## 已知边界

* 40×56 暗光切片上 0/6/7 图标类存在视觉歧义，分歧样本必须人工抽查（`_review/` 与审查页即为此设计）。
* **pose 精度实测（`docs/测试报告.md`）**：在现有暗光手机翻拍素材上 VLM pose 不可用（MPJPE/对角线 ≈1.26、PCK@0.1=0%），pose 预标注建议继续用 YOLO-pose 模型；`tools/eval_pose.py` 已就绪，拿到机器人实拍帧可直接复测。
* 后台复核任务同一时间仅允许一个（本地模型单槽 `-np 1`）。
