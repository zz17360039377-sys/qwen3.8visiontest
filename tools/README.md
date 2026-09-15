# tools/ 脚本索引

## 性能与评测
| 脚本 | 作用 |
|---|---|
| `bench_vlm.py` | VLM 性能基准（延迟/吞吐/显存/thinking 对比）→ `bench_report.md` |
| `eval_testenv.py` | testenv 运行评分：detect/labelme P·R·F1、pose MPJPE/PCK、cls 准确率+混淆矩阵；**随机抽样与全部失败样例可视化** |
| `eval_pose.py` | VLM pose vs 参考来源（YOLO-pose 模型 / 人工标注）：整帧与 `--crop` 裁剪放大双模式 |

## 数据集构建与维护
| 脚本 | 作用 |
|---|---|
| `collect_test_datasets.py` | 按配置区从全盘原始位置收集已标注数据集 → `testenv/datasets/`（校验+去重） |
| `curate_testenv.py` | 裁剪：全局预算内多目标优先、清晰度优先、类型/数据集轮转 |
| `check_testenv.py` | 数据集体检：配对/标签合法性/损坏图/重复/manifest 一致性 → `dataset_check_report.md` |
| `balance_dataset.py` | 类别平衡：每类上限抽稀（固定种子）→ `overflow/` |
| `normalize_dataset.py` | 规范化：MD5 去重、损坏图隔离、统一扩展名、letterbox 等比缩放（默认 dry-run） |

## 标注与复核
| 脚本 | 作用 |
|---|---|
| `recheck_classify.py` | VLM 复核「文件夹=类别」数据集：一致率/混淆矩阵/分歧清单/自动矫正副本 |
| `demo_annotate.py` | 单图标注接口：**提示词原样透传**，支持 detect/pose/caption/seg 四模式、few-shot 示例图、GT-vs-VLM 对比图 |
| `make_radar_detect.py` | radar-2027 TensorRT 权重 + 录播 → 伪标签检测帧（自动切换 radar 环境） |

## 报告配图
| 脚本 | 作用 |
|---|---|
| `render_report_figures.py` | 应用报告全部图表（准确率柱图/混淆矩阵/叠加图/失败样例） |
| `render_showcase.py` | 实际标注效果 3×3 组合大图 |
