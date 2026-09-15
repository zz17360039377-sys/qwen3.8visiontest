# VLM 标注能力测试报告（run = vlm_probe）

- 日期：2026-09-08
- 环境：RTX 4090 24GB / 32 线程 CPU / 30GB RAM，本机 llama-server
- 全部 harness 代码位于 `testenv/work/`（`run_ds.py` 推理 + `aggregate.py` 统计 + `run_all.sh` 编排）

## 1. Harness 设计

**模型服务发现与启动**
1. `nvidia-smi` 确认 GPU；`ss -tlnp` 探测本机端口（4000/8080/7102 等候选）。
2. 按文件名在用户主目录定位服务脚本：`~/bigmodel/start-qwen.sh`（旁边有 `models/Qwen3.8-27B-UD-Q4_K_XL.gguf` + `mmproj-F16.gguf` 与 `llama-server` 日志），确认是 llama.cpp 服务。未读任何脚本/文档源码，仅凭文件名与运行时信息（`/proc/<pid>/cmdline`、`/v1/models`、`/props`、服务日志 grep）判断。
3. 运行 `start-qwen.sh` 启动服务 → `http://127.0.0.1:8080`（OpenAI 兼容），`/v1/models` 上报 capabilities 含 `multimodal`。

**调用方式**
- 纯手写 `urllib` 调 `POST /v1/chat/completions`：user 消息 = `image_url(base64 data URI)` + `text` 提示词。
- 关键参数：`temperature=0`，`chat_template_kwargs.enable_thinking=false`（否则 Qwen3 思考耗尽 token）。
- 预处理：图片最长边缩到 1280（labelme/wide 大图），小于 256 的图放大到 256（cls 小切片）；JPEG q88。
- 输出解析：正则抽数字 + 范围 clamp + 逐行校验（cls id 越界/坐标非法行丢弃），自动剥离 `<think>` 残留；失败即记 parse 失败，不中断。

**并发模型**
- 客户端 `ThreadPoolExecutor` 3 线程流水发送；服务端 `-np 1` 单 slot，计算实际串行（显存 20.5GB 常驻）。
- 批量以 `nohup` 后台进程运行 + sleep 轮询日志，未阻塞单条命令。

**抽样**
- 每套数据集按 `manifest.csv` 随机（seed=42）抽取：detect 2 套 50 张、pose 2 套 30 张、cls 2 套 50 张、labelme 1 套 12 张，共 **7 套 142 张**，四大类全覆盖。

## 2. 效率数据

| dataset | type | n | parse_ok | parse% | lat_mean_s | lat_med_s | lat_max_s | gen_tok_s | img/min | wall_s | gpu_max_MB | prompt_tok | gen_tok | out_lines |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cls_armor_digit_slices | cls | 25 | 25 | 100% | 1.0 | 1.0 | 1.1 | 2.1 | 176.6 | 8 | 20538 | 3050 | 50 | 25 |
| cls_armor_pattern_public | cls | 25 | 25 | 100% | 1.0 | 1.0 | 1.1 | 2.3 | 171.7 | 9 | 20465 | 3937 | 55 | 25 |
| detect_light_luminous | detect | 20 | 20 | 100% | 4.6 | 4.8 | 4.8 | 6.9 | 37.5 | 32 | 20542 | 29620 | 588 | 20 |
| detect_rm2025_armor | detect | 30 | 30 | 100% | 9.0 | 5.6 | 19.2 | 11.6 | 19.4 | 93 | 20537 | 29447 | 3213 | 113 |
| labelme_radar_car | labelme | 12 | 12 | 100% | 16.8 | 16.5 | 24.2 | 15.3 | 9.8 | 74 | 20536 | 12628 | 2866 | 93 |
| pose_droneok | pose | 15 | 15 | 100% | 19.7 | 17.8 | 38.0 | 15.3 | 7.9 | 114 | 20533 | 23895 | 4392 | 24 |
| pose_okbuff2 | pose | 15 | 15 | 100% | 15.7 | 14.9 | 26.1 | 14.5 | 10.6 | 85 | 20534 | 22965 | 3115 | 30 |

- **总吞吐 20.5 张/分钟**（142 张 / 415s 服务端墙钟）；生成速度 2~15 tok/s（输出越长越快摊薄视觉编码开销）。
- **显存**：模型常驻 ~20.5GB（峰值 20542MB），全程平稳；解析成功率 142/142 = **100%**。
- 延迟随图片 token 数与目标数上升：cls 1s → detect 5-9s → pose/labelme 16-20s。

## 3. 遇到的问题与解决

| 问题 | 现象 | 解决 |
|---|---|---|
| 思考模式吃掉全部输出 | 首测 completion=900（截断）、content 为空 | `chat_template_kwargs.enable_thinking=false`，并在解析端兜底剥离 `<think>` |
| 非思考模式下输出退化为全图大框 | `0 0.49 0.50 0.99 0.99`，1s 内作答 | 换 few-shot 式提示词（示例行 + “禁止整图框 + 物体占宽 2-40%”），小框定位明显改善 |
| manifest 列序不一致 | `cls_*` 的 manifest 是 `class,image`，按首列取图全部 FileNotFoundError | 取图列自动探测（校验文件存在） |
| harness 自身两处 NameError（pose 分支变量名、示例行生成器） | pose 任务首轮崩溃 | 修复后重跑，cls/pose 全部补齐 |
| 服务端 vision 是否生效存疑 | prompt_tokens 偏小疑似丢图 | 用合成图（红底蓝圆）做 grounding 测试确认视觉链路正常，token 数与分辨率正相关属正常 |

## 4. 质量观察（粗看，精确指标以 eval 为准）

- cls（armor_pattern）：多数能答对 B1/B2/B3 等图案类；digit_slices 40×56 小切片上数字混淆较多。
- detect：能定位多数装甲板/灯条，类别 id（c0/c1/c2 无语义名）只能靠猜，类别命中受限。
- pose：输出结构 100% 合规（框 + 4/5 关键点 + v），角点大体落在面板上，精度待 PCK 评测。
- labelme：雷达高视角小车全部以 `Car` 名义输出（未区分 robotcar），单图召回偏低（GT 平均数十框/图，模型只报 3-10 框）。

## 5. 产出物

- 预测：`results/vlm_probe/<dataset>/labels/`（142 个文件，格式与数据集 labels 同构）
- 提示词与参数：`results/vlm_probe/<dataset>/notes.md`
- 明细统计：`work/logs/*.jsonl`（逐张耗时/token）、`work/logs/efficiency_table.md`
