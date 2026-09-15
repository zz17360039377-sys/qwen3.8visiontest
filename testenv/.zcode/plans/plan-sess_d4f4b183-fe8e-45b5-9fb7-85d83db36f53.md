# VLM 标注能力测试执行计划（testenv / selflabel_v1）

## 方案结论（已确认）
- **标注引擎**：本机 llama-server（127.0.0.1:8080，加载 Qwen3.8-27B + mmproj 视觉投影，即驱动我的同一本地模型），通过 OpenAI 兼容 API 调用。不接触任何外部服务、不读取项目外文件。
- **规模**：22 个数据集全量，约 2000 张图。
- **评测边界**：不运行 tools/eval_testenv.py 等任何外部脚本；全程不读 datasets/labels/ 标准答案（测试独立性）；质量分不计算，产出物为**全量标注文件 + 完整效率报告**（延迟 / tokens/s / 吞吐 / 进程与 GPU 内存）。
- **关键点约定**：N=4 按您给的「左上、左下、右下、右上」；N=5 = 四角（同序）+中心；N=12 = 沿目标轮廓按 左上→左下→右下→右上 方向均匀取 12 点（角点落在第 1/4/7/10 点）。全部写入 notes.md。
- **代码归属**：全部由我手写，保存在 `testenv/results/selflabel_v1/code/`（符合"保存到当前文件夹"且遵守 限制.md 的输出隔离：数据集目录零写入）。

## 目录与产物
```
testenv/results/selflabel_v1/
├── code/            # 我手写的全部代码
│   ├── config.py        # 路径/API/每数据集参数/缩放策略
│   ├── prompts.py       # 四类任务提示词（含类别表、格式规范、关键点约定、/no_think）
│   ├── preprocess.py    # PIL 预处理：大图降采样(≤2048，密集labelme≤2560)、小图放大×8、JPEG
│   ├── vlm_client.py    # OpenAI 兼容客户端：超时/重试/健康检查/抓取 llama.cpp metrics
│   ├── parsers.py       # 四类输出稳健解析+修复（剥markdown、逗号修复、坐标钳制、v值补齐）
│   ├── metrics.py       # 每图计时、psutil 采服务端RSS、nvidia-smi 采GPU、吞吐聚合
│   ├── report.py        # 生成 metrics_report.md / metrics.json / per_image_log.csv
│   └── harness.py       # 主入口：--smoke / 全量 / --datasets 子集 / 断点续跑
├── <22个数据集>/labels/<stem>.txt   # 每张图一个文件，格式与 限制.md 规范完全一致
├── notes.md             # 提示词全文、参数、预处理策略、关键点约定、局限性
├── per_image_log.csv    # 每图一行：耗时/令牌数/速度/输出行数/解析状态
├── metrics.json         # 原始聚合数据
└── metrics_report.md    # 中文效率总报告
```

## 标注流水线（每张图）
1. 预处理 → base64 → POST /v1/chat/completions（temperature=0, seed=42, 按数据集设 max_tokens）
2. 提示词严格按类型生成：cls（只回类名）/ detect 与 labelme（每行 `class cx cy w h`，归一化[0,1]）/ pose（`class cx cy w h` + N×(x y v)）
3. 稳健解析 → 写 `results/selflabel_v1/<ds>/labels/<stem>.txt`；空文件=无目标（合法）
4. 记录：e2e 墙钟耗时、llama.cpp 返回的 prefill/decode 令牌数与速度、重试次数、解析有效/失败行数
5. 每 10 张采一次服务端进程 RSS（psutil），每 15 张采一次 GPU 显存/利用率（nvidia-smi）

## 统计指标（效率/内存/吞吐）
- 每图：e2e 延迟、模型侧 prefill tok/s 与 decode tok/s、prompt/completion tokens、有效输出行数
- 每数据集 + 全局：images/小时、延迟 p50/p95、成功率、解析失败率
- 内存：llama-server 进程 RSS 均值/峰值、GPU 显存均值/峰值、GPU 利用率、harness 自身 RSS
- 输出：metrics_report.md（中文汇总表）+ metrics.json + per_image_log.csv

## 轻量加速（安全、不改变测试性质）
- 每数据集单独 max_tokens（cls=32、单类detect=512、pose=2048、密集detect/labelme=4096）
- 提示词带 /no_think 与"只输出标注行"约束，抑制冗长推理（首图验证生效，否则回退）
- 超大雷达图(5496×3672、11580×11567)降采样至 2048/2560 长边（坐标归一化不受影响）；cls 11~40px 小切片放大 8× 再送模型
- 断点续跑：已有输出的图直接跳过；API 不可用时指数退避等待并自动恢复

## 执行步骤
1. 创建 `results/selflabel_v1/` 结构，手写全部 8 个代码文件
2. **冒烟**：4 张图（cls/detect/pose/labelme 各 1）验证 API 视觉、格式、解析、计时链路；我本人用 Read 亲自看这几张图并核对 API 输出质量
3. 修复问题后**后台启动全量**（约 2000 张，预计 8~22 小时；GPU 与我的会话共享，属预期）
4. 运行期间定期检查进度与服务健康；结束时 harness 自动生成效率报告
5. 我补写 notes.md，向您汇报最终产物与关键效率数字

## 风险与对策
- 单槽服务排队：e2e 含排队等待，报告中同时给出模型侧纯推理时长，口径分开
- llama-server 被重启：健康检查 + 退避重试 + 续跑，不丢已完成结果
- 密集集（单帧 76~162 目标）降采样后小目标可能漏检：作为能力边界如实记录在 notes.md
- 长时间后台运行：状态全部落文件（日志/CSV/输出），会话上下文压缩不影响续跑