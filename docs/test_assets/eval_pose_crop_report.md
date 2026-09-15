# VLM pose 标注精度评估

- 时间：2026-09-07 21:28:07
- 帧：12 张；参考：模型 /home/dreamchaser/Desktop/模型训练/CNN/yolo_pose_model/armor_pose.pt
- 关键点：4 个（top_left, top_right, bottom_right, bottom_left）
- 匹配阈值：IoU ≥ 0.3（贪心）

## 检测层

- 参考实例 13，VLM 实例 8，匹配 8
- 精确率 P = 100.0%，召回率 R = 61.5%
- 匹配框平均 IoU = 0.023

## 关键点层（参考关键点可见处）

- 平均归一化误差 MPJPE/对角线 = **1.259**（28 个点）
- MPJPE = 213.4 px
- PCK@0.05 = 0/28 = 0.0%
- PCK@0.10 = 0/28 = 0.0%
- 速度：2.58 s/帧（23 帧/分钟）

| 关键点 | 样本 | 平均归一化误差 |
|---|---|---|
| 0 top_left | 8 | 0.931 |
| 1 top_right | 7 | 1.460 |
| 2 bottom_right | 6 | 1.441 |
| 3 bottom_left | 7 | 1.487 |

**结论：关键点误差 >10% 框对角线，VLM pose 只能提供粗框；关键点需人工/模型细化。**

## 最差匹配对比（绿=参考，红=VLM）

- `compare_0.jpg`
- `compare_1.jpg`