#!/usr/bin/env python3
"""render_report_figures -- build the article figures for the application report.

Reads the vlm_probe run (predictions) + testenv datasets (ground truth) and
renders into docs/pic/:
  fig_cls_accuracy.png        classification accuracy bars
  fig_confusion_pattern.png   confusion matrix, armor-pattern set
  fig_confusion_digit.png     confusion matrix, digit-slice set
  fig_luminous_overlay_*.jpg  best detect case: GT (green) vs VLM (red)
  fig_pose_compare_*.jpg      pose case: GT (green) vs VLM (red) keypoints
  fig_cls_fail_*.jpg          misclassified slices with GT/VLM labels

Run:  python3 tools/render_report_figures.py
"""

from __future__ import annotations

import os
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, ROOT)

RUN = os.path.join(ROOT, "testenv", "results", "vlm_probe")
DS = os.path.join(ROOT, "testenv", "datasets")
OUT = os.path.join(ROOT, "docs", "pic")
os.makedirs(OUT, exist_ok=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tools.eval_testenv import (cls_token, find_image_path, greedy_match,  # noqa: E402
                                load_classes, load_yolo)


def confusion(ds_name):
    """Return (labels, matrix, n_ok, n_tot) for a cls dataset+run pair."""
    classes = load_classes(os.path.join(DS, ds_name))
    gt_dir = os.path.join(DS, ds_name, "labels")
    pred_dir = os.path.join(RUN, ds_name, "labels")
    pairs = []
    for f in sorted(os.listdir(gt_dir)):
        stem = os.path.splitext(f)[0]
        gt = open(os.path.join(gt_dir, f)).read().strip()
        pp = os.path.join(pred_dir, stem + ".txt")
        if not os.path.isfile(pp):
            continue
        pred = cls_token(open(pp).read().strip(), classes) or "?"
        pairs.append((gt, pred))
    labels = sorted({g for g, _ in pairs} | {p for _, p in pairs})
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for g, p in pairs:
        m[idx[g], idx[p]] += 1
    ok = sum(m[i][i] for i in range(len(labels)))
    return labels, m, ok, len(pairs)


def fig_confusion(ds_name, out_name, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, m, ok, tot = confusion(ds_name)
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(labels)), max(5, 0.5 * len(labels))))
    im = ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("VLM prediction")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"{title}\naccuracy {ok}/{tot} = {ok / max(1, tot):.0%}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if m[i, j]:
                ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if m[i, j] > m.max() / 2 else "black")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    p = os.path.join(OUT, out_name)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("saved", p)
    return ok, tot


def fig_cls_accuracy(accs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["armor pattern\n(public)", "armor digit\n(dark slices)"]
    vals = [accs[0], accs[1]]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(names, [v * 100 for v in vals], color=["#4c7ef3", "#e5484d"], width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.5, f"{v:.0%}",
                ha="center", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_ylabel("top-1 accuracy (%)")
    ax.set_title("VLM batch classification accuracy\n(Qwen3.8-27B, 25 samples each)")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_cls_accuracy.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("saved", p)


GT = (0, 200, 0)
PRED = (0, 0, 255)


def _load(ds, stem, cell_w=520):
    p = find_image_path(os.path.join(DS, ds), stem)
    img = cv2.imread(p)
    h, w = img.shape[:2]
    if w != cell_w:
        interp = cv2.INTER_NEAREST if w < cell_w else cv2.INTER_AREA
        img = cv2.resize(img, (cell_w, max(1, int(round(h * cell_w / w)))), interpolation=interp)
    return img


def fig_luminous_overlays(n=3):
    """Best detect case: GT green vs VLM red, near-perfect matches."""
    ds, preds = "detect_light_luminous", os.path.join(RUN, "detect_light_luminous", "labels")
    classes = load_classes(os.path.join(DS, ds))
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(preds))
    import random
    rng = random.Random(7)
    cells = []
    for stem in rng.sample(stems, min(n, len(stems))):
        gt = load_yolo(os.path.join(DS, ds, "labels", stem + ".txt"), classes, want_tokens=5)
        pr = load_yolo(os.path.join(preds, stem + ".txt"), classes, want_tokens=5)
        img = _load(ds, stem)
        H, W = img.shape[:2]
        for _c, v, _t in gt:
            cv2.rectangle(img, (int(v[0] * W), int(v[1] * H)), (int(v[2] * W), int(v[3] * H)), GT, 2)
        for _c, v, _t in pr:
            cv2.rectangle(img, (int(v[0] * W), int(v[1] * H)), (int(v[2] * W), int(v[3] * H)), PRED, 2)
        cv2.putText(img, "green=GT  red=VLM", (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(img)
    grid = np.hstack(cells)
    p = os.path.join(OUT, "fig_luminous_overlay.jpg")
    cv2.imwrite(p, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("saved", p)


def fig_pose_compare(n=2):
    """Pose case: GT green vs VLM red keypoints (5-kpt buff set)."""
    ds, preds = "pose_okbuff2", os.path.join(RUN, "pose_okbuff2", "labels")
    classes = load_classes(os.path.join(DS, ds))
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(preds))
    import random
    rng = random.Random(3)
    cells = []
    for stem in rng.sample(stems, min(n, len(stems))):
        gt = load_yolo(os.path.join(DS, ds, "labels", stem + ".txt"), classes, want_tokens=17)
        pr = load_yolo(os.path.join(preds, stem + ".txt"), classes, want_tokens=17)
        img = _load(ds, stem, 640)
        H, W = img.shape[:2]

        def draw(rows, color):
            for _c, v in rows:
                p1 = (int(v[0] * W), int(v[1] * H))
                p2 = (int(v[2] * W), int(v[3] * H))
                cv2.rectangle(img, p1, p2, color, 2)
                pts = [(int(v[4 + 3 * k] * W), int(v[5 + 3 * k] * H)) for k in range(5)]
                for a, b in zip(pts, pts[1:]):
                    cv2.line(img, a, b, color, 1, cv2.LINE_AA)
                for pt in pts:
                    cv2.circle(img, pt, 5, color, -1, cv2.LINE_AA)
                    cv2.circle(img, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)

        draw(gt, GT)
        draw(pr, PRED)
        cv2.putText(img, "green=GT  red=VLM", (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(img)
    grid = np.hstack(cells)
    p = os.path.join(OUT, "fig_pose_compare.jpg")
    cv2.imwrite(p, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("saved", p)


def fig_digit_fails(n=8):
    """Misclassified dark digit slices with GT/VLM labels."""
    ds, preds = "cls_armor_digit_slices", os.path.join(RUN, "cls_armor_digit_slices", "labels")
    classes = load_classes(os.path.join(DS, ds))
    wrong = []
    for f in sorted(os.listdir(preds)):
        stem = os.path.splitext(f)[0]
        gt = open(os.path.join(DS, ds, "labels", f)).read().strip()
        pr = open(os.path.join(preds, f)).read().strip()
        if gt != pr:
            wrong.append((stem, gt, pr))
    import random
    rng = random.Random(1)
    pick = rng.sample(wrong, min(n, len(wrong)))
    cells = []
    for stem, gt, pr in pick:
        img = _load(ds, stem, 120)
        big = cv2.copyMakeBorder(img, 20, 18, 2, 2, cv2.BORDER_CONSTANT, value=(15, 15, 15))
        cv2.putText(big, f"GT:{gt}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(big, f"VLM:{pr}", (3, big.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (60, 60, 255), 1, cv2.LINE_AA)
        cells.append(big)
    cols = 4
    rows_n = (len(cells) + cols - 1) // cols
    cw = max(c.shape[1] for c in cells)
    ch = max(c.shape[0] for c in cells)
    grid = np.full((rows_n * ch, cols * cw, 3), 18, dtype=np.uint8)
    for i, c in enumerate(cells):
        y, x = (i // cols) * ch, (i % cols) * cw
        grid[y:y + c.shape[0], x:x + c.shape[1]] = c
    p = os.path.join(OUT, "fig_digit_fails.jpg")
    cv2.imwrite(p, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("saved", p, f"({len(wrong)} wrong of 25)")


def fig_car_hud_confusion():
    """rm2025_car: GT (green, on real cars) vs VLM (red, on HUD icons)."""
    ds = "detect_rm2025_car"
    run = os.path.join(ROOT, "testenv", "results", "user_prompts")
    classes = load_classes(os.path.join(DS, ds))
    stem = "0497"
    gt = load_yolo(os.path.join(DS, ds, "labels", stem + ".txt"), classes, want_tokens=5)
    pred_path = os.path.join(run, ds, "labels", stem + ".txt")
    if not os.path.isfile(pred_path):
        print("skip fig_car_hud_confusion (no user_prompts prediction)")
        return
    pred = load_yolo(pred_path, classes, want_tokens=5)
    img = _load(ds, stem, 960)
    H, W = img.shape[:2]
    for _c, v, _t in gt:
        cv2.rectangle(img, (int(v[0] * W), int(v[1] * H)),
                      (int(v[2] * W), int(v[3] * H)), GT, 2)
        cv2.putText(img, "GT", (int(v[0] * W), int(v[1] * H) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, GT, 1, cv2.LINE_AA)
    for _c, v, t in pred:
        # 模型输出为 0-1000 归一化坐标
        cx, cy, w, h = (float(x) / 1000 for x in t[1:5])
        cv2.rectangle(img, (int((cx - w / 2) * W), int((cy - h / 2) * H)),
                      (int((cx + w / 2) * W), int((cy + h / 2) * H)), PRED, 2)
    # 黑色遮挡（隐私）: 选手昵称/人脸/学校名（坐标基于 960x540 渲染尺寸）
    for (rx1, ry1, rx2, ry2) in [
        (0, 38, 128, 78),       # 顶部左侧昵称+血条面板
        (132, 0, 302, 36),      # "Hello World 浙江大学"
        (455, 0, 700, 36),      # "南方科技大学 ARTINX"
        (0, 270, 100, 294),     # 摄像头画面上沿的选手昵称
        (92, 280, 190, 400),    # 摄像头画面中的人脸
        (12, 418, 185, 475),    # 左下角红色面板中的选手昵称
    ]:
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (0, 0, 0), -1)
    cv2.putText(img, "green=GT(real cars)  red=VLM(HUD icons)", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    p = os.path.join(OUT, "fig_car_hud_confusion.jpg")
    cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("saved", p)


def main():
    ok1, _ = fig_confusion("cls_armor_pattern_public", "fig_confusion_pattern.png",
                           "armor pattern classification")
    ok2, _ = fig_confusion("cls_armor_digit_slices", "fig_confusion_digit.png",
                           "dark digit-slice classification")
    fig_cls_accuracy([ok1 / 25, ok2 / 25])
    fig_luminous_overlays()
    fig_pose_compare()
    fig_digit_fails()
    fig_car_hud_confusion()


if __name__ == "__main__":
    main()
