#!/usr/bin/env python3
"""render_showcase -- build the "actual VLM annotation results" collage.

One 3x3 grid stitched from the vlm_probe run, covering all 7 tested datasets:
detect / labelme overlays (GT green vs VLM red boxes), pose skeleton rows,
classification cells with GT/VLM class labels. Saved to
docs/pic/fig_showcase.jpg.

Run:  python3 tools/render_showcase.py
"""

from __future__ import annotations

import json
import os
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, ROOT)

from tools.eval_testenv import load_classes, load_yolo, load_labelme_gt  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

RUN = os.path.join(ROOT, "testenv", "results", "vlm_probe")
DS = os.path.join(ROOT, "testenv", "datasets")
OUT = os.path.join(ROOT, "docs", "pic", "fig_showcase.jpg")

GT = (0, 200, 0)
PRED = (0, 0, 255)
CELL_W = 440


def load_img(ds, stem):
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(DS, ds, "images", stem + ext)
        if os.path.isfile(p):
            img = cv2.imread(p)
            h, w = img.shape[:2]
            interp = cv2.INTER_NEAREST if w < CELL_W else cv2.INTER_AREA
            return cv2.resize(img, (CELL_W, max(1, int(round(h * CELL_W / w)))), interpolation=interp)
    return None


def draw_boxes(img, rows, color, tag):
    H, W = img.shape[:2]
    for item in rows:
        name, v = item[0], item[1]
        p1 = (int(v[0] * W), int(v[1] * H))
        p2 = (int(v[2] * W), int(v[3] * H))
        cv2.rectangle(img, p1, p2, color, 2)
        label = f"{tag}:{name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        ty = max(th + 4, p1[1] - 3)
        cv2.rectangle(img, (p1[0], ty - th - 3), (p1[0] + tw + 3, ty), color, -1)
        cv2.putText(img, label, (p1[0] + 1, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (255, 255, 255), 1, cv2.LINE_AA)


def draw_pose(img, rows, color):
    H, W = img.shape[:2]
    for item in rows:
        v = item[1]
        p1 = (int(v[0] * W), int(v[1] * H))
        p2 = (int(v[2] * W), int(v[3] * H))
        cv2.rectangle(img, p1, p2, color, 2)
        n = (len(v) - 4) // 3
        pts = [(int(v[4 + 3 * k] * W), int(v[5 + 3 * k] * H)) for k in range(n)]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, color, 2, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(img, pt, 5, color, -1, cv2.LINE_AA)
            cv2.circle(img, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)


def yolo_rows(ds, stem):
    classes = load_classes(os.path.join(DS, ds))
    return [(r[0], r[1]) for r in load_yolo(os.path.join(DS, ds, "labels", stem + ".txt"),
                                            classes, want_tokens=5)]


def yolo_rows_pose(ds, stem, n_kpt):
    classes = load_classes(os.path.join(DS, ds))
    return [(r[0], r[1]) for r in load_yolo(os.path.join(DS, ds, "labels", stem + ".txt"),
                                            classes, want_tokens=5 + 3 * n_kpt)]


def pred_rows(ds, stem):
    classes = load_classes(os.path.join(DS, ds))
    p = os.path.join(RUN, ds, "labels", stem + ".txt")
    if not os.path.isfile(p):
        return []
    return [(r[0], r[1]) for r in load_yolo(p, classes, want_tokens=5)]


def pred_rows_pose(ds, stem, n_kpt):
    classes = load_classes(os.path.join(DS, ds))
    p = os.path.join(RUN, ds, "labels", stem + ".txt")
    if not os.path.isfile(p):
        return []
    return [(r[0], r[1]) for r in load_yolo(p, classes, want_tokens=5 + 3 * n_kpt)]


def labelme_gt(ds, stem):
    g = load_labelme_gt(os.path.join(DS, ds)).get(stem, [])
    return [(n, b) for n, b in g]


def cls_cell(ds, stem):
    img = load_img(ds, stem)
    gt = open(os.path.join(DS, ds, "labels", stem + ".txt")).read().strip()
    pp = os.path.join(RUN, ds, "labels", stem + ".txt")
    pr = open(pp).read().strip() if os.path.isfile(pp) else "-"
    ok = gt == pr
    cv2.putText(img, f"GT:{gt}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    color = (120, 230, 140) if ok else (60, 60, 255)
    cv2.putText(img, f"VLM:{pr}", (4, img.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1, cv2.LINE_AA)
    return img


def pick_stem(ds, need_boxes=True):
    """从预测目录选一个确定性样本；need_boxes 时优先选 GT 非空的。"""
    pdir = os.path.join(RUN, ds, "labels")
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(pdir))
    fallback = stems[len(stems) // 2] if stems else None
    if not need_boxes:
        return fallback
    for stem in stems:
        rows = yolo_rows(ds, stem) or pred_rows(ds, stem)
        if rows:
            return stem
    return fallback


def main():
    cells = []  # (caption, img)

    # detect: 灯条（近乎完美）
    ds = "detect_light_luminous"
    stem = pick_stem(ds)
    img = load_img(ds, stem)
    draw_boxes(img, yolo_rows(ds, stem), GT, "GT")
    draw_boxes(img, pred_rows(ds, stem), PRED, "VLM")
    cells.append(("detect_light_luminous | P100 R100 IoU0.92", img))

    # detect: 装甲（小目标失败案例）
    ds = "detect_rm2025_armor"
    stem = pick_stem(ds)
    img = load_img(ds, stem)
    draw_boxes(img, yolo_rows(ds, stem), GT, "GT")
    draw_boxes(img, pred_rows(ds, stem), PRED, "VLM")
    cells.append(("detect_rm2025_armor | tiny-target FAIL", img))

    # labelme: 雷达车（密集 GT）
    ds = "labelme_radar_car"
    stem = pick_stem(ds)
    img = load_img(ds, stem)
    draw_boxes(img, labelme_gt(ds, stem), GT, "GT")
    draw_boxes(img, pred_rows(ds, stem), PRED, "VLM")
    cells.append(("labelme_radar_car | dense GT vs sparse VLM", img))

    # pose: okbuff2（框能匹配、关键点粗）
    ds = "pose_okbuff2"
    stem = pick_stem(ds, need_boxes=False)
    img = load_img(ds, stem)
    draw_pose(img, yolo_rows_pose(ds, stem, 5), GT)
    draw_pose(img, pred_rows_pose(ds, stem, 5), PRED)
    cells.append(("pose_okbuff2 (5 kpt) | MPJPE 0.35 diag", img))

    # pose: droneok（小目标完全错位）
    ds = "pose_droneok"
    stem = pick_stem(ds, need_boxes=False)
    img = load_img(ds, stem)
    draw_pose(img, yolo_rows_pose(ds, stem, 4), GT)
    draw_pose(img, pred_rows_pose(ds, stem, 4), PRED)
    cells.append(("pose_droneok (4 kpt) | tiny-target FAIL", img))

    # cls: 图案（一对对错组合）
    ds = "cls_armor_pattern_public"
    stems = sorted(os.listdir(os.path.join(RUN, ds, "labels")))
    shown = 0
    for f in stems:
        stem = os.path.splitext(f)[0]
        gt = open(os.path.join(DS, ds, "labels", f)).read().strip()
        pr = open(os.path.join(RUN, ds, "labels", f)).read().strip()
        if gt != pr or shown < 1:  # 至少放一张正确的
            cells.append((f"cls_armor_pattern | GT:{gt} VLM:{pr}", cls_cell(ds, stem)))
            shown += 1
        if shown >= 2:
            break

    # cls: 数字切片（暗光失败）
    ds = "cls_armor_digit_slices"
    for f in sorted(os.listdir(os.path.join(RUN, ds, "labels")))[:1]:
        stem = os.path.splitext(f)[0]
        gt = open(os.path.join(DS, ds, "labels", f)).read().strip()
        pr = open(os.path.join(RUN, ds, "labels", f)).read().strip()
        cells.append((f"cls_armor_digit (dark) | GT:{gt} VLM:{pr}", cls_cell(ds, stem)))

    # 拼 3x3
    cols, rows_n = 3, 3
    cw = max(c[1].shape[1] for c in cells) + 4
    ch = max(c[1].shape[0] for c in cells) + 26
    grid = np.full((rows_n * ch, cols * cw, 3), 16, dtype=np.uint8)
    for i, (caption, img) in enumerate(cells):
        y, x = (i // cols) * ch, (i % cols) * cw
        grid[y + 20:y + 20 + img.shape[0], x + 2:x + 2 + img.shape[1]] = img
        cv2.putText(grid, caption, (x + 4, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 220, 80), 1, cv2.LINE_AA)
    cv2.imwrite(OUT, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"saved {OUT} ({len(cells)} cells, {grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
