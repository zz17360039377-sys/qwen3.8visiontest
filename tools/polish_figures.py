#!/usr/bin/env python3
"""polish_figures -- regenerate the article figures with a consistent, polished style.

Style contract:
  * dark slate canvas (#14161a), panels with 1px border (#3a4150);
  * GT / ground truth = green (#3fd06c), VLM prediction = red (#ff5a52),
    seg polygon = translucent amber;
  * all text rendered with PIL + Noto Sans CJK (Chinese-capable);
  * legend chips instead of bare text; uniform cell widths.

Figures overwritten into docs/pic/:
  fig_luminous_overlay.jpg  fig_car_hud_confusion.jpg
  fig_pose_showcase.jpg     fig_seg_showcase.jpg
  fig_caption_kitchen.jpg / _horse / _motorcycle (clean photos)

Run:  python3 tools/polish_figures.py
"""

from __future__ import annotations

import json
import os
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, ROOT)

DS = os.path.join(ROOT, "testenv", "datasets")
VLM_PROBE = os.path.join(ROOT, "testenv", "results", "vlm_probe")
USER_RUN = os.path.join(ROOT, "testenv", "results", "user_prompts")
POSE_RUN = os.path.join(ROOT, "testenv", "results", "pose_demo")
SEG_RUN = os.path.join(ROOT, "testenv", "results", "seg_demo")
OUT = os.path.join(ROOT, "docs", "pic")
os.makedirs(OUT, exist_ok=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from tools.eval_testenv import load_classes, load_yolo, load_labelme_gt  # noqa: E402

# ---- style constants ----
BG = (20, 22, 26)
BORDER = (58, 65, 80)
GT_C = (63, 208, 108)      # 绿
VLM_C = (82, 90, 255)      # 红 (BGR)
SEG_C = (60, 176, 255)     # 琥珀 (BGR)
CAP_BG = (10, 12, 16)
CAP_FG = (235, 238, 242)

_CJK_TTC = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_BOLD_TTC = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(size: int, bold: bool = False):
    p = _BOLD_TTC if bold and os.path.isfile(_BOLD_TTC) else _CJK_TTC
    return ImageFont.truetype(p, size)


def find_img(ds_dir, stem):
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(ds_dir, "images", stem + ext)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(stem)


def load_render(ds_dir, stem, cell_w):
    img = cv2.imread(find_img(ds_dir, stem))
    h, w = img.shape[:2]
    interp = cv2.INTER_NEAREST if w < cell_w else cv2.INTER_AREA
    return cv2.resize(img, (cell_w, max(1, int(round(h * cell_w / w)))), interpolation=interp)


def canvas(cells, cols, cell_w, cap_h=30, pad=6):
    """cells: [(caption, img_bgr)] -> BGR grid with caption bars.

    每行高度取该行最高 cell（而非全图最高），避免矮图下方大片空白。"""
    rows_imgs = [cells[r * cols:(r + 1) * cols]
                 for r in range((len(cells) + cols - 1) // cols)]
    row_h = [cap_h + max(c[1].shape[0] for c in row) + pad * 2 for row in rows_imgs]
    cw = cell_w + pad * 2
    grid = np.full((sum(row_h), cols * cw, 3), BG, dtype=np.uint8)
    cap_pos = []
    y = 0
    for row, rh in zip(rows_imgs, row_h):
        for i, (cap, img) in enumerate(row):
            y0 = y + pad
            x = i * cw + pad
            grid[y0 + cap_h:y0 + cap_h + img.shape[0],
                 x:x + img.shape[1]] = img
            cv2.rectangle(grid, (x - 1, y0 + cap_h - 1),
                          (x + img.shape[1], y0 + cap_h + img.shape[0]), BORDER, 1)
            cap_pos.append((x + 2, y0 + 5, cap))
        y += rh
    # 文字最后画（PIL 支持 CJK）
    pil = Image.fromarray(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    f = font(17, bold=True)
    for x, y0, cap in cap_pos:
        d.text((x, y0), cap, font=f, fill=CAP_FG)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def save(grid, name):
    p = os.path.join(OUT, name)
    cv2.imwrite(p, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("saved", p)


def draw_box_alpha(img, box, color, alpha=0.18, thick=2):
    """半透明填充 + 实线边框。box = [x1,y1,x2,y2] 像素。"""
    x1, y1, x2, y2 = box
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)


def legend_chip(grid, x, y, color, text):
    """色卡图例：色块 + 中文文字。"""
    cv2.rectangle(grid, (x, y), (x + 16, y + 16), color, -1)
    img_pil = Image.fromarray(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img_pil)
    d.text((x + 24, y + 1), text, font=font(17), fill=CAP_FG)
    grid[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ------------------------------------------------------------ 1. 灯条检测

def fig_luminous():
    ds = os.path.join(DS, "detect_light_luminous")
    classes = load_classes(ds)
    import glob as _g
    tested = sorted(_g.glob(os.path.join(VLM_PROBE, "detect_light_luminous",
                                         "labels", "*.txt")))
    stem = os.path.splitext(os.path.basename(tested[0]))[0]
    gt = load_yolo(os.path.join(ds, "labels", stem + ".txt"), classes, want_tokens=5)
    pred = load_yolo(os.path.join(VLM_PROBE, "detect_light_luminous", "labels",
                                  stem + ".txt"), classes, want_tokens=5)
    img = load_render(ds, stem, 720)
    H, W = img.shape[:2]
    for _n, v, _t in gt:
        draw_box_alpha(img, [int(v[0] * W), int(v[1] * H), int(v[2] * W), int(v[3] * H)],
                       GT_C)
    for _n, v, _t in pred:
        draw_box_alpha(img, [int(v[0] * W), int(v[1] * H), int(v[2] * W), int(v[3] * H)],
                       VLM_C)
    legend_chip(img, 10, 10, GT_C, "GT 标准答案")
    legend_chip(img, 150, 10, VLM_C, "VLM 预测")
    grid = canvas([(f"detect_light_luminous · IoU 0.918 · P/R 100%", img)], 1, 720)
    save(grid, "fig_luminous_overlay.jpg")


# ------------------------------------------------- 2. 多车检测（含遮挡）

def fig_car_hud():
    ds = os.path.join(DS, "detect_rm2025_car")
    classes = load_classes(ds)
    stem = "0497"
    gt = load_yolo(os.path.join(ds, "labels", stem + ".txt"), classes, want_tokens=5)
    pred = load_yolo(os.path.join(USER_RUN, "detect_rm2025_car", "labels",
                                  stem + ".txt"), classes, want_tokens=5)
    img = load_render(ds, stem, 720)
    H, W = img.shape[:2]
    for _n, v, _t in gt:
        draw_box_alpha(img, [int(v[0] * W), int(v[1] * H), int(v[2] * W), int(v[3] * H)],
                       GT_C)
    for _n, v, t in pred:
        cx, cy, w, h = (float(x) / 1000 for x in t[1:5])   # 0-1000 归一化
        draw_box_alpha(img, [int((cx - w / 2) * W), int((cy - h / 2) * H),
                             int((cx + w / 2) * W), int((cy + h / 2) * H)], VLM_C)
    # 黑色遮挡: 昵称/人脸/学校名（坐标基于 720 宽渲染）
    for (rx1, ry1, rx2, ry2) in [
        (0, 30, 100, 62), (95, 195, 165, 320), (0, 190, 78, 210),
        (95, 0, 220, 26), (330, 0, 465, 26),
    ]:
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (0, 0, 0), -1)
    legend_chip(img, 230, 10, GT_C, "GT 标准答案")
    legend_chip(img, 380, 10, VLM_C, "VLM 预测")
    grid = canvas([("detect_rm2025_car · 干扰物锁错：VLM 全部落到 HUD 图标", img)], 1, 720)
    save(grid, "fig_car_hud_confusion.jpg")


# ------------------------------------------------------- 3. pose 好/坏

def _pose_cell(ds_dir, stem, cell_w, cap, kpt_n, color_only_pred=False):
    if not os.path.isdir(ds_dir):
        ds_dir = os.path.join(DS, ds_dir)
    classes = load_classes(ds_dir)
    img = load_render(ds_dir, stem, cell_w)
    H, W = img.shape[:2]
    gt_rows = load_yolo(os.path.join(ds_dir, "labels", stem + ".txt"), classes,
                        want_tokens=5 + 3 * kpt_n) if os.path.isdir(
        os.path.join(ds_dir, "labels")) else []
    pp = None
    for run in (POSE_RUN, USER_RUN):
        cand = os.path.join(run, os.path.basename(ds_dir), "labels", stem + ".txt")
        if os.path.isfile(cand):
            pp = cand
            break
    pred_rows = load_yolo(pp, classes, want_tokens=5 + 3 * kpt_n) if pp else []

    def draw(rows, color):
        for _n, v, _t in rows:
            p1 = (int(v[0] * W), int(v[1] * H))
            p2 = (int(v[2] * W), int(v[3] * H))
            cv2.rectangle(img, p1, p2, color, 2)
            n = (len(v) - 4) // 3
            pts = [(int(v[4 + 3 * k] * W), int(v[5 + 3 * k] * H)) for k in range(n)]
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, a, b, color, 1, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(img, pt, 5, color, -1, cv2.LINE_AA)
                cv2.circle(img, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)

    if not color_only_pred:
        draw(gt_rows, GT_C)
    draw(pred_rows, VLM_C)
    return (cap, img)


def fig_pose_showcase():
    sz_ds = os.path.join(DS, "pose_pillar_exchange_12kpt")
    ok_ds = os.path.join(DS, "pose_okbuff2")
    cap_ds = os.path.join(ROOT, "testenv", "datasets_extra", "demo_caption")
    sz_stem = "dataset_v9.1__data_side_off_v9.0_compose__frame_000150"
    ok_stem = "hik_20260520_171526_000315+RGB"
    cells = [
        _pose_cell(sz_ds, sz_stem, 440, "GOOD 立柱 12 点 (few-shot)", 12),
        _pose_cell(ok_ds, ok_stem, 440, "PARTIAL 能量机关 5 点 (few-shot+尺寸声明)", 5),
        _pose_cell(cap_ds, "caption_horse", 440, "PARTIAL 骑马骨骼 (日常 12 点)", 12,
                   color_only_pred=True),
        _pose_cell(os.path.join(DS, "pose_final_shenzhen"),
                   "record_20260601_151532_16_000127", 440,
                   "FAIL 暗光小目标 (few-shot)", 4),
        _pose_cell(cap_ds, "caption_kitchen", 440, "FAIL 背身遮挡 (日常 12 点)", 12,
                   color_only_pred=True),
    ]
    grid = canvas(cells, 3, 440)
    save(grid, "fig_pose_showcase.jpg")


# ------------------------------------------------------- 4. 语义分割

def _seg_cell(ds_dir, stem, label, cell_w, cap):
    if not os.path.isdir(ds_dir):
        ds_dir = os.path.join(DS, ds_dir)
    img = load_render(ds_dir, stem, cell_w)
    H, W = img.shape[:2]
    pj = os.path.join(SEG_RUN if "datasets_extra" not in ds_dir else
                      os.path.join(ROOT, "testenv", "results", "seg_demo"),
                      os.path.basename(ds_dir), "labels", stem + ".json")
    if not os.path.isfile(pj):
        pj = os.path.join(RES_FALLBACK, os.path.basename(ds_dir), "labels", stem + ".json")
    poly = []
    if os.path.isfile(pj):
        data = json.load(open(pj, encoding="utf-8"))
        if isinstance(data, list) and data:
            data = data[0]
        poly = data.get("polygon", [])
    if poly:
        pts = np.array([(float(x), float(y)) for x, y in poly], dtype=np.float32)
        mx, my = pts.max(axis=0)
        if mx > W or my > H:   # 0-1000 归一化空间
            pts[:, 0] *= W / 1000.0
            pts[:, 1] *= H / 1000.0
        pts = pts.astype(np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], SEG_C)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        cv2.polylines(img, [pts], True, SEG_C, 2)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img_pil)
    d.text((8, 8), label, font=font(18, bold=True), fill=(255, 255, 255))
    return (cap, cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR))


RES_FALLBACK = os.path.join(ROOT, "testenv", "results", "seg_demo")


def fig_seg_showcase(run="seg_fix"):
    import json as _json
    items = [("caption_motorcycle", "motorcycle"), ("caption_kitchen", "person"),
             ("caption_horse", "horses")]
    cells = []
    for stem, label in items:
        ds_dir = os.path.join(ROOT, "testenv", "datasets_extra", "demo_caption")
        img = load_render(ds_dir, stem, 480)
        H, W = img.shape[:2]
        pj = os.path.join(ROOT, "testenv", "results", run,
                          os.path.basename(ds_dir), "labels", stem + ".json")
        if not os.path.isfile(pj):
            print("skip", stem)
            continue
        data = _json.load(open(pj, encoding="utf-8"))
        if isinstance(data, list) and data:
            data = data[0]
        poly = data.get("polygon", [])
        if poly:
            pts = np.array([(float(x), float(y)) for x, y in poly], dtype=np.float32)
            mx, my = pts.max(axis=0)
            if mx > W or my > H:   # 0-1000 归一化空间
                pts[:, 0] *= W / 1000.0
                pts[:, 1] *= H / 1000.0
            pts = pts.astype(np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], SEG_C)
            img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
            cv2.polylines(img, [pts], True, (0, 0, 255), 2)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img_pil)
        d.text((8, 8), label, font=font(18, bold=True), fill=(255, 255, 255))
        cells.append((f"{label} · {stem}", cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)))
    grid = canvas(cells, 3, 480)
    save(grid, "fig_seg_showcase.jpg")


def main():
    fig_luminous()
    fig_car_hud()
    fig_pose_showcase()
    fig_seg_showcase()


if __name__ == "__main__":
    import json
    main()
