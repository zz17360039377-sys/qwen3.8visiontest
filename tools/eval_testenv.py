#!/usr/bin/env python3
"""eval_testenv -- score test results in testenv/results against testenv/datasets.

Reads one run:  ``testenv/results/<run>/<dataset>/labels/<stem>.txt``  and
compares it with the matching ground truth in ``testenv/datasets/<dataset>/``.
Dataset type comes from the name prefix: ``pose_`` / ``detect_`` / ``cls_`` /
``labelme_``. Prediction class tokens may be numeric ids or class names from
that dataset's classes.txt.

Metrics
  detect / labelme : greedy IoU>=--match-iou, class-aware -> P / R / F1,
                     mean IoU, per-class P/R/F1
  pose             : box IoU>=--pose-match-iou -> per-keypoint MPJPE (px and
                     box-diagonal-normalised) + PCK@thresholds
  cls              : accuracy + confusion matrix

Visualisation (always on): random sample pages of GT (green) vs prediction
(red), plus EVERY failing image (FP/FN boxes, unmatched keypoints, wrong
class), written to  ``results/<run>/vis/<dataset>/``  and linked in the report.

Outputs  ``results/<run>/eval_report.md``  +  ``eval_details.csv``. Run:
  python3 tools/eval_testenv.py --run <name>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
DS_ROOT = os.path.join(ROOT, "testenv", "datasets")
RES_ROOT = os.path.join(ROOT, "testenv", "results")

GT_COLOR = (0, 200, 0)      # 绿 = 标准答案
PRED_COLOR = (0, 0, 255)    # 红 = VLM 预测


# ------------------------------------------------------------------ helpers

def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def greedy_match(preds, gts, thresh):
    """ preds/gts: [(cls, box, ...)] -> [(pi, gi, iou)] class-aware, IoU-desc greedy. """
    pairs = []
    for pi in range(len(preds)):
        pc, pb = preds[pi][0], preds[pi][1]
        for gi in range(len(gts)):
            gc, gb = gts[gi][0], gts[gi][1]
            if pc != gc:
                continue
            v = iou(pb, gb)
            if v >= thresh:
                pairs.append((v, pi, gi))
    pairs.sort(reverse=True)
    up, ug, out = set(), set(), []
    for v, pi, gi in pairs:
        if pi in up or gi in ug:
            continue
        up.add(pi); ug.add(gi)
        out.append((pi, gi, v))
    return out


def load_classes(ds_dir):
    p = os.path.join(ds_dir, "classes.txt")
    if not os.path.isfile(p):
        return []
    return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]


def cls_token(tok, classes):
    """'3'/'B3' -> canonical class NAME; None if unresolvable."""
    if tok == "":
        return None
    if tok.isdigit():
        i = int(tok)
        return classes[i] if i < len(classes) else f"id{i}"
    return tok  # already a name


def load_yolo(path, classes, want_tokens=None):
    """Parse a YOLO txt -> [(cls_name, vals, raw_tokens)]; bad lines skipped.

    vals[0:4] is ALWAYS the box converted to x1/y1/x2/y2 (normalised); a pose
    row keeps its keypoints at vals[4 + 3k : 7 + 3k] = (x, y, v).
    """
    out = []
    if not os.path.isfile(path):
        return out
    for ln in open(path, encoding="utf-8", errors="ignore"):
        t = ln.split()
        if not t:
            continue
        name = cls_token(t[0], classes)
        try:
            vals = [float(x) for x in t[1:]]
        except ValueError:
            continue
        if want_tokens is not None and len(vals) + 1 != want_tokens:
            continue
        cx, cy, w, h = vals[:4]
        vals = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] + vals[4:]
        out.append((name, vals, t))
    return out


def load_labelme_gt(ds_dir):
    """labelme ground truth -> {stem: [(label, [x1,y1,x2,y2])]} (normalised)."""
    gt = {}
    lbl_dir = os.path.join(ds_dir, "labels")
    for jf in sorted(os.listdir(lbl_dir)):
        if not jf.endswith(".json"):
            continue
        try:
            data = json.load(open(os.path.join(lbl_dir, jf), encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        W = float(data.get("imageWidth") or 1)
        H = float(data.get("imageHeight") or 1)
        items = []
        for s in data.get("shapes") or []:
            pts = [list(map(float, p)) for p in (s.get("points") or [])]
            if len(pts) < 2:
                continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            items.append((str(s.get("label")),
                          [min(xs) / W, min(ys) / H, max(xs) / W, max(ys) / H]))
        gt[os.path.splitext(jf)[0]] = items
    return gt


def find_image_path(ds_dir, stem):
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        p = os.path.join(ds_dir, "images", stem + ext)
        if os.path.isfile(p):
            return p
    return None


def image_size(ds_dir, stem):
    from PIL import Image
    p = find_image_path(ds_dir, stem)
    if p is None:
        return None
    from PIL import Image as _I
    with _I.open(p) as im:
        return im.size


def pct(n, d):
    return f"{n / d:.1%}" if d else "-"


# ------------------------------------------------------------------ evaluators

def eval_detect(ds_name, ds_dir, run_dir, args, details, kind="detect"):
    classes = load_classes(ds_dir)
    pred_dir = os.path.join(run_dir, ds_name, "labels")
    if kind == "labelme":
        gt_map = load_labelme_gt(ds_dir)   # {stem: [(label, box xyxy)]}
        stems = set(gt_map)
    else:
        gt_dir = os.path.join(ds_dir, "labels")
        stems = {os.path.splitext(f)[0] for f in os.listdir(gt_dir)}
    gts_of = (lambda stem: gt_map.get(stem, [])) if kind == "labelme" else (
        lambda stem: [(g[0], g[1][:4])
                      for g in load_yolo(os.path.join(ds_dir, "labels", stem + ".txt"),
                                         classes, want_tokens=5)])
    tp = fp = fn = fn_measured = miss_files = 0
    tp_free = fp_free = fn_free = 0
    ious, ious_free = [], []
    per_cls = {}
    per_image = {}
    for stem in sorted(stems):
        gt_rows = gts_of(stem)
        ppath = os.path.join(pred_dir, stem + ".txt")
        if not os.path.isfile(ppath):
            miss_files += 1
            fn += len(gt_rows)
            per_image[stem] = {"preds": [], "gts": [(g[0], g[1][:4]) for g in gt_rows],
                               "fp": 0, "fn": len(gt_rows), "max_err": None,
                               "missing": True}
            continue
        pred_rows = [(p[0], p[1][:4]) for p in
                     load_yolo(ppath, classes, want_tokens=5)]
        matches = greedy_match(pred_rows, gt_rows, args.match_iou)
        matches_free = greedy_match([(p[0], p[1]) for p in pred_rows],
                                    [(g[0], g[1]) for g in gt_rows], args.match_iou)
        hit_p = {pi for pi, _, _ in matches}
        hit_g = {gi for _, gi, _ in matches}
        n_fp = len(pred_rows) - len(matches)
        n_fn = len(gt_rows) - len(matches)
        tp += len(matches)
        fp += n_fp
        fn_measured += n_fn
        fn += n_fn
        tp_free += len(matches_free)
        fp_free += len(pred_rows) - len(matches_free)
        fn_free += len(gt_rows) - len(matches_free)
        for pi, gi, v in matches:
            ious.append(v)
            per_cls.setdefault(gt_rows[gi][0], [0, 0, 0])[0] += 1
        for pi in range(len(pred_rows)):
            if pi not in hit_p:
                per_cls.setdefault(pred_rows[pi][0], [0, 0, 0])[1] += 1
        for gi in range(len(gt_rows)):
            if gi not in hit_g:
                per_cls.setdefault(gt_rows[gi][0], [0, 0, 0])[2] += 1
        for pi, gi, v in matches_free:
            ious_free.append(v)
        per_image[stem] = {"preds": [(p[0], p[1][:4]) for p in pred_rows],
                           "gts": [(g[0], g[1][:4]) for g in gt_rows],
                           "fp": n_fp, "fn": n_fn, "max_err": None}
        for pi, gi, v in matches:
            details.append([ds_name, stem, gt_rows[gi][0], f"{v:.4f}"])
    return {"tp": tp, "fp": fp, "fn": fn, "fn_measured": fn_measured,
            "tp_free": tp_free, "fp_free": fp_free, "fn_free": fn_free,
            "miou_free": sum(ious_free) / len(ious_free) if ious_free else 0.0,
            "miss": miss_files,
            "miou": sum(ious) / len(ious) if ious else 0.0,
            "per_cls": per_cls, "per_image": per_image}


def eval_pose(ds_name, ds_dir, run_dir, args, details):
    classes = load_classes(ds_dir)
    gt_dir = os.path.join(ds_dir, "labels")
    pred_dir = os.path.join(run_dir, ds_name, "labels")
    stems = {os.path.splitext(f)[0] for f in os.listdir(gt_dir)}
    n_kpt = 0
    for stem in sorted(stems):
        rows = load_yolo(os.path.join(gt_dir, stem + ".txt"), classes)
        if rows:
            n_kpt = (len(rows[0][2]) - 4) // 3
            break
    pcks = [float(x) for x in args.pck.split(",") if x.strip()]
    max_pck = max(pcks) if pcks else 0.1
    errs_norm, errs_px = [], []
    per_kpt = {}
    per_image = {}
    matched = miss_files = 0
    n_gt_boxes = n_pred_boxes = 0
    for stem in sorted(stems):
        gt_rows = load_yolo(os.path.join(gt_dir, stem + ".txt"), classes, want_tokens=5 + 3 * n_kpt)
        ppath = os.path.join(pred_dir, stem + ".txt")
        if not os.path.isfile(ppath):
            miss_files += 1
            per_image[stem] = {"gt": [(g[0], g[1]) for g in gt_rows], "pred": [],
                               "matches": [], "max_err": None,
                               "un_gt": len(gt_rows), "un_pr": 0,
                               "missing": True}
            continue
        pred_rows = load_yolo(ppath, classes, want_tokens=5 + 3 * n_kpt)
        n_gt_boxes += len(gt_rows)
        n_pred_boxes += len(pred_rows)
        matches = greedy_match(pred_rows, gt_rows, args.pose_match_iou)
        matched += len(matches)
        size = image_size(ds_dir, stem)
        W, H = size or (1, 1)
        img_max = 0.0
        hit_p = {pi for pi, _, _ in matches}
        hit_g = {gi for _, gi, _ in matches}
        for pi, gi, _v in matches:
            gt_vals, pr_vals = gt_rows[gi][1], pred_rows[pi][1]
            diag = math.hypot((gt_vals[2] - gt_vals[0]) * W,
                              (gt_vals[3] - gt_vals[1]) * H) or 1.0
            for k in range(n_kpt):
                kv = gt_vals[4 + 3 * k: 4 + 3 * k + 3]
                pv = pr_vals[4 + 3 * k: 4 + 3 * k + 3]
                if kv[2] <= 0:
                    continue
                e_px = math.hypot((pv[0] - kv[0]) * W, (pv[1] - kv[1]) * H)
                e_n = e_px / diag
                errs_px.append(e_px)
                errs_norm.append(e_n)
                img_max = max(img_max, e_n)
                per_kpt.setdefault(k, []).append(e_n)
                details.append([ds_name, stem, f"kpt{k}", f"{e_px:.1f}px/{e_n:.4f}"])
        un_gt = len(gt_rows) - len(matches)
        un_pr = len(pred_rows) - len(matches)
        per_image[stem] = {"gt": [(g[0], g[1]) for g in gt_rows],
                           "pred": [(p[0], p[1]) for p in pred_rows],
                           "matches": matches, "max_err": img_max,
                           "un_gt": un_gt, "un_pr": un_pr}
    return {"n_kpt": n_kpt, "matched": matched, "miss": miss_files,
            "n_gt": n_gt_boxes, "n_pred": n_pred_boxes,
            "mpjpe_px": sum(errs_px) / len(errs_px) if errs_px else 0.0,
            "mpjpe_n": sum(errs_norm) / len(errs_norm) if errs_norm else 0.0,
            "n_err": len(errs_norm),
            "pck": {t: sum(1 for e in errs_norm if e <= t) for t in pcks},
            "per_kpt": per_kpt, "pck_list": pcks, "max_pck": max_pck,
            "per_image": per_image}


def eval_cls(ds_name, ds_dir, run_dir, args, details):
    classes = load_classes(ds_dir)
    gt_dir = os.path.join(ds_dir, "labels")
    pred_dir = os.path.join(run_dir, ds_name, "labels")
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(gt_dir))
    n_ok = miss = 0
    confusion = {}
    wrong = {}
    pred_stems = []
    for stem in stems:
        gt = open(os.path.join(gt_dir, stem + ".txt"), encoding="utf-8").read().strip()
        ppath = os.path.join(pred_dir, stem + ".txt")
        if not os.path.isfile(ppath):
            miss += 1
            continue
        pred = open(ppath, encoding="utf-8", errors="ignore").read().strip()
        pred = cls_token(pred, classes) or pred
        pred_stems.append(stem)
        confusion[(gt, pred)] = confusion.get((gt, pred), 0) + 1
        if gt == pred:
            n_ok += 1
        else:
            wrong[stem] = (gt, pred)
        details.append([ds_name, stem, gt, pred, "OK" if gt == pred else "WRONG"])
    n_total = sum(confusion.values()) + miss
    return {"n": n_total, "ok": n_ok, "miss": miss, "confusion": confusion,
            "classes": classes, "wrong": wrong, "pred_stems": pred_stems}


# ------------------------------------------------------------------ 可视化

def _load_render_img(ds_dir, stem, cell_w):
    """Read image; upscale tiny slices (NEAREST) / downscale frames to cell_w."""
    import cv2

    p = find_image_path(ds_dir, stem)
    if p is None:
        return None
    img = cv2.imread(p)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w != cell_w:
        interp = cv2.INTER_NEAREST if w < cell_w else cv2.INTER_AREA
        img = cv2.resize(img, (cell_w, max(1, int(round(h * cell_w / w)))),
                         interpolation=interp)
    return img


def _draw_boxes(img, items, color, tag, W, H):
    import cv2

    for cls_name, box in items:
        p1 = (int(round(box[0] * W)), int(round(box[1] * H)))
        p2 = (int(round(box[2] * W)), int(round(box[3] * H)))
        cv2.rectangle(img, p1, p2, color, 2)
        label = f"{tag}:{cls_name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        ty = max(th + 4, p1[1] - 3)
        cv2.rectangle(img, (p1[0], ty - th - 3), (p1[0] + tw + 3, ty), color, -1)
        cv2.putText(img, label, (p1[0] + 1, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_pose_row(img, vals, color, W, H):
    import cv2

    p1 = (int(round(vals[0] * W)), int(round(vals[1] * H)))
    p2 = (int(round(vals[2] * W)), int(round(vals[3] * H)))
    cv2.rectangle(img, p1, p2, color, 2)
    n_kpt = (len(vals) - 4) // 3
    pts = []
    for k in range(n_kpt):
        x, y, v = vals[4 + 3 * k: 4 + 3 * k + 3]
        pi = (int(round(x * W)), int(round(y * H)), int(v))
        pts.append(pi)
        c = color if v > 0 else (128, 128, 128)
        cv2.circle(img, pi[:2], 4, c, -1, cv2.LINE_AA)
        cv2.circle(img, pi[:2], 4, (255, 255, 255), 1, cv2.LINE_AA)
    for a, b in zip(pts, pts[1:]):
        if a[2] > 0 and b[2] > 0:
            cv2.line(img, a[:2], b[:2], color, 1, cv2.LINE_AA)


def _render_pages(cells, out_prefix, cols, rows_per_page):
    """cells: [(caption, img)] -> numbered jpg pages, returns page paths."""
    import cv2
    import numpy as np

    pages = []
    per_page = cols * rows_per_page
    for pi in range(0, len(cells), per_page):
        chunk = cells[pi: pi + per_page]
        ch = max(c[1].shape[0] for c in chunk) + 22
        cw = max(c[1].shape[1] for c in chunk) + 4
        grid = np.full((rows_per_page * ch, cols * cw, 3), 18, dtype=np.uint8)
        for i, (caption, img) in enumerate(chunk):
            y = (i // cols) * ch
            x = (i % cols) * cw
            grid[y + 18:y + 18 + img.shape[0], x + 2:x + 2 + img.shape[1]] = img
            cv2.putText(grid, caption, (x + 3, y + 13), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (255, 220, 80), 1, cv2.LINE_AA)
        p = f"{out_prefix}_{len(pages)}.jpg"
        cv2.imwrite(p, grid)
        pages.append(p)
    return pages


def visualize_boxes(ds_name, ds_dir, run_dir, result, args):
    """detect/labelme: random sample pages + ALL failure (FP/FN) pages."""
    per = result["per_image"]
    vis_dir = os.path.join(run_dir, "vis", ds_name)
    os.makedirs(vis_dir, exist_ok=True)

    def render(stems, tag):
        cells = []
        for stem in stems:
            img = _load_render_img(ds_dir, stem, 460)
            if img is None:
                continue
            H, W = img.shape[:2]
            info = per[stem]
            _draw_boxes(img, info["gts"], GT_COLOR, "GT", W, H)
            _draw_boxes(img, info["preds"], PRED_COLOR, "VLM", W, H)
            reason = f"FP{info['fp']}/FN{info['fn']}" if tag == "FAIL" else ""
            cells.append((f"{stem[:30]} {reason}".strip(), img))
        return cells

    fail_stems = sorted(s for s, v in per.items()
                        if not v.get("missing") and (v["fp"] or v["fn"]))
    rng = random.Random(0)
    sample_stems = rng.sample(sorted(per), min(args.vis, len(per)))
    cells = render(sample_stems, "sample")
    sample_pages = _render_pages(cells, os.path.join(vis_dir, "sample"), 3, 3)
    n_sample = len(cells)
    truncated = 0
    if fail_stems:
        if len(fail_stems) > args.vis_fail_max:
            truncated = len(fail_stems) - args.vis_fail_max
            fail_stems = fail_stems[: args.vis_fail_max]
        cells = render(fail_stems, "FAIL")
        fail_pages = _render_pages(cells, os.path.join(vis_dir, "fail"), 3, 3)
        n_fail = len(cells)
    else:
        fail_pages, n_fail = [], 0
    note = f"（超限截断 {truncated} 张）" if truncated else ""
    return [f"- 可视化：随机抽样 {n_sample} 张 → `vis/{ds_name}/sample_*.jpg`；"
            f"失败 {n_fail} 张全部画出 → `vis/{ds_name}/fail_*.jpg`{note}"]


def visualize_pose(ds_name, ds_dir, run_dir, result, args):
    """pose: random sample pages + failures = 框漏配 / 关键点误差超 PCK 阈值."""
    per = result["per_image"]
    vis_dir = os.path.join(run_dir, "vis", ds_name)
    os.makedirs(vis_dir, exist_ok=True)

    def render(stems, tag):
        cells = []
        for stem in stems:
            img = _load_render_img(ds_dir, stem, 460)
            if img is None:
                continue
            H, W = img.shape[:2]
            info = per[stem]
            for _c, vals in info["gt"]:
                _draw_pose_row(img, vals, GT_COLOR, W, H)
            for _c, vals in info["pred"]:
                _draw_pose_row(img, vals, PRED_COLOR, W, H)
            reason = ""
            if tag == "FAIL":
                parts = []
                if info["un_gt"] or info["un_pr"]:
                    parts.append(f"漏配GT{info['un_gt']}/FP{info['un_pr']}")
                if info["max_err"] is not None and info["max_err"] > result["max_pck"]:
                    parts.append(f"nErr{info['max_err']:.2f}")
                reason = " ".join(parts)
            cells.append((f"{stem[:30]} {reason}".strip(), img))
        return cells

    fail_stems = sorted(s for s, v in per.items()
                        if not v.get("missing") and
                        (v["un_gt"] or v["un_pr"]
                         or (v["max_err"] is not None and v["max_err"] > result["max_pck"])))
    rng = random.Random(0)
    sample_stems = rng.sample(sorted(per), min(args.vis, len(per)))
    cells = render(sample_stems, "sample")
    sample_pages = _render_pages(cells, os.path.join(vis_dir, "sample"), 3, 3)
    n_sample = len(cells)
    truncated = 0
    if fail_stems:
        if len(fail_stems) > args.vis_fail_max:
            truncated = len(fail_stems) - args.vis_fail_max
            fail_stems = fail_stems[: args.vis_fail_max]
        cells = render(fail_stems, "FAIL")
        fail_pages = _render_pages(cells, os.path.join(vis_dir, "fail"), 3, 3)
        n_fail = len(cells)
    else:
        fail_pages, n_fail = [], 0
    note = f"（超限截断 {truncated} 张）" if truncated else ""
    return [f"- 可视化：随机抽样 {n_sample} 张 → `vis/{ds_name}/sample_*.jpg`；"
            f"失败 {n_fail} 张全部画出（漏配/关键点误差>{result['max_pck']}）→ "
            f"`vis/{ds_name}/fail_*.jpg`{note}"]


def visualize_cls(ds_name, ds_dir, run_dir, result, args):
    """cls: random sample pages (对绿字/错红字) + 全部分错样本."""
    import cv2

    wrong = result["wrong"]
    vis_dir = os.path.join(run_dir, "vis", ds_name)
    os.makedirs(vis_dir, exist_ok=True)

    def render(stems, tag):
        cells = []
        for stem in stems:
            img = _load_render_img(ds_dir, stem, 150)
            if img is None:
                continue
            gt, pred = result["wrong"].get(stem, ("", ""))
            ok = gt == pred or tag == "sample" and gt == pred
            color = (100, 230, 130) if gt == pred else (60, 60, 255)
            cv2.putText(img, f"GT:{gt}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"VLM:{pred}", (3, img.shape[0] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
            cap = f"{stem[:22]} [{tag}]" + ("" if gt == pred else " WRONG")
            cells.append((cap, img))
        return cells

    pred_stems = sorted(result.get("pred_stems") or [])
    rng = random.Random(0)
    sample_stems = rng.sample(pred_stems, min(args.vis, len(pred_stems))) if pred_stems else []
    cells = render(sample_stems, "sample")
    sample_pages = _render_pages(cells, os.path.join(vis_dir, "sample"), 6, 6)
    n_sample = len(cells)
    fail_stems = sorted(wrong)[: args.vis_fail_max]
    cells = render(fail_stems, "FAIL")
    fail_pages = _render_pages(cells, os.path.join(vis_dir, "fail"), 6, 6)
    return [f"- 可视化：随机抽样 {n_sample} 张（对绿字/错红字）→ `vis/{ds_name}/sample_*.jpg`；"
            f"分错 {len(wrong)} 张全部画出 → `vis/{ds_name}/fail_*.jpg`（共 {len(fail_pages)} 页）"]


# --------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score testenv results against ground truth.")
    ap.add_argument("--run", required=True, help="run name under testenv/results (or abs path)")
    ap.add_argument("--datasets", default="", help="comma list; default = all found in the run")
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--pose-match-iou", type=float, default=0.3)
    ap.add_argument("--pck", default="0.05,0.1")
    ap.add_argument("--vis", type=int, default=12,
                    help="random-sample comparison count per dataset (0 = off)")
    ap.add_argument("--vis-fail-max", type=int, default=200,
                    help="max failing images visualised per dataset (failures have priority)")
    args = ap.parse_args()

    run_dir = args.run if os.path.isabs(args.run) else os.path.join(RES_ROOT, args.run)
    if not os.path.isdir(run_dir):
        print(f"[error] run dir not found: {run_dir}", file=sys.stderr)
        return 1
    only = {s.strip() for s in args.datasets.split(",") if s.strip()}

    details = []
    sections = []
    print(f"[eval] run={run_dir}")
    for ds_name in sorted(os.listdir(run_dir)):
        ds_dir = os.path.join(DS_ROOT, ds_name)
        if only and ds_name not in only:
            continue
        if not os.path.isdir(ds_dir):
            if os.path.isdir(os.path.join(run_dir, ds_name, "labels")):
                print(f"  [skip] {ds_name}: no matching dataset")
            continue
        kind = ("pose" if ds_name.startswith("pose_") else
                "cls" if ds_name.startswith("cls_") else
                "labelme" if ds_name.startswith("labelme_") else "detect")

        if kind == "pose":
            r = eval_pose(ds_name, ds_dir, run_dir, args, details)
            lines = [f"### {ds_name}（pose, {r['n_kpt']} 关键点）", "",
                     f"- 框匹配：GT {r['n_gt']} / 预测 {r['n_pred']} / 匹配 {r['matched']}"
                     f"（缺预测文件 {r['miss']}）",
                     f"- MPJPE = **{r['mpjpe_px']:.1f} px**（归一化 **{r['mpjpe_n']:.4f}**，{r['n_err']} 个点）"]
            for t in r["pck_list"]:
                lines.append(f"- PCK@{t} = {r['pck'][t]}/{r['n_err']} = {pct(r['pck'][t], r['n_err'])}")
            if r["per_kpt"]:
                lines += ["", "| 关键点 | 样本 | 平均归一化误差 |", "|---|---|---|"]
                for k in sorted(r["per_kpt"]):
                    e = r["per_kpt"][k]
                    lines.append(f"| {k} | {len(e)} | {sum(e) / len(e):.4f} |")
            score = r["mpjpe_n"]
        elif kind == "cls":
            r = eval_cls(ds_name, ds_dir, run_dir, args, details)
            acc = pct(r["ok"], r["n"] - r["miss"]) if r["n"] - r["miss"] else "-"
            lines = [f"### {ds_name}（分类）", "",
                     f"- 样本 {r['n']}（缺预测 {r['miss']}）→ 准确率 **{acc}**（{r['ok']}/{r['n'] - r['miss']}）"]
            if r["confusion"]:
                labels = sorted({k for pair in r["confusion"] for k in pair})
                lines += ["", "| 真\\预测 | " + " | ".join(labels) + " |",
                          "|---" * (len(labels) + 1) + "|"]
                for g in labels:
                    row = [str(r["confusion"].get((g, p), 0)) for p in labels]
                    lines.append(f"| **{g}** | " + " | ".join(row) + " |")
            score = (r["ok"] / (r["n"] - r["miss"])) if r["n"] - r["miss"] else 0.0
        else:
            r = eval_detect(ds_name, ds_dir, run_dir, args, details, kind=kind)
            f1 = (2 * r["tp"] / (2 * r["tp"] + r["fp"] + r["fn"])) \
                if (2 * r["tp"] + r["fp"] + r["fn"]) else 0.0
            lines = [f"### {ds_name}（{kind}, IoU≥{args.match_iou}）", "",
                     f"- TP {r['tp']} · FP {r['fp']} · FN {r['fn']}（缺预测文件 {r['miss']}）",
                     f"- Precision **{pct(r['tp'], r['tp'] + r['fp'])}** · "
                     f"Recall(全部) **{pct(r['tp'], r['tp'] + r['fn'])}** · "
                     f"Recall(已测) **{pct(r['tp'], r['tp'] + r['fn_measured'])}** · "
                     f"F1 {f1:.3f} · 平均 IoU {r['miou']:.3f}",
                     f"- 类无关定位（类别名不可靠时看这个）："
                     f"P {pct(r['tp_free'], r['tp_free'] + r['fp_free'])} · "
                     f"R(已测) {pct(r['tp_free'], r['tp_free'] + r['fn_free'])} · "
                     f"平均 IoU {r['miou_free']:.3f}"]
            if r["per_cls"]:
                lines += ["", "| 类别 | TP | FP | FN | P | R |", "|---|---|---|---|---|---|"]
                for c in sorted(r["per_cls"]):
                    tp, fp, fn = r["per_cls"][c]
                    lines.append(f"| {c} | {tp} | {fp} | {fn} | {pct(tp, tp + fp)} | {pct(tp, tp + fn)} |")
            score = f1

        if args.vis > 0:
            if kind == "cls":
                lines += visualize_cls(ds_name, ds_dir, run_dir, r, args)
            elif kind == "pose":
                lines += visualize_pose(ds_name, ds_dir, run_dir, r, args)
            else:
                lines += visualize_boxes(ds_name, ds_dir, run_dir, r, args)
        sections.append((ds_name, lines, score))
        print(f"  [done] {ds_name}")

    report = ["# testenv 评测报告", "", f"- run: `{run_dir}`",
              f"- 匹配阈值：detect IoU≥{args.match_iou}，pose 框 IoU≥{args.pose_match_iou}",
              "- 可视化：**绿 = 标准答案(GT)，红 = VLM 预测**；`fail_*` 页覆盖全部失败样例", ""]
    report += ["## 总览", "", "| 数据集 | 得分 |", "|---|---|"]
    for name, _lines, score in sections:
        report.append(f"| {name} | {score:.3f} |" if isinstance(score, float)
                      else f"| {name} | {score} |")
    for name, lines, _score in sections:
        report += ["", *lines]

    out_md = os.path.join(run_dir, "eval_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    out_csv = os.path.join(run_dir, "eval_details.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "image", "class_or_kpt", "iou_or_error", "verdict"])
        for row in details:
            w.writerow(row + [""] * (5 - len(row)))
    print(f"[report] {out_md}")
    print(f"[details] {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
