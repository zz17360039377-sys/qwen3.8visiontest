#!/usr/bin/env python3
"""eval_pose -- measure how accurate the VLM's pose annotation actually is.

One pass over a set of full frames:
  1. run the VLM pose prompt (the production path: resize -> prompt -> parse
     -> NMS);
  2. build reference labels on the same frames from
       * a YOLO-pose model (--ref-model, e.g. the user's armor_pose.pt), and/or
       * existing YOLO-pose label files (--ref-labels);
  3. match VLM instances to reference instances by greedy bbox IoU
     (--match-iou) and compare keypoints.

Metrics: detection precision/recall, bbox IoU, per-keypoint error in pixels
and normalised by the reference box diagonal, MPJPE and PCK@0.05/0.1, plus a
per-keypoint (corner) breakdown. Worst matches are rendered as overlay pages
(green = reference, red = VLM).

Outputs (``--out``, default ``eval_pose_<ts>``): ``eval_pose_report.md``,
``eval_pose_details.csv``, ``compare_N.jpg`` overlay pages.

Run:
  python3 eval_pose.py --video <mp4> --max-images 12 \
      --ref-model ~/Desktop/模型训练/CNN/yolo_pose_model/armor_pose.pt
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.io_utils import extract_frames, list_image_dir, load_config  # noqa: E402
from core.parse_geometry import load_json, nms, parse_to_annotations   # noqa: E402
from core.prompts import build_prompt                                  # noqa: E402
from core.vlm_client import VLMClient, chat_raw                        # noqa: E402

CNN_DIR = os.path.expanduser("~/Desktop/模型训练/CNN")
DEFAULT_REF = os.path.join(CNN_DIR, "yolo_pose_model", "armor_pose.pt")


# ------------------------------------------------------------------ helpers

def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def greedy_match(vlm_boxes, ref_boxes, thresh: float) -> list[tuple[int, int, float]]:
    """Greedy highest-IoU-first matching. Returns [(vi, ri, iou), ...]."""
    pairs = []
    for vi, vb in enumerate(vlm_boxes):
        for ri, rb in enumerate(ref_boxes):
            v = iou(vb, rb)
            if v >= thresh:
                pairs.append((v, vi, ri))
    pairs.sort(reverse=True)
    used_v, used_r, out = set(), set(), []
    for v, vi, ri in pairs:
        if vi in used_v or ri in used_r:
            continue
        used_v.add(vi)
        used_r.add(ri)
        out.append((vi, ri, v))
    return out


def ref_from_model(model, frame: str) -> list[dict]:
    """ultralytics YOLO-pose -> [{box:[x1,y1,x2,y2], kpts:[[x,y,v]xN]}] in px."""
    res = model(frame, verbose=False, device=DEVICE)[0]
    out = []
    names = res.names or {}
    for i, b in enumerate(res.boxes):
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        kps = []
        if res.keypoints is not None:
            xy = res.keypoints.xy[i].tolist()
            cf = (res.keypoints.conf[i].tolist()
                  if res.keypoints.conf is not None else [1.0] * len(xy))
            kps = [[float(x), float(y), 2 if c >= KPT_CONF else 0]
                   for (x, y), c in zip(xy, cf)]
        out.append({"box": [x1, y1, x2, y2], "kpts": kps,
                    "cls": str(names.get(int(b.cls[0]), int(b.cls[0])))})
    return out


def ref_from_labels(path: str, w: int, h: int, n_kpt: int) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = (float(x) for x in parts[1:5])
            box = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
            kps = []
            vals = parts[5:5 + 3 * n_kpt]
            for i in range(n_kpt):
                if 3 * i + 2 < len(vals):
                    kx, ky, v = float(vals[3 * i]), float(vals[3 * i + 1]), int(float(vals[3 * i + 2]))
                    kps.append([kx * w, ky * h, v])
            out.append({"box": box, "kpts": kps, "cls": parts[0]})
    return out


def vlm_pose(args, client: VLMClient, classes, keypoints, frame: str) -> tuple[list[dict], float]:
    """Production pose path; returns (normalised anns, wall seconds)."""
    data_url, w, h = client.resize_path(frame)
    system, user = build_prompt("pose", classes, keypoints, width=w, height=h)
    res = chat_raw(args.api, args.model,
                   [{"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": user}]}],
                   temperature=args.temperature, max_tokens=768)
    anns = parse_to_annotations(res["content"], w, h, "pose", classes, len(keypoints))
    anns = nms(anns, iou_thresh=args.iou, conf=args.conf)
    return anns, res["wall"]


def vlm_pose_crop(args, client: VLMClient, classes, keypoints, frame: str,
                  box: list, margin: float = 0.30, target: int = 640):
    """Zoom-in pose: crop around *box*, upscale, ask for the one instance.

    Returns (anns in FULL-IMAGE normalised coords, wall_s, crop_info or None).
    Only meaningful when the object is actually inside *box* (e.g. a reference
    box) -- this measures keypoint precision independently of detection.
    """
    import cv2

    img = cv2.imread(frame)
    if img is None:
        return [], 0.0, None
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    mx = (x2 - x1) * margin
    my = (y2 - y1) * margin
    cx1, cy1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    cx2, cy2 = min(W, int(x2 + mx)), min(H, int(y2 + my))
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return [], 0.0, None
    crop = img[cy1:cy2, cx1:cx2]
    scale = min(4.0, max(1.0, target / max(crop.shape[:2])))
    if scale > 1.0:
        crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
                          interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", crop)
    if not ok:
        return [], 0.0, None
    import base64
    url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    ch, cw = crop.shape[:2]
    system, user = build_prompt("pose", classes, keypoints, width=cw, height=ch,
                                extra="The crop contains exactly ONE object; return exactly one result.")
    res = chat_raw(args.api, args.model,
                   [{"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": url}},
                        {"type": "text", "text": user}]}],
                   temperature=args.temperature, max_tokens=768)
    anns = parse_to_annotations(res["content"], cw, ch, "pose", classes, len(keypoints))
    anns = nms(anns, iou_thresh=args.iou, conf=args.conf)
    # remap crop-normalised coords -> full-image normalised
    out = []
    for a in anns:
        a = dict(a)
        a["x1"] = (cx1 + a["x1"] * cw) / W
        a["x2"] = (cx1 + a["x2"] * cw) / W
        a["y1"] = (cy1 + a["y1"] * ch) / H
        a["y2"] = (cy1 + a["y2"] * ch) / H
        if a.get("keypoints"):
            a["keypoints"] = [[(cx1 + k[0] * cw) / W, (cy1 + k[1] * ch) / H, k[2]]
                              for k in a["keypoints"]]
        out.append(a)
    info = {"crop": [cx1, cy1, cx2, cy2], "scale": scale}
    return out, res["wall"], info


# ------------------------------------------------------------------ report

def make_compare_pages(items: list, out_dir: str, cell_w: int = 460,
                       cols: int = 2, max_n: int = 12) -> list[str]:
    """Overlay pages for the worst matches: green=reference, red=VLM."""
    import cv2
    import numpy as np

    worst = sorted(items, key=lambda x: x["mean_norm_err"], reverse=True)[:max_n]
    pages = []
    per_page = cols * 3
    SKELETON = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for pi in range(0, len(worst), per_page):
        cells = []
        for it in worst[pi:pi + per_page]:
            img = cv2.imread(it["frame"])
            if img is None:
                continue
            scale = cell_w / img.shape[1]
            img = cv2.resize(img, (cell_w, int(img.shape[0] * scale)))
            for box, kps, col in ((it["ref"]["box"], it["ref"]["kpts"], (0, 255, 0)),
                                  (it["vlm_box"], it["vlm"]["keypoints"], (0, 0, 255))):
                p = [int(round(v * scale)) for v in box]
                cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), col, 2)
                for i, kp in enumerate(kps):
                    if len(kp) > 2 and kp[2] > 0:
                        c = (int(kp[0] * scale), int(kp[1] * scale))
                        cv2.circle(img, c, 4, col, -1)
                        cv2.putText(img, str(i), (c[0] + 4, c[1] - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
                for m, n in SKELETON:
                    if (len(kps) > max(m, n) and kps[m][2] > 0 and kps[n][2] > 0):
                        cv2.line(img, (int(kps[m][0] * scale), int(kps[m][1] * scale)),
                                 (int(kps[n][0] * scale), int(kps[n][1] * scale)), col, 1)
            cv2.putText(img, f"IoU={it['iou']:.2f} nErr={it['mean_norm_err']:.3f}",
                        (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(img)
        if not cells:
            continue
        ch = max(c.shape[0] for c in cells)
        rows_n = (len(cells) + cols - 1) // cols
        grid = np.zeros((rows_n * (ch + 4), cols * (cell_w + 4), 3), dtype=np.uint8)
        for i, c in enumerate(cells):
            y, x = (i // cols) * (ch + 4), (i % cols) * (cell_w + 4)
            grid[y:y + c.shape[0], x:x + c.shape[1]] = c
        p = os.path.join(out_dir, f"compare_{len(pages)}.jpg")
        cv2.imwrite(p, grid)
        pages.append(p)
    return pages


DEVICE = "cpu"       # set by --ref-device in main()
KPT_CONF = 0.3


def main(argv=None) -> int:
    global DEVICE
    ap = argparse.ArgumentParser(description="Measure VLM pose annotation accuracy.")
    ap.add_argument("--images", help="folder of full frames")
    ap.add_argument("--video", help="video file (samples --max-images frames)")
    ap.add_argument("--max-images", type=int, default=12)
    ap.add_argument("--ref-model", default=DEFAULT_REF, help="YOLO-pose .pt ('' disables)")
    ap.add_argument("--ref-labels", help="folder of YOLO-pose .txt labels")
    ap.add_argument("--ref-device", default="cpu", help="cpu keeps GPU free for the VLM")
    ap.add_argument("--crop", action="store_true",
                    help="zoom-in mode: crop around each reference box, upscale, "
                         "measure keypoint precision independently of detection")
    ap.add_argument("--kpt-conf", type=float, default=0.3)
    ap.add_argument("--config", default="config.example.yaml", help="classes/keypoints/skeleton")
    ap.add_argument("--match-iou", type=float, default=0.3)
    ap.add_argument("--conf", type=float, default=0.0)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--api", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    DEVICE = args.ref_device
    KPT_CONF = args.kpt_conf

    classes, keypoints, skeleton = load_config(args.config)
    if not classes or not keypoints:
        print("[error] config must define classes and keypoints", file=sys.stderr)
        return 1
    n_kpt = len(keypoints)

    out_dir = args.out or f"eval_pose_{datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs(out_dir, exist_ok=True)
    if args.video:
        frames = extract_frames(args.video, args.max_images, out_dir)
    elif args.images:
        frames = list_image_dir(args.images)
    else:
        print("[error] --images or --video required", file=sys.stderr)
        return 1
    frames = frames[: args.max_images]
    if not frames:
        print("[error] no frames", file=sys.stderr)
        return 1

    ref_model = None
    if args.ref_model and os.path.exists(os.path.expanduser(args.ref_model)):
        from ultralytics import YOLO
        ref_model = YOLO(os.path.expanduser(args.ref_model))
        print(f"[ref] model {args.ref_model} on {DEVICE}")
    elif args.ref_model:
        print(f"[warn] ref model not found: {args.ref_model}")

    client = VLMClient(api_base=args.api, model=args.model,
                       temperature=args.temperature, max_side=args.max_side)

    rows = []
    matched_items = []
    n_ref_total = n_vlm_total = 0
    walls = []
    for fi, frame in enumerate(frames, 1):
        name = os.path.basename(frame)
        import cv2
        img = cv2.imread(frame)
        if img is None:
            continue
        H, W = img.shape[:2]

        refs = []
        if ref_model is not None:
            refs += ref_from_model(ref_model, frame)
        if args.ref_labels:
            lp = os.path.join(os.path.expanduser(args.ref_labels),
                              os.path.splitext(name)[0] + ".txt")
            if os.path.exists(lp):
                refs += ref_from_labels(lp, W, H, n_kpt)

        walls_one = []
        if args.crop:
            # zoom-in mode: one VLM call per reference box -> 1:1 pairs (keep ref index!)
            anns, pairs = [], []
            for ri, r in enumerate(refs):
                ca, wall, _info = vlm_pose_crop(args, client, classes, keypoints, frame, r["box"])
                walls_one.append(wall)
                if ca:
                    pairs.append((ri, ca[0]))
                    anns.append(ca[0])
            vlm_boxes = [(a["x1"] * W, a["y1"] * H, a["x2"] * W, a["y2"] * H) for a in anns]
            matches = [(i, pairs[i][0], iou(vlm_boxes[i], [r["box"] for r in refs][pairs[i][0]]))
                       for i in range(len(pairs))]
        else:
            try:
                anns, wall = vlm_pose(args, client, classes, keypoints, frame)
            except Exception as e:  # noqa: BLE001
                print(f"  [{fi}/{len(frames)}] {name}: VLM ERROR {e}", file=sys.stderr)
                continue
            walls_one.append(wall)
            vlm_boxes = [(a["x1"] * W, a["y1"] * H, a["x2"] * W, a["y2"] * H) for a in anns]
            matches = greedy_match(vlm_boxes, [r["box"] for r in refs], args.match_iou)
        walls.extend(walls_one)
        n_ref_total += len(refs)
        n_vlm_total += len(anns)

        for vi, ri, iv in matches:
            a, r = anns[vi], refs[ri]
            diag = math.hypot(r["box"][2] - r["box"][0], r["box"][3] - r["box"][1]) or 1.0
            errs, norms, per_kpt = [], [], []
            for ki in range(n_kpt):
                rk = r["kpts"][ki] if ki < len(r["kpts"]) else None
                vk = a["keypoints"][ki] if a["keypoints"] and ki < len(a["keypoints"]) else None
                if not rk or rk[2] <= 0 or not vk:
                    per_kpt.append("")
                    continue
                ex = vk[0] * W - rk[0]
                ey = vk[1] * H - rk[1]
                e = math.hypot(ex, ey)
                errs.append(e)
                norms.append(e / diag)
                per_kpt.append(f"{e:.1f}")
            mean_norm = statistics.mean(norms) if norms else 1.0
            row = {"file": name, "iou": f"{iv:.3f}", "n_kpt_used": len(norms),
                   "mpjpe_px": f"{statistics.mean(errs):.1f}" if errs else "",
                   "mean_norm_err": f"{mean_norm:.4f}",
                   "pck05": sum(1 for x in norms if x <= 0.05),
                   "pck10": sum(1 for x in norms if x <= 0.1)}
            for ki in range(n_kpt):
                row[f"kpt{ki}_px"] = per_kpt[ki] if ki < len(per_kpt) else ""
            rows.append(row)
            matched_items.append({"frame": frame, "ref": r, "vlm": a,
                                  "vlm_box": vlm_boxes[vi], "iou": iv,
                                  "mean_norm_err": mean_norm})
        print(f"  [{fi}/{len(frames)}] {name}: ref={len(refs)} vlm={len(anns)} "
              f"matched={len(matches)} ({wall:.1f}s)")

    if not rows:
        print("[error] no matches -- nothing to report", file=sys.stderr)
        return 1

    with open(os.path.join(out_dir, "eval_pose_details.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- aggregate
    ious = [float(r["iou"]) for r in rows]
    norms = [[float(r["mean_norm_err"])] for r in rows]
    mpjpe = [float(r["mpjpe_px"]) for r in rows if r["mpjpe_px"]]
    pck05 = sum(int(r["pck05"]) for r in rows)
    pck10 = sum(int(r["pck10"]) for r in rows)
    n_kpt_used = sum(int(r["n_kpt_used"]) for r in rows)
    mean_norm = statistics.mean(x[0] for x in norms)

    # per-keypoint breakdown
    kpt_errs: dict[int, list[float]] = {}
    for it in matched_items:
        r, a = it["ref"], it["vlm"]
        W = H = None
        import cv2
        img = cv2.imread(it["frame"])
        if img is None:
            continue
        H, W = img.shape[:2]
        diag = math.hypot(r["box"][2] - r["box"][0], r["box"][3] - r["box"][1]) or 1.0
        for ki in range(n_kpt):
            rk = r["kpts"][ki] if ki < len(r["kpts"]) else None
            vk = a["keypoints"][ki] if a["keypoints"] and ki < len(a["keypoints"]) else None
            if not rk or rk[2] <= 0 or not vk:
                continue
            kpt_errs.setdefault(ki, []).append(
                math.hypot(vk[0] * W - rk[0], vk[1] * H - rk[1]) / diag)

    precision = len(rows) / n_vlm_total if n_vlm_total else 0
    recall = len(rows) / n_ref_total if n_ref_total else 0
    verdict = (
        "**结论：关键点平均误差 <5% 框对角线，VLM pose 可直接产出可用标注。**" if mean_norm < 0.05 else
        "**结论：关键点平均误差 5–10% 框对角线，VLM pose 适合做预标注，建议画布微调后落库。**" if mean_norm < 0.10 else
        "**结论：关键点误差 >10% 框对角线，VLM pose 只能提供粗框；关键点需人工/模型细化。**")

    lines = [
        "# VLM pose 标注精度评估",
        "",
        f"- 时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- 帧：{len(frames)} 张；参考：{'模型 ' + str(args.ref_model) if ref_model else ''}"
        f"{' + 标注 ' + args.ref_labels if args.ref_labels else ''}",
        f"- 关键点：{n_kpt} 个（{', '.join(k['name'] for k in keypoints)}）",
        f"- 匹配阈值：IoU ≥ {args.match_iou}（贪心）",
        "",
        "## 检测层",
        "",
        f"- 参考实例 {n_ref_total}，VLM 实例 {n_vlm_total}，匹配 {len(rows)}",
        f"- 精确率 P = {precision:.1%}，召回率 R = {recall:.1%}",
        f"- 匹配框平均 IoU = {statistics.mean(ious):.3f}",
        "",
        "## 关键点层（参考关键点可见处）",
        "",
        f"- 平均归一化误差 MPJPE/对角线 = **{mean_norm:.3f}**（{n_kpt_used} 个点）",
        f"- MPJPE = {statistics.mean(mpjpe):.1f} px" if mpjpe else "- 无像素误差数据",
        f"- PCK@0.05 = {pck05}/{n_kpt_used} = {pck05 / max(1, n_kpt_used):.1%}",
        f"- PCK@0.10 = {pck10}/{n_kpt_used} = {pck10 / max(1, n_kpt_used):.1%}",
        f"- 速度：{statistics.mean(walls):.2f} s/帧（{60 / statistics.mean(walls):.0f} 帧/分钟）",
        "",
        "| 关键点 | 样本 | 平均归一化误差 |",
        "|---|---|---|",
    ]
    for ki in range(n_kpt):
        errs = kpt_errs.get(ki)
        nm = keypoints[ki]["name"] if ki < len(keypoints) else f"kpt{ki}"
        lines.append(f"| {ki} {nm} | {len(errs)} | {statistics.mean(errs):.3f} |" if errs
                     else f"| {ki} {nm} | 0 | - |")
    lines += ["", verdict, ""]
    pages = make_compare_pages(matched_items, out_dir)
    if pages:
        lines.append("## 最差匹配对比（绿=参考，红=VLM）")
        lines.append("")
        lines += [f"- `{os.path.basename(p)}`" for p in pages]
    report = os.path.join(out_dir, "eval_pose_report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[report] {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
