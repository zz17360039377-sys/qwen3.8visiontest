#!/usr/bin/env python3
"""demo_annotate -- annotate ONE dataset image with YOUR OWN prompt.

The prompt is yours and is passed VERBATIM as the user message (plus the
image). This script only handles: image encoding -> model call -> response
parsing -> prediction file -> optional GT-vs-VLM overlay.

Usage:
  python3 tools/demo_annotate.py --dataset pose_okbuff2 --pick best \
      --prompt-file myprompt.txt --run demo --show

  --dataset   dataset name under testenv/datasets/
  --stem/--pick  image stem, or "best"/"multi" to auto-pick (multi = most objects)
  --prompt-file / --prompt  YOUR prompt text (verbatim user message)
  --system    optional system prompt (default: none)
  --strict    append a generic JSON output contract to your prompt
  --run       results run name (default "demo")
  --show      render GT(green) vs VLM(red) overlay next to the raw reply

Prediction lands in  testenv/results/<run>/<dataset>/labels/<stem>.txt  in the
standard eval format, so tools/eval_testenv.py --run <run> works afterwards.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, ROOT)

API = "http://127.0.0.1:8080/v1/chat/completions"
DS_ROOT = os.path.join(ROOT, "testenv", "datasets")
RES_ROOT = os.path.join(ROOT, "testenv", "results")
MAXSIDE = 1280

STRICT_SUFFIX = (
    "\n\nRespond ONLY with a single JSON object, no prose:\n"
    '{"results": [{"label": "<class name>", "bbox": [x1, y1, x2, y2]'
    + ', "keypoints": [[x1, y1, v], ...]}' + '], "score": 0..1}]}. '
    "Coordinates in integer pixels of the image. Use an empty list if none."
)


def load_classes(ds):
    root = ds if os.path.isdir(ds) else os.path.join(DS_ROOT, ds)
    p = os.path.join(root, "classes.txt")
    return [l.strip() for l in open(p, encoding="utf-8")] if os.path.isfile(p) else []


def gt_rows(ds, stem):
    """Ground truth rows as (cls_name, [x1,y1,x2,y2] + kpts)."""
    from tools.eval_testenv import load_yolo
    root = ds if os.path.isdir(ds) else os.path.join(DS_ROOT, ds)
    classes = load_classes(ds)
    p = os.path.join(root, "labels", stem + ".txt")
    return [(r[0], r[1]) for r in load_yolo(p, classes)]


def n_kpt_of(ds, stem):
    rows = gt_rows(ds, stem)
    return (len(rows[0][1]) - 4) // 3 if rows else 0


def pick_stem(ds, mode):
    from tools.eval_testenv import load_yolo
    root = ds if os.path.isdir(ds) else os.path.join(DS_ROOT, ds)
    lbl_dir = os.path.join(root, "labels")
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(lbl_dir))
    best, best_key = None, (-1, -1, -1)
    for s in stems:
        rows = gt_rows(ds, s)
        if not rows:
            continue
        boxes = [r[1][:4] for r in rows]
        area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes) / len(boxes)
        key = (len(rows), round(area, 6), 1)  # 多目标优先，其次平均框大
        if mode == "multi" and key > best_key:
            best, best_key = s, key
    if mode == "best" or best is None:
        # best: 平均框最大的单目标图（目标大更好演示）
        for s in stems:
            rows = gt_rows(ds, s)
            if not rows:
                continue
            boxes = [r[1][:4] for r in rows]
            area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes) / len(boxes)
            if area > best_key[1]:
                best, best_key = s, (1, area, 1)
    return best or stems[0]


def call_vlm(img_bytes: bytes, prompt: str, system: str | None, temperature: float):
    import time
    import urllib.request

    b64 = base64.b64encode(img_bytes).decode()
    content = [{"type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + b64}},
               {"type": "text", "text": prompt}]
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": content}]
    body = json.dumps({"model": "qwen", "messages": messages,
                       "temperature": temperature,
                       "max_tokens": 2048,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(API, data=body,
                                 headers={"Content-Type": "application/json"})
    for attempt in range(2):
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
        wall = time.time() - t0
        msg = data["choices"][0]["message"]
        usage = data.get("usage", {})
        content = (msg.get("content") or "").strip()
        if content:
            return content, wall, usage
        body["max_tokens"] = max(body.get("max_tokens", 2048), 4096)
        req = urllib.request.Request(API, data=body,
                                     headers={"Content-Type": "application/json"})
    return "", wall, usage


def _qwen_native_anns(content: str, W: int, H: int):
    """Qwen 原生标注标签： <points ...> 与 <box>(x,y),(x,y)</box>（0-1000 归一化）。"""
    import re as _re
    anns = []
    for m in _re.finditer(r'<points\s+x1="([\d.]+)"\s+y1="([\d.]+)"([^>]*)>', content):
        x, y = float(m.group(1)), float(m.group(2))
        x, y = x / 1000.0 if x > 1.5 else x, y / 1000.0 if y > 1.5 else y
        alt = ""
        am = _re.search(r'alt="([^"]*)"', m.group(3))
        if am:
            alt = am.group(1)
        anns.append({"label": alt or "target", "class_id": 0,
                     "x1": max(0.0, x - 0.01), "y1": max(0.0, y - 0.012),
                     "x2": min(1.0, x + 0.01), "y2": min(1.0, y + 0.012),
                     "keypoints": None, "score": 0.9})
    for m in _re.finditer(r"<box>\s*\(([\d.]+),\s*([\d.]+)\)\s*,\s*\(([\d.]+),\s*([\d.]+)\)\s*</box>", content):
        x1, y1, x2, y2 = (float(g) / 1000.0 for g in m.groups())
        anns.append({"label": "target", "class_id": 0, "x1": x1, "y1": y1,
                     "x2": x2, "y2": y2, "keypoints": None, "score": 0.9})
    return anns


def parse_response(content: str, W: int, H: int, classes, n_kpt):
    """Reuse the project parser: pixel/normalised autodetect, label->id, clamp."""
    from core.parse_geometry import parse_to_annotations
    if "<points" in content or "<box>" in content:
        return "native", _qwen_native_anns(content, W, H)
    mode = "pose" if n_kpt else "detect"
    anns = parse_to_annotations(content, W, H, mode, classes, n_kpt)
    return mode, anns


def render_overlay(ds, stem, anns, out_path):
    import cv2
    import numpy as np

    from core.draw import draw_overlay
    classes = load_classes(ds)
    gt = gt_rows(ds, stem)
    src = find_img(ds, stem)
    img = cv2.imread(src)
    H, W = img.shape[:2]
    # GT 绿
    g = [{"class_id": classes.index(r[0]) if r[0] in classes else 0, "label": r[0],
          "x1": r[1][0], "y1": r[1][1], "x2": r[1][2], "y2": r[1][3],
          "keypoints": [r[1][4 + 3 * k: 7 + 3 * k] for k in range((len(r[1]) - 4) // 3)],
          "score": 0} for r in gt]
    draw_overlay(src, g, "pose" if any(len(r[1]) > 4 for r in gt) else "detect",
                 out_path + ".gt.tmp.jpg", classes)
    base = cv2.imread(out_path + ".gt.tmp.jpg")
    # VLM 红
    for a in anns:
        p1 = (int(a["x1"] * W), int(a["y1"] * H))
        p2 = (int(a["x2"] * W), int(a["y2"] * H))
        cv2.rectangle(base, p1, p2, (0, 0, 255), 2)
        cv2.putText(base, f"VLM:{a['label']}", (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        for kp in a.get("keypoints") or []:
            if len(kp) > 2 and kp[2] > 0:
                c = (int(kp[0] * W), int(kp[1] * H))
                cv2.circle(base, c, 4, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(base, c, 4, (255, 255, 255), 1, cv2.LINE_AA)
    grid = np.hstack([base])
    cv2.imwrite(out_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    os.remove(out_path + ".gt.tmp.jpg")


def find_img(ds, stem):
    root = ds if os.path.isdir(ds) else os.path.join(DS_ROOT, ds)
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(root, "images", stem + ext)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(stem)


def _wrap_text(text, font, max_w):
    from PIL import Image, ImageDraw
    d = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    lines, cur = [], ""
    for ch in text:
        if d.textlength(cur + ch, font=font) > max_w:
            lines.append(cur); cur = ch
        else:
            cur += ch
    lines.append(cur)
    return lines


def _cjk_font(size):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
              "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"):
        if os.path.isfile(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def handle_extra_mode(args, stem, content, im, wall, usage):
    """caption=图片描述；seg=语义分割多边形（无 GT，能力演示）。"""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    res = os.path.join(RES_ROOT, args.run, os.path.basename(args.dataset.rstrip("/")))
    vis = os.path.join(RES_ROOT, args.run, "vis")
    os.makedirs(vis, exist_ok=True)
    if args.mode == "caption":
        cap_dir = os.path.join(res, "captions")
        os.makedirs(cap_dir, exist_ok=True)
        with open(os.path.join(cap_dir, stem + ".md"), "w", encoding="utf-8") as f:
            f.write(f"# {stem}\n\n{content}\n")
        img = im.copy()
        font = _cjk_font(22)
        lines = _wrap_text(" ".join(content.splitlines()), font, img.size[0] - 20)
        panel = Image.new("RGB", (img.size[0], 30 * len(lines) + 20), (18, 18, 18))
        d = ImageDraw.Draw(panel)
        for i, ln in enumerate(lines):
            d.text((10, 12 + 30 * i), ln, font=font, fill=(230, 230, 230))
        combo = Image.new("RGB", (img.size[0], img.size[1] + panel.size[1]))
        combo.paste(img, (0, 0)); combo.paste(panel, (0, img.size[1]))
        p = os.path.join(vis, f"caption_{stem}.jpg")
        combo.convert("RGB").save(p, quality=92)
        print(f"[caption] {args.dataset}/{stem}  {wall:.1f}s  描述已存 {cap_dir}")
        print("---- 描述 ----\n" + content[:600])
        return f"对比图: {p}"
    # seg
    from core.parse_geometry import load_json
    data = load_json(content)
    if isinstance(data, list) and data:
        data = data[0]                      # 模型可能返回顶层数组
    polys = []
    if isinstance(data, dict):
        for key in ("polygon", "polygons", "points", "segmentation"):
            v = data.get(key)
            if isinstance(v, list) and v and isinstance(v[0], (list, tuple)):
                polys = v   # [x,y] 或 [x,y,v] 点均可
                break
    W, H = im.size
    overlay = im.copy().convert("RGB")
    if polys:
        pts = [(float(x), float(y)) for x, y in polys]
        # 模型可能用自己的尺度：越界时等比缩回图片范围
        mx = max(p[0] for p in pts); my = max(p[1] for p in pts)
        if mx > W or my > H:
            sx, sy = (W - 1) / mx, (H - 1) / my
            pts = [(x * sx, y * sy) for x, y in pts]
        mask = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], (60, 160, 255))
        overlay = cv2.addWeighted(np.array(overlay), 0.6, mask, 0.4, 0)
        overlay = cv2.polylines(overlay, [np.array(pts, dtype=np.int32)], True, (0, 0, 255), 2)
    if not isinstance(overlay, np.ndarray):
        overlay = np.array(overlay)
    p = os.path.join(vis, f"seg_{stem}.jpg")
    cv2.imwrite(p, overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
    pj = os.path.join(res, "labels")
    os.makedirs(pj, exist_ok=True)
    with open(os.path.join(pj, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump({"polygon": polys, "raw": content[:2000]}, f, ensure_ascii=False, indent=1)
    print(f"[seg] {args.dataset}/{stem}  {wall:.1f}s  多边形点数={len(polys)}")
    print("---- 原始回复 ----\n" + content[:400])
    return f"分割叠加图: {p}"


def pick_stem_in(ds_dir, mode):
    from tools.eval_testenv import load_yolo
    lbl_dir = os.path.join(ds_dir, "labels")
    if not os.path.isdir(lbl_dir):
        stems = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(ds_dir, "images")))
        return stems[len(stems) // 2] if stems else None
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(lbl_dir))
    best, best_key = stems[len(stems) // 2] if stems else None, (-1, -1)
    for s in stems:
        rows = [(r[0], r[1]) for r in load_yolo(os.path.join(lbl_dir, s + ".txt"), [])]
        if not rows:
            continue
        area = sum((r[1][2] - r[1][0]) * (r[1][3] - r[1][1]) for r in rows) / len(rows)
        key = (len(rows), round(area, 6)) if mode == "multi" else (round(area, 6), len(rows))
        if key > best_key:
            best, best_key = s, key
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Annotate one demo image with YOUR prompt.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--stem", help="image stem; omit to auto-pick")
    ap.add_argument("--pick", choices=["best", "multi"], default="multi",
                    help="auto-pick: best=平均框最大, multi=目标数最多 (default)")
    ap.add_argument("--prompt-file")
    ap.add_argument("--prompt")
    ap.add_argument("--system", default=None)
    ap.add_argument("--strict", action="store_true", help="追加通用 JSON 输出契约")
    ap.add_argument("--run", default="demo")
    ap.add_argument("--mode", choices=["auto", "detect", "pose", "cls", "caption", "seg"],
                    default="auto", help="caption/seg 为日常图演示模式（无 GT）")
    ap.add_argument("--show", action="store_true", help="渲染 GT(绿) vs VLM(红) 对比图")
    ap.add_argument("--hint", default="", help="追加在用户提示词后的元信息（如数据集类别名）")
    ap.add_argument("--calibrate", action="store_true",
                    help="自动校准模型坐标系（合成标记图实测线性映射并校正预测）")
    ap.add_argument("--kpt", type=int, default=0,
                    help="pose: 关键点数（无 GT 的演示图必须显式指定）")
    ap.add_argument("--fewshot", type=int, default=0,
                    help="pose: 用 K 张画好 GT 的示例图做 few-shot（随目标图一起发）")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args(argv)

    prompt = args.prompt or (open(args.prompt_file, encoding="utf-8").read() if args.prompt_file else "")
    if not prompt.strip():
        print("[error] 需要 --prompt 或 --prompt-file（你的提示词，原样发给模型）")
        return 1
    if args.strict:
        prompt += STRICT_SUFFIX

    ds_dir = args.dataset if os.path.isdir(args.dataset) else os.path.join(DS_ROOT, args.dataset)
    if not os.path.isdir(ds_dir):
        print(f"[error] dataset 不存在: {args.dataset}")
        return 1
    args.dataset = ds_dir          # 下游函数对目录路径自适应
    stem = args.stem or pick_stem_in(ds_dir, args.pick)
    classes = load_classes(args.dataset)
    n_kpt = args.kpt
    if not n_kpt and args.mode in ("auto", "pose"):
        try:
            n_kpt = n_kpt_of(args.dataset, stem)
        except Exception:  # noqa: BLE001 - 演示目录可能没有 GT
            n_kpt = 0
    src = find_img(args.dataset, stem)

    from PIL import Image
    im = Image.open(src).convert("RGB")
    W, H = im.size
    scale = min(1.0, MAXSIDE / max(W, H))
    if scale < 1:
        im = im.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    full_prompt = prompt + (("\n\n" + args.hint) if args.hint else "")
    # 尺寸声明是必要的 grounding 元数据（生产提示词同样包含）
    full_prompt += (f"\n\nThe image is {im.size[0]} x {im.size[1]} pixels "
                    f"(width x height, top-left origin). Report every coordinate as "
                    f"integer pixels in this space.")

    if args.mode == "pose" and args.fewshot > 0:
        # few-shot：K 张画好 GT 的示例图 + 目标图
        import cv2 as _cv2
        lbl_dir = os.path.join(ds_dir, "labels")
        by_cls = {}
        for f in sorted(os.listdir(lbl_dir)):
            st = os.path.splitext(f)[0]
            if st == stem:
                continue
            rows = gt_rows(ds_dir, st)
            if rows:
                by_cls.setdefault(rows[0][0], []).append((st, rows))
        examples = []
        for cls_name in sorted(by_cls):
            if len(examples) >= args.fewshot:
                break
            st, rows = by_cls[cls_name][0]
            examples.append((st, rows))
        parts = []
        for st, rows in examples:
            p = find_img(ds_dir, st)
            ex_im = _cv2.imread(p)
            EH, EW = ex_im.shape[:2]
            for _n, v in rows:
                _cv2.rectangle(ex_im, (int(v[0] * EW), int(v[1] * EH)),
                               (int(v[2] * EW), int(v[3] * EH)), (0, 220, 0), 2)
                n = (len(v) - 4) // 3
                for k in range(n):
                    c = (int(v[4 + 3 * k] * EW), int(v[5 + 3 * k] * EH))
                    _cv2.circle(ex_im, c, 5, (0, 160, 255), -1, _cv2.LINE_AA)
                    _cv2.putText(ex_im, str(k), (c[0] + 6, c[1] - 4),
                                 _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 160, 255), 1, _cv2.LINE_AA)
            ok, buf_ex = _cv2.imencode(".jpg", ex_im)
            parts.append({"type": "image_url",
                          "image_url": {"url": "data:image/jpeg;base64," +
                                        base64.b64encode(buf_ex.tobytes()).decode()}})
        parts.append({"type": "image_url",
                      "image_url": {"url": "data:image/jpeg;base64," +
                                    base64.b64encode(buf.getvalue()).decode()}})
        parts.append({"type": "text", "text": full_prompt})
        messages = ([{"role": "system", "content": args.system}] if args.system else []) + [
            {"role": "user", "content": parts}]
        body = json.dumps({"model": "qwen", "messages": messages,
                           "temperature": args.temperature, "max_tokens": 2048,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()
        import time as _t
        import urllib.request
        req = urllib.request.Request(API, data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = _t.time()
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
        wall = _t.time() - t0
        usage = data.get("usage", {})
        content = (data["choices"][0]["message"].get("content") or "").strip()
    else:
        content, wall, usage = call_vlm(buf.getvalue(), full_prompt, args.system, args.temperature)
    if args.mode == "auto":
        gt = gt_rows(args.dataset, stem)
        n_kpt = (len(gt[0][1]) - 4) // 3 if gt and len(gt[0][1]) > 4 else 0
        args.mode = "pose" if n_kpt else "detect"
    if args.mode in ("caption", "seg"):
        out = handle_extra_mode(args, stem, content, im, wall, usage)
        print(out)
        return 0
    mode, anns = parse_response(content, im.size[0], im.size[1],
                                [{"name": c} for c in classes], n_kpt)
    if args.calibrate and anns:
        from tools.calibrate_coords import fit_transform
        tr = fit_transform(im.size[0], im.size[1])
        if tr:
            ax, bx = tr["x"]; ay, by = tr["y"]
            for a in anns:
                a["x1"], a["x2"] = ax * a["x1"] + bx, ax * a["x2"] + bx
                a["y1"], a["y2"] = ay * a["y1"] + by, ay * a["y2"] + by
                for kp in a.get("keypoints") or []:
                    kp[0], kp[1] = ax * kp[0] + bx, ay * kp[1] + by
                    kp[0], kp[1] = min(1, max(0, kp[0])), min(1, max(0, kp[1]))

    out_dir = os.path.join(RES_ROOT, args.run, os.path.basename(args.dataset.rstrip("/")), "labels")
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    for a in anns:
        cx = (a["x1"] + a["x2"]) / 2
        cy = (a["y1"] + a["y2"]) / 2
        w = a["x2"] - a["x1"]
        h = a["y2"] - a["y1"]
        row = [str(a["class_id"]), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
        for kp in a.get("keypoints") or []:
            row += [f"{kp[0]:.6f}", f"{kp[1]:.6f}", str(int(kp[2]) if len(kp) > 2 else 0)]
        lines.append(" ".join(row))
    with open(os.path.join(out_dir, stem + ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    show_path = None
    if args.show:
        show_dir = os.path.join(RES_ROOT, args.run, "vis")
        os.makedirs(show_dir, exist_ok=True)
        show_path = os.path.join(show_dir, f"{os.path.basename(args.dataset)}_{stem}.jpg")
        render_overlay(args.dataset, stem, anns, show_path)

    print(f"[demo] {args.dataset}/{stem}  {mode}  {wall:.1f}s  "
          f"gen {usage.get('completion_tokens', '?')} tok @ "
          f"{usage.get('completion_tokens', 0) / max(wall, 0.1):.1f} tok/s")
    print(f"  解析出 {len(anns)} 个实例 -> {os.path.relpath(os.path.join(out_dir, stem + '.txt'))}")
    print(f"  原始回复前 200 字: {content[:200]!r}")
    if show_path:
        print(f"  对比图: {show_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
