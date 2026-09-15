#!/usr/bin/env python3
"""recheck_classify -- VLM re-verification of a folder-per-class dataset.

The dataset layout is ``<root>/<class_name>/<image files>`` (what the user's
CNN training pipeline produces). Each sampled image is re-classified by the
local Qwen VLM and compared with its folder label. The original dataset is
NEVER modified -- everything is written to an output dir.

Outputs (``--out``, default ``recheck_<timestamp>``):
  * ``recheck_report.md``   -- overall agreement, per-class table, confusion
    matrix, throughput and an auto-generated feasibility verdict (中文);
  * ``recheck_results.csv`` -- one row per image: file, folder_label,
    vlm_label, vlm_conf, agree, wall_s, parse_error. This is the input the
    webapp review page and ``--corrected`` consume;
  * ``disagreement_N.jpg``  -- montage pages of disagreeing samples;
  * optionally ``--corrected DIR`` -- a corrected COPY of the dataset:
      - agree                    -> <corrected>/<class>/
      - disagree & conf >= HIGH  -> <corrected>/<vlm label>/ (+corrections.csv)
      - everything else          -> <corrected>/_review/<class>/ (human review)

Tiny slices (40x56) are upscaled xN before sending (--upscale). Reference
patterns (--refs dir holding ``<class>.png``) are prepended as few-shot
images. Runs are resumable: with the default ``--policy skip`` an existing
results csv is honoured and remaining images are appended.

Run:
  python3 recheck_classify.py --dataset ~/Desktop/模型训练/CNN/datasets/train_2000 \
      --sample 50 --upscale 6 --refs "~/Desktop/模型训练/CNN/参考原始分类" --extra "..."
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import os
import random
import statistics
import sys
import time
from datetime import datetime

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.parse_geometry import load_json  # noqa: E402
from core.prompts import build_classify_prompt  # noqa: E402
from core.vlm_client import chat_raw  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SKIP_PREFIXES = ("_", ".")          # _lowconf / _review / hidden dirs
CSV_FIELDS = ["file", "folder_label", "vlm_label", "vlm_conf", "agree",
              "wall_s", "parse_error", "scaled_w", "scaled_h"]

ARMOR_HINT = (
    "The image is a small grayscale crop of a RoboMaster armor panel, upscaled "
    "from a low-resolution video frame; pixelation, blur and low contrast are "
    "normal. The classes are official RoboMaster armor markings: '1'..'4' are "
    "the plain digits 1,2,3,4 (infantry robots); '0' is the SENTRY emblem (a "
    "robot/hammer-shaped icon, NOT a digit); '6' is the OUTPOST emblem (a "
    "narrow tower with a round head and an H-like body); '7' is the BASE "
    "emblem (a wide trapezoid fortress). Judge ONLY by the white marking "
    "shape on the dark background."
)


# ------------------------------------------------------------------ dataset

def scan_dataset(root: str) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Return (class_names, items) where items are (abs_path, relpath, label)."""
    classes = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith(SKIP_PREFIXES)
    )
    items = []
    for cls in classes:
        d = os.path.join(root, cls)
        for fn in sorted(os.listdir(d)):
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                p = os.path.join(d, fn)
                items.append((p, os.path.join(cls, fn), cls))
    return classes, items


def stratified_sample(items: list, per_class: int, seed: int) -> list:
    rng = random.Random(seed)
    by_cls: dict[str, list] = {}
    for it in items:
        by_cls.setdefault(it[2], []).append(it)
    out = []
    for cls in sorted(by_cls):
        pool = by_cls[cls]
        out.extend(pool if per_class <= 0 else rng.sample(pool, min(per_class, len(pool))))
    return out


# ------------------------------------------------------------------- image

def _data_url(im: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def load_image(path: str, upscale: int = 1, autocontrast: bool = False) -> tuple[Image.Image, int, int]:
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):  # composite alpha over black (slice-like)
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (0, 0, 0, 255))
        bg.alpha_composite(im)
        im = bg.convert("RGB")
    elif im.mode != "RGB":
        im = im.convert("RGB")
    if upscale > 1:
        im = im.resize((im.size[0] * upscale, im.size[1] * upscale), Image.LANCZOS)
    if autocontrast:
        from PIL import ImageOps
        im = ImageOps.autocontrast(im, cutoff=1)
    return im, im.size[0], im.size[1]


def classify_one(api: str, model: str, path: str, classes: list[dict],
                 upscale: int, ref_images: list[Image.Image], extra: str,
                 temperature: float, max_tokens: int,
                 autocontrast: bool = False) -> dict:
    """Classify one image; retry once on a missing/unknown label."""
    target, w, h = load_image(path, upscale, autocontrast=autocontrast)
    n_refs = len(ref_images)
    system, user = build_classify_prompt(classes, extra=extra, n_reference_images=n_refs)

    def call(user_text: str) -> dict:
        imgs = list(ref_images) + [target]
        parts = [{"type": "image_url", "image_url": {"url": _data_url(im)}} for im in imgs]
        parts.append({"type": "text", "text": user_text})
        return chat_raw(api, model,
                        [{"role": "system", "content": system},
                         {"role": "user", "content": parts}],
                        temperature=temperature, max_tokens=max_tokens)

    res = call(user)
    valid_names = {c["name"] for c in classes}
    label, score, err = _extract(res["content"], valid_names)
    if label is None:  # one stricter retry
        res = call(user + "\nIMPORTANT: answer with the exact JSON object, label must be one of: "
                   + ", ".join(sorted(valid_names)))
        label, score, err = _extract(res["content"], valid_names)
    return {"label": label, "score": score, "parse_error": err, "wall": res["wall"],
            "scaled_w": w, "scaled_h": h}


def _extract(content: str, valid_names: set) -> tuple[str | None, float, str]:
    data = load_json(content)
    if not isinstance(data, dict):
        return None, 0.0, "no_json"
    label = str(data.get("label") or data.get("class") or "").strip()
    try:
        score = float(data.get("score", data.get("conf", 0)) or 0)
    except (TypeError, ValueError):
        score = 0.0
    if not label:
        return None, 0.0, "empty_label"
    if label not in valid_names:
        # tolerate "class 3" / "armor 3" style answers
        for name in sorted(valid_names, key=len, reverse=True):
            if name in label:
                return name, score, ""
        return None, score, f"unknown_label:{label}"
    return label, score, ""


# ------------------------------------------------------------------ outputs

def append_row(csv_path: str, row: dict, new_file: bool) -> None:
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file or not exists:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def load_rows(csv_path: str) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_disagreement_pages(rows: list, root: str, out_dir: str,
                            cell_w: int = 150, cols: int = 6,
                            max_pages: int = 12) -> list[str]:
    """Tile disagreeing samples with a red frame + 'T:<folder> V:<vlm>' caption."""
    import cv2
    import numpy as np

    bad = [r for r in rows if r["agree"] == "0" and r["vlm_label"]]
    if not bad:
        return []
    pages = []
    per_page = cols * 6
    for pi in range(0, len(bad), per_page):
        if len(pages) >= max_pages:
            break
        chunk = bad[pi: pi + per_page]
        cells = []
        for r in chunk:
            img = cv2.imread(os.path.join(root, r["file"]))
            if img is None:
                continue
            h0, w0 = img.shape[:2]
            scale = cell_w / w0
            cell = cv2.resize(img, (cell_w, max(1, int(h0 * scale))), interpolation=cv2.INTER_CUBIC)
            ch = cell.shape[0]
            cell = cv2.copyMakeBorder(cell, 18, 3, 3, 3, cv2.BORDER_CONSTANT, value=(0, 0, 255))
            caption = f"T:{r['folder_label']} V:{r['vlm_label'][:12]}"
            cv2.putText(cell, caption, (5, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cell[14:18, :] = (0, 0, 255)
            cells.append(cell)
        if not cells:
            continue
        rows_n = (len(cells) + cols - 1) // cols
        ch = max(c.shape[0] for c in cells)
        cw = max(c.shape[1] for c in cells)
        grid = np.zeros((rows_n * (ch + 4), cols * (cw + 4), 3), dtype=np.uint8)
        for i, c in enumerate(cells):
            y = (i // cols) * (ch + 4)
            x = (i % cols) * (cw + 4)
            grid[y:y + c.shape[0], x:x + c.shape[1]] = c
        p = os.path.join(out_dir, f"disagreement_{len(pages)}.jpg")
        cv2.imwrite(p, grid)
        pages.append(p)
    return pages


def write_report(path: str, rows: list, classes: list, root: str, total_images: int,
                 args, pages: list[str]) -> None:
    ok_rows = [r for r in rows if not r["parse_error"]]
    agree_rows = [r for r in ok_rows if r["agree"] == "1"]
    n = len(rows)
    n_ok = len(ok_rows)
    n_agree = len(agree_rows)
    acc = n_agree / n_ok if n_ok else 0.0

    def conf_mean(rs: list) -> str:
        vals = [float(r["vlm_conf"]) for r in rs if r["vlm_conf"]]
        return f"{statistics.mean(vals):.2f}" if vals else "-"

    walls = [float(r["wall_s"]) for r in rows if r["wall_s"]]
    per_min = 60.0 / (statistics.mean(walls)) if walls else 0.0
    hours_all = total_images / per_min / 60.0 if per_min else 0.0

    lines = [
        "# 分类复核报告（recheck）",
        "",
        f"- 时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- 数据集：`{root}`（共 {total_images} 张，本次复核 {n} 张）",
        f"- 参数：每类抽样={args.sample or '全部'}  upscale=×{args.upscale}  "
        f"few-shot 参考={'有' if args.refs else '无'}  seed={args.seed}  "
        f"温度={args.temperature}",
        f"- 类别：{'、'.join(classes)}",
        "",
        "## 总体",
        "",
        f"- 有效解析：{n_ok}/{n}",
        f"- **一致率：{acc:.1%}**（{n_agree}/{n_ok}，VLM 标签 == 文件夹标签）",
        f"- 一致样本平均置信度：{conf_mean(agree_rows)}；"
        f"分歧样本平均置信度：{conf_mean([r for r in ok_rows if r['agree'] == '0'])}",
        f"- 速度：{statistics.mean(walls):.2f} s/张（{per_min:.0f} 张/分钟）→ "
        f"全量 {total_images} 张 ≈ **{hours_all:.1f} 小时**",
        "",
        "## 每类明细",
        "",
        "| 类别 | 样本 | 有效 | 一致 | 一致率 | conf(一致) | conf(分歧) |",
        "|---|---|---|---|---|---|---|",
    ]
    for cls in classes:
        rs = [r for r in rows if r["folder_label"] == cls]
        ro = [r for r in rs if not r["parse_error"]]
        ra = [r for r in ro if r["agree"] == "1"]
        rd = [r for r in ro if r["agree"] == "0"]
        lines.append(
            f"| {cls} | {len(rs)} | {len(ro)} | {len(ra)} "
            f"| {len(ra) / len(ro):.1%} | {conf_mean(ra)} | {conf_mean(rd)} |")

    lines += ["", "## 混淆矩阵（行=文件夹标签，列=VLM 标签）", ""]
    vlm_labels = sorted({r["vlm_label"] for r in ok_rows} | {c for c in classes})
    head = "| 文件夹\\VLM | " + " | ".join(vlm_labels) + " | 无解析 |"
    sep = "|---" * (len(vlm_labels) + 2) + "|"
    lines += [head, sep]
    for cls in classes:
        cells = []
        for vl in vlm_labels:
            cells.append(str(sum(1 for r in ok_rows if r["folder_label"] == cls and r["vlm_label"] == vl)))
        nerr = sum(1 for r in rows if r["folder_label"] == cls and r["parse_error"])
        lines.append(f"| **{cls}** | " + " | ".join(cells) + f" | {nerr} |")

    verdict = (
        "**结论：VLM 一致率 ≥90%，可直接作为复核裁判自动矫正；分歧样本再人工抽查。**"
        if acc >= 0.9 else
        "**结论：VLM 一致率 70–90%，适合粗筛——高置信分歧自动改档、其余人工确认。**"
        if acc >= 0.7 else
        "**结论：VLM 一致率 <70%，只宜做低置信参考；矫正需以人工审查为主。**"
    )
    lines += ["", "## 结论", "", verdict, ""]
    if pages:
        lines.append("## 分歧样本拼图（红框：T=文件夹标签，V=VLM 标签）")
        lines.append("")
        lines += [f"- `{os.path.basename(p)}`" for p in pages]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_corrected(corrected: str, rows: list, root: str, conf_high: float) -> str:
    """Write a corrected COPY of the dataset (originals untouched)."""
    import shutil

    os.makedirs(corrected, exist_ok=True)
    corrections = []
    copied = review = 0
    for r in rows:
        src = os.path.join(root, r["file"])
        if not os.path.exists(src):
            continue
        safe = r["file"].replace(os.sep, "__")
        if r["agree"] == "1" or not r["vlm_label"]:
            dst_cls = r["folder_label"]
        elif float(r["vlm_conf"] or 0) >= conf_high:
            dst_cls = r["vlm_label"]
            corrections.append((r["file"], r["folder_label"], r["vlm_label"], r["vlm_conf"]))
        else:
            dst_cls = os.path.join("_review", r["folder_label"])
            review += 1
        d = os.path.join(corrected, dst_cls)
        os.makedirs(d, exist_ok=True)
        shutil.copy2(src, os.path.join(d, safe))
        copied += 1
    with open(os.path.join(corrected, "corrections.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "from", "to", "vlm_conf"])
        w.writerows(corrections)
    print(f"[corrected] copied={copied}  auto-fixed={len(corrections)}  to-review={review}"
          f"  -> {corrected}")
    return corrected


# --------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="VLM re-verification of a folder-per-class dataset.")
    ap.add_argument("--dataset", required=True, help="dataset root (<root>/<class>/<image>)")
    ap.add_argument("--sample", type=int, default=50, help="images per class (0 = all)")
    ap.add_argument("--upscale", type=int, default=6, help="upscale factor for small slices")
    ap.add_argument("--autocontrast", action="store_true",
                    help="stretch contrast before sending (dark slices)")
    ap.add_argument("--refs", default="", help="dir of <class>.png reference patterns (few-shot)")
    ap.add_argument("--extra", default="auto",
                    help="extra domain hint for the prompt; 'auto' = use the built-in "
                         "RoboMaster hint when all class names are digits, '' = none")
    ap.add_argument("--conf-high", type=float, default=0.80,
                    help="min VLM confidence to auto-apply a correction")
    ap.add_argument("--corrected", default="", help="write corrected dataset copy here")
    ap.add_argument("--out", default="", help="output dir (default recheck_<timestamp>)")
    ap.add_argument("--api", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--policy", choices=["skip", "overwrite"], default="skip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap total images (smoke test)")
    args = ap.parse_args(argv)

    root = os.path.expanduser(args.dataset)
    if not os.path.isdir(root):
        print(f"[error] dataset not found: {root}", file=sys.stderr)
        return 1
    classes, items = scan_dataset(root)
    if not classes:
        print(f"[error] no class folders under {root}", file=sys.stderr)
        return 1

    out_dir = args.out or f"recheck_{datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "recheck_results.csv")

    sample = stratified_sample(items, args.sample, args.seed)
    if args.limit > 0:
        sample = sample[: args.limit]

    done: set[str] = set()
    if args.policy == "skip" and os.path.exists(csv_path):
        done = {r["file"] for r in load_rows(csv_path)}
        print(f"[resume] {len(done)} images already done, skipping them")

    ref_images: list[Image.Image] = []
    ref_classes: list[str] = []
    if args.refs:
        refs_dir = os.path.expanduser(args.refs)
        for cls in classes:
            p = os.path.join(refs_dir, f"{cls}.png")
            if os.path.exists(p):
                ref_images.append(load_image(p)[0])
                ref_classes.append(cls)
        if ref_images and ref_classes != classes:
            print(f"[warn] reference images cover {ref_classes} but classes are {classes}; "
                  "references are matched by position -- check ordering!")

    class_dicts = [{"name": c, "description": ""} for c in classes]
    extra = args.extra
    if extra == "auto":  # built-in hint only when the classes look like armor ids
        extra = ARMOR_HINT if set(classes) <= set("012345678") else ""
        if extra:
            print("[recheck] using built-in RoboMaster hint (--extra '' to disable)")
    print(f"[recheck] dataset={root}  classes={classes}  total={len(items)}  "
          f"todo={len([s for s in sample if s[1] not in done])}  upscale=×{args.upscale}  "
          f"refs={len(ref_images)}  -> {out_dir}")

    t0 = time.time()
    new_file = not os.path.exists(csv_path)
    n_done = n_err = 0
    for i, (path, rel, label) in enumerate(sample, 1):
        if rel in done:
            continue
        try:
            r = classify_one(args.api, args.model, path, class_dicts, args.upscale,
                             ref_images, extra, args.temperature, args.max_tokens,
                             autocontrast=args.autocontrast)
            row = {
                "file": rel, "folder_label": label,
                "vlm_label": r["label"] or "", "vlm_conf": f"{r['score']:.3f}" if r["label"] else "",
                "agree": int(r["label"] == label) if r["label"] else "",
                "wall_s": f"{r['wall']:.3f}", "parse_error": r["parse_error"],
                "scaled_w": r["scaled_w"], "scaled_h": r["scaled_h"],
            }
            if not r["label"]:
                n_err += 1
        except Exception as e:  # noqa: BLE001
            row = {"file": rel, "folder_label": label, "vlm_label": "", "vlm_conf": "",
                   "agree": "", "wall_s": "", "parse_error": f"{type(e).__name__}: {e}",
                   "scaled_w": "", "scaled_h": ""}
            n_err += 1
        append_row(csv_path, row, new_file)
        new_file = False
        n_done += 1
        if i % 25 == 0 or i == len(sample):
            dt = time.time() - t0
            eta = dt / n_done * (len(sample) - len(done) - n_done)
            print(f"  [{i}/{len(sample)}] done={n_done} err={n_err} "
                  f"({dt:.0f}s, ETA {eta:.0f}s)")

    rows = load_rows(csv_path)
    pages = make_disagreement_pages(rows, root, out_dir)
    report = os.path.join(out_dir, "recheck_report.md")
    write_report(report, rows, classes, root, len(items), args, pages)
    print(f"[report] {report}")

    if args.corrected:
        write_corrected(os.path.expanduser(args.corrected), rows, root, args.conf_high)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
