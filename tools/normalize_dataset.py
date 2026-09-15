#!/usr/bin/env python3
"""normalize_dataset -- cleaning / normalisation for a folder-per-class dataset.

Steps (each optional, all reported in ``normalize_report.csv``):
  * ``--fix-broken``  unreadable images (PIL verify/open) -> ``_broken/``;
  * ``--dedup``       duplicate CONTENT (decoded-pixel hash) -> ``_dedup/`` --
                      the first occurrence in sorted order is kept;
  * ``--ext jpg|png`` convert every image to one extension (re-encoded);
  * ``--resize WxH``  equal-ratio letterbox into a fixed canvas, padding 0
                      (black) -- NEVER one-side stretching (the user's CNN
                      pipeline iron rule: 等比缩放 + 填边, 禁止单边拉伸);
  * ``--grayscale``   convert to 8-bit grayscale.

Dry-run by default: without ``--apply`` nothing on disk changes; the report
lists exactly what would happen. ``_``-prefixed dirs are invisible to
dataset scanners (recheck/webapp skip them).

Run:
  python3 normalize_dataset.py --dataset <dir> --dedup --resize 40x56 [--apply]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import sys
from datetime import datetime

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SKIP_PREFIXES = ("_", ".")


def iter_images(root: str):
    for cls in sorted(os.listdir(root)):
        d = os.path.join(root, cls)
        if not os.path.isdir(d) or cls.startswith(SKIP_PREFIXES):
            continue
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            if os.path.isfile(p) and os.path.splitext(fn)[1].lower() in IMG_EXTS:
                yield cls, fn, p


def unique_dst(dst: str) -> str:
    if not os.path.exists(dst):
        return dst
    stem, ext = os.path.splitext(dst)
    for k in range(1, 200):
        cand = f"{stem}_k{k}{ext}"
        if not os.path.exists(cand):
            return cand
    raise ValueError(f"no free name for {dst}")


def open_image(path: str):
    """Return a verified PIL image or None if broken."""
    try:
        with Image.open(path) as im:
            im.load()
            return im.copy()
    except Exception:  # noqa: BLE001
        return None


def content_hash(im: Image.Image) -> str:
    return hashlib.md5(f"{im.size}{im.mode}".encode() + im.tobytes()).hexdigest()


def letterbox(im: Image.Image, w: int, h: int, grayscale: bool) -> Image.Image:
    scale = min(w / im.size[0], h / im.size[1])
    nw, nh = max(1, round(im.size[0] * scale)), max(1, round(im.size[1] * scale))
    im2 = im.resize((nw, nh), Image.LANCZOS)
    if grayscale:
        im2 = im2.convert("L")
        canvas = Image.new("L", (w, h), 0)
        canvas.paste(im2, ((w - nw) // 2, (h - nh) // 2))
    else:
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        canvas.paste(im2, ((w - nw) // 2, (h - nh) // 2))
    return canvas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dataset cleaning/normalisation (dry-run by default).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fix-broken", action="store_true")
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--ext", choices=["jpg", "png"], help="unify extension (re-encodes)")
    ap.add_argument("--resize", metavar="WxH", help='equal-ratio letterbox, e.g. "40x56"')
    ap.add_argument("--grayscale", action="store_true")
    ap.add_argument("--quality", type=int, default=92, help="jpeg quality when re-encoding")
    ap.add_argument("--apply", action="store_true", help="write changes (default dry-run)")
    args = ap.parse_args(argv)

    root = os.path.abspath(os.path.expanduser(args.dataset))
    if not os.path.isdir(root):
        print(f"[error] dataset not found: {root}", file=sys.stderr)
        return 1

    resize = None
    if args.resize:
        try:
            w, h = args.resize.lower().split("x")
            resize = (int(w), int(h))
        except ValueError:
            print("[error] --resize expects WxH, e.g. 40x56", file=sys.stderr)
            return 1

    rows = []
    seen: dict[str, str] = {}
    n_broken = n_dupes = n_changed = 0

    for cls, fn, path in iter_images(root):
        rel = os.path.join(cls, fn)
        action, detail = "", ""
        im = open_image(path)
        if im is None:
            n_broken += 1
            action, detail = "broken", ""
            if args.fix_broken and args.apply:
                d = os.path.join(root, "_broken", cls)
                os.makedirs(d, exist_ok=True)
                shutil.move(path, unique_dst(os.path.join(d, fn)))
                action = "broken->moved"
            rows.append({"file": rel, "action": action or "broken(detected)",
                         "detail": detail})
            continue

        hsh = content_hash(im)
        if args.dedup and hsh in seen:
            n_dupes += 1
            action, detail = "duplicate", f"same as {seen[hsh]}"
            if args.apply:
                d = os.path.join(root, "_dedup", cls)
                os.makedirs(d, exist_ok=True)
                shutil.move(path, unique_dst(os.path.join(d, fn)))
                action = "duplicate->moved"
            rows.append({"file": rel, "action": action, "detail": detail})
            continue
        seen[hsh] = rel

        ext = os.path.splitext(fn)[1].lower().lstrip(".")
        want_ext = args.ext or ext
        need_convert = (want_ext != ext) or bool(resize) or (args.grayscale and im.mode != "L")
        if need_convert:
            n_changed += 1
            action = "convert"
            detail = f"{ext}->{want_ext}" + (f" letterbox{resize[0]}x{resize[1]}" if resize else "") + \
                     (" gray" if args.grayscale and im.mode != "L" else "")
            if args.apply:
                out = letterbox(im, *resize, grayscale=args.grayscale) if resize else \
                    (im.convert("L") if args.grayscale and im.mode != "L" else im)
                new_name = fn
                if want_ext != ext:
                    new_name = os.path.splitext(fn)[0] + "." + want_ext
                dst = unique_dst(os.path.join(root, cls, new_name))
                if want_ext == "png":
                    out.save(dst)
                else:
                    out.convert("RGB").save(dst, quality=args.quality)
                if dst != path and os.path.exists(path):
                    os.remove(path)
                action = "converted"
        if action or detail:
            rows.append({"file": rel, "action": action, "detail": detail})

    report = os.path.join(root, f"normalize_report_{datetime.now():%Y%m%d_%H%M%S}.csv")
    with open(report, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "action", "detail"])
        w.writeheader()
        w.writerows(rows)

    mode = "APPLY(已写入)" if args.apply else "DRY-RUN(未改动, 加 --apply 执行)"
    print(f"[normalize] dataset={root}  mode={mode}")
    print(f"  broken={n_broken}  duplicates={n_dupes}  to-convert={n_changed}")
    print(f"  report -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
