#!/usr/bin/env python3
"""balance_dataset -- per-class cap / thinning for a folder-per-class dataset.

Rules (aligned with the user's CNN pipeline ``supplement_dataset.py``):
  * every class keeps at most ``--target`` images;
  * when a class exceeds the target, a FIXED-SEED random subset is kept and
    the surplus is MOVED to ``overflow/<class>/`` (never deleted);
  * classes below the target are only reported (no augmentation here -- the
    reference-image augmentation of the original pipeline is project-specific).

Dry-run by default: without ``--apply`` nothing is moved, only
``balance_report.csv`` + the console table are produced.

Run:
  python3 balance_dataset.py --dataset <dir> --target 2000 [--apply]
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
from datetime import datetime

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SKIP_PREFIXES = ("_", ".")


def scan(root: str) -> dict[str, list[str]]:
    out = {}
    for cls in sorted(os.listdir(root)):
        d = os.path.join(root, cls)
        if not os.path.isdir(d) or cls.startswith(SKIP_PREFIXES):
            continue
        out[cls] = sorted(f for f in os.listdir(d)
                          if os.path.splitext(f)[1].lower() in IMG_EXTS)
    return out


def unique_dst(dst: str) -> str:
    if not os.path.exists(dst):
        return dst
    stem, ext = os.path.splitext(dst)
    for k in range(1, 200):
        cand = f"{stem}_k{k}{ext}"
        if not os.path.exists(cand):
            return cand
    raise ValueError(f"no free name for {dst}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-class cap / thinning (dry-run by default).")
    ap.add_argument("--dataset", required=True, help="dataset root (<root>/<class>/<image>)")
    ap.add_argument("--target", type=int, default=2000, help="max images per class (0 = report only)")
    ap.add_argument("--overflow", default="", help="surplus dir (default <dataset>/../overflow)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--apply", action="store_true", help="actually move files (default dry-run)")
    args = ap.parse_args(argv)

    root = os.path.abspath(os.path.expanduser(args.dataset))
    if not os.path.isdir(root):
        print(f"[error] dataset not found: {root}", file=sys.stderr)
        return 1
    overflow = os.path.abspath(os.path.expanduser(
        args.overflow or os.path.join(os.path.dirname(root.rstrip(os.sep)), "overflow")))
    target = args.target

    by_cls = scan(root)
    if not by_cls:
        print("[error] no class folders found", file=sys.stderr)
        return 1

    rows = []
    moved_total = 0
    for cls, files in by_cls.items():
        n = len(files)
        moved = deficit = 0
        if target > 0 and n > target:
            rng = random.Random(args.seed + hash(cls) % 100000)
            keep = set(rng.sample(files, target))
            surplus = [f for f in files if f not in keep]
            moved = len(surplus)
            if args.apply:
                dst_dir = os.path.join(overflow, cls)
                os.makedirs(dst_dir, exist_ok=True)
                for f in surplus:
                    shutil.move(os.path.join(root, cls, f), unique_dst(os.path.join(dst_dir, f)))
            moved_total += moved
        elif target > 0 and n < target:
            deficit = target - n
        rows.append({"class": cls, "before": n, "kept": min(n, target) if target else n,
                     "moved_to_overflow": moved, "deficit": deficit,
                     "action": "moved" if moved and args.apply else
                               ("would-move" if moved else "ok")})

    report = os.path.join(root, f"balance_report_{datetime.now():%Y%m%d_%H%M%S}.csv")
    with open(report, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class", "before", "kept", "moved_to_overflow",
                                          "deficit", "action"])
        w.writeheader()
        w.writerows(rows)

    mode = "APPLY(已移动)" if args.apply else "DRY-RUN(未改动, 加 --apply 执行)"
    print(f"[balance] dataset={root}  target={target}  overflow={overflow}  mode={mode}")
    print(f"{'class':<12}{'before':>8}{'kept':>8}{'overflow':>10}{'deficit':>9}  action")
    for r in rows:
        print(f"{r['class']:<12}{r['before']:>8}{r['kept']:>8}{r['moved_to_overflow']:>10}"
              f"{r['deficit']:>9}  {r['action']}")
    print(f"[balance] would move {moved_total} file(s)" if not args.apply
          else f"[balance] moved {moved_total} file(s)")
    print(f"[balance] report -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
