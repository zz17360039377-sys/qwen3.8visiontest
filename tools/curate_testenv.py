#!/usr/bin/env python3
"""curate_testenv -- prune testenv/datasets to ONE global image budget.

Keeps the highest-value images across ALL types (detect / pose / cls / labelme):
  1. images with MORE objects in one photo come first (多目标优先);
  2. ties broken by sharpness (Laplacian variance, 高质量优先) then size;
  3. types AND their datasets are served round-robin, so every type and every
     dataset variety survives.

Everything not selected is DELETED from testenv/datasets (images + labels +
manifest rows). Original sources on disk are untouched -- re-run
tools/collect_test_datasets.py to restore. Per-dataset README counts and the
master datasets/README.md are regenerated.

Run:  python3 tools/curate_testenv.py [--total 2000]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
DS_ROOT = os.path.join(ROOT, "testenv", "datasets")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def type_of(name: str) -> str:
    return ("pose" if name.startswith("pose_") else
            "cls" if name.startswith("cls_") else
            "labelme" if name.startswith("labelme_") else "detect")


def sharpness(path: str) -> float:
    import cv2

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -1.0
    if max(img.shape) > 512:
        s = 512 / max(img.shape)
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
    import numpy as np
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def count_objects(label_path: str, kind: str) -> int:
    if kind == "labelme":
        import json
        try:
            data = json.load(open(label_path, encoding="utf-8"))
            return len(data.get("shapes") or [])
        except Exception:  # noqa: BLE001
            return 0
    n = 0
    for ln in open(label_path, encoding="utf-8", errors="ignore"):
        if ln.strip():
            n += 1
    return n


def scan_dataset(ds_dir: str, kind: str) -> list[dict]:
    """All valid items of one dataset, scored. [{img, lbl, stem, n_obj, sharp, area}]"""
    img_dir = os.path.join(ds_dir, "images")
    lbl_dir = os.path.join(ds_dir, "labels")
    items = []
    for fn in sorted(os.listdir(img_dir)):
        if not fn.lower().endswith(IMG_EXTS):
            continue
        stem = os.path.splitext(fn)[0]
        img_p = os.path.join(img_dir, fn)
        if kind == "labelme":
            lbl_p = os.path.join(lbl_dir, stem + ".json")
        else:
            lbl_p = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.isfile(lbl_p):
            continue
        n_obj = count_objects(lbl_p, kind)
        if n_obj <= 0:
            continue  # 无标注内容 = 无效
        sp = sharpness(img_p)
        if sp < 0:
            continue  # 图片不可解码
        from PIL import Image
        with Image.open(img_p) as im:
            w, h = im.size
        items.append({"img": img_p, "lbl": lbl_p, "stem": stem, "fn": fn,
                      "n_obj": n_obj, "sharp": sp, "area": w * h})
    return items


def rewrite_manifest(ds_dir: str, kind: str, kept: list[dict]) -> None:
    mp = os.path.join(ds_dir, "manifest.csv")
    if not os.path.isfile(mp):
        return
    kept_stems = {it["stem"] for it in kept}
    rows = list(csv.reader(open(mp, encoding="utf-8")))
    head, body = rows[0], rows[1:]
    new_body = [r for r in body if any(s in " ".join(r) for s in kept_stems)]
    with open(mp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(new_body)


def update_readme(ds_dir: str, n: int) -> None:
    p = os.path.join(ds_dir, "README.md")
    if not os.path.isfile(p):
        return
    lines = open(p, encoding="utf-8").read().splitlines()
    out = []
    for ln in lines:
        if ln.startswith("- images:"):
            ln = f"- images: {n} (curated: 多目标+清晰度优先, 每类合计限额)"
        out.append(ln)
    open(p, "w", encoding="utf-8").write("\n".join(out) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Prune testenv/datasets to one global image budget.")
    ap.add_argument("--total", type=int, default=2000, help="max images across ALL types (default 2000)")
    args = ap.parse_args()

    # group datasets by type
    groups: dict[str, list[str]] = {}
    for name in sorted(os.listdir(DS_ROOT)):
        d = os.path.join(DS_ROOT, name)
        if os.path.isdir(d):
            groups.setdefault(type_of(name), []).append(name)

    ranked = {}
    for kind, names in sorted(groups.items()):
        budget = args.total  # informational only; the real gate is the global loop
        for name in names:
            items = scan_dataset(os.path.join(DS_ROOT, name), kind)
            items.sort(key=lambda it: (-it["n_obj"], -it["sharp"], -it["area"]))
            ranked[name] = items
            print(f"[{kind}] {name}: {len(items)} valid items")

        print(f"[{kind}] scanned {sum(len(v) for v in ranked.values())} valid items")

    # global round-robin: cycle types, then their datasets, one best image each
    kept: dict[str, list[dict]] = {n: [] for ns in groups.values() for n in ns}
    total = 0
    while total < args.total:
        progressed = False
        for kind in sorted(groups):
            for name in groups[kind]:
                if total >= args.total:
                    break
                pool = ranked[name]
                if pool:
                    it = pool.pop(0)
                    kept[name].append(it)
                    total += 1
                    progressed = True
            if total >= args.total:
                break
        if not progressed:
            break

    # delete everything not kept
    if True:
        n_del = 0
        names_all = [n for ns in groups.values() for n in ns]
        for name in names_all:
            ds_dir = os.path.join(DS_ROOT, name)
            kept_stems = {it["stem"] for it in kept[name]}
            img_dir = os.path.join(ds_dir, "images")
            lbl_dir = os.path.join(ds_dir, "labels")
            for fn in os.listdir(img_dir):
                if os.path.splitext(fn)[0] not in kept_stems:
                    os.remove(os.path.join(img_dir, fn))
                    n_del += 1
            for fn in os.listdir(lbl_dir):
                if os.path.splitext(fn)[0] not in kept_stems:
                    os.remove(os.path.join(lbl_dir, fn))
            rewrite_manifest(ds_dir, "yolo", kept[name])
            update_readme(ds_dir, len(kept[name]))
            if not kept[name]:
                print(f"  [drop] {name}: 0 kept, removing empty dataset dir")
                shutil.rmtree(ds_dir, ignore_errors=True)
        print(f"[global] kept {total} (budget {args.total}), deleted {n_del} images")

    # regenerate master README
    rows = []
    for name in sorted(os.listdir(DS_ROOT)):
        d = os.path.join(DS_ROOT, name)
        if not os.path.isdir(d):
            continue
        ni = len(os.listdir(os.path.join(d, "images")))
        cp = os.path.join(d, "classes.txt")
        ncls = len(open(cp).read().split()) if os.path.exists(cp) else 0
        sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(d) for f in fs)
        n_inst = 0
        for lf in os.listdir(os.path.join(d, "labels")):
            n_inst += sum(1 for ln in open(os.path.join(d, "labels", lf),
                                           encoding="utf-8", errors="ignore") if ln.strip())
        rows.append((name, type_of(name), ni, ncls, sz // (1024 * 1024), n_inst))
    with open(os.path.join(DS_ROOT, "README.md"), "w", encoding="utf-8") as f:
        f.write("# 测试数据集总清单\n\n"
                "每套结构：`images/` + `labels/`（同名对应）+ `classes.txt` + `manifest.csv` + `README.md`。\n"
                "已按「全部合计 ≤ 限额、多目标优先、清晰度优先、类型与数据集轮转保多样性」整理。\n"
                "规则见 [../限制.md](../限制.md)。\n\n"
                "| 数据集 | 类型 | 图片 | 标注框/实例数 | 类别数 | 体积 |\n|---|---|---|---|---|---|\n")
        for name, kind, ni, ncls, mb, n_inst in rows:
            f.write(f"| {name} | {kind} | {ni} | {n_inst} | {ncls} | {mb}MB |\n")
        f.write("\n注：`detect_radar_*_selfmade` 为 radar-2027 权重对录播的伪标签；"
                "`c<N>` 类别名占位表示源数据未提供。\n")
    print("[done] master README regenerated")
    return 0


if __name__ == "__main__":
    import shutil
    raise SystemExit(main())
