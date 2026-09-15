#!/usr/bin/env python3
"""check_testenv -- sanity-audit every dataset under testenv/datasets.

Checks, per dataset:
  * pairing          : orphan images (no label) / orphan labels (no image)
  * image integrity  : decodable, min size, non-image junk in images/
  * label validity   : token count (detect 5 / pose 5+3N), class id in range,
                       coords in [0,1], visibility in {0,1,2}, degenerate boxes,
                       empty files; cls -> label text must be a known class;
                       labelme -> json parses, shapes >= 1, points in bounds
  * classes.txt      : exists / non-empty
  * manifest.csv     : rows match actual files (if present)
  * duplicates       : identical content inside a dataset and across datasets

Prints a per-dataset report + writes testenv/dataset_check_report.md.
Run:  python3 tools/check_testenv.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
DS_ROOT = os.path.join(ROOT, "testenv", "datasets")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def md5(p: str) -> str:
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def type_of(name: str) -> str:
    return ("pose" if name.startswith("pose_") else
            "cls" if name.startswith("cls_") else
            "labelme" if name.startswith("labelme_") else "detect")


def main(argv=None) -> int:
    problems = {}       # dataset -> [issue strings]
    global_dups = {}    # content hash -> [paths]
    datasets = sorted(d for d in os.listdir(DS_ROOT)
                      if os.path.isdir(os.path.join(DS_ROOT, d)))

    for name in datasets:
        d = os.path.join(DS_ROOT, name)
        kind = type_of(name)
        issues: list[str] = []
        img_dir, lbl_dir = os.path.join(d, "images"), os.path.join(d, "labels")
        classes = []
        cp = os.path.join(d, "classes.txt")
        if os.path.isfile(cp):
            classes = [l.strip() for l in open(cp, encoding="utf-8") if l.strip()]
        if not classes:
            issues.append("classes.txt 缺失或为空")

        imgs = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir)
                if f.lower().endswith(IMG_EXTS)}
        lbls = set(os.listdir(lbl_dir))
        junk = [f for f in os.listdir(img_dir) if not f.lower().endswith(IMG_EXTS)]
        if junk:
            issues.append(f"images/ 内有非图片文件 {len(junk)} 个: {junk[:3]}")

        # pairing
        orphan_img = [s for s in imgs if (s + ".json" if kind == "labelme" else s + ".txt") not in lbls]
        orphan_lbl = [f for f in sorted(lbls)
                      if os.path.splitext(f)[0] not in imgs]
        if orphan_img:
            issues.append(f"孤儿图片(无标签) {len(orphan_img)}: {orphan_img[:3]}")
        if orphan_lbl:
            issues.append(f"孤儿标签(无图片) {len(orphan_lbl)}: {orphan_lbl[:3]}")

        # label content
        n_empty = n_bad_tokens = n_bad_cls = n_bad_coord = n_degen = n_tiny = 0
        n_kpt_expected = None
        from PIL import Image
        small_imgs = []
        for stem, fn in sorted(imgs.items()):
            lp = os.path.join(lbl_dir, stem + (".json" if kind == "labelme" else ".txt"))
            if not os.path.isfile(lp):
                continue
            if kind == "labelme":
                try:
                    data = json.load(open(lp, encoding="utf-8"))
                    shapes = data.get("shapes") or []
                except Exception as e:  # noqa: BLE001
                    n_bad_tokens += 1
                    if len(issues) < 8:
                        issues.append(f"labelme json 解析失败 {stem}: {e}")
                    continue
                if not shapes:
                    n_empty += 1
                W, H = float(data.get("imageWidth") or 0), float(data.get("imageHeight") or 0)
                for s in shapes:
                    for x, y in s.get("points", []):
                        if not (0 <= x <= (W or 1e9) and 0 <= y <= (H or 1e9)):
                            n_bad_coord += 1
                            break
            else:
                lines = [l for l in open(lp, encoding="utf-8", errors="ignore").read().splitlines()
                         if l.strip()]
                if not lines:
                    n_empty += 1
                    continue
                for ln in lines:
                    t = ln.split()
                    if kind == "cls":
                        if t[0] not in classes:
                            n_bad_cls += 1
                        continue
                    if len(t) == 5:
                        n_tok, has_kpt = 5, False
                    else:
                        n_tok = len(t)
                        has_kpt = True
                        if n_kpt_expected is None:
                            n_kpt_expected = (n_tok - 5) // 3
                    if n_kpt_expected is not None and has_kpt and n_tok != 5 + 3 * n_kpt_expected:
                        n_bad_tokens += 1
                        continue
                    if len(t) != 5 and not has_kpt:
                        n_bad_tokens += 1
                        continue
                    try:
                        cid = int(float(t[0]))
                        cx, cy, w, h = (float(v) for v in t[1:5])
                        kps = [float(v) for v in t[5:]]
                    except ValueError:
                        n_bad_tokens += 1
                        continue
                    if classes and not (0 <= cid < len(classes)):
                        n_bad_cls += 1
                        continue
                    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
                            and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                        n_bad_coord += 1
                        continue
                    if w <= 0 or h <= 0:
                        n_degen += 1
                        continue
                    if w < 0.002 and h < 0.002:
                        n_tiny += 1
                        continue
                    if has_kpt:
                        for i in range(0, len(kps), 3):
                            if not (0 <= kps[i] <= 1 and 0 <= kps[i + 1] <= 1
                                    and int(kps[i + 2]) in (0, 1, 2)):
                                n_bad_coord += 1
                                break
            # image integrity (decode + size) — 每张都查
            ip = os.path.join(img_dir, fn)
            try:
                with Image.open(ip) as im:
                    w, h = im.size
                if w < 8 or h < 8:
                    small_imgs.append(stem)
            except Exception as e:  # noqa: BLE001
                issues.append(f"图片损坏 {fn}: {e}")

        # manifest consistency (rows ↔ files)
        mp = os.path.join(d, "manifest.csv")
        if os.path.isfile(mp):
            rows = [r for r in csv.reader(open(mp, encoding="utf-8")) if r][1:]
            man_stems = set()
            for r in rows:  # cls 的第一列是类名，两列都纳入
                for cell in r:
                    man_stems.add(os.path.splitext(os.path.basename(cell))[0])
            lost = [s for s in (os.path.splitext(f)[0] for f in os.listdir(img_dir))
                    if s not in man_stems]
            if lost:
                issues.append(f"manifest 缺行 {len(lost)}: {lost[:3]}")

        # duplicates inside dataset
        local_dups = 0
        seen = {}
        for fn in sorted(os.listdir(img_dir)):
            hh = md5(os.path.join(img_dir, fn))
            if hh in seen:
                local_dups += 1
            else:
                seen[hh] = fn
                global_dups.setdefault(hh, []).append(f"{name}/{fn}")

        # summarize issues
        stat_bits = []
        if n_empty:
            stat_bits.append(f"空标签 {n_empty}")
        if n_bad_tokens:
            stat_bits.append(f"token数异常 {n_bad_tokens}")
        if n_bad_cls:
            stat_bits.append(f"类别id越界/未知 {n_bad_cls}")
        if n_bad_coord:
            stat_bits.append(f"坐标越界 {n_bad_coord}")
        if n_degen:
            stat_bits.append(f"退化框(w/h<=0) {n_degen}")
        if n_tiny:
            stat_bits.append(f"极小框(宽高均<0.2%) {n_tiny}")
        if small_imgs:
            stat_bits.append(f"过小图 {len(small_imgs)}")
        if local_dups:
            stat_bits.append(f"库内重复图 {local_dups}")
        issues.extend(stat_bits)
        if issues:
            problems[name] = issues

    # cross-dataset duplicates
    cross = {hh: paths for hh, paths in global_dups.items() if len({p.split("/")[0] for p in paths}) > 1}
    if cross:
        problems.setdefault("(跨数据集)", []).append(
            f"跨数据集内容重复 {len(cross)} 组，例: {list(cross.values())[:3]}")

    # report
    lines = ["# testenv 数据集体检报告", ""]
    total_issues = sum(len(v) for v in problems.values())
    lines.append(f"- 数据集 {len(datasets)} 套；发现问题 {total_issues} 处"
                 f"（涉及 {len(problems)} 个对象）" if problems else
                 f"- 数据集 {len(datasets)} 套；**未发现异常**")
    lines.append("")
    for name in datasets:
        iss = problems.get(name, [])
        mark = "⚠️ " if iss else "✅ "
        lines.append(f"- {mark}**{name}**" + ("".join(f"\n  - {i}" for i in iss)))
    if "(跨数据集)" in problems:
        lines.append(f"- ⚠️ **(跨数据集)**：" + "; ".join(problems["(跨数据集)"]))
    out = os.path.join(os.path.dirname(DS_ROOT), "dataset_check_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[report] {out}")
    return 0 if not problems else 2


import csv  # noqa: E402  (used for manifest check)

if __name__ == "__main__":
    raise SystemExit(main())
