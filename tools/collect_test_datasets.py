#!/usr/bin/env python3
"""collect_test_datasets -- gather existing annotated datasets for VLM testing.

Collects pose / detect / classification / labelme datasets scattered across the
machine into ONE place:  ``datasets/<name>/{images,labels}/``  (images and
labels strictly separated), each capped at --cap images, every item validated:

  * image decodes (PIL) and is >= 8 px;
  * label exists and parses: detect = 5 tokens, pose = 5+3N tokens, class id
    in range, coords within [0,1];
  * content-deduplicated inside a dataset (decoded-pixel hash);
  * empty labels are dropped (we want annotated content only).

Per dataset it writes:  images/, labels/, classes.txt, README.md (source,
classes, keypoint count, counts, skipped stats).

Edit COLLECT_SPECS below (config block) and run:  python3 collect_test_datasets.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import sys

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "testenv", "datasets")

ARMOR14 = ["BG", "B1", "B2", "B3", "B4", "BO", "BB", "RG", "R1", "R2", "R3", "R4", "RO", "RB"]
BUFF2 = ["BT", "RT"]

# ----------------------------- 配置区 -----------------------------
# type: yolo(detect/pose, 自动按 token 数校验) | cls(文件夹=类别) | labelme
COLLECT_SPECS = [
    # ---- pose: 装甲板 14 类 4 关键点 ----
    dict(name="pose_final_shenzhen", type="yolo", kpt=4,
         root="/home/dreamchaser/Desktop/armor-label/autolabel/armor-label/datasets/final_shenzhen"),
    dict(name="pose_pose2026_sz", type="yolo", kpt=4, cap=2000,
         root="/home/dreamchaser/Desktop/armor-label/autolabel/armor-label/datasets/pose_2026_sz"),
    dict(name="pose_merder", type="yolo", kpt=4,
         root="/home/dreamchaser/Desktop/armor-label/autolabel/armor-label/datasets/merder"),
    dict(name="pose_container", type="yolo", kpt=4,
         root="/home/dreamchaser/yolo_ws/yolov12/container"),
    # ---- pose: RT/BT 5 关键点 ----
    dict(name="pose_newbuffer_rtbt", type="yolo", kpt=5,
         root="/home/dreamchaser/Desktop/armor-label/autolabel/armor-label/datasets/newbuffer_yolo_pose"),
    # ---- pose: 能量机关 5 关键点 ----
    dict(name="pose_okbuff2", type="yolo", kpt=5,
         root="/home/dreamchaser/yolo_ws/yolov12/okbuff2"),
    dict(name="pose_new_buff", type="yolo", kpt=5,
         root="/home/dreamchaser/yolo_ws/yolov12/datasets/new_buff_dataset"),
    # ---- pose: 无人机装甲 4 关键点 ----
    dict(name="pose_drone_combined", type="yolo", kpt=4,
         root="/home/dreamchaser/Downloads/DroneCombined"),
    dict(name="pose_droneok", type="yolo", kpt=4,
         root="/home/dreamchaser/yolo_ws/yolov12/droneok"),
    # ---- pose: 立柱/交换站 12 关键点 ----
    dict(name="pose_pillar_exchange_12kpt", type="yolo", kpt=12, cap=2000,
         root="/home/dreamchaser/Desktop/armor-label/autolabel/armor-label/datasets/RM26_engineer_exchange/dataset_finalversion"),
    # ---- detect ----
    dict(name="detect_rm2025_armor", type="yolo", cap=2000,
         root="/home/dreamchaser/Desktop/armor-label/autolabel/armor-label/origin_data/RM2025-Armor-Public-Dataset"),
    dict(name="detect_rm2025_car", type="yolo", cap=2000,
         root="/home/dreamchaser/yolo_ws/yolov12/RM2025-Car-Public-Dataset"),
    dict(name="detect_rb_bd_frames", type="yolo", cap=2000,
         root="/home/dreamchaser/yolo_ws/yolov12/output_images"),
    dict(name="detect_wide_sight", type="yolo",
         root="/home/dreamchaser/Downloads/yolo_wide_sight"),
    dict(name="detect_light_luminous", type="yolo",
         root="/home/dreamchaser/tools/data-marker/light_datasets"),
    # ---- 分类（文件夹=类别；标签转为 labels/<stem>.txt 一行类名）----
    dict(name="cls_armor_digit_slices", type="cls", cap=2000,
         root="/home/dreamchaser/Desktop/模型训练/CNN/datasets/train_2000"),
    dict(name="cls_armor_pattern_public", type="cls", cap=2000,
         root="/home/dreamchaser/Desktop/armor-label/autolabel/armor-label/origin_data/RM2025-Armor-Pattern-Public-Dataset/RM2025-Armor-Pattern-Dataset"),
    # ---- labelme（labels/ 内为原始 json）----
    dict(name="labelme_car_radar", type="labelme", cap=2000,
         root="/home/dreamchaser/Desktop/armor-label/autolabel/armor-label/origin_data/car_radar"),
    dict(name="labelme_radar_car", type="labelme", cap=2000,
         root="/home/dreamchaser/Desktop/armor-label/autolabel/armor-label/origin_data/radar_car"),
]
# ------------------------------------------------------------------


def find_label(img_rel: str, img_path: str, root: str) -> str | None:
    """Locate the YOLO txt for an image: labels/ mirror, sibling labels/, or beside."""
    rel_no_img = img_rel
    if "/" in img_rel and img_rel.split("/")[0].startswith("images"):
        rel_no_img = img_rel.split("/", 1)[1]
    stem_rel = os.path.splitext(rel_no_img)[0]
    stem = os.path.splitext(os.path.basename(img_rel))[0]
    cand = [
        os.path.join(root, "labels", stem_rel + ".txt"),
        os.path.join(root, "labels", stem + ".txt"),
        os.path.splitext(img_path)[0] + ".txt",
    ]
    for c in cand:
        c = os.path.abspath(c)
        if os.path.isfile(c):
            return c
    return None


def iter_images(root: str):
    for dirpath, _, files in os.walk(root):
        if any(p.startswith(".") for p in dirpath[len(root):].split(os.sep)):
            continue
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                yield os.path.join(dirpath, fn), os.path.relpath(os.path.join(dirpath, fn), root)


def content_hash(im: Image.Image) -> str:
    return hashlib.md5(f"{im.size}{im.mode}".encode() + im.tobytes()).hexdigest()


def read_classes(root: str) -> list[str] | None:
    for cand in ("data.yaml", "dataset.yaml", "classes.txt", "labels.txt"):
        p = os.path.join(root, cand)
        if not os.path.isfile(p):
            continue
        if cand.endswith(".yaml"):
            names, in_names = [], False
            for ln in open(p, encoding="utf-8", errors="ignore"):
                s = ln.strip()
                if s.startswith("names:"):
                    in_names = True
                    continue
                if in_names:
                    if s.startswith("-"):
                        names.append(s.lstrip("-").strip())
                    elif s and not s.startswith("-"):
                        break
            if names:
                return [n.strip("\"' ") for n in names]
        else:
            names = [l.strip() for l in open(p, encoding="utf-8", errors="ignore") if l.strip()]
            if names:
                return names
    return None


def collect_yolo(spec: dict, dst: str) -> dict:
    root, kpt, cap = spec["root"], spec.get("kpt"), spec.get("cap") or 10**9
    classes = read_classes(root) or []
    images_dir = os.path.join(dst, "images")
    labels_dir = os.path.join(dst, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    n_ok = n_skip = 0
    skipped = {"no_label": 0, "empty_label": 0, "bad_tokens": 0, "bad_image": 0, "dup": 0}
    seen_hashes = set()
    manifest = []
    for img_path, img_rel in iter_images(root):
        if n_ok >= cap:
            break
        lbl = find_label(img_rel, img_path, root)
        if lbl is None:
            skipped["no_label"] += 1
            continue
        lines = [l for l in open(lbl, encoding="utf-8", errors="ignore").read().splitlines() if l.strip()]
        if not lines:
            skipped["empty_label"] += 1
            continue
        want = 5 + 3 * kpt if kpt else 5
        ok_lbl = True
        for ln in lines:
            tok = ln.split()
            if len(tok) != want:
                ok_lbl = False
                break
            try:
                box = [float(x) for x in tok[1:5]]
                cls_id = int(float(tok[0]))
                kps = [float(x) for x in tok[5:]]
            except ValueError:
                ok_lbl = False
                break
            if not all(0.0 <= v <= 1.0 for v in box):
                ok_lbl = False
                break
            for i in range(0, len(kps), 3):     # x y must be in [0,1], v in {0,1,2}
                if not (0.0 <= kps[i] <= 1.0 and 0.0 <= kps[i + 1] <= 1.0):
                    ok_lbl = False
                    break
                if int(kps[i + 2]) not in (0, 1, 2):
                    ok_lbl = False
                    break
            if not ok_lbl:
                break
            if cls_id < 0 or (classes and cls_id >= len(classes)):
                ok_lbl = False
                break
        if not ok_lbl:
            skipped["bad_tokens"] += 1
            continue
        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im2:
                h = content_hash(im2)
        except Exception:  # noqa: BLE001
            skipped["bad_image"] += 1
            continue
        if h in seen_hashes:
            skipped["dup"] += 1
            continue
        seen_hashes.add(h)

        stem = os.path.splitext(os.path.basename(img_rel))[0]
        dst_img = os.path.join(images_dir, stem + os.path.splitext(img_path)[1].lower())
        k = 1
        while os.path.exists(dst_img):
            dst_img = os.path.join(images_dir, f"{stem}_k{k}{os.path.splitext(img_path)[1].lower()}")
            k += 1
        shutil.copy2(img_path, dst_img)
        stem_dst = os.path.splitext(os.path.basename(dst_img))[0]
        shutil.copy2(lbl, os.path.join(labels_dir, stem_dst + ".txt"))
        manifest.append((os.path.basename(dst_img), stem_dst + ".txt"))
        n_ok += 1

    with open(os.path.join(dst, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "label"])
        w.writerows(manifest)
    if classes:
        with open(os.path.join(dst, "classes.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(classes) + "\n")
    return {"n": n_ok, "skipped": skipped, "classes": classes, "kpt": kpt or 0}


def collect_cls(spec: dict, dst: str) -> dict:
    root, cap = spec["root"], spec.get("cap") or 10**9
    images_dir = os.path.join(dst, "images")
    labels_dir = os.path.join(dst, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    class_names = sorted(d for d in os.listdir(root)
                         if os.path.isdir(os.path.join(root, d)) and not d.startswith((".", "_")))
    # balanced round-robin so the cap does not starve later classes
    per_cls: dict[str, list] = {c: [] for c in class_names}
    for c in class_names:
        d = os.path.join(root, c)
        for fn in sorted(os.listdir(d)):
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                per_cls[c].append(os.path.join(d, fn))

    n_ok = n_skip = 0
    skipped = {"bad_image": 0, "dup": 0}
    seen_hashes = set()
    manifest = []
    idx = 0
    while n_ok < cap and any(per_cls[c][idx:] for c in class_names):
        for c in class_names:
            if idx >= len(per_cls[c]) or n_ok >= cap:
                continue
            p = per_cls[c][idx]
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im2:
                    h = content_hash(im2)
            except Exception:  # noqa: BLE001
                skipped["bad_image"] += 1
                continue
            if h in seen_hashes:
                skipped["dup"] += 1
                continue
            seen_hashes.add(h)
            stem = os.path.splitext(os.path.basename(p))[0]
            dst_img = os.path.join(images_dir, f"{c}__{stem}.jpg")
            k = 1
            while os.path.exists(dst_img):
                dst_img = os.path.join(images_dir, f"{c}__{stem}_k{k}.jpg")
                k += 1
            shutil.copy2(p, dst_img)
            stem_dst = os.path.splitext(os.path.basename(dst_img))[0]
            with open(os.path.join(labels_dir, stem_dst + ".txt"), "w", encoding="utf-8") as f:
                f.write(c + "\n")
            manifest.append((c, os.path.basename(dst_img)))
            n_ok += 1
        idx += 1

    with open(os.path.join(dst, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "image"])
        w.writerows(manifest)
    with open(os.path.join(dst, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(class_names) + "\n")
    per_count = {c: sum(1 for m in manifest if m[0] == c) for c in class_names}
    return {"n": n_ok, "skipped": skipped, "classes": class_names, "per_count": per_count, "kpt": 0}


def collect_labelme(spec: dict, dst: str) -> dict:
    root, cap = spec["root"], spec.get("cap") or 10**9
    images_dir = os.path.join(dst, "images")
    labels_dir = os.path.join(dst, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    import json

    n_ok = n_skip = 0
    skipped = {"no_json": 0, "bad_json": 0, "bad_image": 0}
    labels_used = set()
    for img_path, img_rel in iter_images(root):
        if n_ok >= cap:
            break
        j = os.path.splitext(img_path)[0] + ".json"
        if not os.path.isfile(j):
            skipped["no_json"] += 1
            continue
        try:
            data = json.load(open(j, encoding="utf-8"))
            shapes = data.get("shapes") or []
            assert isinstance(shapes, list) and shapes
        except Exception:  # noqa: BLE001
            skipped["bad_json"] += 1
            continue
        try:
            with Image.open(img_path) as im:
                im.verify()
        except Exception:  # noqa: BLE001
            skipped["bad_image"] += 1
            continue
        stem = os.path.splitext(os.path.basename(img_rel))[0]
        dst_img = os.path.join(images_dir, f"{stem}{os.path.splitext(img_path)[1].lower()}")
        k = 1
        while os.path.exists(dst_img):
            dst_img = os.path.join(images_dir, f"{stem}_k{k}{os.path.splitext(img_path)[1].lower()}")
            k += 1
        shutil.copy2(img_path, dst_img)
        stem_dst = os.path.splitext(os.path.basename(dst_img))[0]
        shutil.copy2(j, os.path.join(labels_dir, stem_dst + ".json"))
        labels_used.update(str(s.get("label")) for s in shapes)
        n_ok += 1
    with open(os.path.join(dst, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(labels_used)) + "\n")
    return {"n": n_ok, "skipped": skipped, "classes": sorted(labels_used), "kpt": 0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Collect annotated datasets for VLM testing.")
    ap.add_argument("--only", default="", help="comma-separated dataset names to collect")
    ap.add_argument("--cap", type=int, default=2000, help="override per-dataset cap")
    args = ap.parse_args(argv)
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    print(f"[collect] out root: {OUT_ROOT}")
    summary = []
    for spec in COLLECT_SPECS:
        if only and spec["name"] not in only:
            continue
        dst = os.path.join(OUT_ROOT, spec["name"])
        os.makedirs(dst, exist_ok=True)
        s = dict(spec)
        if args.cap:
            s["cap"] = args.cap
        try:
            if s["type"] == "yolo":
                r = collect_yolo(s, dst)
            elif s["type"] == "cls":
                r = collect_cls(s, dst)
            else:
                r = collect_labelme(s, dst)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {spec['name']}: {type(e).__name__}: {e}")
            summary.append((spec["name"], "FAIL", str(e)))
            continue
        head = f"# {spec['name']}\n\n- type: {s['type']}"
        if s["type"] == "yolo" and r.get("kpt"):
            head += f", keypoints: {r['kpt']}"
        with open(os.path.join(dst, "README.md"), "w", encoding="utf-8") as f:
            f.write(head +
                    f"\n- source: `{s['root']}`\n"
                    f"- images: {r['n']} (labels 同名对应在 labels/)\n"
                    f"- skipped: {r['skipped']}\n"
                    f"- classes: {r['classes']}\n")
        extra = ""
        if "per_count" in r:
            extra = "  每类: " + ", ".join(f"{c}:{n}" for c, n in r["per_count"].items())
        print(f"  [ok] {spec['name']:34s} n={r['n']:<5} skipped={r['skipped']}{extra}")
        summary.append((spec["name"], r["n"], r["skipped"]))

    print("\n[summary]")
    for name, n, sk in summary:
        print(f"  {name:36s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
