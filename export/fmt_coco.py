"""COCO exporter -- ``annotations/instances.json`` (+ original images).

Standard COCO layout; pose tasks additionally fill
``categories[].keypoints`` / ``categories[].skeleton`` (1-based index pairs)
and each annotation's ``keypoints``/``num_keypoints`` in absolute pixels.
``score`` is kept per annotation (non-standard but widely tolerated).
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime

def write(out_dir: str, records: list, classes: list, keypoints: list,
          skeleton: list, copy_images: bool) -> dict:
    is_pose = bool(keypoints)
    images_dir = os.path.join(out_dir, "images")
    ann_dir = os.path.join(out_dir, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    if copy_images:
        os.makedirs(images_dir, exist_ok=True)

    images = []
    annotations = []
    used_names: set[str] = set()
    ann_id = 1
    for img_id, rec in enumerate(records, 1):
        src = rec["path"]
        name = os.path.basename(src)
        stem, ext = os.path.splitext(name)
        k = 1
        while name in used_names:  # guarantee unique file_name per image
            name = f"{stem}_{k}{ext}"
            k += 1
        used_names.add(name)
        if copy_images and os.path.exists(src):
            dst = os.path.join(images_dir, name)
            if os.path.abspath(dst) != os.path.abspath(src):
                shutil.copy2(src, dst)
        images.append({
            "id": img_id,
            "file_name": f"images/{name}" if copy_images else os.path.abspath(src),
            "width": int(rec.get("width") or 0),
            "height": int(rec.get("height") or 0),
        })
        for a in rec.get("anns") or []:
            x1, y1 = a["x1"] * rec["width"], a["y1"] * rec["height"]
            x2, y2 = a["x2"] * rec["width"], a["y2"] * rec["height"]
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            entry = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(a["class_id"]) + 1,  # COCO is 1-based
                "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2),
                "segmentation": [],
                "iscrowd": 0,
                "score": round(float(a.get("score") or 0), 4),
            }
            if is_pose:
                kps = []
                n_visible = 0
                for kp in a.get("keypoints") or []:
                    kx = kp[0] * rec["width"]
                    ky = kp[1] * rec["height"]
                    v = int(kp[2]) if len(kp) > 2 else 0
                    kps.extend([round(kx, 2), round(ky, 2), v])
                    if v > 0:
                        n_visible += 1
                entry["keypoints"] = kps
                entry["num_keypoints"] = n_visible
            annotations.append(entry)
            ann_id += 1

    category = {
        "id": 1,
        "name": "object",
        "supercategory": "none",
    }
    if is_pose:
        category["keypoints"] = [k.get("name", f"kpt{i}") for i, k in enumerate(keypoints)]
        category["skeleton"] = [[i + 1, j + 1] for i, j in (skeleton or [])]
    categories = []
    for i, c in enumerate(classes):
        cc = dict(category)
        cc["id"] = i + 1
        cc["name"] = str(c.get("name", f"class{i}"))
        categories.append(cc)

    data = {
        "info": {
            "description": "Created by selflabel (local VLM auto-annotation)",
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    out_json = os.path.join(ann_dir, "instances_pose.json" if is_pose else "instances_default.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    return {"format": "coco", "mode": "pose" if is_pose else "detect",
            "json": out_json, "images_dir": images_dir if copy_images else None,
            "n_images": len(images), "n_annotations": len(annotations)}
