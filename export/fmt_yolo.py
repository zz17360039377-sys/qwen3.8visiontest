"""YOLO exporter -- ``labels/*.txt`` + ``classes.txt`` (+ original images).

Rows:
  detect : ``<cls> cx cy w h``
  pose   : ``<cls> cx cy w h  x1 y1 v1 ... xN yN vN``

Layout: originals are copied to ``images/`` (what training expects); overlay
previews live in ``previews/`` and are written by the caller, not here.
"""

from __future__ import annotations

import os
import shutil
from typing import Iterable

def ann_to_yolo_line(ann: dict, mode: str) -> str:
    mode = mode.lower()
    cx = (ann["x1"] + ann["x2"]) / 2.0
    cy = (ann["y1"] + ann["y2"]) / 2.0
    w = ann["x2"] - ann["x1"]
    h = ann["y2"] - ann["y1"]
    head = f"{ann['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
    if mode == "pose":
        kps = ann.get("keypoints") or []
        parts = [head]
        for kp in kps:
            kx, ky, v = kp[0], kp[1], int(kp[2]) if len(kp) > 2 else 0
            parts.append(f"{kx:.6f} {ky:.6f} {max(0, min(2, v))}")
        return " ".join(parts)
    return head


def annotations_to_yolo(anns: Iterable[dict], mode: str) -> str:
    """Return the full .txt content (one line per instance, trailing newline)."""
    lines = [ann_to_yolo_line(a, mode) for a in anns]
    return "\n".join(lines) + "\n" if lines else ""


def write_classes(classes: list[dict], out_path: str) -> None:
    """Write a classes.txt with one name per line (index = class id)."""
    with open(out_path, "w", encoding="utf-8") as f:
        for c in classes:
            f.write(str(c.get("name", "")).strip() + "\n")


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def write(out_dir: str, records: list, classes: list, keypoints: list,
          skeleton: list, copy_images: bool) -> dict:
    mode = "pose" if keypoints else "detect"
    labels_dir = os.path.join(out_dir, "labels")
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(labels_dir, exist_ok=True)

    n_anns = 0
    for rec in records:
        stem = _stem(rec["path"])
        with open(os.path.join(labels_dir, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write(annotations_to_yolo(rec["anns"], mode))
        n_anns += len(rec["anns"])
        if copy_images and os.path.exists(rec["path"]):
            os.makedirs(images_dir, exist_ok=True)
            dst = os.path.join(images_dir, os.path.basename(rec["path"]))
            if os.path.abspath(dst) != os.path.abspath(rec["path"]):
                shutil.copy2(rec["path"], dst)

    write_classes(classes, os.path.join(out_dir, "classes.txt"))
    return {"format": "yolo", "mode": mode, "labels_dir": labels_dir,
            "images_dir": images_dir if copy_images else None,
            "n_images": len(records), "n_annotations": n_anns}
