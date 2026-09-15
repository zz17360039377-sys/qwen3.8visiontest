"""Draw annotation overlays (detect boxes / pose skeleton + keypoints).

Pure preview -- the .txt YOLO labels are the source of truth, this is just a
human-check image. Coordinates in the annotations are normalised [0, 1] and are
scaled here to the full-resolution image.
"""

from __future__ import annotations

import cv2
import numpy as np

# (B, G, R) palette cycled by class id
_PALETTE = [
    (0, 200, 0), (200, 120, 0), (0, 120, 220), (220, 0, 160),
    (0, 200, 200), (200, 200, 0), (120, 0, 220), (0, 0, 200),
    (200, 80, 200), (80, 200, 80), (0, 255, 255), (255, 128, 0),
]


def _color(class_id: int) -> tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


def draw_overlay(
    image_path: str,
    anns: list[dict],
    mode: str,
    out_path: str,
    classes: list[dict],
    skeleton: list | None = None,
) -> None:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"cannot read image: {image_path}")
    H, W = img.shape[:2]
    mode = mode.lower()

    for ann in anns:
        cid = int(ann["class_id"])
        col = _color(cid)
        name = ann.get("label") or (classes[cid]["name"] if cid < len(classes) else str(cid))

        x1 = int(round(ann["x1"] * W))
        y1 = int(round(ann["y1"] * H))
        x2 = int(round(ann["x2"] * W))
        y2 = int(round(ann["y2"] * H))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)

        label = f"{name}"
        if ann.get("score") and float(ann["score"]) > 0:
            label += f" {float(ann['score']):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y1 - 6 if y1 - 6 - th > 0 else y1 + th + 6
        cv2.rectangle(img, (x1, ty - th - 2), (x1 + tw + 4, ty + 2), col, -1)
        cv2.putText(img, label, (x1 + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if mode == "pose":
            kps = ann.get("keypoints") or []
            pts = [(int(round(k[0] * W)), int(round(k[1] * H)), (k[2] if len(k) > 2 else 0)) for k in kps]
            if skeleton:
                for edge in skeleton:
                    try:
                        i, j = int(edge[0]), int(edge[1])
                    except Exception:
                        continue
                    if i < len(pts) and j < len(pts):
                        pi, pj = pts[i], pts[j]
                        if pi[2] > 0 and pj[2] > 0:
                            cv2.line(img, pi[:2], pj[:2], col, 2, cv2.LINE_AA)
            for px, py, v in pts:
                if v > 0:
                    c = (0, 255, 255)
                else:
                    c = (0, 0, 255)
                cv2.circle(img, (px, py), 4, c, -1, cv2.LINE_AA)
                cv2.circle(img, (px, py), 4, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, img)
