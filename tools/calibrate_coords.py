#!/usr/bin/env python3
"""calibrate_coords -- empirically measure the VLM's coordinate space.

Generates a calibration image (4 colored dots at KNOWN positions on the target
image size), asks the VLM to report each dot's pixel coordinates, then fits a
per-axis linear map (scale + offset) from reported -> true coordinates.

The fitted transform is exactly what's needed to correct model predictions
(pose/detect) whose coordinate space doesn't match the real image.

Run:  python3 tools/calibrate_coords.py --width 640 --height 427
Output: docs/pic/ + coords_transform.json  {"x": [a, b], "y": [a, b], ...}
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, ROOT)

from tools.demo_annotate import call_vlm  # noqa: E402

MARKERS = [  # (name, BGR, rel_x, rel_y)
    ("white", (255, 255, 255), 0.12, 0.12),
    ("red", (0, 0, 255), 0.88, 0.15),
    ("green", (0, 255, 0), 0.10, 0.88),
    ("blue", (255, 120, 0), 0.85, 0.85),
]


def make_calibration_image(W: int, H: int, out_path: str):
    import cv2
    import numpy as np

    img = np.full((H, W, 3), 24, dtype=np.uint8)
    truth = {}
    for name, color, rx, ry in MARKERS:
        x, y = int(rx * W), int(ry * H)
        cv2.circle(img, (x, y), max(10, W // 60), color, -1)
        cv2.putText(img, name, (x + 14, y + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA)
        truth[name] = (x, y)
    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return truth


def fit_transform(W: int, H: int, out_path: str | None = None):
    """在指定图像尺寸上实测模型坐标系, 返回 per-axis 线性映射 (a, b)。

    true = a * reported + b —— 应用于模型预测即可校正到真实像素空间。
    """
    import cv2
    import numpy as np
    from tools.demo_annotate import call_vlm

    cal_path = os.path.join(ROOT, "testenv", "results",
                            f"_calib_{W}x{H}.jpg")
    truth = make_calibration_image(W, H, cal_path)
    buf = io.BytesIO()
    cv2.imwrite("_c.jpg", cv2.imread(cal_path), [cv2.IMWRITE_JPEG_QUALITY, 95])
    content, _w, _u = call_vlm(open("_c.jpg", "rb").read(),
        "The image contains 4 colored circles on a dark background: white, red, "
        "green, blue. Report the pixel coordinates (x, y) of the CENTER of each "
        "colored circle. Respond ONLY with JSON: "
        '{"dots": [{"color": "white", "x": <int>, "y": <int>}, ...]} '
        "using integer pixel coordinates of the image.", None, 0.0)
    os.remove("_c.jpg")
    try:
        data = json.loads(content[content.find("{"): content.rfind("}") + 1])
    except Exception:  # noqa: BLE001
        return None
    reported = {d.get("color"): (float(d.get("x", 0)), float(d.get("y", 0)))
                for d in data.get("dots", []) if isinstance(d, dict)}
    fits = {}
    for axis, ai in (("x", 0), ("y", 1)):
        rep, tru = [], []
        for name, _color, rx, ry in MARKERS:
            if name in reported:
                rep.append(reported[name][ai])
                tru.append(truth[name][ai])
        if len(rep) >= 2:
            A = np.vstack([rep, np.ones(len(rep))]).T
            (a, b), *_ = np.linalg.lstsq(A, np.array(tru), rcond=None)
            fits[axis] = (float(a), float(b))
    return fits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure the VLM coordinate space.")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=427)
    ap.add_argument("--out", default=os.path.join(ROOT, "coords_transform.json"))
    args = ap.parse_args(argv)

    import cv2
    import numpy as np

    cal_path = os.path.join(ROOT, "testenv", "results", "_calib_img.jpg")
    os.makedirs(os.path.dirname(cal_path), exist_ok=True)
    truth = make_calibration_image(args.width, args.height, cal_path)

    prompt = (
        "The image contains 4 colored circles on a dark background: white, red, "
        "green, blue. Report the pixel coordinates (x, y) of the CENTER of each "
        "colored circle. Respond ONLY with JSON: "
        '{"dots": [{"color": "white", "x": <int>, "y": <int>}, ...]} '
        "using integer pixel coordinates of the image."
    )
    buf = io.BytesIO()
    cv2.imwrite("_calib_send.jpg", cv2.imread(cal_path), [cv2.IMWRITE_JPEG_QUALITY, 95])
    content, wall, usage = call_vlm(open("_calib_send.jpg", "rb").read(), prompt, None, 0.0)
    print(f"model ({wall:.1f}s): {content[:300]}")
    try:
        data = json.loads(content[content.find("{"): content.rfind("}") + 1])
    except Exception as e:  # noqa: BLE001
        print(f"[error] 无法解析模型回复: {e}")
        return 1

    reported = {d.get("color"): (float(d.get("x", 0)), float(d.get("y", 0)))
                for d in data.get("dots", []) if isinstance(d, dict)}

    # 拟合 per-axis 线性映射: true = a * reported + b
    import numpy as np
    fits = {}
    for axis, ai in (("x", 0), ("y", 1)):
        rep, tru = [], []
        for name, color, rx, ry in MARKERS:
            if name in reported:
                rep.append(reported[name][ai])
                tru.append(truth[name][ai])
        if len(rep) >= 2:
            A = np.vstack([rep, np.ones(len(rep))]).T
            (a, b), *_ = np.linalg.lstsq(A, np.array(tru), rcond=None)
            fits[axis] = (float(a), float(b))
            resid = max(abs(a * r + b - t) for r, t in zip(rep, tru))
            print(f"{axis} 轴: true = {a:.4f} * reported + {b:.2f}   最大残差 {resid:.1f}px"
                  f"  (n={len(rep)})")
        else:
            print(f"{axis} 轴: 标记点不足")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"width": args.width, "height": args.height,
                   "x": fits.get("x"), "y": fits.get("y"),
                   "reported": {k: list(v) for k, v in reported.items()},
                   "truth": {k: list(v) for k, v in truth.items()}},
                  f, ensure_ascii=False, indent=1)
    print(f"transform -> {args.out}")
    os.remove("_calib_send.jpg") if os.path.isfile("_calib_send.jpg") else None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
