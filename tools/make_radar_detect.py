#!/usr/bin/env python3
"""make_radar_detect -- build a small radar-view DETECT set from recordings.

Runs the radar-2027 TensorRT weight (armor.engine, classes GRAY/RED/BLUE) over
frames of a recorded video, keeps frames that contain confident detections and
writes them as a YOLO detect dataset:

    testenv/datasets/detect_radar_selfmade/{images,labels}/ + classes.txt

The model's own predictions become the labels -- this is MODEL-generated
ground truth (伪标签), useful for VLM prompt comparison, NOT human-verified.

The script re-execs itself with the radar-2027 venv python (the only
environment whose TensorRT version matches the exported engine).

Run:  python3 tools/make_radar_detect.py            (any python)
"""

from __future__ import annotations

import argparse
import os
import sys

# ----------------------------- 配置区（默认值，可被命令行覆盖） -----------------------------
RADAR_PY = "/home/dreamchaser/rader_2027/radar/bin/python"
TRT_LIB = ("/home/dreamchaser/Downloads/TensorRT-10.12.0.36.Linux.x86_64-gnu."
           "cuda-12.9/TensorRT-10.12.0.36/targets/x86_64-linux-gnu/lib")
ENGINE = "/home/dreamchaser/rader_2027/weights/armor.engine"
VIDEO = "/home/dreamchaser/rader_2027/demo/test_video.avi"
OUT_NAME = "detect_radar_selfmade"
TARGET = 300          # 保留帧数上限
CONF = 0.45           # 检测置信度阈值
FRAME_STRIDE = 2      # 每隔 N 帧取一帧
# ------------------------------------------------------------------


def _reexec_if_needed() -> None:
    """Re-run this script under the radar venv python with TRT libs on the path."""
    if os.path.abspath(sys.executable) == os.path.abspath(RADAR_PY):
        return
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = TRT_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    print(f"[re-exec] switching to radar env: {RADAR_PY}")
    os.execve(RADAR_PY, [RADAR_PY, os.path.abspath(__file__), "--in-radar-env"]
              + sys.argv[1:], env)


def main() -> int:
    global ENGINE, VIDEO, OUT_NAME, TARGET, CONF, FRAME_STRIDE
    ap = argparse.ArgumentParser(description="Radar-engine pseudo-label frames -> YOLO detect set.")
    ap.add_argument("--engine", default=ENGINE)
    ap.add_argument("--video", default=VIDEO)
    ap.add_argument("--out", default=OUT_NAME, help="dataset dir name under testenv/datasets/")
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--stride", type=int, default=FRAME_STRIDE)
    ap.add_argument("--classes", default="", help="comma names if engine has no metadata")
    args, _unknown = ap.parse_known_args()  # 容忍 re-exec 的 --in-radar-env 标记
    ENGINE, VIDEO, TARGET, CONF, FRAME_STRIDE = args.engine, args.video, args.target, args.conf, args.stride
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "testenv", "datasets", args.out)
    _reexec_if_needed()
    import cv2
    import numpy as np
    from ultralytics import YOLO

    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "labels"), exist_ok=True)

    model = YOLO(ENGINE, task="detect")
    names = model.names or {i: n for i, n in enumerate(
        [c.strip() for c in args.classes.split(",") if c.strip()] or ["obj"])}
    print(f"[radar] engine={ENGINE} classes={names}")

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"[error] cannot open video: {VIDEO}")
        return 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[radar] video: {VIDEO}  frames={total}  stride={FRAME_STRIDE}  conf>={CONF}")

    n_kept = n_empty = n_boxes = 0
    idx = -1
    while n_kept < TARGET:
        ok, frame = cap.read()
        idx += 1
        if not ok:
            break
        if idx % FRAME_STRIDE:
            continue
        res = model(frame, verbose=False, conf=CONF)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            n_empty += 1
            continue
        H, W = frame.shape[:2]
        lines = []
        for b in boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            cls = int(b.cls[0])
            cx = (x1 + x2) / 2 / W
            cy = (y1 + y2) / 2 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        name = f"radar_{idx:06d}.jpg"
        cv2.imwrite(os.path.join(OUT, "images", name), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        with open(os.path.join(OUT, "labels", name.replace(".jpg", ".txt")),
                  "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        n_kept += 1
        n_boxes += len(lines)
        if n_kept % 25 == 0:
            print(f"  kept {n_kept}/{TARGET} (frame {idx}/{total}, empty skipped: {n_empty})")
    cap.release()

    with open(os.path.join(OUT, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(names[i] for i in sorted(names)) + "\n")
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"# detect_radar_selfmade\n\n"
                f"- source video: `{VIDEO}`\n"
                f"- label source: rader_2027 `armor.engine` 伪标签（模型自动生成，非人工）\n"
                f"- classes: {names}\n"
                f"- frames kept: {n_kept} (conf >= {CONF}, stride {FRAME_STRIDE})\n"
                f"- total boxes: {n_boxes} (empty skipped: {n_empty})\n")
    print(f"[done] kept={n_kept} boxes={n_boxes} empty-skipped={n_empty} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
