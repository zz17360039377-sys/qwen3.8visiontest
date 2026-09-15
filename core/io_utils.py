"""Shared file/config helpers for the entry apps (CLI, web, tools)."""

from __future__ import annotations

import os

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_inline_classes(s: str) -> list[dict]:
    """Parse inline classes into ``[{name, description}, ...]``.

    Classes are separated by ``;``; within a class the first ``=`` splits the
    name from its description, so a description may freely contain commas::

        "armor=a small armor module"
        "cat=a feline; dog=canine; ball"
    """
    out = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            name, desc = part.split("=", 1)
            out.append({"name": name.strip(), "description": desc.strip()})
        else:
            out.append({"name": part, "description": ""})
    return out


def load_config(path: str) -> tuple[list[dict], list[dict], list]:
    """Read a YAML task config -> (classes, keypoints, skeleton)."""
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("classes") or [], data.get("keypoints") or [], data.get("skeleton") or []


def list_image_dir(d: str) -> list[str]:
    return [
        os.path.join(d, fn)
        for fn in sorted(os.listdir(d))
        if os.path.splitext(fn)[1].lower() in IMG_EXTS
    ]


def extract_frames(video: str, step: int, out_dir: str) -> list[str]:
    """Dump every --step-th frame of *video* into out_dir/_frames as jpg."""
    import cv2

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video}")
    frames_dir = os.path.join(out_dir, "_frames")
    os.makedirs(frames_dir, exist_ok=True)
    files = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            p = os.path.join(frames_dir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(p, frame)
            files.append(p)
        idx += 1
    cap.release()
    return files
