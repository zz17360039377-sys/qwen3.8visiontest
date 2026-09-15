#!/usr/bin/env python3
"""selflabel -- prompt-driven VLM auto-annotation to YOLO format.

Drives the local Qwen VLM to emit detection boxes (``--mode detect``) or pose
keypoints (``--mode pose``) for a set of images / a video, parses the answer
defensively, and writes YOLO ``.txt`` labels plus a preview overlay.

The class set, class descriptions, and the number of pose keypoints N are all
PER-TASK: supply them via ``--config`` (YAML) or ``--classes`` / ``--kpt``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.draw import draw_overlay
from core.io_utils import (IMG_EXTS, extract_frames, list_image_dir,
                           load_config, parse_inline_classes)
from core.parse_geometry import nms, parse_to_annotations
from core.prompts import build_prompt
from core.vlm_client import VLMClient, VLMError
from export.fmt_yolo import write_classes


# ------------------------------------------------------------------- runner

def run_image(
    client: VLMClient,
    fp: str,
    mode: str,
    classes: list[dict],
    keypoints: list[dict],
    extra: str,
    skeleton: list,
    args: argparse.Namespace,
    out_dir: str,
) -> list[dict]:
    """Annotate one image; returns the (normalised) annotations, [] if none."""
    stem = os.path.splitext(os.path.basename(fp))[0]
    label_path = os.path.join(out_dir, "labels", stem + ".txt")
    if args.policy == "skip" and os.path.exists(label_path):
        print(f"    {stem}: skip (label exists)")
        return []

    data_url, w, h = client.resize_path(fp)
    system, user = build_prompt(mode, classes, keypoints, width=w, height=h, extra=extra)
    content = client.send(data_url, system, user)

    n_kpt = len(keypoints) if mode == "pose" else 0
    anns = parse_to_annotations(content, w, h, mode, classes, n_kpt)
    anns = nms(anns, iou_thresh=args.iou, max_det=args.max_det, conf=args.conf)

    if not args.no_draw:
        prev_dir = os.path.join(out_dir, "previews")
        os.makedirs(prev_dir, exist_ok=True)
        draw_overlay(fp, anns, mode, os.path.join(prev_dir, stem + "_annot.jpg"), classes, skeleton)
    return anns


def run_classify_mode(
    args: argparse.Namespace,
    files: list[str],
    classes: list[dict],
) -> int:
    """Single-label classification: copy each image into <out>/<class>/."""
    from recheck_classify import classify_one, load_image

    out_dir = args.out
    conf_min = args.conf_min
    csv_path = os.path.join(out_dir, "classify_summary.csv")
    done: set[str] = set()
    new_file = True
    if args.policy == "skip" and os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            done = {os.path.basename(r["file"]) for r in csv.DictReader(f)}
        new_file = False
        print(f"[resume] {len(done)} already classified")

    ref_images = []
    if args.refs:
        refs_dir = os.path.expanduser(args.refs)
        for c in classes:
            p = os.path.join(refs_dir, f"{c['name']}.png")
            if os.path.exists(p):
                ref_images.append(load_image(p)[0])

    client = VLMClient(api_base=args.api, model=args.model, temperature=args.temperature,
                       max_tokens=args.max_tokens, thinking=args.thinking,
                       max_side=args.max_side)
    if not client.ping():
        print(f"[warn] server at {client.base_url} did not respond to ping; continuing anyway.")

    counts: dict[str, int] = {}
    lowconf = fails = 0
    t_start = time.time()
    new_file_header_needed = new_file and not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as csv_f:
        w = csv.writer(csv_f)
        if new_file_header_needed:
            w.writerow(["file", "class", "conf", "wall_s"])
        for i, fp in enumerate(files, 1):
            name = os.path.basename(fp)
            if name in done:
                continue
            try:
                r = classify_one(args.api, args.model, fp, classes, args.upscale,
                                 ref_images, args.prompt, args.temperature, 96,
                                 autocontrast=args.autocontrast)
                if not r["label"]:
                    fails += 1
                    print(f"[{i}/{len(files)}] {name}: PARSE FAIL ({r['parse_error']})",
                          file=sys.stderr)
                    continue
                label = r["label"]
                dst_dir = os.path.join(out_dir, "_lowconf" if r["score"] < conf_min else label)
                os.makedirs(dst_dir, exist_ok=True)
                if not os.path.exists(os.path.join(dst_dir, name)):
                    shutil.copy2(fp, os.path.join(dst_dir, name))
                counts[label] = counts.get(label, 0) + 1
                if r["score"] < conf_min:
                    lowconf += 1
                w.writerow([name, label, f"{r['score']:.3f}", f"{r['wall']:.2f}"])
                csv_f.flush()
                print(f"[{i}/{len(files)}] {name}: {label} ({r['score']:.2f})")
            except (VLMError, ValueError) as e:
                fails += 1
                print(f"[{i}/{len(files)}] {name}: ERROR {e}", file=sys.stderr)
    dt = time.time() - t_start
    print(f"[done] classified={sum(counts.values())} lowconf(<{conf_min})={lowconf} "
          f"fail={fails} in {dt:.0f}s -> {out_dir}")
    for c in sorted(counts):
        print(f"    {c}: {counts[c]}")
    return 0 if fails == 0 else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Prompt-driven VLM auto-annotation.")
    ap.add_argument("--mode", required=True, choices=["detect", "pose", "classify"],
                    help="annotation type")
    ap.add_argument("--image", help="a single image file")
    ap.add_argument("--images", help="a folder of images")
    ap.add_argument("--video", help="a video file (sampled every --video-step frames)")
    ap.add_argument("--video-step", type=int, default=5)

    ap.add_argument("--config", help="YAML config: classes, keypoints, skeleton")
    ap.add_argument("--classes", help='inline classes "name1=desc1;name2=desc2" (overrides config)')
    ap.add_argument("--kpt", type=int, help="pose: number of keypoints (if config has none)")
    ap.add_argument("--prompt", default="", help="extra free-text guidance appended to the prompt")

    ap.add_argument("--format", choices=["yolo", "coco"], default="yolo",
                    help="detect/pose output format (default yolo)")
    ap.add_argument("--conf-min", type=float, default=0.30,
                    help="classify: below this score -> _lowconf/")
    ap.add_argument("--upscale", type=int, default=1,
                    help="classify: upscale factor before sending (small slices -> 6)")
    ap.add_argument("--autocontrast", action="store_true",
                    help="classify: stretch contrast before sending (dark slices)")
    ap.add_argument("--refs", default="", help="classify: dir of <class>.png few-shot references")

    ap.add_argument("--api", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--thinking", action="store_true", help="enable the model's reasoning block (slower)")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-side", type=int, default=1280, help="downscale max image side before sending")

    ap.add_argument("--conf", type=float, default=0.0, help="min model score to keep (default 0.0 = keep all)")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    ap.add_argument("--max-det", type=int, default=0, help="max instances to keep (0 = no limit)")

    ap.add_argument("--out", default=None, help="output dir (default out_<mode>_<timestamp>)")
    ap.add_argument("--policy", choices=["overwrite", "skip"], default="overwrite")
    ap.add_argument("--no-draw", action="store_true", help="skip writing preview overlay images")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N inputs (for testing)")
    args = ap.parse_args(argv)

    # ---- resolve classes ----------------------------------------------------
    classes, cfg_kpts, skeleton = [], [], []
    if args.classes:
        classes = parse_inline_classes(args.classes)
    elif args.config:
        classes, cfg_kpts, skeleton = load_config(args.config)
    if not classes:
        classes = [{"name": "object", "description": (args.prompt or "the target object")}]

    # ---- inputs -------------------------------------------------------------
    if not (args.image or args.images or args.video):
        ap.error("one of --image / --images / --video is required")
    out_dir = args.out or f"out_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs(out_dir, exist_ok=True)

    if args.video:
        files = extract_frames(args.video, args.video_step, out_dir)
    else:
        files = [args.image] if args.image else list_image_dir(args.images)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        print("no input images found")
        return 1

    # ---- classify branch ----------------------------------------------------
    if args.mode == "classify":
        return run_classify_mode(args, files, classes)

    # ---- resolve pose keypoints ---------------------------------------------
    keypoints = cfg_kpts
    if args.mode == "pose" and not keypoints:
        n = args.kpt or 4
        print(f"[warn] no `keypoints` defined; using {n} generic keypoints "
              f"(add `keypoints` with descriptions to a config for better grounding).")
        keypoints = [{"name": f"kpt{i}", "description": f"keypoint {i + 1}"} for i in range(n)]
    elif args.mode == "detect":
        keypoints = []

    os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)
    write_classes(classes, os.path.join(out_dir, "classes.txt"))

    client = VLMClient(
        api_base=args.api,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
        max_side=args.max_side,
    )
    if not client.ping():
        print(f"[warn] server at {client.base_url} did not respond to ping; continuing anyway.")

    n_kpt = len(keypoints) if args.mode == "pose" else 0
    print(f"[selflabel] mode={args.mode}  images={len(files)}  classes={len(classes)}  "
          f"N_kpt={n_kpt}  thinking={args.thinking}  format={args.format}  -> {out_dir}")

    ok = fail = skipped = 0
    records: list[dict] = []
    t_start = time.time()
    for i, fp in enumerate(files, 1):
        name = os.path.basename(fp)
        try:
            anns = run_image(client, fp, args.mode, classes, keypoints, args.prompt,
                             skeleton, args, out_dir)
            ok += 1
            from PIL import Image
            with Image.open(fp) as im:
                w0, h0 = im.size
            records.append({"path": fp, "width": w0, "height": h0, "anns": anns})
            print(f"[{i}/{len(files)}] {name}: {len(anns)} instance(s)")
        except (VLMError, ValueError) as e:
            fail += 1
            print(f"[{i}/{len(files)}] {name}: ERROR {e}", file=sys.stderr)

    import exporters
    manifest = exporters.write_dataset(
        args.format, out_dir, records, classes,
        keypoints=keypoints if args.mode == "pose" else [],
        skeleton=skeleton, copy_images=True)
    dt = time.time() - t_start
    print(f"[done] ok={ok} fail={fail}  in {dt:.1f}s  "
          f"{args.format}: {manifest['n_images']} images / {manifest['n_annotations']} anns "
          f"-> {out_dir}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
