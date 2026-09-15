#!/usr/bin/env python3
"""bench_vlm -- benchmark the local Qwen VLM (llama.cpp) for annotation workloads.

Measures, per scenario and per round:
  * wall-clock latency and llama.cpp ``timings`` (prompt eval / generation tok/s);
  * whether the answer parsed as the expected JSON;
  * predicted label vs the ground-truth folder name for slice scenarios.

Scenarios
  text_baseline        tiny text-only JSON reply (generation speed reference)
  recheck_xK           40x56 dataset slice upscaled xK, single-label classify
                       (the production recheck prompt wording)
  detect_frame         full-frame detect prompt (2 armor classes)
  pose_frame           full-frame pose prompt (4 box-corner keypoints)
  *_thinking           same prompt with enable_thinking=true (2 rounds)

Inputs default to the user's real dataset: ``--slice-dir`` is a
folder-per-class dataset (folder name = ground-truth label), ``--video`` is a
full-frame source from which a few frames are extracted.

Outputs ``bench_report.md`` + ``bench_results.csv`` in ``--outdir``.

Run:  python3 bench_vlm.py [--rounds 5] [--slice-dir DIR] [--video FILE]
"""

from __future__ import annotations

import argparse
import base64
import csv
import glob
import io
import json
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.parse_geometry import load_json, nms, parse_to_annotations  # noqa: E402
from core.prompts import build_prompt                                 # noqa: E402
from core.vlm_client import chat_raw                                  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CNN_DIR = os.path.expanduser("~/Desktop/模型训练/CNN")

# slice-class classes: folder names in the dataset == armor ids
ARMOR_CLASSES = [
    {"name": "0", "description": "digit 0 pattern - guard (sentry) robot armor"},
    {"name": "1", "description": "digit 1 pattern - infantry robot number 1"},
    {"name": "2", "description": "digit 2 pattern - infantry robot number 2"},
    {"name": "3", "description": "digit 3 pattern - infantry robot number 3"},
    {"name": "4", "description": "digit 4 pattern - infantry robot number 4"},
    {"name": "6", "description": "digit 6 pattern - outpost armor"},
    {"name": "7", "description": "digit 7 pattern - base armor"},
]

CLASSIFY_SYSTEM = (
    "You are a strict image classifier for RoboMaster armor panel digits. "
    "You answer with exactly one JSON object and nothing else."
)

# full-frame task classes (same as config.example.yaml)
FRAME_CLASSES = [
    {"name": "red_armor",
     "description": "A small bright-red armor/target module, roughly rectangular, high saturation."},
    {"name": "blue_armor",
     "description": "A small blue armor/target module, roughly rectangular, clearly distinct from red."},
]
FRAME_KEYPOINTS = [
    {"name": "top_left", "description": "The top-left corner of the armor module."},
    {"name": "top_right", "description": "The top-right corner of the armor module."},
    {"name": "bottom_right", "description": "The bottom-right corner of the armor module."},
    {"name": "bottom_left", "description": "The bottom-left corner of the armor module."},
]


def classify_user_prompt() -> str:
    lines = [f'  "{c["name"]}" -- {c["description"]}' for c in ARMOR_CLASSES]
    return (
        "Classify the armor digit pattern in the image into exactly ONE of these classes:\n"
        + "\n".join(lines)
        + "\n\nThe image is a small grayscale crop (upscaled, possibly pixelated) showing ONE "
        "white digit-like pattern on a dark background. Judge by the digit shape only.\n"
        'Respond ONLY with: {"label": "<one of: 0,1,2,3,4,6,7>", "score": <number 0..1>}'
    )


# ------------------------------------------------------------------ imaging

def _pil_data_url(im, quality: int = 90) -> str:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def slice_data_url(path: str, upscale: int) -> tuple[str, int, int]:
    """Upscale a small slice xN (LANCZOS) and return (data_url, w, h)."""
    from PIL import Image

    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    w, h = im.size
    im = im.resize((w * upscale, h * upscale), Image.LANCZOS)
    return _pil_data_url(im), im.size[0], im.size[1]


def frame_data_url(path: str, max_side: int) -> tuple[str, int, int]:
    """Downscale a full frame to max_side (like vlm_client) and return parts."""
    from PIL import Image

    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    w, h = im.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        im = im.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))))
    return _pil_data_url(im), im.size[0], im.size[1]


# ---------------------------------------------------------------- sampling

def pick_slices(slice_dir: str, seed: int = 0) -> list[tuple[str, str]]:
    """One random slice per class folder -> [(path, true_label), ...]."""
    rng = random.Random(seed)
    out = []
    for cls in sorted(os.listdir(slice_dir)):
        d = os.path.join(slice_dir, cls)
        if not os.path.isdir(d) or cls.startswith(("_", ".")):
            continue
        files = [f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in IMG_EXTS]
        if files:
            out.append((os.path.join(d, rng.choice(files)), cls))
    return out


def extract_frames(video: str, n: int, out_dir: str) -> list[str]:
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    paths = []
    for i in range(1, n + 1):
        want = int(total * i / (n + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, want)
        ok, frame = cap.read()
        if not ok:
            continue
        p = os.path.join(out_dir, f"frame_{want:06d}.jpg")
        cv2.imwrite(p, frame)
        paths.append(p)
    cap.release()
    return paths


# ---------------------------------------------------------------- transport

def chat(api: str, model: str, messages: list, thinking: bool,
         temperature: float, max_tokens: int, timeout: int = 600) -> dict:
    """Thin wrapper over vlm_client.chat_raw (kept for bench call sites)."""
    return chat_raw(api, model, messages, thinking=thinking,
                    temperature=temperature, max_tokens=max_tokens, timeout=timeout)


def vram_used_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return -1


def model_name(api: str, fallback: str) -> str:
    try:
        r = requests.get(api.rstrip("/") + "/models", timeout=5)
        return str(r.json()["data"][0]["id"])
    except Exception:
        return fallback


# ---------------------------------------------------------------- scenarios

CSV_FIELDS = [
    "scenario", "round", "ok", "wall_s", "prompt_n", "prompt_ms",
    "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second",
    "cache_n", "parse_ok", "vlm_label", "true_label", "agree", "n_instances",
    "finish", "error", "timings_json",
]


def record(rows: list, scenario: str, rnd: int, res: dict | None,
           parse: dict | None, err: str | None = None) -> None:
    row = {k: "" for k in CSV_FIELDS}
    row.update({"scenario": scenario, "round": rnd, "ok": res is not None})
    if res is not None:
        t = res["timings"]
        row.update({
            "wall_s": f"{res['wall']:.3f}",
            "prompt_n": t.get("prompt_n", ""),
            "prompt_ms": t.get("prompt_ms", ""),
            "prompt_per_second": t.get("prompt_per_second", ""),
            "predicted_n": t.get("predicted_n", ""),
            "predicted_ms": t.get("predicted_ms", ""),
            "predicted_per_second": t.get("predicted_per_second", ""),
            "cache_n": t.get("cache_n", ""),
            "finish": res["finish"],
            "timings_json": json.dumps(t, ensure_ascii=False) if t else "",
        })
    if parse:
        row.update({k: parse.get(k, "") for k in
                    ("parse_ok", "vlm_label", "true_label", "agree", "n_instances")})
    if err:
        row["error"] = err
    rows.append(row)


def run_classify_round(args, rnd: int, slices: list, upscale: int,
                       thinking: bool, rows: list) -> None:
    path, true_label = slices[(rnd - 1) % len(slices)]
    scenario = f"recheck_x{upscale}" + ("_thinking" if thinking else "")
    try:
        url, w, h = slice_data_url(path, upscale)
        res = chat(
            args.api, args.model,
            [{"role": "system", "content": CLASSIFY_SYSTEM},
             {"role": "user", "content": [
                 {"type": "image_url", "image_url": {"url": url}},
                 {"type": "text", "text": classify_user_prompt()}]}],
            thinking=thinking, temperature=args.temperature, max_tokens=96,
        )
    except Exception as e:  # noqa: BLE001
        record(rows, scenario, rnd, None, None, err=f"{type(e).__name__}: {e}")
        return
    data = load_json(res["content"])
    label = str(data.get("label", "")).strip() if isinstance(data, dict) else ""
    score = float(data.get("score", 0)) if isinstance(data, dict) and data.get("score") else 0.0
    record(rows, scenario, rnd, res, {
        "parse_ok": bool(label),
        "vlm_label": f"{label}({score:.2f})" if label else "",
        "true_label": true_label,
        "agree": int(label == true_label) if label else "",
    })


def run_frame_round(args, rnd: int, frames: list, mode: str,
                    thinking: bool, rows: list) -> None:
    path = frames[(rnd - 1) % len(frames)]
    scenario = f"{mode}_frame" + ("_thinking" if thinking else "")
    try:
        url, w, h = frame_data_url(path, args.max_side)
        system, user = build_prompt(
            mode, FRAME_CLASSES, FRAME_KEYPOINTS, width=w, height=h)
        res = chat(
            args.api, args.model,
            [{"role": "system", "content": system},
             {"role": "user", "content": [
                 {"type": "image_url", "image_url": {"url": url}},
                 {"type": "text", "text": user}]}],
            thinking=thinking, temperature=args.temperature, max_tokens=768,
        )
    except Exception as e:  # noqa: BLE001
        record(rows, scenario, rnd, None, None, err=f"{type(e).__name__}: {e}")
        return
    anns = parse_to_annotations(res["content"], w, h, mode, FRAME_CLASSES,
                                len(FRAME_KEYPOINTS) if mode == "pose" else 0)
    anns = nms(anns, iou_thresh=0.5)
    record(rows, scenario, rnd, res, {
        "parse_ok": load_json(res["content"]) is not None,
        "n_instances": len(anns),
    })


def run_text_round(args, rnd: int, rows: list) -> None:
    try:
        res = chat(
            args.api, args.model,
            [{"role": "user",
              "content": f'(round {rnd}) Reply with ONLY the JSON object {{"ok": true}}.'}],
            thinking=False, temperature=0.0, max_tokens=32,
        )
    except Exception as e:  # noqa: BLE001
        record(rows, "text_baseline", rnd, None, None, err=f"{type(e).__name__}: {e}")
        return
    record(rows, "text_baseline", rnd, res,
           {"parse_ok": load_json(res["content"]) is not None})


# ---------------------------------------------------------------- reporting

def _pctl(vals: list, p: int) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def aggregate(rows: list) -> dict:
    out = {}
    for r in rows:
        out.setdefault(r["scenario"], []).append(r)
    stats = {}
    for name, rs in out.items():
        walls = [float(r["wall_s"]) for r in rs if r["wall_s"]]
        tok_s = [float(r["predicted_per_second"]) for r in rs if r["predicted_per_second"]]
        pr_s = [float(r["prompt_per_second"]) for r in rs if r["prompt_per_second"]]
        pr_n = [float(r["prompt_n"]) for r in rs if r["prompt_n"]]
        agree = [r for r in rs if r["agree"] != ""]
        stats[name] = {
            "n": len(rs),
            "ok": sum(1 for r in rs if r["ok"] == True),  # noqa: E712
            "parse_ok": sum(1 for r in rs if r["parse_ok"] == True),  # noqa: E712
            "wall_mean": statistics.mean(walls) if walls else 0.0,
            "wall_p50": _pctl(walls, 50),
            "wall_p95": _pctl(walls, 95),
            "tok_s": statistics.mean(tok_s) if tok_s else 0.0,
            "prompt_s": statistics.mean(pr_s) if pr_s else 0.0,
            "prompt_n": statistics.mean(pr_n) if pr_n else 0.0,
            "agree": f"{sum(r['agree'] for r in agree)}/{len(agree)}" if agree else "-",
        }
    return stats


def write_report(path: str, stats: dict, args, vram: int, n_slices: int,
                 n_frames: int, csv_path: str) -> None:
    def throughput(name: str) -> str:
        st = stats.get(name)
        if not st or not st["wall_mean"]:
            return "-"
        per_min = 60.0 / st["wall_mean"]
        h14k = 14000 / per_min / 60.0
        return f"{per_min:.1f} 张/分钟；1.4 万张 ≈ {h14k:.1f} 小时"

    lines = [
        "# VLM 性能基准报告",
        "",
        f"- 时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- 服务：{args.api}  模型：{model_name(args.api, args.model)}",
        f"- GPU 显存占用：{vram} MiB",
        f"- 轮数：{args.rounds}（thinking 场景固定 2 轮）",
        f"- 切片来源：{args.slice_dir}（{n_slices} 张，每类 1 张，seed=0）",
        f"- 整帧来源：{args.video or args.frames_dir or '无'}（{n_frames} 帧，max_side={args.max_side}）",
        "",
        "| 场景 | 轮数 | 请求成功 | JSON解析 | 平均耗时(s) | p50(s) | p95(s) | 生成 tok/s | prompt tok/s | prompt_n(≈文本+视觉token) | 标签一致 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    order = ["text_baseline", "recheck_x3", "recheck_x6", "recheck_x9",
             "detect_frame", "pose_frame", "recheck_x6_thinking", "detect_frame_thinking"]
    for name in order + [k for k in stats if k not in order]:
        st = stats.get(name)
        if not st:
            continue
        lines.append(
            f"| {name} | {st['n']} | {st['ok']} | {st['parse_ok']} "
            f"| {st['wall_mean']:.2f} | {st['wall_p50']:.2f} | {st['wall_p95']:.2f} "
            f"| {st['tok_s']:.1f} | {st['prompt_s']:.1f} | {st['prompt_n']:.0f} "
            f"| {st['agree']} |")

    lines += ["", "## 吞吐结论", ""]
    for name in ("recheck_x3", "recheck_x6", "recheck_x9"):
        if name in stats:
            lines.append(f"- `{name}`：{throughput(name)}")
    for name in ("detect_frame", "pose_frame"):
        if name in stats:
            st = stats[name]
            lines.append(f"- `{name}`：{60.0 / st['wall_mean']:.1f} 张/分钟" if st["wall_mean"] else f"- `{name}`：无数据")
    lines += [
        "",
        "## 说明",
        "",
        "- `prompt_n` 含文本 prompt 与视觉 token；与 `text_baseline` 的差值 ≈ 该分辨率下的视觉 token 数。",
        "- llama.cpp 对相同前缀有 KV 缓存，第 1 轮偏慢属正常；不同切片/帧之间不共享图像 token。",
        "- 原始逐轮数据见 `" + os.path.basename(csv_path) + "`。",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark the local Qwen VLM for annotation workloads.")
    ap.add_argument("--api", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--slice-dir", default=os.path.join(CNN_DIR, "datasets", "train_2000"))
    ap.add_argument("--video", default=None, help="full-frame video (default: auto-find under CNN/converted_mp4)")
    ap.add_argument("--frames-dir", default=None, help="existing full-frame images (overrides --video)")
    ap.add_argument("--upscales", default="3,6,9")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)

    base = args.api.rstrip("/").rsplit("/v1", 1)[0]
    try:
        requests.get(base + "/health", timeout=5).raise_for_status()
    except Exception as e:  # noqa: BLE001
        print(f"[error] VLM 服务未就绪 ({base}/health): {e}", file=__import__("sys").stderr)
        return 1

    out_dir = args.outdir or f"bench_{datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs(out_dir, exist_ok=True)

    slices = pick_slices(args.slice_dir)
    if not slices:
        print(f"[error] no slices found under {args.slice_dir}")
        return 1

    frames: list[str] = []
    if args.frames_dir:
        frames = sorted(os.path.join(args.frames_dir, f) for f in os.listdir(args.frames_dir)
                        if os.path.splitext(f)[1].lower() in IMG_EXTS)
    else:
        video = args.video
        if not video:
            cands = glob.glob(os.path.join(CNN_DIR, "converted_mp4", "**", "*.mp4"), recursive=True)
            video = cands[0] if cands else None
        if video:
            print(f"[frames] extracting from {video}")
            frames = extract_frames(video, args.rounds, os.path.join(out_dir, "frames"))
    print(f"[bench] slices={len(slices)}  frames={len(frames)}  rounds={args.rounds}  -> {out_dir}")

    vram = vram_used_mib()
    rows: list = []
    t_start = time.time()

    for rnd in range(1, args.rounds + 1):
        run_text_round(args, rnd, rows)
        print(f"[round {rnd}/{args.rounds}] text done  ({time.time() - t_start:.0f}s)")
    for k in [int(x) for x in str(args.upscales).split(",") if x.strip()]:
        for rnd in range(1, args.rounds + 1):
            run_classify_round(args, rnd, slices, k, thinking=False, rows=rows)
        print(f"[round {args.rounds}/{args.rounds}] recheck_x{k} done  ({time.time() - t_start:.0f}s)")
    if frames:
        for mode in ("detect", "pose"):
            for rnd in range(1, args.rounds + 1):
                run_frame_round(args, rnd, frames, mode, thinking=False, rows=rows)
            print(f"[round {args.rounds}/{args.rounds}] {mode}_frame done  ({time.time() - t_start:.0f}s)")
    # thinking comparison: 2 rounds each
    for rnd in range(1, min(2, args.rounds) + 1):
        run_classify_round(args, rnd, slices, 6, thinking=True, rows=rows)
        if frames:
            run_frame_round(args, rnd, frames, "detect", thinking=True, rows=rows)
        print(f"[round {rnd}/2] thinking done  ({time.time() - t_start:.0f}s)")

    csv_path = os.path.join(out_dir, "bench_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    stats = aggregate(rows)
    report = os.path.join(out_dir, "bench_report.md")
    write_report(report, stats, args, vram, len(slices), len(frames), csv_path)

    print(f"[done] {len(rows)} calls in {time.time() - t_start:.0f}s")
    for name, st in stats.items():
        print(f"  {name:28s} n={st['n']}  wall={st['wall_mean']:.2f}s (p95 {st['wall_p95']:.2f})"
              f"  tok/s={st['tok_s']:.1f}  agree={st['agree']}")
    print(f"[report] {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
