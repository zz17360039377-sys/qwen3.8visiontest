#!/usr/bin/env python3
"""VLM labeling harness (self-written). Calls local llama-server OpenAI API.

Usage: run_ds.py --dataset NAME --type detect|pose|cls|labelme --n N --workers W
Dataset meta (classes, kpts) is auto-read from testenv/datasets/<NAME>/{classes.txt,README.md}.
Outputs:
  testenv/results/vlm_probe/<NAME>/labels/<stem>.txt
  testenv/work/logs/<NAME>.jsonl  (per-image stats)
"""
import argparse, base64, io, json, os, re, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

ROOT = "/home/dreamchaser/selflabel/testenv"
API = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "/home/dreamchaser/bigmodel/models/Qwen3.8-27B-UD-Q4_K_XL.gguf"
TEST = "vlm_probe"
MAXSIDE = 1280
LOGDIR = f"{ROOT}/work/logs"

os.makedirs(LOGDIR, exist_ok=True)

gpu_stop = threading.Event()
gpu_samples = []

def gpu_sampler():
    while not gpu_stop.is_set():
        try:
            out = os.popen("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").read().strip()
            gpu_samples.append(int(out.splitlines()[0]))
        except Exception:
            pass
        time.sleep(2)

def read_meta(ds):
    d = f"{ROOT}/datasets/{ds}"
    classes = [l.strip() for l in open(f"{d}/classes.txt") if l.strip()]
    kpts = None
    for line in open(f"{d}/README.md"):
        m = re.search(r"keypoints:\s*(\d+)", line)
        if m:
            kpts = int(m.group(1)); break
    return d, classes, kpts

def pick_images(d, n, seed=42):
    import random
    manifest = f"{d}/manifest.csv"
    lines = [l.strip().split(",") for l in open(manifest).read().splitlines()[1:] if l.strip()]
    imgdir = f"{d}/images"
    # manifest column order varies (class,image vs image,label): pick the column that exists as files
    col = 0
    for c in range(min(2, len(lines[0]))):
        if os.path.exists(os.path.join(imgdir, lines[0][c])):
            col = c
            break
    imgs = [r[col] for r in lines if len(r) > col and r[col]]
    random.Random(seed).shuffle(imgs)
    return imgs[:n]

def prep_image(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = MAXSIDE / max(w, h)
    if scale < 1:
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    if max(w, h) < 256:  # upscale tiny slices for the VLM
        s = 256 / max(w, h)
        im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode(), im.size

def call_api(img_b64, prompt, temperature=0.0, max_tokens=900):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img_b64}},
                {"type": "text", "text": prompt},
            ]},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    dt = time.time() - t0
    u = resp.get("usage", {})
    content = resp["choices"][0]["message"]["content"] or ""
    return clean_think(content), dt, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)

def clean_think(txt):
    # strip any <think>...</think> blocks the model may still emit
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()

def clamp(v):
    return min(1.0, max(0.0, v))

def parse_detect(txt, ncls):
    lines_out = []
    for raw in txt.splitlines():
        raw = raw.strip().strip("`*")
        if not raw or raw.upper().startswith(("NONE", "NO ", "无")):
            continue
        nums = re.findall(r"-?\d+\.?\d*", raw)
        if len(nums) < 5:
            continue
        try:
            vals = [float(x) for x in nums[:5]]
        except ValueError:
            continue
        c = int(vals[0])
        if c < 0 or c >= ncls:
            # maybe model wrote class name? handled by caller mapping earlier
            continue
        cx, cy, w, h = (clamp(v) for v in vals[1:5])
        if w <= 0.001 or h <= 0.001:
            continue
        lines_out.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines_out

def parse_pose(txt, ncls, nkpt):
    lines_out = []
    for raw in txt.splitlines():
        raw = raw.strip().strip("`*")
        if not raw or raw.upper().startswith(("NONE", "NO ", "无")):
            continue
        nums = re.findall(r"-?\d+\.?\d*", raw)
        need = 5 + 3 * nkpt
        if len(nums) < need:
            continue
        try:
            vals = [float(x) for x in nums[:need]]
        except ValueError:
            continue
        c = int(vals[0])
        if c < 0 or c >= ncls:
            continue
        cx, cy, w, h = (clamp(v) for v in vals[1:5])
        kps = []
        ok = True
        for i in range(nkpt):
            x, y, v = vals[5 + 3 * i], vals[6 + 3 * i], vals[7 + 3 * i]
            vi = int(round(v))
            if vi not in (0, 1, 2):
                ok = False; break
            kps.append(f"{clamp(x):.6f} {clamp(y):.6f} {vi}")
        if ok:
            lines_out.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} " + " ".join(kps))
    return lines_out

def parse_cls(txt, classes):
    t = txt.strip().splitlines()
    if not t:
        return None
    line = t[0].strip().strip("`* .")
    m = re.search(r"[BRS]?\d+|[A-Za-z]+\d*|\d+", line)
    if not m:
        return None
    tok = m.group(0)
    for c in classes:
        if tok.upper() == c.upper():
            return c
    if tok.isdigit() and int(tok) < len(classes):
        return classes[int(tok)]
    return None

def build_prompt(dtype, classes, nkpt):
    cl = ", ".join(f"{i}={c}" for i, c in enumerate(classes))
    if dtype == "detect":
        return (
            "You are a precise YOLO annotation engine. Silently locate EVERY object instance in the image.\n"
            f"Classes (id=name): {cl}.\n"
            "Then output ONLY YOLO lines, one per object: class_id cx cy w h\n"
            "- class_id is the integer id listed above\n"
            "- cx cy w h are floats in [0,1] normalized by image width/height, 4+ decimals\n"
            "- one TIGHT box per object, objects are typically 2-40% of image width; never output a whole-image box\n"
            "Example output:\n0 0.6465 0.7551 0.1868 0.1592\n2 0.1877 0.7654 0.1008 0.1628\n"
            "If there are no objects output exactly: NONE"
        )
    if dtype == "pose":
        nk = nkpt
        example = "0 0.5925 0.4708 0.0861 0.1444 " + " ".join(
            f"0.63{i:02d} 0.4{7+i} 2" for i in range(min(nk, 3)))
        if nk > 3:
            example += " " + " ".join(f"0.5{5+i:02d} 0.4{2+i} 2" for i in range(nk - 3))
        return (
            "You are a precise pose annotation engine. Silently locate EVERY object and its keypoints.\n"
            f"Classes (id=name): {cl}.\n"
            f"Each object has exactly {nk} keypoints in this order:\n"
            + kp_order_note(classes) +
            "\nThen output ONLY one line per object:\n"
            f"class_id cx cy w h x1 y1 v1 x2 y2 v2 ... ({nk} keypoints)\n"
            "- cx,cy,w,h: bounding box center/size normalized to [0,1]\n"
            "- xi,yi: keypoint position normalized to [0,1]\n"
            "- vi: 2 = clearly visible, 1 = occluded/at edge, 0 = not visible\n"
            f"Example line ({nk} kpts):\n{example}\n"
            "Mark keypoints precisely on the physical feature points. "
            "If there are no objects output exactly: NONE"
        )
    if dtype == "cls":
        return (
            "Classify this image into exactly one of these classes:\n" + cl + "\n"
            "Answer with ONLY the class name (or its id), nothing else."
        )
    if dtype == "labelme":
        return (
            "You are a precise annotation engine for a stationary radar/overhead camera view. "
            "Silently locate EVERY object in the image.\n"
            f"Classes (name): {cl}.\n"
            "Then output ONLY lines, one per object: class_name cx cy w h\n"
            "- class_name is the exact class name string (e.g. robotcar)\n"
            "- cx cy w h are floats in [0,1] normalized by image width/height, 4+ decimals\n"
            "- one TIGHT box per object; objects may be small\n"
            "Example output:\nrobotcar 0.5123 0.4311 0.0540 0.0380\n"
            "If there are no objects output exactly: NONE"
        )

def kp_order_note(classes):
    # generic note; armor-style 4 kpts are corner points
    return ("For armor-panel-like objects: top-left, top-right, bottom-right, bottom-left corners of the light-bar/panel.\n"
            "For buff/target objects: the 5 marked feature points in reading order (top, left, center-ish, right, bottom as visible).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--type", required=True, choices=["detect", "pose", "cls", "labelme"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=900)
    args = ap.parse_args()

    d, classes, kpts = read_meta(args.dataset)
    imgs = pick_images(d, args.n)
    outdir = f"{ROOT}/results/{TEST}/{args.dataset}/labels"
    os.makedirs(outdir, exist_ok=True)
    prompt = build_prompt(args.type, classes, kpts or 4)
    logf = open(f"{LOGDIR}/{args.dataset}.jsonl", "w")
    lock = threading.Lock()
    done = [0]

    def one(img):
        stem = os.path.splitext(img)[0]
        rec = {"dataset": args.dataset, "image": img, "ok": False, "n_lines": 0}
        try:
            b64, size = prep_image(f"{d}/images/{img}")
            t_pre = time.time()
            txt, dt, ptok, ctok = call_api(b64, prompt, max_tokens=args.max_tokens)
            rec.update(t_total=dt, prompt_tokens=ptok, completion_tokens=ctok,
                       tok_s=(ctok / dt) if dt > 0 else 0, img_px=size)
            if args.type == "detect":
                lines = parse_detect(txt, len(classes))
            elif args.type == "pose":
                lines = parse_pose(txt, len(classes), kpts or 4)
            elif args.type == "cls":
                c = parse_cls(txt, classes)
                lines = [c] if c else []
            else:
                # labelme: map class names to ids tolerant parse
                lines = parse_labelme(txt, classes)
            rec["n_lines"] = len(lines)
            rec["ok"] = True
            rec["raw_first"] = txt[:120].replace("\n", " | ")
            with open(f"{outdir}/{stem}.txt", "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
        except Exception as e:
            rec["err"] = str(e)[:200]
        with lock:
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            done[0] += 1
            print(f"[{args.dataset}] {done[0]}/{len(imgs)} {img} ok={rec['ok']} lines={rec['n_lines']} t={rec.get('t_total',0):.1f}s", flush=True)

    t0 = time.time()
    th = threading.Thread(target=gpu_sampler, daemon=True); th.start()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(one, imgs))
    wall = time.time() - t0
    gpu_stop.set()
    stats = {"dataset": args.dataset, "type": args.type, "n": len(imgs), "wall_s": wall,
             "img_per_min": len(imgs) / wall * 60, "workers": args.workers,
             "gpu_mem_max_mb": max(gpu_samples) if gpu_samples else None,
             "prompt": prompt}
    open(f"{LOGDIR}/{args.dataset}_meta.json", "w").write(json.dumps(stats, indent=2, ensure_ascii=False))
    print("META", json.dumps(stats))

def parse_labelme(txt, classes):
    lines_out = []
    name2id = {c.upper(): c for c in classes}
    for raw in txt.splitlines():
        raw = raw.strip().strip("`*")
        if not raw or raw.upper().startswith(("NONE", "NO ", "无")):
            continue
        parts = raw.split()
        nums = re.findall(r"-?\d+\.?\d*", raw)
        if len(nums) < 4:
            continue
        name = parts[0]
        for k, v in name2id.items():
            if k in name.upper():
                name = v; break
        else:
            name = classes[0]
        try:
            vals = [float(x) for x in nums[-4:]]
        except ValueError:
            continue
        cx, cy, w, h = (clamp(v) for v in vals)
        if w <= 0.001 or h <= 0.001:
            continue
        lines_out.append(f"{name} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines_out

if __name__ == "__main__":
    main()
