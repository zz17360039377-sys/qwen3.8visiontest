#!/usr/bin/env python3
"""selflabel web app -- VLM auto-annotation UI + dataset review + background jobs.

Serves static/index.html and a set of JSON endpoints. The annotation canvas
reuses the exact same modules as the CLI (vlm_client, prompts,
parse_geometry, exporters), so exports are byte-identical.

Deliberately stdlib-only (http.server.ThreadingHTTPServer). The local model
is a single-slot server (``-np 1``): VLM work is serialised -- interactive
``/annotate`` behind a lock, batch rechecks as one background job thread.

Endpoints
---------
  GET  /                     editor page (and /static/... assets)
  GET  /health               {"ok", "server", "model"}
  POST /upload               multipart (field "file") -> {"filename", "stem"}
  POST /annotate             {filename, mode, classes, keypoints, ...} -> {"items", ...}
  POST /export               {format: "labels"|"classes"|"zip", ...} -> file

  dataset review (folder-per-class datasets under --root, realpath-guarded):
  GET  /api/datasets         {"root", "datasets": [{"name","classes":[{name,count}]}], "default_extra"}
  GET  /api/images           ?ds=&cls=&start=&count= -> {"total","files":[...]}
  GET  /img                  ?ds=&cls=&name= -> image bytes
  POST /api/move             {ds, cls, names, to}
  POST /api/delete           {ds, cls, names, confirm}
  POST /api/newclass         {ds, name}
  GET  /api/recheck          ?ds= -> suggestions from the newest matching recheck csv
  POST /api/jobs             {ds, cls?, upscale?, extra?, sample?} -> {"id"} (one at a time)
  GET  /api/jobs             list of job summaries
  GET  /api/jobs/<id>        progress detail
  POST /api/jobs/<id>/cancel

Run:  python3 webapp.py [--port 8089] [--root <datasets parent dir>]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from core.prompts import build_prompt                      # noqa: E402
from core.parse_geometry import parse_to_annotations, nms  # noqa: E402
from core.vlm_client import VLMClient, VLMError            # noqa: E402
from export import exporters                               # noqa: E402
from export.fmt_yolo import annotations_to_yolo            # noqa: E402,F401
from export.fmt_yolo import write_classes                  # noqa: E402,F401

STATIC_DIR = os.path.join(_REPO, "static")
SCRIPT_DIR = _REPO
DEFAULT_API = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "qwen"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

try:  # domain hint prefilled in the review UI (optional import)
    from recheck_classify import ARMOR_HINT, classify_one, append_row, load_rows, scan_dataset
except Exception:  # noqa: BLE001 - review endpoints degrade gracefully without it
    ARMOR_HINT = ""
    classify_one = append_row = load_rows = scan_dataset = None

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}

# serialises VLM calls (single-slot local server)
_CALL_LOCK = threading.Lock()

# background recheck jobs: one at a time, state guarded by _JOBS_LOCK
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_JOB_THREAD: threading.Thread | None = None
_CANCEL = threading.Event()

REVIEW_ROOT = SCRIPT_DIR  # overridden by --root


def _make_client(p: dict) -> VLMClient:
    return VLMClient(
        api_base=str(p.get("api") or DEFAULT_API),
        model=str(p.get("model") or DEFAULT_MODEL),
        temperature=float(p.get("temperature", 0.1)),
        max_tokens=int(p.get("max_tokens", 2048)),
        thinking=bool(p.get("thinking", False)),
        max_side=int(p.get("max_side", 1280)),
    )


def _parse_keypoint_text(text: str) -> list[dict]:
    """'name=desc\\nname2=desc2' -> [{'name','description'}, ...] (one per line)."""
    out = []
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        name, desc = (ln.split("=", 1) + [""])[:2] if "=" in ln else (ln, "")
        out.append({"name": name.strip(), "description": desc.strip()})
    return out


def _multipart_file(body: bytes, ctype: str) -> tuple[str, bytes] | None:
    """Return (filename, data) of the first uploaded file part, else None."""
    m = re.search(r'boundary="?([^";]+)"?', ctype)
    if not m:
        return None
    boundary = "--" + m.group(1)
    for part in body.split(boundary.encode("latin-1")):
        part = part.strip(b"\r\n")
        if not part or b"filename=" not in part:
            continue
        head, _, data = part.partition(b"\r\n\r\n")
        hm = re.search(r'filename="([^"]*)"', head.decode("latin-1", "replace"))
        filename = os.path.basename(unquote(hm.group(1))) if hm else "upload"
        if data.endswith(b"\r\n"):
            data = data[:-2]
        return filename, data
    return None


# ------------------------------------------------------------ dataset paths

def _is_image(fn: str) -> bool:
    return os.path.splitext(fn)[1].lower() in IMG_EXTS


def _class_counts(ds_path: str) -> list[dict]:
    out = []
    for name in sorted(os.listdir(ds_path)):
        d = os.path.join(ds_path, name)
        if not os.path.isdir(d) or name.startswith((".", "_")):
            continue
        n = sum(1 for f in os.listdir(d) if _is_image(f))
        if n or any(not f.startswith((".", "_")) for f in os.listdir(d)):
            out.append({"name": name, "count": n})
    return out


def _looks_like_dataset(d: str) -> bool:
    try:
        entries = os.listdir(d)
    except OSError:
        return False
    for s in entries:
        sd = os.path.join(d, s)
        if os.path.isdir(sd) and not s.startswith((".", "_")):
            if any(_is_image(f) for f in os.listdir(sd)):
                return True
    return False


def _is_container(d: str) -> bool:
    """A dir holding >=2 dataset-like subdirs is a parent, not a dataset."""
    n = 0
    try:
        entries = os.listdir(d)
    except OSError:
        return False
    for s in entries:
        sd = os.path.join(d, s)
        if os.path.isdir(sd) and not s.startswith((".", "_")) and _looks_like_dataset(sd):
            n += 1
            if n >= 2:
                return True
    return False


def _datasets_map() -> dict[str, str]:
    """name -> abs path for every dataset-looking dir under REVIEW_ROOT."""
    out: dict[str, str] = {}
    root = os.path.realpath(REVIEW_ROOT)
    if _looks_like_dataset(root) and not _is_container(root):
        out[os.path.basename(root) or "root"] = root
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return out
    for s in entries:
        d = os.path.join(root, s)
        if (os.path.isdir(d) and not s.startswith((".", "_"))
                and _looks_like_dataset(d) and not _is_container(d)):
            out.setdefault(s, os.path.realpath(d))
    return out


def _resolve_ds(ds: str) -> str:
    m = _datasets_map()
    if ds not in m:
        raise ValueError(f"unknown dataset {ds!r} (found: {sorted(m)})")
    return m[ds]


def _safe_path(root: str, *parts: str) -> str:
    p = os.path.realpath(os.path.join(root, *parts))
    if p != root and not p.startswith(root + os.sep):
        raise ValueError("path escapes dataset root")
    return p


def _unique_dst(dst: str) -> str:
    if not os.path.exists(dst):
        return dst
    stem, ext = os.path.splitext(dst)
    for k in range(1, 200):
        cand = f"{stem}_k{k}{ext}"
        if not os.path.exists(cand):
            return cand
    raise ValueError("cannot find a free filename")


# ------------------------------------------------------------ recheck jobs

def _find_recheck_csv(ds_path: str) -> str | None:
    """Newest recheck csv whose file paths resolve under *ds_path*."""
    cands = [os.path.join(ds_path, "recheck_results.csv")]
    own = sorted(glob_dirs(SCRIPT_DIR, "recheck_*"), key=lambda p: os.path.getmtime(p), reverse=True)
    cands += [os.path.join(d, "recheck_results.csv") for d in own]
    for csv_path in cands:
        if not os.path.exists(csv_path):
            continue
        try:
            rows = load_rows(csv_path)
        except Exception:  # noqa: BLE001
            continue
        for r in rows[:5]:
            if os.path.exists(os.path.join(ds_path, r.get("file", ""))):
                return csv_path
    return None


def glob_dirs(parent: str, pattern: str) -> list[str]:
    import glob as _g

    return [d for d in _g.glob(os.path.join(parent, pattern)) if os.path.isdir(d)]


def _run_job(job_id: str, params: dict) -> None:
    job = _JOBS[job_id]
    try:
        ds_path = _resolve_ds(params["ds"])
        classes_dirnames = [c["name"] for c in _class_counts(ds_path)]
        class_dicts = [{"name": c, "description": ""} for c in classes_dirnames]
        _, items = scan_dataset(ds_path)
        if params.get("cls"):
            items = [it for it in items if it[2] == params["cls"]]
        sample = int(params.get("sample") or 0)
        if sample > 0:  # per-class cap
            from recheck_classify import stratified_sample
            items = stratified_sample(items, sample, int(params.get("seed") or 0))
        done = set()
        out_dir = os.path.join(SCRIPT_DIR, f"recheck_web_{datetime.now():%Y%m%d_%H%M%S}")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "recheck_results.csv")
        if os.path.exists(csv_path):
            done = {r["file"] for r in load_rows(csv_path)}
        job.update({"total": len(items), "csv": csv_path, "status": "running"})
        upscale = int(params.get("upscale") or 6)
        extra = str(params.get("extra") or "")
        api = str(params.get("api") or DEFAULT_API)
        model = str(params.get("model") or DEFAULT_MODEL)
        ref_images = []
        if params.get("refs"):
            from recheck_classify import load_image
            refs_dir = os.path.expanduser(str(params["refs"]))
            ref_images = [load_image(os.path.join(refs_dir, f"{c}.png"))[0]
                          for c in classes_dirnames
                          if os.path.exists(os.path.join(refs_dir, f"{c}.png"))]
        new_file = True
        for path, rel, label in items:
            if _CANCEL.is_set():
                job["status"] = "cancelled"
                break
            if rel in done:
                job["done"] += 1
                continue
            try:
                r = classify_one(api, model, path, class_dicts, upscale,
                                 ref_images, extra, 0.1, 96)
                row = {"file": rel, "folder_label": label,
                       "vlm_label": r["label"] or "",
                       "vlm_conf": f"{r['score']:.3f}" if r["label"] else "",
                       "agree": int(r["label"] == label) if r["label"] else "",
                       "wall_s": f"{r['wall']:.3f}", "parse_error": r["parse_error"],
                       "scaled_w": r["scaled_w"], "scaled_h": r["scaled_h"]}
                if not r["label"]:
                    job["errors"] += 1
            except Exception as e:  # noqa: BLE001
                row = {"file": rel, "folder_label": label, "vlm_label": "", "vlm_conf": "",
                       "agree": "", "wall_s": "", "parse_error": f"{type(e).__name__}: {e}",
                       "scaled_w": "", "scaled_h": ""}
                job["errors"] += 1
            with _CALL_LOCK:
                append_row(csv_path, row, new_file)
            new_file = False
            job["done"] += 1
            job["disagree"] += 1 if (row["agree"] == "0") else 0
        if job["status"] == "running":
            job["status"] = "done"
    except Exception as e:  # noqa: BLE001
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        job["finished"] = time.time()
        _CANCEL.clear()


def _start_job(params: dict) -> str:
    global _JOB_THREAD
    with _JOBS_LOCK:
        running = any(j["status"] in ("running", "pending")
                      for j in _JOBS.values() if j is not None)
        if running:
            raise RuntimeError("another recheck job is running (cancel it first)")
        job_id = datetime.now().strftime("%H%M%S") + f"_{len(_JOBS) + 1}"
        _JOBS[job_id] = {"id": job_id, "status": "pending", "ds": params.get("ds", ""),
                         "cls": params.get("cls") or "", "done": 0, "total": 0,
                         "errors": 0, "disagree": 0, "started": time.time(),
                         "finished": None, "csv": ""}
        _CANCEL.clear()
        t = threading.Thread(target=_run_job, args=(job_id, dict(params)), daemon=True)
        _JOB_THREAD = t
        t.start()
        return job_id


# ------------------------------------------------------------------ handler

class Handler(BaseHTTPRequestHandler):
    server_version = "selflabel/1.1"
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------- helpers
    def _send_bytes(self, code: int, data: bytes, ctype: str) -> None:
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True  # client went away (lazy-img abort etc.)

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _send_json(self, obj, code: int = 200) -> None:
        self._send_bytes(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                         "application/json; charset=utf-8")

    def _send_text(self, text: str, code: int, filename: str | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(text.encode("utf-8"))))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(text.encode("utf-8"))

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except (ValueError, UnicodeDecodeError):
            return {}

    def log_message(self, *args) -> None:  # keep the console quiet
        pass

    def _serve_file(self, path: str) -> None:
        if not os.path.isfile(path):
            self._send_json({"error": "not found"}, 404)
            return
        ctype = _CONTENT_TYPES.get(os.path.splitext(path)[1].lower(), "application/octet-stream")
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if path.endswith((".html", ".js", ".css")):
            self.send_header("Cache-Control", "no-cache")  # dev-friendly: always revalidate
        self.end_headers()
        self.wfile.write(data)

    # --------------------------------------------------------------- routes
    def do_GET(self) -> None:
        url = urlparse(self.path)
        path = unquote(url.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if path in ("/", "/index.html"):
                self._serve_file(os.path.join(STATIC_DIR, "index.html"))
            elif path.startswith("/static/"):
                self._serve_file(os.path.join(STATIC_DIR, path[len("/static/"):]))
            elif path == "/health":
                c = _make_client({})
                self._send_json({"ok": c.ping(), "server": c.base_url, "model": c.model})
            elif path == "/api/datasets":
                self._api_datasets()
            elif path == "/api/images":
                self._api_images(q)
            elif path == "/img":
                self._img(q)
            elif path == "/api/recheck":
                self._api_recheck(q)
            elif path == "/api/jobs":
                self._send_json({"jobs": [dict(j) for j in _JOBS.values()]})
            elif path.startswith("/api/jobs/") and path.endswith("/cancel"):
                jid = path[len("/api/jobs/"):-len("/cancel")]
                if _CANCEL.is_set() or not _JOBS.get(jid):
                    self._send_json({"error": "no such job or already cancelled"}, 404)
                    return
                _CANCEL.set()
                self._send_json({"ok": True})
            elif path.startswith("/api/jobs/"):
                jid = path[len("/api/jobs/"):]
                if jid not in _JOBS:
                    self._send_json({"error": "no such job"}, 404)
                    return
                self._send_json(_JOBS[jid])
            else:
                self._send_json({"error": "not found"}, 404)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001
            self._send_json({"error": f"unexpected error: {e}"}, 500)

    def do_POST(self) -> None:
        path = unquote(self.path.split("?", 1)[0])
        try:
            if path == "/upload":
                self._upload()
            elif path == "/annotate":
                self._annotate()
            elif path == "/export":
                self._export()
            elif path == "/api/move":
                self._api_move()
            elif path == "/api/delete":
                self._api_delete()
            elif path == "/api/newclass":
                self._api_newclass()
            elif path == "/api/jobs":
                self._api_jobs_start()
            else:
                self._send_json({"error": "not found"}, 404)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001
            self._send_json({"error": f"unexpected error: {e}"}, 500)

    # ------------------------------------------------------------ annotate
    def _upload(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        got = _multipart_file(body, self.headers.get("Content-Type", ""))
        if not got:
            self._send_json({"error": "no file part in upload (field 'file')"}, 400)
            return
        filename, data = got
        if not data:
            self._send_json({"error": "uploaded file is empty"}, 400)
            return
        with open(os.path.join(STATIC_DIR, filename), "wb") as f:
            f.write(data)
        stem = os.path.splitext(filename)[0]
        self._send_json({"filename": filename, "stem": stem, "ok": True})

    def _annotate(self) -> None:
        p = self._read_json()
        filename = str(p.get("filename") or "")
        mode = str(p.get("mode") or "detect").lower()
        if mode not in ("detect", "pose"):
            self._send_json({"error": "mode must be 'detect' or 'pose'"}, 400)
            return
        path = os.path.join(STATIC_DIR, os.path.basename(filename))
        if not filename or not os.path.exists(path):
            self._send_json({"error": f"image not found on server: {filename!r} (upload it first)"}, 404)
            return

        classes = _coerce_classes(p.get("classes"))
        if not classes:
            self._send_json({"error": "at least one class is required"}, 400)
            return
        keypoints = _coerce_keypoints(p.get("keypoints"))
        if mode == "pose" and not keypoints:
            n = int(p.get("kpt", 4) or 4)
            keypoints = [{"name": f"kpt{i}", "description": f"keypoint {i + 1}"} for i in range(n)]

        client = _make_client(p)
        with _CALL_LOCK:  # the local model handles one request at a time
            try:
                data_url, w, h = client.resize_path(path)
                system, user = build_prompt(
                    mode, classes, keypoints, width=w, height=h, extra=str(p.get("prompt") or "")
                )
                content = client.send(data_url, system, user)
                n_kpt = len(keypoints) if mode == "pose" else 0
                anns = parse_to_annotations(content, w, h, mode, classes, n_kpt)
                anns = nms(
                    anns,
                    iou_thresh=float(p.get("iou", 0.5)),
                    max_det=int(p.get("max_det", 0) or 0),
                    conf=float(p.get("conf", 0.0)),
                )
            except VLMError as e:
                self._send_json({"error": f"VLM error: {e}"}, 400)
                return
            except Exception as e:  # noqa: BLE001 - surface anything unexpected to the UI
                self._send_json({"error": f"unexpected error: {e}"}, 500)
                return
        self._send_json({
            "items": anns,
            "n": len(anns),
            "w": w,
            "h": h,
            "classes": classes,
            "keypoints": keypoints,
            "mode": mode,
        })

    def _export(self) -> None:
        p = self._read_json()
        fmt = str(p.get("format") or "labels")
        if fmt == "classes":
            classes = [str(c.get("name", "")).strip() for c in (p.get("classes") or [])]
            text = "\n".join(classes) + "\n" if classes else ""
            self._send_text(text, 200, "classes.txt")
        elif fmt == "labels":
            mode = str(p.get("mode") or "detect").lower()
            if mode not in ("detect", "pose"):
                self._send_json({"error": "mode must be 'detect' or 'pose'"}, 400)
                return
            items = p.get("items") or []
            text = annotations_to_yolo(items, mode)
            self._send_text(text, 200, f"{os.path.splitext(p.get('filename') or 'labels')[0]}_{mode}.txt")
        elif fmt == "zip":
            self._export_zip(p)
        else:
            self._send_json({"error": "format must be 'labels', 'classes' or 'zip'"}, 400)

    def _export_zip(self, p: dict) -> None:
        from PIL import Image

        fmt = str(p.get("fmt") or "yolo").lower()
        classes = _coerce_classes(p.get("classes")) or [{"name": "object", "description": ""}]
        keypoints = _coerce_keypoints(p.get("keypoints"))
        skeleton = [list(map(int, e)) for e in (p.get("skeleton") or [])
                    if isinstance(e, (list, tuple)) and len(e) == 2]
        records = []
        for entry in p.get("images") or []:
            fname = os.path.basename(str(entry.get("filename") or ""))
            path = os.path.join(STATIC_DIR, fname)
            if not fname or not os.path.exists(path):
                continue
            with Image.open(path) as im:
                w, h = im.size
            anns = entry.get("items") or []
            records.append({"path": path, "width": w, "height": h, "anns": anns})
        if not records:
            self._send_json({"error": "no annotated images to export"}, 400)
            return
        tmp = tempfile.mkdtemp(prefix="selflabel_export_")
        try:
            manifest = exporters.write_dataset(fmt, tmp, records, classes, keypoints, skeleton)
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for root, _, files in os.walk(tmp):
                    for fn in files:
                        fp = os.path.join(root, fn)
                        z.write(fp, os.path.relpath(fp, tmp))
            self._send_bytes(200, buf.getvalue(), "application/zip")
            print(f"[export] {fmt}: {manifest}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ------------------------------------------------------------- review
    def _api_datasets(self) -> None:
        ds = [{"name": name, "classes": _class_counts(path)}
              for name, path in _datasets_map().items()]
        self._send_json({"root": REVIEW_ROOT, "datasets": ds, "default_extra": ARMOR_HINT})

    def _api_images(self, q: dict) -> None:
        ds = _resolve_ds(q.get("ds", ""))
        cls = q.get("cls") or ""
        d = _safe_path(ds, cls) if cls else ds
        if not os.path.isdir(d):
            self._send_json({"error": "no such class dir"}, 404)
            return
        names = sorted(f for f in os.listdir(d) if _is_image(f))
        start = max(0, int(q.get("start") or 0))
        count = min(2000, max(1, int(q.get("count") or 500)))
        self._send_json({"total": len(names), "files": names[start:start + count]})

    def _img(self, q: dict) -> None:
        ds = _resolve_ds(q.get("ds", ""))
        p = _safe_path(ds, q.get("cls") or "", q.get("name") or "")
        if not os.path.isfile(p) or not _is_image(p):
            self._send_json({"error": "no such image"}, 404)
            return
        with open(p, "rb") as f:
            data = f.read()
        try:
            self.send_response(200)
            self.send_header("Content-Type", _CONTENT_TYPES.get(os.path.splitext(p)[1].lower(),
                                                                "application/octet-stream"))
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=60")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _api_move(self) -> None:
        p = self._read_json()
        ds = _resolve_ds(str(p.get("ds") or ""))
        cls = str(p.get("cls") or "")
        to = str(p.get("to") or "")
        names = [str(n) for n in (p.get("names") or [])]
        if not to or not names:
            raise ValueError("fields 'to' and 'names' are required")
        _validate_name(to)
        dst_dir = _safe_path(ds, to)
        os.makedirs(dst_dir, exist_ok=True)
        moved = []
        for name in names:
            src = _safe_path(ds, cls, name)
            if not os.path.isfile(src):
                continue
            dst = _unique_dst(_safe_path(ds, to, os.path.basename(name)))
            shutil.move(src, dst)
            moved.append(os.path.basename(dst))
        self._send_json({"moved": len(moved), "files": moved})

    def _api_delete(self) -> None:
        p = self._read_json()
        ds = _resolve_ds(str(p.get("ds") or ""))
        cls = str(p.get("cls") or "")
        names = [str(n) for n in (p.get("names") or [])]
        if not p.get("confirm"):
            self._send_json({"error": "delete requires confirm:true (double confirm in UI)"}, 400)
            return
        deleted = 0
        for name in names:
            f = _safe_path(ds, cls, name)
            if os.path.isfile(f):
                os.remove(f)
                deleted += 1
        self._send_json({"deleted": deleted})

    def _api_newclass(self) -> None:
        p = self._read_json()
        ds = _resolve_ds(str(p.get("ds") or ""))
        name = str(p.get("name") or "")
        _validate_name(name)
        d = _safe_path(ds, name)
        os.makedirs(d, exist_ok=True)
        self._send_json({"ok": True, "name": name})

    def _api_recheck(self, q: dict) -> None:
        ds_name = q.get("ds") or ""
        ds = _resolve_ds(ds_name)
        csv_path = _find_recheck_csv(ds) if load_rows else None
        if not csv_path:
            self._send_json({"found": False, "ds": ds_name})
            return
        suggestions = {}
        for r in load_rows(csv_path):
            if r.get("vlm_label"):
                try:
                    conf = float(r.get("vlm_conf") or 0)
                except ValueError:
                    conf = 0.0
                suggestions[r["file"]] = {"label": r["vlm_label"], "conf": conf,
                                          "agree": r.get("agree") == "1"}
        self._send_json({"found": True, "ds": ds_name, "csv": csv_path,
                         "suggestions": suggestions})

    def _api_jobs_start(self) -> None:
        if classify_one is None:
            self._send_json({"error": "recheck_classify.py not importable"}, 500)
            return
        p = self._read_json()
        if not p.get("ds"):
            raise ValueError("field 'ds' is required")
        try:
            job_id = _start_job(p)
        except RuntimeError as e:
            self._send_json({"error": str(e)}, 409)
            return
        self._send_json({"id": job_id})


def _coerce_classes(raw) -> list[dict]:
    return [
        {"name": str(c.get("name") or f"class{i}").strip(),
         "description": str(c.get("description") or "").strip()}
        for i, c in enumerate(raw or [])
        if isinstance(c, dict) and str(c.get("name") or "").strip()
    ]


def _coerce_keypoints(raw) -> list[dict]:
    return [
        {"name": str(k.get("name") or f"kpt{i}").strip(),
         "description": str(k.get("description") or "").strip()}
        for i, k in enumerate(raw or [])
        if isinstance(k, dict) and str(k.get("name") or "").strip()
    ]


def _validate_name(name: str) -> None:
    if not name or name.startswith((".", "_")) or re.search(r"[\\/:*?\"<>|]", name):
        raise ValueError(f"invalid class name: {name!r}")
    if len(name) > 80:
        raise ValueError("class name too long")


def main(argv=None) -> int:
    global REVIEW_ROOT
    ap = argparse.ArgumentParser(description="selflabel web app (stdlib http.server).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--root", default=SCRIPT_DIR,
                    help="parent dir containing folder-per-class datasets (review tab)")
    args = ap.parse_args(argv)
    REVIEW_ROOT = os.path.abspath(os.path.expanduser(args.root))
    if not os.path.isdir(STATIC_DIR):
        raise SystemExit(f"static dir not found: {STATIC_DIR}")
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[selflabel web] http://{args.host}:{args.port}  (model {DEFAULT_API})")
    print(f"[selflabel web] review root: {REVIEW_ROOT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
