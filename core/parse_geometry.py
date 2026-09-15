"""Robust parsing of the VLM's grounding answer into clean geometry.

The model (even under ``json_object`` mode) is only *mostly* well-behaved, so this
module is deliberately defensive:

  * strips markdown fences (```json ... ```) and any surrounding prose;
  * finds and loads a JSON object or array, tolerating stray text around it;
  * auto-detects the coordinate space: if ANY coordinate exceeds 1.5 the answer
    is in PIXELS (divide x by image width, y by image height); otherwise it is
    already normalised to [0, 1];
  * clamps everything to [0, 1], enforces x1<x2 / y1<y2, drops degenerate boxes;
  * maps the free-text label back to a class id;
  * fills missing pose keypoints as [0,0,0] (absent) so the YOLO field count
    stays correct.

Plus a greedy NMS for merging duplicate boxes.
"""

from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _load_json(text: str):
    """Return the first parseable JSON object/array found in *text*, else None."""
    text = _strip_fences(text)
    if not text:
        return None
    # 1) the whole thing parses
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2) grab the outermost {...} or [...]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            chunk = text[start : end + 1]
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                continue
    # 3) last resort: pull out any top-level object bodies like {...} in order
    objs = re.findall(r"\{[^{}]*\}", text)
    for o in objs:
        try:
            return json.loads(o)
        except json.JSONDecodeError:
            continue
    return None


def load_json(text: str):
    """Public alias of :func:`_load_json` for classify/recheck callers."""
    return _load_json(text)


def _first(d: dict, keys: list[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _to_bbox(raw) -> tuple[float, float, float, float] | None:
    """Normalise a bbox given as [x1,y1,x2,y2] or [cx,cy,w,h] to (x1,y1,x2,y2)."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        raw = [raw.get(k) for k in ("x1", "y1", "x2", "y2")]
        if any(v is None for v in raw):
            raw = [raw.get(k) for k in ("cx", "cy", "w", "h")]
            if any(v is None for v in raw):
                return None
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        a = [float(v) for v in raw[:4]]
    except (TypeError, ValueError):
        return None
    return tuple(a)  # type: ignore[return-value]


def _looks_like_pixel(coords, width: int, height: int) -> bool:
    for c in coords:
        try:
            if abs(float(c)) > 1.5:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _match_class(label, classes: list[dict]) -> tuple[int, str]:
    """Map a free-text label to (class_id, canonical_name). Falls back to id 0."""
    if isinstance(label, (int, float)):
        i = int(label)
        if 0 <= i < len(classes):
            return i, str(classes[i]["name"])
        return 0, str(classes[0]["name"])
    s = str(label or "").strip().lower()
    if not s:
        return 0, str(classes[0]["name"])
    # exact
    for i, c in enumerate(classes):
        if str(c["name"]).strip().lower() == s:
            return i, str(c["name"])
    # substring either way
    for i, c in enumerate(classes):
        cn = str(c["name"]).strip().lower()
        if cn in s or s in cn:
            return i, str(c["name"])
    return 0, str(label)


def _all_coords(bbox, kps) -> list:
    vals = list(bbox) if bbox else []
    for kp in kps or []:
        try:
            vals.extend([kp[0], kp[1]])
        except Exception:
            pass
    return vals


def parse_to_annotations(
    content: str,
    width: int,
    height: int,
    mode: str,
    classes: list[dict],
    n_kpt: int,
) -> list[dict]:
    """Parse raw model output into a list of annotation dicts.

    Each annotation: ``{class_id, label, x1, y1, x2, y2 (normalised),
    keypoints: [[x,y,v] x n_kpt] or None, score}``.
    """
    data = _load_json(content)
    raws = []
    if isinstance(data, list):
        raws = data
    elif isinstance(data, dict):
        for key in ("results", "detections", "objects", "items", "annos", "annotations"):
            if isinstance(data.get(key), list):
                raws = data[key]
                break
        else:
            # single object that itself looks like a detection
            if _first(data, ("bbox", "box")) is not None:
                raws = [data]
    if not raws:
        return []

    coords = []
    for r in raws:
        if not isinstance(r, dict):
            continue
        bb = _to_bbox(_first(r, ("bbox", "box", "bboxes", "bbox_2d")))
        kps = _first(r, ("keypoints", "kps", "landmarks"))
        if bb is None:
            continue
        coords.append(bb)
        if kps:
            coords.extend(kps if isinstance(kps, list) else [kps])
    flat = [v for c in coords if isinstance(c, (list, tuple)) for v in c]
    pixel = _looks_like_pixel(flat, width, height)

    out = []
    for r in raws:
        if not isinstance(r, dict):
            continue
        bb = _to_bbox(_first(r, ("bbox", "box", "bboxes", "bbox_2d")))
        if bb is None:
            continue
        x1, y1, x2, y2 = bb
        if pixel:
            x1, x2 = x1 / width, x2 / width
            y1, y2 = y1 / height, y2 / height
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        x1, y1, x2, y2 = _clamp01(x1), _clamp01(y1), _clamp01(x2), _clamp01(y2)
        if (x2 - x1) <= 1e-4 or (y2 - y1) <= 1e-4:
            continue  # degenerate box

        class_id, label = _match_class(_first(r, ("label", "name", "class")), classes)

        kps_raw = _first(r, ("keypoints", "kps", "landmarks")) if mode == "pose" else None
        kps = None
        if mode == "pose":
            kps = []
            src = kps_raw if isinstance(kps_raw, list) else []
            for i in range(n_kpt):
                kp = src[i] if i < len(src) else None
                if isinstance(kp, (list, tuple)) and len(kp) >= 2:
                    kx, ky = float(kp[0]), float(kp[1])
                    v = int(kp[2]) if len(kp) > 2 and kp[2] is not None else 1
                else:
                    kx, ky, v = 0.0, 0.0, 0
                if pixel:
                    kx, ky = kx / width, ky / height
                kps.append([_clamp01(kx), _clamp01(ky), max(0, min(2, v))])

        try:
            score = float(_first(r, ("score", "conf", "confidence"), 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0

        out.append(
            {
                "class_id": class_id,
                "label": label,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "keypoints": kps,
                "score": score,
            }
        )
    return out


# ------------------------------------------------------------------------ NMS


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(
    anns: list[dict],
    iou_thresh: float = 0.5,
    max_det: int = 0,
    conf: float = 0.0,
) -> list[dict]:
    """Conf filter + greedy NMS + optional count cap. Returns a new list."""
    kept = [a for a in anns if a["score"] >= conf]

    def box(a):
        return (a["x1"], a["y1"], a["x2"], a["y2"])

    # score desc, tie-break by box area desc (scores are often 0 / unreliable)
    kept.sort(key=lambda a: (a["score"], (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])), reverse=True)
    boxes = [box(a) for a in kept]
    indices = list(range(len(kept)))
    picked = []
    while indices:
        i = indices.pop(0)
        picked.append(i)
        remaining = []
        for j in indices:
            if _iou(boxes[i], boxes[j]) < iou_thresh:
                remaining.append(j)
        indices = remaining
    result = [kept[i] for i in picked]
    if max_det and len(result) > max_det:
        result = result[:max_det]
    return result
