"""Dataset exporters -- one writer module per on-disk format.

Every writer consumes the same in-memory shape and only decides how it is
serialised, so adding a format means writing one small module and registering
it (one line in ``REGISTRY`` below):

  records : ``[{path, width, height, anns: [ann]}]``
  ann     : ``{label, class_id, x1, y1, x2, y2 (normalised 0..1),
               keypoints: [[x, y, v] x N] or None, score}``
  classes : ``[{name, description}]``
  keypoints/skeleton : pose metadata (skeleton edges as index pairs)

A writer has the signature::

    write(out_dir, records, classes, keypoints, skeleton, copy_images) -> manifest

``manifest`` is a small dict describing what was written (counts + paths).
``pose`` vs ``detect`` is inferred from whether ``keypoints`` is non-empty.
"""

from __future__ import annotations

import os
from typing import Callable

REGISTRY: dict[str, Callable] = {}


def register(name: str) -> Callable:
    def deco(fn: Callable) -> Callable:
        REGISTRY[name] = fn
        return fn
    return deco


def list_formats() -> list[str]:
    return sorted(REGISTRY)


def write_dataset(
    fmt: str,
    out_dir: str,
    records: list[dict],
    classes: list[dict],
    keypoints: list[dict] | None = None,
    skeleton: list | None = None,
    copy_images: bool = True,
) -> dict:
    """Dispatch to the format writer registered under *fmt*."""
    fmt = fmt.lower()
    if fmt not in REGISTRY:
        raise ValueError(f"unknown format {fmt!r} (have: {list_formats()})")
    os.makedirs(out_dir, exist_ok=True)
    return REGISTRY[fmt](out_dir, records, classes, keypoints or [],
                         skeleton or [], copy_images)


# registration (kept at the bottom to avoid import cycles: the fmt modules
# never import this module back -- we call their ``write`` from here)
from . import fmt_coco as _coco_mod  # noqa: E402
from . import fmt_yolo as _yolo_mod  # noqa: E402

register("yolo")(_yolo_mod.write)
register("coco")(_coco_mod.write)