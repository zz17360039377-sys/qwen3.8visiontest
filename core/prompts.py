"""Prompt builders for detect / pose annotation.

Design (per task, nothing hard-coded to any dataset):
  * ``classes`` is a list of ``{"name": str, "description": str}``. The
    *description* is what tells the model what to look for -- each class name is
    paired with its description in the prompt.
  * For pose, ``keypoints`` is a list of ``{"name": str, "description": str}``.
    The keypoint COUNT ``N`` is simply ``len(keypoints)`` and is decided per
    task. Each keypoint carries its own distinct semantic description, because
    otherwise the model collapses keypoints onto the box corners.
  * The image's pixel size (W x H, after downscale) is stated up front and the
    model is asked for INTEGER PIXEL coordinates -- this anchors the coordinate
    space reliably. Normalisation to YOLO happens later in the parser.
"""

from __future__ import annotations

from typing import Optional

SYSTEM_DETECT = (
    "You are a precise object detector. You look at the image and locate every "
    "requested object. You respond with a single valid JSON object and nothing "
    "else - no prose, no markdown, no code fences."
)

SYSTEM_POSE = (
    "You are a precise object pose / landmark annotator. You look at the image "
    "and, for every requested object, locate its bounding box AND each of its "
    "named keypoints. You respond with a single valid JSON object and nothing "
    "else - no prose, no markdown, no code fences."
)

SYSTEM_CLASSIFY = (
    "You are a strict single-label image classifier. You look at the image and "
    "assign exactly one of the listed classes. You respond with a single valid "
    "JSON object and nothing else - no prose, no markdown, no code fences."
)


def _class_block(classes: list[dict]) -> str:
    lines = []
    for i, c in enumerate(classes):
        name = str(c.get("name", "")).strip()
        desc = str(c.get("description", "")).strip()
        lines.append(f'  id {i}: "{name}" -- {desc}' if desc else f'  id {i}: "{name}"')
    return "\n".join(lines)


def _keypoint_block(keypoints: list[dict]) -> str:
    lines = []
    for i, k in enumerate(keypoints):
        name = str(k.get("name", "")).strip()
        desc = str(k.get("description", "")).strip()
        lines.append(f'  kpt {i}: "{name}" -- {desc}' if desc else f'  kpt {i}: "{name}"')
    return "\n".join(lines)


def build_prompt(
    mode: str,
    classes: list[dict],
    keypoints: Optional[list[dict]] = None,
    width: int = 0,
    height: int = 0,
    extra: str = "",
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for one image of size width x height."""
    mode = mode.lower()
    if mode not in ("detect", "pose"):
        raise ValueError(f"mode must be 'detect' or 'pose', got {mode!r}")
    if not classes:
        raise ValueError("at least one class is required")

    cls_block = _class_block(classes)
    size = (
        f"The image is {width} x {height} pixels (width x height, top-left origin). "
        f"Report every coordinate as INTEGER PIXELS: x in [0, {width}], y in [0, {height}]."
    )
    head = "Locate every instance of the following classes in the image:\n" + cls_block
    if extra and extra.strip():
        head += "\n\nAdditional guidance: " + extra.strip()

    if mode == "detect":
        user = (
            head
            + "\n\nFor EVERY instance output one object: "
            '{"label": "<class name exactly as listed>", "bbox": [x1, y1, x2, y2], '
            '"score": <number 0..1>}.\n'
            + size
            + " The bbox is [left, top, right, bottom] with left<right and top<bottom.\n"
            'Return {"results": []} when no instance is present.\n'
            'Respond with ONLY the JSON object {"results": [ ... ]}.'
        )
        return SYSTEM_DETECT, user

    if not keypoints:
        raise ValueError("pose mode requires a non-empty keypoints list")
    n = len(keypoints)
    kp_block = _keypoint_block(keypoints)
    user = (
        head
        + "\n\nEach instance ALSO has these "
        + str(n)
        + " keypoints, IN THIS EXACT ORDER (index i below is kpt i):\n"
        + kp_block
        + "\nPlace each keypoint at the EXACT position of its own named feature - "
        "do NOT put keypoints on the box corners.\n"
        "Each keypoint is [x, y, v]: x,y are the pixel position of THAT point, and "
        "v is visibility (2 = clearly visible, 1 = partly occluded, 0 = not visible).\n"
        + size
        + "\nFor EVERY instance output one object: "
        '{"label": "<class name exactly as listed>", "bbox": [x1, y1, x2, y2], '
        '"keypoints": [[x, y, v], ... ' + str(n) + " entries in order], "
        '"score": <number 0..1>}.\n'
        'Return {"results": []} when no instance is present.\n'
        'Respond with ONLY the JSON object {"results": [ ... ]}.'
    )
    return SYSTEM_POSE, user


def build_classify_prompt(
    classes: list[dict],
    extra: str = "",
    n_reference_images: int = 0,
) -> tuple[str, str]:
    """Return (system, user) for single-label classification of one image.

    With ``n_reference_images`` > 0 the caller prepends that many reference
    images to the message (in class-list order); the user prompt then explains
    that layout and states the LAST image is the sample to classify.
    ``extra`` is free-text domain guidance inserted before the task line.
    """
    if not classes:
        raise ValueError("at least one class is required")
    block = _class_block(classes)
    user = (
        "Classify the image into exactly ONE of these classes:\n" + block
        + "\n\nOutput the single most likely class -- one label per image, no hedging.\n"
        'Respond with ONLY the JSON object '
        '{"label": "<class name exactly as listed>", "score": <number 0..1>}.'
    )
    if extra and extra.strip():
        user = extra.strip() + "\n\n" + user
    if n_reference_images:
        user = (
            f"The message contains {n_reference_images + 1} images: the first "
            f"{n_reference_images} are REFERENCE patterns shown in the same order as "
            "the class list above, and the LAST image is the sample to classify.\n\n"
            + user
        )
    return SYSTEM_CLASSIFY, user
