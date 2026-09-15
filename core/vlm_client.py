"""OpenAI-compatible client for the local Qwen VLM served by llama.cpp.

The server exposes ``/v1/chat/completions``. Flow for one image:

  1. ``resize_bytes`` / ``resize_path`` -- downscale the image to a fixed max
     side (so the coordinate space the model sees is always the same size) and
     return a base64 data URL **plus the resulting width/height**.
  2. build the prompt using that exact width/height (see ``prompts.py``);
  3. ``send`` -- post image + prompt with ``response_format: json_object``.

Thinking/reasoning is OFF by default (``thinking=False``): it is ~2.3-30x faster
with equal grounding quality. Pass ``thinking=True`` / CLI ``--thinking`` to
enable it. Only ``message.content`` is returned -- never ``reasoning_content``.
"""

from __future__ import annotations

import base64
import io
import os
import time

import requests


class VLMError(RuntimeError):
    """Raised when the model call fails or returns no usable answer."""


def chat_raw(
    api_base: str,
    model: str,
    messages: list,
    thinking: bool = False,
    temperature: float = 0.1,
    max_tokens: int = 512,
    timeout: int = 600,
    json_mode: bool = True,
) -> dict:
    """One low-level ``/chat/completions`` call (multi-image capable).

    Returns ``{content, finish, timings, usage, wall}`` -- ``timings`` is
    llama.cpp's throughput block (prompt_n/prompt_ms/predicted_*/ *_per_second),
    ``wall`` is client-side wall-clock seconds. Raises ``requests`` exceptions
    on transport errors; empty content is returned as-is for the caller to
    decide (bench/recheck measure parse failures instead of hiding them).
    """
    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    t0 = time.perf_counter()
    r = requests.post(api_base.rstrip("/") + "/chat/completions", json=body, timeout=timeout)
    wall = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    return {
        "content": (choice["message"].get("content") or "").strip(),
        "finish": choice.get("finish_reason"),
        "timings": data.get("timings") or {},
        "usage": data.get("usage") or {},
        "wall": wall,
    }


class VLMClient:
    def __init__(
        self,
        api_base: str = "http://127.0.0.1:8080/v1",
        model: str = "qwen",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        thinking: bool = False,
        timeout: int = 600,
        max_side: int = 1280,
        api_key: str | None = None,
    ) -> None:
        self.url = api_base.rstrip("/") + "/chat/completions"
        self.base_url = api_base.rstrip("/").rsplit("/v1", 1)[0]
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.timeout = timeout
        self.max_side = max_side
        self.api_key = api_key or os.environ.get("VLM_API_KEY", "")

    # ------------------------------------------------------------------ image

    def _resize_bytes(self, data: bytes) -> tuple[str, int, int]:
        from PIL import Image

        im = Image.open(io.BytesIO(data))
        if im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, self.max_side / max(w, h))
        if scale < 1.0:
            im = im.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88)
        data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        return data_url, im.size[0], im.size[1]

    def resize_path(self, path: str) -> tuple[str, int, int]:
        with open(path, "rb") as f:
            return self._resize_bytes(f.read())

    def resize_bytes(self, data: bytes) -> tuple[str, int, int]:
        return self._resize_bytes(data)

    # -------------------------------------------------------------- transport

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = "Bearer " + self.api_key
        return h

    def _post(self, body: dict) -> str:
        r = requests.post(self.url, json=body, headers=self._headers(), timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        finish = data["choices"][0].get("finish_reason")
        if not content.strip():
            raise VLMError(f"empty content (finish_reason={finish})")
        return content.strip()

    def send(self, data_url: str, system: str, user: str) -> str:
        """Post image + prompt; return the model's ``content`` string.

        Retries once with a bigger token budget if the answer comes back empty
        (which happens when a reasoning trace eats the whole budget).
        """
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": user},
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": bool(self.thinking)},
        }

        last_err: Exception | None = None
        for attempt in range(2):
            try:
                return self._post(body)
            except VLMError as e:
                last_err = e
                if attempt == 0:
                    body["max_tokens"] = max(self.max_tokens, 4096)
                    time.sleep(0.4)
            except requests.RequestException as e:
                last_err = e
                if attempt == 0:
                    time.sleep(1.0)
        raise VLMError(f"VLM call failed after retries: {last_err}")

    def ping(self) -> bool:
        try:
            r = requests.get(self.base_url, timeout=5)
            return r.status_code < 500
        except requests.RequestException:
            return False
