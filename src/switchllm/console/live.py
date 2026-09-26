"""Live model calls through OpenRouter (or any OpenAI-compatible endpoint)."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from ..providers import OpenAICompatibleProvider


class LiveClient:
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1", timeout: float = 120.0):
        self._http = OpenAICompatibleProvider(base_url, api_key, max_tokens_param="max_tokens", timeout=timeout)

    def chat(self, model_id: str, messages: list[dict[str, str]], effort: str = "low",
             max_tokens: int = 2048) -> tuple[str, int, int, int]:
        """Returns (text, input tokens, output tokens, latency in ms)."""
        body: dict[str, Any] = {"model": model_id, "messages": messages, "max_tokens": max_tokens,
                                "reasoning": {"effort": effort}}
        started = time.monotonic()
        data = self._http._post("/chat/completions", body)
        latency = int((time.monotonic() - started) * 1000)
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        usage = data.get("usage") or {}
        return text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)), latency


def parse_json(text: str) -> dict[str, Any]:
    """First JSON object in a model reply (models like to wrap it in prose or fences)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    return json.loads(m.group(0))
