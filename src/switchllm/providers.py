"""Model providers. Customers keep their own vendor contracts and keys."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from .catalog import Effort, ModelSpec, Mode, approx_tokens, estimate_output_tokens


class ProviderError(RuntimeError):
    pass


@dataclass
class CompletionRequest:
    model: ModelSpec
    messages: list[dict[str, str]]
    effort: Effort
    mode: Mode
    max_output_tokens: int = 4096
    hints: dict[str, Any] = field(default_factory=dict)  # router context, e.g. difficulty


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    confidence: float | None = None  # None = no verifier signal; treated as accepted
    executed_mode: Mode | None = None  # set when the provider ran a different mode than asked


class Provider(Protocol):
    def complete(self, req: CompletionRequest) -> Completion: ...


class MockProvider:
    """Keyless, deterministic provider for offline development and benchmarks.

    A tier-t model handles tasks below CAPACITY[t] difficulty, fails above it,
    and fails one in BOUNDARY_FAILURE_EVERY tasks right at it (chosen by
    prompt hash, so runs are reproducible). Failures report low confidence,
    which exercises escalation.
    """

    CAPACITY = {1: 2, 2: 4, 3: 5}
    BOUNDARY_FAILURE_EVERY = 4

    def complete(self, req: CompletionRequest) -> Completion:
        difficulty = int(req.hints.get("difficulty", 3))
        capacity = self.CAPACITY.get(req.model.tier, 5)
        prompt = req.messages[-1]["content"]
        ok = difficulty < capacity or (
            difficulty == capacity
            and (capacity == 5 or int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % self.BOUNDARY_FAILURE_EVERY)
        )
        snippet = prompt if len(prompt) <= 60 else prompt[:57] + "..."
        return Completion(
            text=f"[{req.model.id} | {req.effort.value} effort | {req.mode.value}] answer to: {snippet}",
            input_tokens=sum(approx_tokens(m["content"]) for m in req.messages),
            output_tokens=estimate_output_tokens(int(req.hints.get("base_output_tokens", 300)), req.effort, req.mode),
            confidence=0.9 if ok else 0.35,
        )


class OpenAICompatibleProvider:
    """Any OpenAI-compatible chat-completions endpoint: OpenAI itself, or the
    gateway a company already runs (LiteLLM, Portkey, OpenRouter, vLLM, ...).

    Batch and agentic modes run as a single synchronous call for now and are
    reported (and billed) as such via Completion.executed_mode.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        max_tokens_param: str = "max_completion_tokens",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_tokens_param = max_tokens_param
        self.timeout = timeout

    def complete(self, req: CompletionRequest) -> Completion:
        body: dict[str, Any] = {
            "model": req.model.api_name,
            "messages": req.messages,
            self.max_tokens_param: req.max_output_tokens,
        }
        if req.model.supports_effort:
            body["reasoning_effort"] = req.effort.value
        data = self._post("/chat/completions", body)
        try:
            text = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage", {})
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"unexpected response shape: {e}") from e
        executed = req.mode if req.mode is Mode.EXTENDED_THINKING else Mode.SINGLE
        return Completion(
            text=text,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            executed_mode=executed,
        )

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + path, data=json.dumps(body).encode(), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise ProviderError(f"HTTP {e.code} from {self.base_url}: {e.read()[:500]!r}") from e
        except urllib.error.URLError as e:
            raise ProviderError(f"cannot reach {self.base_url}: {e.reason}") from e


def build_providers(configs: dict[str, Any]) -> dict[str, Provider]:
    """Instantiate providers from a policy's [providers.*] tables."""
    providers: dict[str, Provider] = {"mock": MockProvider()}
    for name, cfg in configs.items():
        if cfg.type == "mock":
            providers[name] = MockProvider()
        elif cfg.type == "openai_compatible":
            if not cfg.base_url:
                raise ValueError(f"provider {name!r} needs base_url")
            key = os.environ.get(cfg.api_key_env) if cfg.api_key_env else None
            providers[name] = OpenAICompatibleProvider(cfg.base_url, key, max_tokens_param=cfg.max_tokens_param)
        else:
            raise ValueError(f"provider {name!r}: unknown type {cfg.type!r}")
    return providers
