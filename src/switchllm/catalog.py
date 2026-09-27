"""Model catalog, effort levels, execution modes and the cost model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Effort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return _EFFORT_ORDER.index(self)

    @staticmethod
    def lowest(*efforts: "Effort") -> "Effort":
        return min(efforts, key=lambda e: e.rank)


_EFFORT_ORDER = [Effort.LOW, Effort.MEDIUM, Effort.HIGH]


class Mode(str, Enum):
    SINGLE = "single"
    EXTENDED_THINKING = "extended_thinking"
    AGENTIC = "agentic"
    BATCH = "batch"


# The same model at high effort emits many more (reasoning) tokens than at low
# effort -- this is the 5-10x cost spread nobody governs today.
EFFORT_TOKEN_MULTIPLIER = {Effort.LOW: 1.0, Effort.MEDIUM: 2.5, Effort.HIGH: 6.0}
MODE_TOKEN_MULTIPLIER = {
    Mode.SINGLE: 1.0,
    Mode.EXTENDED_THINKING: 2.0,
    Mode.AGENTIC: 4.0,  # several model calls per task
    Mode.BATCH: 1.0,
}
MODE_PRICE_MULTIPLIER = {
    Mode.SINGLE: 1.0,
    Mode.EXTENDED_THINKING: 1.0,
    Mode.AGENTIC: 1.0,
    Mode.BATCH: 0.5,  # vendors discount asynchronous batch work
}


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider: str
    tier: int  # 1 = cheapest, 3 = frontier
    input_per_mtok: float
    output_per_mtok: float
    name: str | None = None  # provider-side model name; defaults to id
    supports_effort: bool = False  # accepts a reasoning-effort parameter

    @property
    def api_name(self) -> str:
        return self.name or self.id

    def cost(self, input_tokens: int, output_tokens: int, mode: Mode = Mode.SINGLE) -> float:
        usd = (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1_000_000
        return usd * MODE_PRICE_MULTIPLIER[mode]


def token_scale(effort: Effort, mode: Mode) -> float:
    return EFFORT_TOKEN_MULTIPLIER[effort] * MODE_TOKEN_MULTIPLIER[mode]


def estimate_output_tokens(base: int, effort: Effort, mode: Mode) -> int:
    return int(base * token_scale(effort, mode))


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)
