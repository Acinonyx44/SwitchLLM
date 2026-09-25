from datetime import datetime, timezone

import pytest

from switchllm import Policy, SwitchLLM
from switchllm.audit import AuditLog
from switchllm.providers import Completion, CompletionRequest

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)


class ScriptedProvider:
    """Accepts answers only from models at or above pass_tier."""

    def __init__(self, pass_tier: int):
        self.pass_tier = pass_tier
        self.calls: list[CompletionRequest] = []

    def complete(self, req: CompletionRequest) -> Completion:
        self.calls.append(req)
        ok = req.model.tier >= self.pass_tier
        return Completion(f"answer from {req.model.id}", 100, 200, confidence=0.9 if ok else 0.2)


@pytest.fixture
def policy() -> Policy:
    return Policy.example()


@pytest.fixture
def engine(policy) -> SwitchLLM:
    return SwitchLLM(policy, audit=AuditLog(), clock=lambda: NOW)
