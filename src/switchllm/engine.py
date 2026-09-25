"""The SwitchLLM engine: route, execute with verification and escalation,
attach a cost receipt, and write the audit record."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .audit import AuditLog
from .catalog import Effort, ModelSpec, Mode, token_scale
from .classifier import TaskProfile, classify
from .directory import RoleDirectory, StaticDirectory, resolve_role
from .policy import Policy, Role
from .providers import Completion, CompletionRequest, Provider, ProviderError, build_providers
from .router import RouteDecision, route


@dataclass
class Attempt:
    model: str
    effort: str
    mode: str
    cost_usd: float
    confidence: float | None
    accepted: bool


@dataclass
class Receipt:
    cost_usd: float
    baseline_usd: float  # what the pre-SwitchLLM default would have cost

    @property
    def savings_usd(self) -> float:
        return self.baseline_usd - self.cost_usd

    @property
    def savings_pct(self) -> float:
        return 100.0 * self.savings_usd / self.baseline_usd if self.baseline_usd > 0 else 0.0


@dataclass
class RunResult:
    text: str
    served_by: str
    decision: RouteDecision
    attempts: list[Attempt]
    receipt: Receipt
    verified: bool
    shadow: bool = False
    routed_cost_usd: float | None = None  # shadow mode: what routing would have cost
    routed_text: str | None = None  # shadow mode with shadow_execute_routed
    routed_verified: bool | None = None


class SwitchLLM:
    def __init__(
        self,
        policy: Policy,
        providers: dict[str, Provider] | None = None,
        directory: RoleDirectory | None = None,
        audit: AuditLog | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        self.policy = policy
        self.providers = providers if providers is not None else build_providers(policy.providers)
        self.directory = directory or StaticDirectory(policy.users)
        self.audit = audit or AuditLog()
        self._now = clock or (lambda: datetime.now(timezone.utc))

    @classmethod
    def from_files(cls, policy_path: str | Path | None = None, audit_path: str | Path | None = None) -> "SwitchLLM":
        policy = Policy.load(policy_path) if policy_path else Policy.example()
        return cls(policy, audit=AuditLog(audit_path))

    # -- public API ----------------------------------------------------------

    def role_for(self, user: str) -> Role:
        return resolve_role(self.policy, self.directory, user)

    def month_spend(self, user: str) -> float:
        now = self._now()
        return self.audit.spend(user, now.replace(day=1, hour=0, minute=0, second=0, microsecond=0))

    def explain(self, user: str, prompt: str, urgency: str = "interactive") -> RouteDecision:
        role = self.role_for(user)
        spend = self.month_spend(user) if role.monthly_budget_usd is not None else 0.0
        return route(self.policy, role, classify(prompt), user=user, month_spend_usd=spend, urgency=urgency)

    def run(
        self,
        user: str,
        prompt: str,
        history: list[dict[str, str]] | None = None,
        urgency: str = "interactive",
    ) -> RunResult:
        decision = self.explain(user, prompt, urgency)
        messages = [*(history or []), {"role": "user", "content": prompt}]
        result = self._run_shadow(decision, messages) if self.policy.shadow else self._run_live(decision, messages)
        self._record(result, prompt)
        return result

    def spend_report(self, since: datetime | None = None) -> dict[str, Any]:
        return self.audit.report(since)

    # -- execution -------------------------------------------------------------

    def _run_live(self, decision: RouteDecision, messages: list[dict[str, str]]) -> RunResult:
        attempts, model, completion, verified = self._cascade(decision, messages)
        cost = sum(a.cost_usd for a in attempts)
        baseline = self._rescale_cost(
            completion, decision.effort, completion.executed_mode or decision.mode,
            self.policy.baseline, self.policy.baseline_effort, self.policy.baseline_mode,
        )
        return RunResult(
            text=completion.text,
            served_by=model.id,
            decision=decision,
            attempts=attempts,
            receipt=Receipt(cost, baseline),
            verified=verified,
        )

    def _run_shadow(self, decision: RouteDecision, messages: list[dict[str, str]]) -> RunResult:
        """Serve the baseline exactly as today; measure what routing would have done."""
        p = self.policy
        completion, cost = self._call(p.baseline, p.baseline_effort, p.baseline_mode, messages, decision.profile)
        attempts = [Attempt(p.baseline.id, p.baseline_effort.value, p.baseline_mode.value, cost,
                            completion.confidence, True)]
        routed_text = routed_verified = None
        if p.shadow_execute_routed:
            r_attempts, _, r_completion, routed_verified = self._cascade(decision, messages)
            routed_cost = sum(a.cost_usd for a in r_attempts)
            routed_text = r_completion.text
        else:
            routed_cost = self._rescale_cost(
                completion, p.baseline_effort, completion.executed_mode or p.baseline_mode,
                decision.model, decision.effort, decision.mode,
            )
        return RunResult(
            text=completion.text,
            served_by=p.baseline.id,
            decision=decision,
            attempts=attempts,
            receipt=Receipt(cost, cost),
            verified=True,
            shadow=True,
            routed_cost_usd=routed_cost,
            routed_text=routed_text,
            routed_verified=routed_verified,
        )

    def _cascade(
        self, decision: RouteDecision, messages: list[dict[str, str]]
    ) -> tuple[list[Attempt], ModelSpec, Completion, bool]:
        """Try the routed model; escalate up the ladder while the verifier rejects."""
        attempts: list[Attempt] = []
        for model in [decision.model, *decision.escalation]:
            completion, cost = self._call(model, decision.effort, decision.mode, messages, decision.profile)
            accepted = completion.confidence is None or completion.confidence >= self.policy.confidence_threshold
            attempts.append(Attempt(model.id, decision.effort.value, decision.mode.value, cost,
                                    completion.confidence, accepted))
            if accepted:
                return attempts, model, completion, True
        # Nothing passed verification: serve the strongest answer we got, flagged.
        return attempts, model, completion, False

    def _call(
        self, model: ModelSpec, effort: Effort, mode: Mode, messages: list[dict[str, str]], profile: TaskProfile
    ) -> tuple[Completion, float]:
        provider = self.providers.get(model.provider)
        if provider is None:
            raise ProviderError(f"no provider configured for {model.provider!r} (model {model.id!r})")
        request = CompletionRequest(
            model=model,
            messages=messages,
            effort=effort,
            mode=mode,
            hints={
                "task_class": profile.task_class,
                "difficulty": profile.difficulty,
                "base_output_tokens": profile.base_output_tokens,
            },
        )
        completion = provider.complete(request)
        cost = model.cost(completion.input_tokens, completion.output_tokens, completion.executed_mode or mode)
        return completion, cost

    @staticmethod
    def _rescale_cost(
        completion: Completion, from_effort: Effort, from_mode: Mode,
        to_model: ModelSpec, to_effort: Effort, to_mode: Mode,
    ) -> float:
        """Estimate what the same answer would have cost at another model/effort/mode."""
        out = completion.output_tokens / token_scale(from_effort, from_mode) * token_scale(to_effort, to_mode)
        return to_model.cost(completion.input_tokens, int(out), to_mode)

    def _record(self, result: RunResult, prompt: str) -> None:
        d = result.decision
        record: dict[str, Any] = {
            "ts": self._now().isoformat(),
            "user": d.user,
            "role": d.role,
            "task_class": d.profile.task_class,
            "difficulty": d.profile.difficulty,
            "routed_model": d.model.id,
            "served_by": result.served_by,
            "effort": d.effort.value,
            "mode": d.mode.value,
            "attempts": [asdict(a) for a in result.attempts],
            "escalated": len(result.attempts) > 1,
            "verified": result.verified,
            "cost_usd": result.receipt.cost_usd,
            "baseline_usd": result.receipt.baseline_usd,
            "shadow": result.shadow,
            "reasons": d.reasons,
            # Prompts are not stored; the hash supports dedupe and joins.
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }
        if result.shadow:
            record["routed_cost_usd"] = result.routed_cost_usd
            record["routed_verified"] = result.routed_verified
        self.audit.append(record)
