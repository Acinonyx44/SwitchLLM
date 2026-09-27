"""Turn (role, task profile, spend so far) into a routing decision.

Order of precedence, each step recorded as a human-readable reason:
  1. difficulty picks a starting tier and effort
  2. the task class's quality floor can raise the tier
  3. the role's caps can lower tier and effort -- never below the floor
  4. the role's monthly budget, paced by run rate, throttles effort, then tier
  5. urgency and task shape pick the execution mode
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .catalog import Effort, ModelSpec, Mode, estimate_output_tokens
from .classifier import TaskProfile
from .policy import Policy, PolicyError, Role

_TIER_FOR_DIFFICULTY = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}
_EFFORT_FOR_DIFFICULTY = {1: Effort.LOW, 2: Effort.LOW, 3: Effort.MEDIUM, 4: Effort.HIGH, 5: Effort.HIGH}
# Projected month-end spend / budget at which routing steps down.
PACE_EFFORT_AT, PACE_TIER_AT, PACE_FLOOR_AT = 0.9, 1.0, 1.2


@dataclass
class RouteDecision:
    user: str
    role: str
    profile: TaskProfile
    model: ModelSpec
    effort: Effort
    mode: Mode
    escalation: list[ModelSpec]
    est_cost_usd: float
    est_baseline_usd: float
    reasons: list[str] = field(default_factory=list)

    @property
    def est_savings_pct(self) -> float:
        if self.est_baseline_usd <= 0:
            return 0.0
        return 100.0 * (1 - self.est_cost_usd / self.est_baseline_usd)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user": self.user,
            "role": self.role,
            "task_class": self.profile.task_class,
            "difficulty": self.profile.difficulty,
            "model": self.model.id,
            "effort": self.effort.value,
            "mode": self.mode.value,
            "escalation": [m.id for m in self.escalation],
            "est_cost_usd": round(self.est_cost_usd, 6),
            "est_baseline_usd": round(self.est_baseline_usd, 6),
            "est_savings_pct": round(self.est_savings_pct, 1),
            "reasons": self.reasons,
            "signals": self.profile.signals,
        }


def route(
    policy: Policy,
    role: Role,
    profile: TaskProfile,
    *,
    user: str,
    month_spend_usd: float = 0.0,
    month_progress: float = 1.0,
    urgency: str = "interactive",
) -> RouteDecision:
    reasons: list[str] = []
    d = profile.difficulty

    tier = _TIER_FOR_DIFFICULTY[d]
    effort = _EFFORT_FOR_DIFFICULTY[d]
    reasons.append(f"{profile.task_class} task, difficulty {d}/5 -> tier {tier}, {effort.value} effort")

    floor = policy.rule_for(profile.task_class).min_tier
    if floor > tier:
        tier = floor
        reasons.append(f"'{profile.task_class}' has a tier-{floor} quality floor")

    if tier > role.max_tier:
        tier = max(role.max_tier, floor)
        if floor > role.max_tier:
            reasons.append(f"quality floor (tier {floor}) overrides role '{role.name}' cap (tier {role.max_tier})")
        else:
            reasons.append(f"role '{role.name}' is capped at tier {role.max_tier}")
    ceiling = max(role.max_tier, floor)

    if effort.rank > role.max_effort.rank:
        effort = role.max_effort
        reasons.append(f"role '{role.name}' is capped at {effort.value} effort")

    if role.monthly_budget_usd is not None:
        budget = role.monthly_budget_usd
        used = month_spend_usd / budget if budget > 0 else 1.0
        # Pace by run rate: project month-end spend and step down early.
        projected = month_spend_usd / max(month_progress, 1e-6)
        pace = projected / budget if budget > 0 else float("inf")
        on_pace = f"on pace for ${projected:.2f} of ${budget:.2f}"
        if used >= 1.0:
            tier, ceiling, effort = floor, floor, Effort.LOW
            reasons.append(f"monthly budget exhausted (${month_spend_usd:.2f}) -> floor tier, low effort")
        elif pace >= PACE_FLOOR_AT:
            tier, ceiling, effort = floor, floor, Effort.LOW
            reasons.append(f"{on_pace} -> floor tier, low effort")
        elif pace >= PACE_TIER_AT:
            ceiling = max(floor, ceiling - 1)
            tier, effort = max(floor, min(tier, ceiling)), Effort.LOW
            reasons.append(f"{on_pace} -> low effort, top tier held back")
        elif pace >= PACE_EFFORT_AT and effort.rank > Effort.MEDIUM.rank:
            effort = Effort.MEDIUM
            reasons.append(f"{on_pace} -> effort throttled to medium")

    mode = _pick_mode(role, profile, effort, urgency)
    if mode is not Mode.SINGLE:
        reasons.append(f"execution mode: {mode.value}")

    model = _pick_model(policy, tier, role)
    if model.tier != tier:
        reasons.append(f"no permitted tier-{tier} model; using tier {model.tier}")
    escalation = _escalation_ladder(policy, model.tier, ceiling, role)

    out = estimate_output_tokens(profile.base_output_tokens, effort, mode)
    est = model.cost(profile.input_tokens, out, mode)
    base_out = estimate_output_tokens(profile.base_output_tokens, policy.baseline_effort, policy.baseline_mode)
    est_base = policy.baseline.cost(profile.input_tokens, base_out, policy.baseline_mode)

    return RouteDecision(
        user=user,
        role=role.name,
        profile=profile,
        model=model,
        effort=effort,
        mode=mode,
        escalation=escalation,
        est_cost_usd=est,
        est_baseline_usd=est_base,
        reasons=reasons,
    )


def _pick_mode(role: Role, profile: TaskProfile, effort: Effort, urgency: str) -> Mode:
    allowed = role.allowed_modes
    if urgency == "batch" and Mode.BATCH in allowed:
        return Mode.BATCH
    if profile.needs_tools and Mode.AGENTIC in allowed:
        return Mode.AGENTIC
    if profile.difficulty >= 4 and effort is Effort.HIGH and Mode.EXTENDED_THINKING in allowed:
        return Mode.EXTENDED_THINKING
    return Mode.SINGLE


def _pick_model(policy: Policy, tier: int, role: Role) -> ModelSpec:
    # Prefer the requested tier, then the nearest tier above, then below.
    for t in [tier, *range(tier + 1, 4), *range(tier - 1, 0, -1)]:
        m = policy.cheapest_at_tier(t, role.providers)
        if m is not None:
            return m
    raise PolicyError(f"role '{role.name}' has no permitted models")


def _escalation_ladder(policy: Policy, start_tier: int, ceiling: int, role: Role) -> list[ModelSpec]:
    ladder = []
    for t in range(start_tier + 1, ceiling + 1):
        m = policy.cheapest_at_tier(t, role.providers)
        if m is not None:
            ladder.append(m)
    return ladder
