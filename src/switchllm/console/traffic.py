"""Simulated company traffic, replayed through a policy to price it.

The traffic (who asked what, how hard, how long) is generated once per
seed; costing it under any policy is a pure function, so a policy preview can
replay 30 days of requests in well under a second.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from .policy import TIERS, TYPICAL_DIFFICULTY, budget_for, clamp_tier, resolve, tier_index

HEADCOUNT = {"support_agent": 60, "engineer": 50, "analyst": 30, "contractor": 25,
             "default": 15, "legal": 10, "executive": 10}
REQUESTS_PER_PERSON_PER_DAY = {"support_agent": 12, "engineer": 14, "analyst": 11, "contractor": 5,
                               "default": 5, "legal": 8, "executive": 6}

# Task mix per role: (task class, weight).
MIX: dict[str, list[tuple[str, float]]] = {
    "support_agent": [("customer_reply", 55), ("customer_escalation", 8), ("summarize", 12), ("draft_email", 10),
                      ("qa_factual", 8), ("quantitative_reasoning", 4), ("legal_clause_analysis", 1),
                      ("financial_analysis", 1), ("translate", 1)],
    "engineer": [("code_generate", 30), ("code_debug", 20), ("code_review", 10), ("code_explain", 10),
                 ("architecture_design", 4), ("summarize", 6), ("draft_document", 5), ("qa_factual", 8),
                 ("data_extraction", 4), ("general_chat", 3)],
    "analyst": [("quantitative_reasoning", 22), ("data_analysis", 15), ("financial_analysis", 12),
                ("research_synthesis", 12), ("summarize", 15), ("draft_document", 8), ("code_generate", 6),
                ("data_extraction", 6), ("strategic_analysis", 4)],
    "contractor": [("draft_email", 25), ("code_generate", 20), ("summarize", 20), ("qa_factual", 15),
                   ("rewrite_edit", 10), ("financial_analysis", 3), ("code_review", 4), ("legal_clause_analysis", 3)],
    "default": [("draft_email", 25), ("summarize", 25), ("qa_factual", 20), ("rewrite_edit", 10),
                ("brainstorm", 10), ("general_chat", 10)],
    "legal": [("legal_clause_analysis", 35), ("compliance_check", 20), ("summarize", 20), ("draft_document", 15),
              ("draft_email", 10)],
    "executive": [("strategic_analysis", 20), ("summarize", 30), ("draft_email", 20), ("financial_analysis", 10),
                  ("draft_document", 10), ("brainstorm", 10)],
}

UTILITY_IN, UTILITY_OUT = 900, 28  # classifier call
EFFORT_OUT = {"low": 1.0, "medium": 1.6, "high": 2.5}
THINKING_OUT = 1.8


@dataclass(frozen=True)
class Request:
    day: int  # index into the window
    role: str
    task_class: str
    difficulty: str
    needs: str  # weakest tier that answers it correctly
    in_tokens: int
    out_tokens: int  # at low effort, single call


def generate(days: list[date], seed: int = 7) -> list[Request]:
    rng = random.Random(seed)
    reqs: list[Request] = []
    for di, _ in enumerate(days):
        for role, heads in HEADCOUNT.items():
            classes, weights = zip(*MIX[role])
            n = int(heads * REQUESTS_PER_PERSON_PER_DAY[role] * rng.uniform(0.85, 1.15))
            for _ in range(n):
                tc = rng.choices(classes, weights)[0]
                base = tier_index(TYPICAL_DIFFICULTY.get(tc, "medium"))
                d = max(0, min(2, base + rng.choices([-1, 0, 1], [20, 65, 15])[0]))
                # Most tasks are answerable at their difficulty tier; a few need one more.
                needs = min(2, d + (1 if rng.random() < 0.06 else 0))
                reqs.append(Request(di, role, tc, TIERS[d], TIERS[needs],
                                    int(rng.lognormvariate(6.0, 0.6)), int(rng.lognormvariate(5.6, 0.5))))
    return reqs


def price(pool: dict[str, Any], model_tier: str, tin: int, tout: int) -> float:
    m = next(x for x in pool["models"] if x["tier"] == model_tier)
    return (tin * m["price_in"] + tout * m["price_out"]) / 1e6


def baseline_price(pool: dict[str, Any], tin: int, tout: int) -> float:
    m = next(x for x in pool["models"] if x["name"] == pool["baseline"])
    return (tin * m["price_in"] + tout * m["price_out"]) / 1e6


def cost_request(policy: dict, pool: dict, r: Request, cache: dict) -> dict[str, Any]:
    """Route one simulated request; returns tier used, cost, baseline and flags."""
    key = (r.role, r.task_class)
    cell = cache.get(key)
    if cell is None:
        cell = cache[key] = resolve(policy, r.role, r.task_class)
    cost = price(pool, "small", UTILITY_IN, UTILITY_OUT)
    if cell["blocked"]:
        return {"blocked": True, "tier": None, "cost": cost, "baseline": 0.0, "escalations": 0}
    out = r.out_tokens * EFFORT_OUT[cell["effort"]] * (THINKING_OUT if cell["execution"] == "thinking" else 1.0)
    out = int(out)
    tier = clamp_tier(r.difficulty, cell["floor"], cell["ceiling"])
    escalations = 0
    while True:
        cost += price(pool, tier, r.in_tokens, out)
        if not cell["verify"]:
            break
        cost += price(pool, "small", r.in_tokens + out, 20)  # verifier
        if tier_index(tier) >= tier_index(r.needs) or tier == cell["ceiling"]:
            break
        tier = TIERS[tier_index(tier) + 1]
        escalations += 1
    return {"blocked": False, "tier": tier, "cost": cost, "baseline": baseline_price(pool, r.in_tokens, out),
            "escalations": escalations}


def replay(policy: dict, pool: dict, reqs: list[Request]) -> list[dict[str, Any]]:
    cache: dict = {}
    return [cost_request(policy, pool, r, cache) for r in reqs]


def window(today: date, days: int = 30) -> list[date]:
    start = today - timedelta(days=days - 1)
    return [start + timedelta(i) for i in range(days) if (start + timedelta(i)).weekday() < 5]


def role_budgets(policy: dict) -> dict[str, float]:
    return {role: budget_for(policy, role) for role in HEADCOUNT}
