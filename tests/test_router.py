import pytest

from switchllm import Effort, Mode, PolicyError, classify, route
from switchllm.classifier import TaskProfile


def profile(task_class="chat", difficulty=1, needs_tools=False):
    return TaskProfile(task_class, difficulty, needs_tools, input_tokens=100, base_output_tokens=300)


def test_easy_task_goes_to_cheapest_tier(policy):
    d = route(policy, policy.roles["quant_research"], profile(difficulty=1), user="a")
    assert d.model.tier == 1 and d.effort is Effort.LOW and d.mode is Mode.SINGLE
    assert [m.tier for m in d.escalation] == [2, 3]
    assert d.est_cost_usd < d.est_baseline_usd


def test_hard_task_gets_frontier_and_thinking_for_permitted_role(policy):
    d = route(policy, policy.roles["quant_research"], profile("math", 5), user="a")
    assert d.model.tier == 3 and d.effort is Effort.HIGH and d.mode is Mode.EXTENDED_THINKING
    assert d.escalation == []


def test_role_caps_tier_and_effort(policy):
    d = route(policy, policy.roles["sales"], profile("math", 5), user="b")
    assert d.model.tier == 2 and d.effort is Effort.MEDIUM and d.mode is Mode.SINGLE
    assert any("capped at tier 2" in r for r in d.reasons)


def test_quality_floor_overrides_role_cap(policy):
    d = route(policy, policy.roles["sales"], profile("legal", 1), user="b")
    assert d.model.tier == 3
    assert any("overrides" in r for r in d.reasons)


def test_quality_floor_raises_easy_task(policy):
    d = route(policy, policy.roles["engineering"], profile("code", 1), user="c")
    assert d.model.tier == 2


def test_budget_throttles_then_drops_to_floor(policy):
    role = policy.roles["quant_research"]  # $500 budget
    hard = profile("analysis", 5)
    assert route(policy, role, hard, user="a", month_spend_usd=0).effort is Effort.HIGH
    throttled = route(policy, role, hard, user="a", month_spend_usd=450)
    assert throttled.effort is Effort.MEDIUM and throttled.model.tier == 3
    exhausted = route(policy, role, hard, user="a", month_spend_usd=500)
    assert exhausted.model.tier == 1 and exhausted.effort is Effort.LOW and exhausted.escalation == []


def test_exhausted_budget_still_respects_quality_floor(policy):
    d = route(policy, policy.roles["sales"], profile("legal", 3), user="b", month_spend_usd=1000)
    assert d.model.tier == 3


def test_modes(policy):
    quant, sales = policy.roles["quant_research"], policy.roles["sales"]
    assert route(policy, quant, profile(needs_tools=True), user="a").mode is Mode.AGENTIC
    assert route(policy, sales, profile(needs_tools=True), user="b").mode is Mode.SINGLE  # not allowed
    assert route(policy, sales, profile(), user="b", urgency="batch").mode is Mode.BATCH


def test_batch_is_cheaper(policy):
    role = policy.roles["general"]
    p = classify("Summarize these meeting notes")
    assert route(policy, role, p, user="x", urgency="batch").est_cost_usd < route(policy, role, p, user="x").est_cost_usd


def test_provider_allow_list(policy):
    from dataclasses import replace

    role = replace(policy.roles["general"], providers=frozenset({"nobody"}))
    with pytest.raises(PolicyError, match="no permitted models"):
        route(policy, role, profile(), user="x")
