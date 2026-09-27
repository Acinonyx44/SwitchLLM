from dataclasses import replace

import pytest

from switchllm import SwitchLLM
from switchllm.audit import AuditLog
from switchllm.providers import MockProvider

from .conftest import NOW, ScriptedProvider


def test_role_resolution(engine):
    assert engine.role_for("alice@acme.com").name == "quant_research"
    assert engine.role_for("ALICE@acme.com").name == "quant_research"
    assert engine.role_for("bob@acme.com").name == "sales"
    assert engine.role_for("stranger@acme.com").name == "general"


def test_live_run_writes_receipt_and_audit(engine):
    result = engine.run("bob@acme.com", "Quick tl;dr of this thread")
    assert result.served_by == "mock-small"
    assert result.verified
    assert 0 < result.receipt.cost_usd < result.receipt.baseline_usd
    (record,) = engine.audit.records()
    assert record["role"] == "sales" and record["task_class"] == "summarize"
    assert record["cost_usd"] == pytest.approx(result.receipt.cost_usd)
    assert "prompt" not in record and len(record["prompt_sha256"]) == 64


def test_escalates_until_verifier_accepts(policy):
    provider = ScriptedProvider(pass_tier=2)
    engine = SwitchLLM(policy, providers={"mock": provider}, audit=AuditLog(), clock=lambda: NOW)
    result = engine.run("alice@acme.com", "hello there")
    assert [a.model for a in result.attempts] == ["mock-small", "mock-mid"]
    assert [a.accepted for a in result.attempts] == [False, True]
    assert result.served_by == "mock-mid" and result.verified
    assert result.receipt.cost_usd == pytest.approx(sum(a.cost_usd for a in result.attempts))
    assert engine.audit.records()[0]["escalated"]


def test_unverified_answer_is_flagged(policy):
    provider = ScriptedProvider(pass_tier=4)  # nothing passes
    engine = SwitchLLM(policy, providers={"mock": provider}, audit=AuditLog(), clock=lambda: NOW)
    result = engine.run("bob@acme.com", "hello there")  # sales: ladder stops at tier 2
    assert [a.model for a in result.attempts] == ["mock-small", "mock-mid"]
    assert not result.verified
    assert engine.spend_report()["totals"]["unverified"] == 1


def test_budget_accumulates_from_audit_log(policy):
    policy = replace(policy, roles={**policy.roles, "sales": replace(policy.roles["sales"], monthly_budget_usd=0.01)})
    engine = SwitchLLM(policy, audit=AuditLog(), clock=lambda: NOW)
    prompt = "Refactor this python function"
    first = engine.run("bob@acme.com", prompt)
    assert first.decision.model.tier == 2
    assert engine.month_spend("bob@acme.com") >= 0.01
    second = engine.explain("bob@acme.com", prompt)
    assert any("budget exhausted" in r for r in second.reasons)
    assert second.model.tier == 2  # code floor holds even when broke


def test_shadow_mode_serves_baseline_and_projects_savings(policy):
    engine = SwitchLLM(replace(policy, shadow=True), audit=AuditLog(), clock=lambda: NOW)
    result = engine.run("bob@acme.com", "Quick tl;dr of this thread")
    assert result.shadow and result.served_by == policy.baseline_model
    assert result.receipt.savings_usd == 0
    assert result.routed_cost_usd < result.receipt.cost_usd
    totals = engine.spend_report()["totals"]
    assert totals["savings_usd"] == 0
    assert totals["shadow_projected_savings_usd"] > 0


def test_shadow_can_execute_routed_path(policy):
    engine = SwitchLLM(replace(policy, shadow=True, shadow_execute_routed=True), audit=AuditLog(), clock=lambda: NOW)
    result = engine.run("bob@acme.com", "Quick tl;dr of this thread")
    assert result.routed_text and "mock-small" in result.routed_text
    assert result.routed_verified is True


def test_history_is_sent_to_provider(policy):
    provider = ScriptedProvider(pass_tier=1)
    engine = SwitchLLM(policy, providers={"mock": provider}, audit=AuditLog(), clock=lambda: NOW)
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    engine.run("bob@acme.com", "and now?", history=history)
    assert provider.calls[0].messages == [*history, {"role": "user", "content": "and now?"}]


def test_mock_provider_is_deterministic(engine):
    a = engine.run("alice@acme.com", "Compare the trade-offs of these two strategies")
    b = engine.run("alice@acme.com", "Compare the trade-offs of these two strategies")
    assert [x.model for x in a.attempts] == [x.model for x in b.attempts]
    assert a.receipt.cost_usd == b.receipt.cost_usd


def test_mock_boundary_failures_trigger_escalation():
    from switchllm.catalog import Effort, ModelSpec, Mode
    from switchllm.providers import CompletionRequest

    small = ModelSpec("s", "mock", 1, 1, 1)
    outcomes = {
        MockProvider().complete(CompletionRequest(small, [{"role": "user", "content": f"task {i}"}],
                                                  Effort.LOW, Mode.SINGLE, hints={"difficulty": 2})).confidence
        for i in range(20)
    }
    assert outcomes == {0.9, 0.35}
