import json
import threading
import urllib.request
from datetime import date
from http.cookiejar import CookieJar

import pytest

from switchllm.console.engine import ConsoleEngine
from switchllm.console.nlpolicy import parse
from switchllm.console.policy import DEFAULT_POLICY, resolve
from switchllm.console.server import ConsoleServer, page_html

TODAY = date(2026, 9, 25)


@pytest.fixture(scope="module")
def engine_factory():
    return lambda **kw: ConsoleEngine(today=TODAY, **kw)


@pytest.fixture
def engine(engine_factory):
    return engine_factory()


def test_resolution_layers():
    # Task-class quality floor beats the role cap; role effort overrides.
    cell = resolve(DEFAULT_POLICY, "support_agent", "code_review")
    assert (cell["floor"], cell["ceiling"], cell["effort"]) == ("large", "large", "low")
    assert resolve(DEFAULT_POLICY, "contractor", "financial_analysis")["blocked"]
    legal = resolve(DEFAULT_POLICY, "legal", "summarize")
    assert legal["floor"] == "medium" and legal["effort"] == "high"
    assert resolve(DEFAULT_POLICY, "support_agent", "customer_escalation")["ceiling"] == "large"


def test_every_scenario_does_what_it_says(engine):
    by_id = {}
    for s in engine.scenarios:
        by_id[s["id"]] = engine.ask(s["persona"], s["task"], compare=True)
    refund = by_id["prorated-refund"]["receipt"]
    assert [a["tier"] for a in refund["attempts"]] == ["small", "medium"]
    assert refund["attempts"][0]["passed"] is False and refund["attempts"][1]["passed"] is True
    assert by_id["blocked-valuation"]["status"] == "blocked"
    assert by_id["valuation"]["receipt"]["final_tier"] == "large"
    assert by_id["policy-summary"]["receipt"]["final_tier"] == "medium"  # legal's floor lifts it
    assert by_id["double-charge"]["receipt"]["cost_usd"] < by_id["double-charge"]["receipt"]["baseline_cost_usd"]
    assert all("compare" in r for r in by_id.values() if r["status"] == "ok")


def test_free_text_is_classified_and_routed(engine):
    r = engine.ask("dana", "Write a Python function that parses ISO dates, with unit tests.")
    assert r["classification"]["task_class"] == "code_generate"
    assert r["receipt"]["final_tier"] in ("medium", "large")  # engineers' code floor
    assert "Simulated answer" in r["answer"]


def test_nl_policy_parser():
    p = parse("Cap contractors at $3 a month and block them from code review")
    assert p["patch"]["roles"]["contractor"]["budget_usd_month"] == 3.0
    assert "code_review" in p["patch"]["roles"]["contractor"]["denied_task_classes"]
    p = parse("Code reviews can start on the medium model and escalate if needed")
    assert p["patch"] == {"task_classes": {"code_review": {"floor": "medium"}}} and not p["unparsed"]
    p = parse("Analysts always get the frontier model with extended thinking for quantitative reasoning")
    assert p["patch"]["roles"]["analyst"]["rules"]["quantitative_reasoning"] == {"floor": "large", "execution": "thinking"}
    assert not parse("Give everyone a pony")["ok"]


def test_preview_apply_and_dashboard(engine):
    before = engine.dashboard()["kpis"]["cost"]
    pv = engine.preview("Analysts always get the frontier model with extended thinking for quantitative reasoning")
    assert pv["ok"] and pv["projection"]["delta"] > 0 and pv["diff"][0] == "--- policy.yaml"
    out = engine.apply("Analysts always get the frontier model with extended thinking for quantitative reasoning")
    assert out["version"] == 2 and out["dashboard"]["kpis"]["cost"] > before
    with pytest.raises(ValueError):
        engine.apply("Give everyone a pony")


def test_budget_exhaustion_blocks_requests(engine):
    engine.apply("Cap contractors at $0.01 a month")
    r = engine.ask("sam", "Draft a short email asking the vendor to resend the Q3 invoice with our PO number.")
    assert r["status"] == "budget"


def test_export_and_offline_page(engine):
    data = engine.export()
    assert set(data["ladders"]) == {"hybrid", "open"}
    runs = data["ladders"]["hybrid"]["runs"]
    assert len(runs) == 2 * len(engine.scenarios)
    html = page_html(data)
    assert "window.SWITCHLLM_STATIC=" in html and html.count("<script>") == 2


class FakeLive:
    """Stands in for OpenRouter: classifier, worker and verifier replies."""

    def __init__(self):
        self.calls = []

    def chat(self, model_id, messages, effort="low", max_tokens=2048):
        self.calls.append(model_id)
        text = messages[-1]["content"]
        if text.startswith("Classify"):
            return '{"task_class": "quantitative_reasoning", "difficulty": "small", "confidence": 0.9}', 900, 30, 300
        if text.startswith("You are a strict reviewer"):
            ok = "120b" in self.calls[-2]
            return json.dumps({"pass": ok, "score": 90 if ok else 40, "issues": "" if ok else "wrong"}), 400, 20, 200
        return f"answer from {model_id}", 300, 80, 900


def test_live_mode_uses_real_calls_and_escalates(engine_factory):
    fake = FakeLive()
    e = engine_factory(live_client=fake)
    r = e.ask("priya", "What's 17% of $240?")
    assert r["answer_kind"] == "live" and r["answer"] == "answer from openai/gpt-oss-120b"
    assert [a["tier"] for a in r["receipt"]["attempts"]] == ["small", "medium"]
    assert r["classification"]["source"] == "model"


def test_live_failure_falls_back_to_simulation(engine_factory):
    class Broken:
        def chat(self, *a, **k):
            raise ConnectionError("no network")

    r = engine_factory(live_client=Broken()).ask("priya", "Reply to this customer: thanks for the update!")
    assert r["answer_kind"] == "fallback" and "no network" in r["fallback_reason"]
    assert r["status"] == "ok"


def test_server_sessions_are_isolated(engine_factory):
    server = ConsoleServer(("127.0.0.1", 0), engine_factory)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def client():
        return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))

    def call(opener, path, body=None):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        return json.loads(opener.open(req).read())

    try:
        a, b = client(), client()
        assert call(a, "/api/health")["ok"]
        assert call(a, "/api/meta")["version"] == 1
        assert call(a, "/api/policy/apply", {"instruction": "Engineers only need medium effort for code generation"})["version"] == 2
        assert call(a, "/api/policy")["version"] == 2
        assert call(b, "/api/policy")["version"] == 1  # another viewer is unaffected
        r = call(b, "/api/ask", {"persona": "priya", "task": "Reply to this customer: where is my order?", "compare": False})
        assert r["status"] == "ok"
        rid = r["receipt"]["id"]
        assert call(b, "/api/feedback", {"receipt_id": rid, "liked": False})["option"]["tier"] == "medium"
        up = call(b, "/api/override", {"receipt_id": rid, "reason": "too vague"})["run"]["receipt"]["id"]
        pending = call(b, "/api/override", {"receipt_id": up, "reason": "VIP"})
        assert pending["status"] == "pending" and call(a, "/api/approvals") == []  # per-viewer queue
        assert call(b, "/api/approvals/decide", {"id": pending["approval"]["id"], "approve": True})["run"]["status"] == "ok"
        with pytest.raises(urllib.error.HTTPError) as err:
            call(b, "/api/ask", {"persona": "nobody", "task": "hi"})
        assert err.value.code == 400
        assert b"<title>SwitchLLM Console</title>" in a.open(base + "/").read()
    finally:
        server.shutdown()
        server.server_close()


def test_pacing_by_run_rate(engine):
    from switchllm.console.pacing import apply, pace

    p = pace(spent=55, cap=100, today=date(2026, 9, 15))  # on pace for $110
    assert p["action"] == "tier" and p["projected"] == 110.0
    cell = resolve(DEFAULT_POLICY, "engineer", "code_generate")
    paced = apply(cell, p)
    assert paced["effort"] == "low" and paced["ceiling"] == "medium" and paced["layers"][-1] == "budget.pacing"
    assert apply(resolve(DEFAULT_POLICY, "legal", "legal_clause_analysis"), pace(90, 100, date(2026, 9, 5)))["floor"] == "large"

    engine.apply("Cap engineers at $85 a month")
    r = engine.ask("dana", "Write a Python function merge_intervals(intervals) that merges overlapping intervals, with a few tests.")
    assert r["pacing"]["action"] != "none" and "budget.pacing" in r["receipt"]["applied"]


def test_feedback_self_serve_rerun_then_manager_approval(engine):
    r = engine.ask("priya", "Reply to this customer: where is my order?")
    assert r["receipt"]["final_tier"] == "small"
    fb = engine.feedback(r["receipt"]["id"], liked=False)
    assert fb["option"]["tier"] == "medium" and not fb["option"]["needs_approval"]
    rerun = engine.request_override(r["receipt"]["id"], "too generic")
    assert rerun["status"] == "done" and rerun["run"]["receipt"]["final_tier"] == "medium"
    assert rerun["run"]["receipt"]["override"]["kind"] == "self"

    # Support is capped at medium for replies, so frontier needs a manager.
    second = rerun["run"]["receipt"]["id"]
    assert engine.feedback(second, liked=False)["option"]["needs_approval"]
    pending = engine.request_override(second, "VIP customer")
    assert pending["status"] == "pending"
    with pytest.raises(ValueError):
        engine.request_override(second, "again")  # one open request per answer
    assert engine.approvals_view()[0]["status"] == "pending"
    out = engine.decide(pending["approval"]["id"], approve=True, approver="Support lead")
    assert out["run"]["receipt"]["final_tier"] == "large"
    assert out["run"]["receipt"]["override"] == {"kind": "manager", "by": "Support lead", "reason": "VIP customer",
                                                 "from": second, "approval": pending["approval"]["id"]}
    with pytest.raises(ValueError):
        engine.decide(pending["approval"]["id"], approve=True)
    kpis = engine.dashboard()["kpis"]
    assert kpis["disliked"] == 2 and kpis["overrides"] == 2 and kpis["pending_approvals"] == 0


def test_denied_override_and_top_tier(engine):
    r = engine.ask("leo", "Explain the risk to us in this clause: \"Vendor's total liability shall not exceed the fees paid by Customer in the twelve (12) months preceding the claim.\"")
    fb = engine.feedback(r["receipt"]["id"], liked=False)
    assert fb["option"] is None and "strongest" in fb["message"]
    assert engine.feedback(r["receipt"]["id"], liked=True)["option"] is None
    r = engine.ask("sam", "Draft a short email asking the vendor to resend the Q3 invoice with our PO number.")
    second = engine.request_override(engine.request_override(r["receipt"]["id"])["run"]["receipt"]["id"])
    assert second["status"] == "pending"  # contractors are capped at medium
    assert engine.decide(second["approval"]["id"], approve=False)["approval"]["status"] == "denied"
