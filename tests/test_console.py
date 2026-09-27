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
        with pytest.raises(urllib.error.HTTPError) as err:
            call(b, "/api/ask", {"persona": "nobody", "task": "hi"})
        assert err.value.code == 400
        assert b"<title>SwitchLLM Console</title>" in a.open(base + "/").read()
    finally:
        server.shutdown()
        server.server_close()
