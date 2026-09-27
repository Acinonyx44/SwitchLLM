import asyncio
import json

import pytest

from switchllm.cli import main
from switchllm.providers import OpenAICompatibleProvider


def test_cli_route_run_report(tmp_path, capsys):
    log = str(tmp_path / "audit.jsonl")
    assert main(["--audit-log", log, "route", "--json", "--user", "bob@acme.com", "Review this contract clause"]) == 0
    assert json.loads(capsys.readouterr().out)["model"] == "mock-frontier"

    assert main(["--audit-log", log, "run", "--user", "bob@acme.com", "Quick tl;dr please"]) == 0
    assert "saved" in capsys.readouterr().out

    assert main(["--audit-log", log, "report", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["totals"]["requests"] == 1


def test_cli_chat(tmp_path, capsys, monkeypatch):
    lines = iter(["Quick tl;dr please", "/why Prove the theorem", "/report", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(lines))
    assert main(["--audit-log", str(tmp_path / "a.jsonl"), "chat", "--user", "alice@acme.com"]) == 0
    out = capsys.readouterr().out
    assert "role: quant_research" in out and "mock-small" in out and "1 requests" in out


def test_mcp_tools(engine):
    pytest.importorskip("mcp")
    from switchllm.mcp_server import build_server

    server = build_server(engine, "bob@acme.com")

    async def exercise():
        names = {t.name for t in await server.list_tools()}
        result = await server.call_tool("run_task", {"prompt": "Quick tl;dr please"})
        return names, result

    names, result = asyncio.run(exercise())
    assert names == {"run_task", "explain_route", "spend_report"}
    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert payload["route"]["role"] == "sales"
    assert payload["receipt"]["cost_usd"] < payload["receipt"]["baseline_usd"]


def test_openai_compatible_request_shape(monkeypatch):
    from switchllm.catalog import Effort, ModelSpec, Mode
    from switchllm.providers import CompletionRequest

    sent = {}

    def fake_post(self, path, body):
        sent.update(path=path, body=body)
        return {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 7}}

    monkeypatch.setattr(OpenAICompatibleProvider, "_post", fake_post)
    model = ModelSpec("gw-small", "gateway", 1, 1, 1, name="vendor-small", supports_effort=True)
    req = CompletionRequest(model, [{"role": "user", "content": "x"}], Effort.MEDIUM, Mode.BATCH, 1000)
    c = OpenAICompatibleProvider("https://gw.example/v1").complete(req)
    assert sent["path"] == "/chat/completions"
    assert sent["body"]["model"] == "vendor-small"
    assert sent["body"]["reasoning_effort"] == "medium"
    assert sent["body"]["max_completion_tokens"] == 1000
    assert (c.text, c.input_tokens, c.output_tokens) == ("hi", 5, 7)
    assert c.executed_mode is Mode.SINGLE  # batch not yet submitted asynchronously
