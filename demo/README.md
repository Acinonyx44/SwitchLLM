# SwitchLLM Console demo

`index.html` is the SwitchLLM Console. It's a single self-contained page: open
it in a browser and it works offline, with no server.

It has three views:

- **Ask (employee):** five personas (support agent, quant analyst, engineer,
  legal counsel, contractor) and 13 sample requests. Each request animates
  through the steps: classify, apply the role's policy, pick a model tier,
  verify, escalate if needed, and print a cost receipt. An optional
  side-by-side run compares the answer and cost against the always-frontier
  default.
- **Policy (admin):** a matrix of where each team and task type starts, team
  budgets, and plain-English policy changes. Each change shows a preview of the
  YAML diff and a replay of 30 days of traffic priced under the new policy.
- **Savings (CIO):** 30 days of simulated traffic for a 200-person company:
  42,725 requests, $612 spent versus $1,378 always-frontier, 55.6% saved.
  Includes cumulative spend, a breakdown by team and by task type, and every
  request's receipt.

There are two model ladders to switch between:

- **Hybrid:** gpt-oss-20b, then gpt-oss-120b, then Claude Opus 5.5.
- **All open-weights:** gpt-oss-20b, then gpt-oss-120b, then DeepSeek V4 Pro.

Prices are OpenRouter list prices from September 2026.

## How it works

The page embeds a replay export (`window.SWITCHLLM_STATIC`) that the demo
server generated on 2026-09-25. In replay mode, model answers are pre-written
samples. Classification, policy, routing, verification and costs were computed
by the engine when the export was generated.

When `SWITCHLLM_STATIC` is absent, the page talks to a live demo server
instead, through `/api/meta`, `/api/ask`, `/api/policy`,
`/api/policy/preview`, `/api/policy/apply`, `/api/dashboard`, `/api/reset`
and `/api/replay`. It expects that server to be started with
`uv run switchllm demo`. That server isn't in this repository yet.
