# SwitchLLM Console demo

The Console shows SwitchLLM from all three seats:

- **Ask (employee):** five personas and 13 sample requests. You can also type
  any request of your own. Each request is classified, checked against the
  policy for the sender's role, routed to the cheapest model that clears the
  bar, verified, escalated if needed, and priced on a receipt. The optional
  compare mode runs the always-frontier default side by side.
- **Policy (admin):** describe a change in plain English, for example "Cap
  contractors at $3 a month and block them from code review". SwitchLLM shows
  the YAML diff and the teams and task types it affects. It then replays 30
  days of company traffic, about 44,000 requests, to price the change before
  you apply it.
- **Budget pacing and overrides:** each team's spend is projected to month
  end. A team on pace to overshoot its cap is stepped down early: effort
  first, then the top tier. Employees can reject an answer and re-run it one
  tier up. Within policy the re-run happens immediately. Above the cap it goes
  to a manager, who approves or denies it in the Override requests card on
  the Policy tab.
- **Savings (CIO):** spend compared with always-frontier by team and task
  type, budgets month to date, and every request sent from the Ask tab.

## Three ways to show it

| Situation | Command | What you get |
| --- | --- | --- |
| No setup, no network | Open `demo/index.html` in a browser | Offline replay of the sample requests and policy examples |
| Laptop demo with your own requests and policy edits | `switchllm demo --open` | The full live engine at http://localhost:8000 with simulated model calls |
| Real answers from real models | `OPENROUTER_API_KEY=... switchllm demo --live --open` | Real classifier, answer, verifier and compare calls through OpenRouter |

To host it for others, run `switchllm demo --host 0.0.0.0 --port 8000`, or
use the container:

```bash
docker build -t switchllm .
docker run -p 8000:8000 switchllm                                              # simulated models
docker run -p 8000:8000 -e OPENROUTER_API_KEY=... -e SWITCHLLM_LIVE=1 switchllm # real models
```

This works on any container host (Render, Fly.io, Railway, Cloud Run). The
server reads `PORT` from the environment.

**Several people at once.** Each browser gets its own session, so viewers can
change the policy and send requests without affecting each other. Simulated
traffic and replays are shared, so each extra viewer costs almost nothing.
Idle sessions expire after 4 hours, and at most 200 are kept.

**If the server stops mid-demo,** the page switches to its built-in offline
replay and offers to reconnect when the server is back.

## What's real and what's simulated

Classification rules, policy resolution, routing, verification and
escalation logic, receipts, the policy parser, the traffic replay and the
dashboards all run in the engine (`src/switchllm/console/`).

In simulated mode:
- Model answers to the sample requests are pre-written.
- Typed-in requests get a placeholder answer that says so.
- Token counts are modelled from prompt and answer length.
- The 30 days of company traffic are generated, not observed.

In live mode, the classifier, the answers, the verifier and the compare run
are real model calls. If a call fails, the page says so and falls back to
simulation for that request.

Prices are OpenRouter list prices as of September 2026 and live in
`src/switchllm/console/engine.py`.

To regenerate the offline page after changing the engine:

```bash
switchllm demo --export demo/index.html
```
