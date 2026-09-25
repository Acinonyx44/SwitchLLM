# SwitchLLM

**Role-based model routing for the enterprise.** SwitchLLM removes the "which
model, which effort level, which mode?" decision from every employee. Each task
is classified, the employee's role is looked up from the identity provider, and
the task runs on the right model at the right reasoning effort and execution
mode, under a budget policy the company writes once.

Employees get one text box with no dropdowns. IT and FinOps get one policy, one
audit log, and a cost receipt on every request.

## Demo

Open [`demo/index.html`](demo/index.html) in a browser for the SwitchLLM
Console, which runs offline. It has three views: **Ask**, where an employee
sends a request and sees the route and cost receipt; **Policy**, where an
admin edits the policy in plain English and previews its effect by replaying
traffic; and **Savings**, a 30-day spend dashboard for the CIO. See
[`demo/README.md`](demo/README.md).

## What's in this repo

This repo contains the role-based routing core, runnable offline against a
keyless mock provider:

| Piece | File | What it does |
|---|---|---|
| Policy | `src/switchllm/policy.py`, `example_policy.toml` | Models and tiers, roles (tier cap, effort cap, allowed modes, monthly budget, provider allow-list), task-class quality floors, the baseline setting |
| Identity | `src/switchllm/directory.py` | User → IdP groups → role. A static directory for now; Okta, Entra, Slack and Glean adapters plug in behind the same interface |
| Classifier | `src/switchllm/classifier.py` | Heuristic task class, difficulty (1–5) and tool need, with the signals behind each call |
| Router | `src/switchllm/router.py` | Chooses tier, effort and mode. Order of precedence: difficulty, then quality floor, then role caps, then budget, then mode. Every step is recorded as a reason |
| Engine | `src/switchllm/engine.py` | Runs the chosen route, escalates up the tier ladder while the verifier rejects, attaches a cost receipt, supports shadow mode, writes the audit log |
| Audit | `src/switchllm/audit.py` | Append-only JSONL log, monthly spend per user, and a spend and savings report by role, task class and model |
| Providers | `src/switchllm/providers.py` | Mock provider, plus any OpenAI-compatible endpoint (such as a company's existing LiteLLM, Portkey or OpenRouter gateway) |
| Connector | `src/switchllm/mcp_server.py` | MCP tools `run_task`, `explain_route` and `spend_report` for Claude, ChatGPT, Copilot or Cursor |
| Chat CLI | `src/switchllm/cli.py` | `switchllm chat`: a role-based chat where every message is routed and receipted |

### Routing rules

1. **Difficulty sets the starting point.** Difficulty 1–2 starts at tier 1 with
   low effort, 3 at tier 2 with medium effort, and 4–5 at tier 3 with high
   effort.
2. **Quality floors raise it.** For example, `legal` tasks are never routed
   below tier 3, whoever asks.
3. **Role caps lower it, but never below the floor.** For example, `sales`
   tops out at tier 2 and medium effort. When the floor and the cap conflict,
   the floor wins and the audit log records the override.
4. **Budgets throttle.** At 80% of a user's monthly budget, effort drops to
   medium. At 100%, requests run at the floor tier with low effort and no
   escalation.
5. **Mode.** `batch` applies when the caller marks the task non-urgent, the
   role allows it, and the vendor discount applies. `agentic` applies when the
   task needs tools. `extended_thinking` applies to hard, high-effort tasks.
   Otherwise the request is a single call.
6. **Escalation.** If the verifier's confidence falls below the threshold, the
   task is retried one tier up, until it is accepted or reaches the role's
   ceiling. If nothing passes, the answer is marked unverified.

### Shadow mode

Set `mode = "shadow"` in the policy. Traffic is served exactly as it is today
(the baseline), and each audit record stores what routing would have cost. Set
`shadow_execute_routed = true` to also run the routed path and see whether it
would have passed verification. This is the quality-versus-cost comparison
admins see before any policy goes live.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# Role-based chat. alice is quant_research, bob is sales, anyone else is general.
switchllm chat --user bob@acme.com

# Why would this go where it goes?
switchllm route --user bob@acme.com "Review this contract clause on indemnification"

# Spend and savings from the audit log (.switchllm/audit.jsonl)
switchllm report

# MCP connector over stdio
SWITCHLLM_USER=alice@acme.com switchllm mcp

pytest
```

To use your own policy, pass `--policy path/to/policy.toml` or set
`SWITCHLLM_POLICY`. To route real traffic, add a `[providers.<name>]` table
of type `openai_compatible` that points at your gateway, then map your models
to tiers. There's a commented example in `example_policy.toml`.

## Status and next milestones

Built:
- Role-based routing core
- Tier floors and caps
- Per-user budgets
- Escalation
- Shadow mode
- Cost receipts and audit log
- Chat CLI
- MCP connector (stdio)

Next:
- Live (non-mock) benchmark on real models across roughly 200 mixed-difficulty
  tasks
- IdP sync (Okta, Entra) behind `RoleDirectory`
- Plain-English policy authoring that compiles to the policy schema
  (`define_policy` tool)
- A real verifier (LLM judge) in place of provider-reported confidence
- True async batch submission and an agentic tool loop
- A hosted MCP connector with OAuth identity, plus a shadow-mode dashboard
- A learned router, trained on the audit log with a quality-minus-cost reward,
  to replace the heuristic classifier
