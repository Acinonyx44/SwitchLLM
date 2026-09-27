"""SwitchLLM as an MCP connector for Claude, ChatGPT, Copilot or Cursor.

The host assistant hands each task to run_task; SwitchLLM picks the model,
effort and execution mode for the employee's role and returns the answer
with its route and cost receipt.

Identity comes from the deployment (SWITCHLLM_USER for a local stdio
install; the OAuth subject for a hosted connector), never from a tool
argument, so the model cannot claim a more privileged role.
"""

from __future__ import annotations

import os
from typing import Any

from .engine import SwitchLLM

INSTRUCTIONS = (
    "SwitchLLM routes work to the right model for this employee under company policy. "
    "Send self-contained tasks to run_task instead of answering them yourself. "
    "Use explain_route to show why a task would go where it goes, and spend_report for usage."
)


def build_server(engine: SwitchLLM, user: str):
    from mcp.server import MCPServer

    server = MCPServer("switchllm", instructions=INSTRUCTIONS)

    @server.tool()
    def run_task(prompt: str, urgency: str = "interactive") -> dict[str, Any]:
        """Run a task on the model, effort level and execution mode that company
        policy selects for the current employee. urgency: "interactive" or "batch"."""
        result = engine.run(user, prompt, urgency=urgency)
        return {
            "answer": result.text,
            "served_by": result.served_by,
            "verified": result.verified,
            "route": result.decision.to_dict(),
            "receipt": {
                "cost_usd": round(result.receipt.cost_usd, 6),
                "baseline_usd": round(result.receipt.baseline_usd, 6),
                "savings_pct": round(result.receipt.savings_pct, 1),
            },
            "shadow": result.shadow,
        }

    @server.tool()
    def explain_route(prompt: str, urgency: str = "interactive") -> dict[str, Any]:
        """Explain where a task would be routed and why, without running it."""
        return engine.explain(user, prompt, urgency).to_dict()

    @server.tool()
    def spend_report() -> dict[str, Any]:
        """Spend, baseline and savings from the audit log, by role, task class and model."""
        return engine.spend_report()

    return server


def main() -> None:
    user = os.environ.get("SWITCHLLM_USER")
    if not user:
        raise SystemExit("set SWITCHLLM_USER to the employee's identity (e.g. alice@acme.com)")
    engine = SwitchLLM.from_files(
        os.environ.get("SWITCHLLM_POLICY"),
        os.environ.get("SWITCHLLM_AUDIT_LOG", ".switchllm/audit.jsonl"),
    )
    build_server(engine, user).run("stdio")


if __name__ == "__main__":
    main()
