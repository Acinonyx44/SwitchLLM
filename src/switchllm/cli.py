"""Command line: role-based chat, route explanations, one-off runs and reports."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .engine import RunResult, SwitchLLM
from .router import RouteDecision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="switchllm", description=__doc__)
    parser.add_argument("--policy", default=os.environ.get("SWITCHLLM_POLICY"),
                        help="policy TOML (default: bundled example policy)")
    parser.add_argument("--audit-log", default=os.environ.get("SWITCHLLM_AUDIT_LOG", ".switchllm/audit.jsonl"))
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="interactive chat; every message is routed for the user's role")
    chat.add_argument("--user", required=True)

    for name, help_ in [("route", "explain where a prompt would go, without running it"),
                        ("run", "route and run a single prompt")]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--user", required=True)
        p.add_argument("--batch", action="store_true", help="non-urgent: allow batch execution")
        p.add_argument("--json", action="store_true")
        p.add_argument("prompt")

    report = sub.add_parser("report", help="spend and savings from the audit log")
    report.add_argument("--json", action="store_true")

    sub.add_parser("mcp", help="serve the MCP connector over stdio (reads SWITCHLLM_USER)")

    args = parser.parse_args(argv)

    if args.command == "mcp":
        from .mcp_server import main as mcp_main
        mcp_main()
        return 0

    engine = SwitchLLM.from_files(args.policy, args.audit_log)

    if args.command == "chat":
        return _chat(engine, args.user)
    if args.command == "route":
        decision = engine.explain(args.user, args.prompt, "batch" if args.batch else "interactive")
        print(json.dumps(decision.to_dict(), indent=2) if args.json else _format_decision(decision))
        return 0
    if args.command == "run":
        result = engine.run(args.user, args.prompt, urgency="batch" if args.batch else "interactive")
        if args.json:
            print(json.dumps({"answer": result.text, "route": result.decision.to_dict(),
                              "cost_usd": result.receipt.cost_usd, "baseline_usd": result.receipt.baseline_usd},
                             indent=2))
        else:
            print(result.text)
            print(_format_receipt(result))
        return 0
    if args.command == "report":
        data = engine.spend_report()
        print(json.dumps(data, indent=2) if args.json else _format_report(data))
        return 0
    return 1


def _chat(engine: SwitchLLM, user: str) -> int:
    role = engine.role_for(user)
    print(f"SwitchLLM chat as {user} (role: {role.name}). One text box, no dropdowns.")
    print("Commands: /why <prompt>  /report  /quit")
    history: list[dict[str, str]] = []
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return 0
        if line == "/report":
            print(_format_report(engine.spend_report()))
            continue
        if line.startswith("/why "):
            print(_format_decision(engine.explain(user, line[5:])))
            continue
        result = engine.run(user, line, history=history)
        history += [{"role": "user", "content": line}, {"role": "assistant", "content": result.text}]
        print(f"\nassistant> {result.text}")
        print(_format_receipt(result))


def _format_decision(d: RouteDecision) -> str:
    lines = [
        f"role {d.role} -> {d.model.id} (tier {d.model.tier}) | {d.effort.value} effort | {d.mode.value}",
        f"est. ${d.est_cost_usd:.4f} vs baseline ${d.est_baseline_usd:.4f} ({d.est_savings_pct:.0f}% saved)",
    ]
    lines += [f"  - {r}" for r in d.reasons]
    if d.escalation:
        lines.append("  escalates to: " + " -> ".join(m.id for m in d.escalation))
    lines.append("  signals: " + "; ".join(d.profile.signals))
    return "\n".join(lines)


def _format_receipt(r: RunResult) -> str:
    d = r.decision
    path = " -> ".join(a.model for a in r.attempts)
    flag = "" if r.verified else "  [UNVERIFIED]"
    if r.shadow:
        return (f"  [shadow] served {r.served_by} ${r.receipt.cost_usd:.4f}; routing would have used "
                f"{d.model.id} for ~${r.routed_cost_usd:.4f}")
    return (f"  [{d.role}: {path} | {d.effort.value} | {d.mode.value} | "
            f"${r.receipt.cost_usd:.4f} vs ${r.receipt.baseline_usd:.4f} baseline, "
            f"{r.receipt.savings_pct:.0f}% saved]{flag}")


def _format_report(data: dict) -> str:
    t = data["totals"]
    lines = [f"{t['requests']} requests | spent ${t['cost_usd']:.4f} | baseline ${t['baseline_usd']:.4f} | "
             f"saved ${t['savings_usd']:.4f} | {t['escalations']} escalations"]
    if "shadow_projected_savings_usd" in t:
        lines.append(f"shadow: {t['shadow_requests']} requests, projected savings "
                     f"${t['shadow_projected_savings_usd']:.4f}")
    for section in ("by_role", "by_task_class", "by_model"):
        lines.append(section.replace("_", " ") + ":")
        for key, s in data[section].items():
            lines.append(f"  {key:<16} {s['requests']:>4} req  ${s['cost_usd']:.4f}  saved ${s['savings_usd']:.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
