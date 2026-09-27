"""Append-only audit log: one JSON record per routed request.

It is the source of truth for cost receipts, budgets, the spend report,
savings-share billing and, later, training data for the learned router.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


class AuditLog:
    def __init__(self, path: str | Path | None = None):
        """path=None keeps records in memory only."""
        self.path = Path(path) if path else None
        self._memory: list[dict[str, Any]] = []

    def append(self, record: dict[str, Any]) -> None:
        if self.path is None:
            self._memory.append(record)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def records(self, since: datetime | None = None) -> list[dict[str, Any]]:
        if self.path is None:
            rows = list(self._memory)
        elif self.path.exists():
            with self.path.open() as f:
                rows = [json.loads(line) for line in f if line.strip()]
        else:
            rows = []
        if since is not None:
            rows = [r for r in rows if datetime.fromisoformat(r["ts"]) >= since]
        return rows

    def spend(self, user: str, since: datetime) -> float:
        return sum(r["cost_usd"] for r in self.records(since) if r["user"].lower() == user.lower())

    def report(self, since: datetime | None = None) -> dict[str, Any]:
        rows = self.records(since)
        return {
            "totals": _summarize(rows),
            "by_role": {k: _summarize(v) for k, v in _group(rows, "role").items()},
            "by_task_class": {k: _summarize(v) for k, v in _group(rows, "task_class").items()},
            "by_model": {k: _summarize(v) for k, v in _group(rows, "served_by").items()},
        }


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return dict(sorted(groups.items()))


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cost = sum(r["cost_usd"] for r in rows)
    baseline = sum(r["baseline_usd"] for r in rows)
    live = [r for r in rows if not r.get("shadow")]
    shadow = [r for r in rows if r.get("shadow")]
    summary: dict[str, Any] = {
        "requests": len(rows),
        "cost_usd": round(cost, 6),
        "baseline_usd": round(baseline, 6),
        "savings_usd": round(sum(r["baseline_usd"] - r["cost_usd"] for r in live), 6),
        "escalations": sum(1 for r in rows if r.get("escalated")),
        "unverified": sum(1 for r in rows if not r.get("verified", True)),
    }
    if shadow:
        # What routing *would* have saved on traffic that was served at baseline.
        summary["shadow_requests"] = len(shadow)
        summary["shadow_projected_savings_usd"] = round(
            sum(r["baseline_usd"] - r["routed_cost_usd"] for r in shadow), 6
        )
    return summary
