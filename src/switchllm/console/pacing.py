"""Budget pacing: steer each team toward its monthly cap by run rate.

A hard stop at 100% of budget arrives too late (day 25, mid-deadline). Pacing
projects month-end spend from the spend so far and steps routing down early
and gently: effort first, then the tier ceiling. Quality floors still win.
"""

from __future__ import annotations

import calendar
from datetime import date
from typing import Any

from .policy import EFFORTS, TIERS, tier_index

# Projected month-end spend / cap -> action.
THRESHOLDS = [(1.2, "floor"), (1.0, "tier"), (0.9, "effort")]


def pace(spent: float, cap: float, today: date) -> dict[str, Any]:
    days = calendar.monthrange(today.year, today.month)[1]
    elapsed = max(1, today.day)
    projected = spent / elapsed * days
    ratio = projected / cap if cap > 0 else float("inf")
    action = next((a for limit, a in THRESHOLDS if ratio >= limit), "none")
    return {"spent": round(spent, 4), "cap": cap, "projected": round(projected, 2),
            "ratio": round(ratio, 3), "day": elapsed, "days": days, "action": action}


def apply(cell: dict[str, Any], pacing: dict[str, Any]) -> dict[str, Any]:
    """Route with pacing applied; never below the route's floor."""
    action = pacing["action"]
    if cell.get("blocked") or action == "none":
        return cell
    out = dict(cell)
    if action == "effort":
        out["effort"] = min(out["effort"], "medium", key=EFFORTS.index)
    elif action == "tier":
        out["effort"] = "low"
        out["ceiling"] = TIERS[max(tier_index(out["floor"]), tier_index(out["ceiling"]) - 1)]
    else:  # floor
        out["effort"] = "low"
        out["ceiling"] = out["floor"]
    if out["execution"] == "thinking" and out["effort"] == "low":
        out["execution"] = "cascade" if out["verify"] else "single"
    out["start"] = TIERS[min(tier_index(out["start"]), tier_index(out["ceiling"]))]
    if out != cell:
        out["layers"] = [*cell["layers"], "budget.pacing"]
    return out


def describe(pacing: dict[str, Any], label: str) -> str:
    head = f"{label} on pace for ${pacing['projected']:,.2f} of a ${pacing['cap']:,.2f} monthly cap"
    return head + {
        "none": ", no change",
        "effort": ", so effort is capped at medium",
        "tier": ", so effort is low and the top tier is held back",
        "floor": ", so requests run on the floor tier only",
    }[pacing["action"]]
