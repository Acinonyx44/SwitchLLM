"""Console policy: layered rules resolved per (role, task class).

Layers apply in order -- defaults, task_classes.<class>, roles.<role>.defaults,
roles.<role>.rules.<class> -- and every resolved route records the layers
that shaped it. A role's floor can only raise the tier and its ceiling can
only lower it, but a task class's quality floor always wins over a role cap.
"""

from __future__ import annotations

import copy
import difflib
from typing import Any

TIERS = ["small", "medium", "large"]
EFFORTS = ["low", "medium", "high"]
EXECUTIONS = ["single", "cascade", "thinking", "batch"]
ROUTE_FIELDS = ("floor", "ceiling", "effort", "execution", "verify")

# Where a typical request of each class starts; the classifier refines it per request.
TYPICAL_DIFFICULTY = {
    "general_chat": "small", "qa_factual": "small", "summarize": "small", "draft_email": "small",
    "rewrite_edit": "small", "translate": "small", "classify_label": "small", "customer_reply": "small",
    "code_explain": "small", "brainstorm": "small", "data_extraction": "small",
    "customer_escalation": "medium", "code_generate": "medium", "code_debug": "medium",
    "quantitative_reasoning": "medium", "data_analysis": "medium", "draft_document": "medium",
    "research_synthesis": "medium", "math_proof": "large", "code_review": "large",
    "architecture_design": "large", "strategic_analysis": "large", "legal_clause_analysis": "large",
    "compliance_check": "large", "financial_analysis": "large", "medical_or_safety": "large",
    "agentic_task": "large",
}

MATRIX_CLASSES = ["customer_reply", "summarize", "code_generate", "code_review",
                  "quantitative_reasoning", "legal_clause_analysis", "financial_analysis"]

_SINGLE = {"effort": "low", "execution": "single", "verify": False}
_DEEP = {"floor": "large", "effort": "high", "execution": "thinking"}

DEFAULT_POLICY: dict[str, Any] = {
    "defaults": {"floor": "small", "ceiling": "large", "effort": "medium", "execution": "cascade",
                 "verify": True, "max_output": 4096},
    "task_classes": {
        "qa_factual": {"ceiling": "medium", **_SINGLE},
        "general_chat": {"ceiling": "small", **_SINGLE},
        "summarize": dict(_SINGLE),
        "draft_email": {"ceiling": "medium", **_SINGLE},
        "rewrite_edit": {"ceiling": "medium", **_SINGLE},
        "translate": dict(_SINGLE),
        "brainstorm": dict(_SINGLE),
        "classify_label": {"ceiling": "medium", **_SINGLE},
        "data_extraction": {"effort": "low", "execution": "single", "verify": True},
        "code_explain": dict(_SINGLE),
        "customer_reply": {"ceiling": "medium", **_SINGLE},
        "code_review": dict(_DEEP),
        "architecture_design": dict(_DEEP),
        "math_proof": {"floor": "medium", "effort": "high"},
        "research_synthesis": {"floor": "medium", "effort": "high"},
        "strategic_analysis": dict(_DEEP),
        "legal_clause_analysis": dict(_DEEP),
        "compliance_check": dict(_DEEP),
        "financial_analysis": dict(_DEEP),
        "medical_or_safety": dict(_DEEP),
        "agentic_task": {"floor": "large", "effort": "high"},
    },
    "roles": {
        "default": {"description": "Anyone without a more specific role."},
        "analyst": {"description": "Research / quant / business analysts.", "budget_usd_month": 250.0,
                    "rules": {"quantitative_reasoning": {"floor": "medium", "effort": "high"},
                              "data_analysis": {"floor": "medium"}}},
        "engineer": {"description": "Software engineers.", "budget_usd_month": 450.0,
                     "rules": {"code_generate": {"floor": "medium", "effort": "high"},
                               "code_debug": {"floor": "medium", "effort": "high"}}},
        "support_agent": {"description": "Customer support.", "budget_usd_month": 25.0,
                          "defaults": {"ceiling": "medium", "effort": "low"},
                          "rules": {"customer_escalation": {"ceiling": "large", "effort": "medium",
                                                            "execution": "cascade", "verify": True}},
                          "denied_task_classes": ["legal_clause_analysis", "financial_analysis"]},
        "legal": {"description": "Legal and compliance.", "budget_usd_month": 120.0,
                  "defaults": {"floor": "medium", "effort": "high"}},
        "executive": {"description": "Leadership.", "budget_usd_month": 60.0,
                      "defaults": {"floor": "medium"}},
        "contractor": {"description": "External contractors.", "budget_usd_month": 5.0,
                       "defaults": {"ceiling": "medium", "effort": "low"},
                       "denied_task_classes": ["legal_clause_analysis", "financial_analysis",
                                               "compliance_check", "strategic_analysis"]},
    },
    "budget_usd_month_default": 15.0,
}


def tier_index(t: str) -> int:
    return TIERS.index(t)


def clamp_tier(t: str, floor: str, ceiling: str) -> str:
    return TIERS[max(tier_index(floor), min(tier_index(t), tier_index(ceiling)))]


def resolve(policy: dict[str, Any], role: str, task_class: str) -> dict[str, Any]:
    """The route a (role, task class) pair gets under this policy."""
    r = policy["roles"].get(role) or policy["roles"]["default"]
    role_key = role if role in policy["roles"] else "default"
    if task_class in r.get("denied_task_classes", []):
        return {"blocked": True, "layers": [f"roles.{role_key}.denied_task_classes"]}

    cell = {k: policy["defaults"][k] for k in ROUTE_FIELDS}
    layers = ["defaults"]
    tc = policy["task_classes"].get(task_class)
    if tc:
        cell.update({k: v for k, v in tc.items() if k in ROUTE_FIELDS})
        layers.append(f"task_classes.{task_class}")
    class_floor = cell["floor"]

    rd = r.get("defaults")
    if rd:
        if "floor" in rd:
            cell["floor"] = max(cell["floor"], rd["floor"], key=tier_index)
        if "ceiling" in rd:
            cell["ceiling"] = min(cell["ceiling"], rd["ceiling"], key=tier_index)
        cell.update({k: rd[k] for k in ("effort", "execution", "verify") if k in rd})
        layers.append(f"roles.{role_key}.defaults")
    rule = r.get("rules", {}).get(task_class)
    if rule:
        cell.update({k: v for k, v in rule.items() if k in ROUTE_FIELDS})
        layers.append(f"roles.{role_key}.rules.{task_class}")

    # A task class's quality floor beats any role cap.
    cell["floor"] = max(cell["floor"], class_floor, key=tier_index)
    if tier_index(cell["floor"]) > tier_index(cell["ceiling"]):
        cell["ceiling"] = cell["floor"]
    cell["start"] = clamp_tier(TYPICAL_DIFFICULTY.get(task_class, "medium"), cell["floor"], cell["ceiling"])
    return {"blocked": False, **cell, "layers": layers}


def budget_for(policy: dict[str, Any], role: str) -> float:
    r = policy["roles"].get(role) or {}
    return float(r.get("budget_usd_month", policy.get("budget_usd_month_default", 0.0)))


def known_classes(policy: dict[str, Any]) -> list[str]:
    names = set(TYPICAL_DIFFICULTY) | set(policy["task_classes"])
    for r in policy["roles"].values():
        names |= set(r.get("rules", {})) | set(r.get("denied_task_classes", []))
    return sorted(names)


def merge_patch(policy: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(policy)

    def merge(dst: dict, src: dict) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst[k] = copy.deepcopy(v)

    merge(out, patch)
    return out


def to_yaml(obj: Any, indent: int = 0) -> str:
    """Minimal YAML emitter for the policy (dicts, lists of scalars, scalars)."""
    pad = "  " * indent
    lines: list[str] = []
    for k, v in obj.items():
        if isinstance(v, dict):
            lines.append(f"{pad}{k}:")
            lines.append(to_yaml(v, indent + 1))
        elif isinstance(v, list):
            lines.append(f"{pad}{k}:")
            lines += [f"{pad}- {_scalar(x)}" for x in v]
        else:
            lines.append(f"{pad}{k}: {_scalar(v)}")
    return "\n".join(line for line in lines if line)


def _scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def yaml_diff(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    return list(difflib.unified_diff(
        to_yaml(before).splitlines(), to_yaml(after).splitlines(),
        "policy.yaml", "policy.yaml (proposed)", n=2, lineterm=""))
