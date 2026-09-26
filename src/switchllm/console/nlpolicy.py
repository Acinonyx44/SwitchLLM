"""Plain-English policy changes -> a policy patch.

Deterministic rules cover the common shapes admins write ("engineers only
need medium effort for code generation", "cap contractors at $3 a month and
block them from code review"). Clauses the rules don't understand come back
as `unparsed` rather than being guessed at.
"""

from __future__ import annotations

import re
from typing import Any

from .policy import DEFAULT_POLICY

TEAMS = [
    (r"support( agents?| team)?", "support_agent"),
    (r"engineers?|engineering|developers?", "engineer"),
    (r"analysts?|quants?|research(ers)?", "analyst"),
    (r"legal( team)?|lawyers?|counsel", "legal"),
    (r"executives?|leadership|execs?", "executive"),
    (r"contractors?|vendors?|externals?", "contractor"),
    (r"everyone( else)?|all (employees|staff|teams)|the whole company", "*"),
]
CLASSES = [
    (r"code generation|writing code|generat\w* code|code gen", "code_generate"),
    (r"code reviews?|reviewing code|security reviews?", "code_review"),
    (r"debugging|code debug\w*", "code_debug"),
    (r"quantitative reasoning|quant(itative)? (work|tasks|questions)|math", "quantitative_reasoning"),
    (r"legal (clause|contract)s?( analysis)?|contract review", "legal_clause_analysis"),
    (r"financial analysis|valuations?|financial modell?ing", "financial_analysis"),
    (r"customer escalations?|escalations", "customer_escalation"),
    (r"customer repl(y|ies)", "customer_reply"),
    (r"summar(y|ies|ization|isation|ize|ise)", "summarize"),
    (r"e-?mails?", "draft_email"),
    (r"architecture( design)?|system design", "architecture_design"),
    (r"strategic analysis|strategy", "strategic_analysis"),
    (r"compliance( checks?)?", "compliance_check"),
    (r"research synthesis", "research_synthesis"),
    (r"translations?|translating", "translate"),
    (r"data analysis", "data_analysis"),
]
TIER_WORDS = [(r"frontier|large|best|strongest|top", "large"), (r"medium|mid(-tier)?|middle", "medium"),
              (r"small|cheap(est)?|smallest|basic", "small")]
FILLER = re.compile(r"^(escalate( if needed| when needed)?|if needed|please|as needed)?$", re.I)


def _find(table: list[tuple[str, str]], text: str) -> str | None:
    for pat, value in table:
        if re.search(rf"\b(?:{pat})\b", text, re.I):
            return value
    return None


def _strip(table: list[tuple[str, str]], text: str) -> str:
    for pat, _ in table:
        text = re.sub(rf"\b(?:{pat})\b", " ", text, flags=re.I)
    return text


def parse(instruction: str, policy: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = policy or DEFAULT_POLICY
    clauses = [c.strip(" .") for c in re.split(r"\s*(?:;|,|\band\b|\bbut\b)\s*", instruction) if c.strip(" .")]
    patch: dict[str, Any] = {}
    actions: list[str] = []
    unparsed: list[str] = []
    team = task = None

    for clause in clauses:
        text = clause.lower()
        team = _find(TEAMS, text) or (team if re.search(r"\b(them|they|their)\b", text) else team)
        task = _find(CLASSES, text) or task
        rest = _strip(CLASSES, _strip(TEAMS, text))
        changes: dict[str, Any] = {}
        role_changes: dict[str, Any] = {}

        if m := re.search(r"\b(low|medium|high)\s+(reasoning\s+)?effort", rest):
            changes["effort"] = m.group(1)
        tier = _find([(p + r")\s+(?:model|tier", t) for p, t in TIER_WORDS], rest)
        if tier:
            if re.search(r"\b(cap|capped|at most|no more than|limit\w*|max(imum)?|only up to)\b", rest):
                changes["ceiling"] = tier
            else:  # "start on", "always get", "at least", "use"
                changes["floor"] = tier
        if re.search(r"extended thinking|think(ing)? mode", rest):
            changes["execution"] = "thinking"
        if re.search(r"\b(no|skip|without)\b.*\bverif", rest):
            changes["verify"] = False
        elif re.search(r"\balways verify|\bverif(y|ied|ication) (required|always)", rest):
            changes["verify"] = True
        if m := re.search(r"\$\s*(\d+(?:\.\d+)?)\s*(k)?\s*(?:a|per|/)\s*month", rest):
            role_changes["budget_usd_month"] = float(m.group(1)) * (1000 if m.group(2) else 1)
        block = re.search(r"\b(block|deny|ban|stop|prevent|can'?t|cannot|shouldn'?t)\b", text)
        unblock = re.search(r"\b(unblock|allow|let)\b", text)

        if not (changes or role_changes or block or unblock):
            if not FILLER.match(rest.strip()):
                unparsed.append(clause)
            continue
        if team is None and (role_changes or block or unblock):
            unparsed.append(clause + " (which team?)")
            continue

        if team and team != "*":
            r = patch.setdefault("roles", {}).setdefault(team, {})
            if role_changes:
                r.update(role_changes)
                actions.append(f"{team}: monthly budget → ${role_changes['budget_usd_month']:,.2f}")
            if (block or unblock) and task:
                current = list(policy["roles"].get(team, {}).get("denied_task_classes", []))
                denied = r.get("denied_task_classes", current)
                if block and task not in denied:
                    denied = [*denied, task]
                    actions.append(f"{team}: block {task}")
                elif unblock and task in denied:
                    denied = [c for c in denied if c != task]
                    actions.append(f"{team}: allow {task}")
                r["denied_task_classes"] = denied
            elif (block or unblock) and not task:
                unparsed.append(clause + " (block which task type?)")
            if changes:
                if task:
                    r.setdefault("rules", {}).setdefault(task, {}).update(changes)
                    actions += [f"{team} · {task}: {k} → {_fmt(v)}" for k, v in changes.items()]
                else:
                    r.setdefault("defaults", {}).update(changes)
                    actions += [f"{team} (all tasks): {k} → {_fmt(v)}" for k, v in changes.items()]
        elif changes:
            if task:
                patch.setdefault("task_classes", {}).setdefault(task, {}).update(changes)
                actions += [f"all teams · {task}: {k} → {_fmt(v)}" for k, v in changes.items()]
            else:
                patch.setdefault("defaults", {}).update(changes)
                actions += [f"company default: {k} → {_fmt(v)}" for k, v in changes.items()]
        elif role_changes and team == "*":
            patch["budget_usd_month_default"] = role_changes["budget_usd_month"]
            actions.append(f"default budget → ${role_changes['budget_usd_month']:,.2f}")

    return {"ok": bool(actions), "actions": actions, "unparsed": unparsed, "patch": patch, "parser": "rules"}


def _fmt(v: Any) -> str:
    return ("on" if v else "off") if isinstance(v, bool) else str(v)
