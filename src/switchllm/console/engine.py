"""The console engine behind `switchllm demo`.

Serves the three console views -- Ask (employee), Policy (admin), Savings
(CIO) -- from one policy, one routing engine and one audit trail. Model calls
are simulated by default (pre-written answers for the sample requests,
modelled token counts) and go to real models in live mode.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from collections import OrderedDict, defaultdict
from datetime import date, datetime, timezone
from importlib import resources
from typing import Any

from . import classifier, pacing, traffic
from .live import LiveClient, parse_json
from .nlpolicy import parse as parse_instruction
from .policy import (DEFAULT_POLICY, MATRIX_CLASSES, TIERS, budget_for, clamp_tier, known_classes,
                     merge_patch, resolve, tier_index, to_yaml, yaml_diff)

LADDERS: dict[str, dict[str, Any]] = {
    "hybrid": {
        "label": "Open models + frontier",
        "blurb": "gpt-oss for routine work, Claude Opus 5.5 for the hard tail",
        "pool": {"name": "hybrid", "baseline": "claude-opus-5.5", "utility": "gpt-oss-20b", "models": [
            {"name": "gpt-oss-20b", "tier": "small", "model_id": "openai/gpt-oss-20b", "price_in": 0.03, "price_out": 0.13},
            {"name": "gpt-oss-120b", "tier": "medium", "model_id": "openai/gpt-oss-120b", "price_in": 0.15, "price_out": 0.6},
            {"name": "claude-opus-5.5", "tier": "large", "model_id": "anthropic/claude-opus-5.5", "price_in": 4.0, "price_out": 20.0},
        ]},
    },
    "open": {
        "label": "All open-weights",
        "blurb": "gpt-oss-20b → gpt-oss-120b → DeepSeek V4 Pro",
        "pool": {"name": "open", "baseline": "deepseek-v4-pro", "utility": "gpt-oss-20b", "models": [
            {"name": "gpt-oss-20b", "tier": "small", "model_id": "openai/gpt-oss-20b", "price_in": 0.03, "price_out": 0.13},
            {"name": "gpt-oss-120b", "tier": "medium", "model_id": "openai/gpt-oss-120b", "price_in": 0.15, "price_out": 0.6},
            {"name": "deepseek-v4-pro", "tier": "large", "model_id": "deepseek/deepseek-v4-pro", "price_in": 0.913, "price_out": 1.825},
        ]},
    },
}
EFFORT_OUT = traffic.EFFORT_OUT
ROLE_ORDER = ["default", "analyst", "engineer", "support_agent", "legal", "executive", "contractor"]


# Shared across every viewer's engine: simulated traffic and its replays are
# pure functions of (day window, seed) and (ladder, policy), so sessions reuse them.
_CACHE_LOCK = threading.Lock()
_TRAFFIC: dict[tuple, list] = {}
_REPLAYS: "OrderedDict[tuple, list[dict]]" = OrderedDict()
_REPLAY_CACHE_SIZE = 64


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _tok(text: str) -> int:
    return max(1, len(text) // 4)


class ConsoleEngine:
    def __init__(self, live_client: LiveClient | None = None, today: date | None = None, seed: int = 7):
        data = json.loads(resources.files("switchllm.console").joinpath("scenarios.json").read_text())
        self.personas: list[dict] = data["personas"]
        self.scenarios: list[dict] = data["scenarios"]
        self.samples: dict[str, dict] = data["samples"]
        self.policy_examples: list[str] = data["policy_examples"]
        self.role_labels: dict[str, str] = data["role_labels"]
        self.live = live_client
        self.today = today or date.today()
        self.days = traffic.window(self.today)
        self.seed = seed
        with _CACHE_LOCK:
            key = (self.today, seed)
            if key not in _TRAFFIC:
                _TRAFFIC[key] = traffic.generate(self.days, seed)
            self.requests = _TRAFFIC[key]
        self._counter = 0
        self.reset("hybrid")

    # -- state ------------------------------------------------------------------

    def reset(self, ladder: str | None = None) -> dict[str, Any]:
        if ladder is not None:
            if ladder not in LADDERS:
                raise ValueError(f"unknown model ladder {ladder!r}")
            self.ladder = ladder
        self.policy = copy.deepcopy(DEFAULT_POLICY)
        self.version = 1
        self.history: list[dict] = []
        self.session: list[dict] = []  # receipts from the Ask view
        self.approvals: list[dict] = []  # override requests waiting on a manager
        self._asks: dict[str, dict] = {}  # receipt id -> what was asked, for re-runs
        return self.meta()

    @property
    def pool(self) -> dict[str, Any]:
        return LADDERS[self.ladder]["pool"]

    def _model(self, tier: str) -> dict[str, Any]:
        return next(m for m in self.pool["models"] if m["tier"] == tier)

    def _by_name(self, name: str) -> dict[str, Any]:
        return next(m for m in self.pool["models"] if m["name"] == name)

    @staticmethod
    def _price(m: dict, tin: int, tout: int) -> float:
        return (tin * m["price_in"] + tout * m["price_out"]) / 1e6

    def meta(self) -> dict[str, Any]:
        return {
            "live": self.live is not None,
            "ladder": self.ladder,
            "ladders": {k: {"label": v["label"], "blurb": v["blurb"]} for k, v in LADDERS.items()},
            "pool": self.pool,
            "personas": self.personas,
            "scenarios": self.scenarios,
            "policy_examples": self.policy_examples,
            "role_labels": self.role_labels,
            "headcount": traffic.HEADCOUNT,
            "employees": sum(traffic.HEADCOUNT.values()),
            "version": self.version,
        }

    # -- ask ----------------------------------------------------------------------

    def ask(self, persona_id: str, task: str, compare: bool = False,
            force_tier: str | None = None, override: dict | None = None) -> dict[str, Any]:
        """Route and answer a request. `force_tier` + `override` re-run it on a
        chosen tier after feedback, self-serve or manager-approved."""
        persona = next((p for p in self.personas if p["id"] == persona_id), None)
        if persona is None:
            raise ValueError(f"unknown persona {persona_id!r}")
        task = (task or "").strip()
        if not task:
            raise ValueError("empty request")
        role = persona["role"]
        scenario = next((s for s in self.scenarios if s["persona"] == persona_id and _norm(s["task"]) == _norm(task)), None)
        sample = self.samples.get(scenario["id"]) if scenario else None
        utility = self._by_name(self.pool["utility"])
        steps: list[dict] = []
        answer_kind, fallback_reason = ("live" if self.live else "sample"), None

        cls = None
        if self.live:
            try:
                cls, step = self._live_classify(task, utility)
                steps.append(step)
            except Exception as e:  # noqa: BLE001 -- fall back to rules, say so
                fallback_reason = f"classifier: {e}"
        if cls is None:
            if scenario:  # replay of the classifier call recorded for this sample
                cls = {"task_class": scenario["task_class"], "difficulty": sample["difficulty"] or "medium",
                       "confidence": 0.92, "source": "model"}
                tin = 800 + _tok(task)
                steps.append(self._step("classifier", utility, "low", tin, 28, 450))
            else:
                cls = classifier.classify(task)

        cell = resolve(self.policy, role, cls["task_class"])
        receipt = self._receipt_base(persona, role, cls, task)
        receipt["steps"] = steps
        budget = {"limit": budget_for(self.policy, role), "spent": round(self._spent_mtd().get(role, 0.0), 4)}
        pace = pacing.pace(budget["spent"], budget["limit"], self.today)
        pace["message"] = pacing.describe(pace, self.role_labels.get(role, role))
        base = {"persona": persona_id, "role": role, "task": task, "policy_version": self.version,
                "ladder": self.ladder, "answer_kind": answer_kind, "classification": cls, "budget": budget,
                "pacing": pace}

        if cell["blocked"]:
            receipt.update(plan=self._plan(resolve(self.policy, "default", cls["task_class"])),
                           applied=cell["layers"] + ["defaults"], status="blocked",
                           cost_usd=sum(s["cost_usd"] for s in steps), baseline_cost_usd=0.0,
                           start_tier="none", final_model="none", final_tier="none")
            self._record(receipt)
            return {**base, "status": "blocked", "receipt": receipt,
                    "message": f"role '{role}' is not allowed to run task class '{cls['task_class']}'"}
        if budget["spent"] >= budget["limit"] > 0 and not (override and override.get("kind") == "manager"):
            return {**base, "status": "budget",
                    "message": f"{self.role_labels.get(role, role)} have used ${budget['spent']:.2f} of their "
                               f"${budget['limit']:.2f} monthly budget. An admin can raise it on the Policy tab."}

        policy_ceiling = cell["ceiling"]
        if force_tier:
            if force_tier not in TIERS:
                raise ValueError(f"unknown tier {force_tier!r}")
            cell = {**cell, "floor": force_tier, "ceiling": force_tier,
                    "layers": [*cell["layers"], f"override.{override['kind'] if override else 'self'}"]}
            receipt["override"] = override or {"kind": "self"}
        else:
            cell = pacing.apply(cell, pace)
        plan = self._plan(cell)
        start = clamp_tier(cls["difficulty"], cell["floor"], cell["ceiling"])
        receipt.update(plan=plan, applied=cell["layers"], start_tier=start)
        needs = self._needs(cls, sample, task)

        answer, attempts = None, []
        if self.live:
            try:
                answer, attempts = self._live_cascade(task, cell, start, steps, utility)
            except Exception as e:  # noqa: BLE001
                fallback_reason = f"{type(e).__name__}: {e}"
                answer_kind = "fallback"
                steps[:] = [s for s in steps if s["role"] == "classifier"]
        if answer is None:
            answer = sample["answer"] if sample else self._sim_answer(persona, cls, cell, start)
            attempts = self._sim_cascade(cell, start, needs, sample, steps, utility, answer, task)

        final = attempts[-1]
        worker = [s for s in steps if s["role"] == "worker"][-1]
        baseline = self._by_name(self.pool["baseline"])
        receipt.update(
            final_model=final["model"], final_tier=final["tier"],
            escalations=len(attempts) - 1, attempts=attempts, status="ok",
            cost_usd=sum(s["cost_usd"] for s in steps),
            baseline_model=baseline["name"],
            baseline_cost_usd=self._price(baseline, worker["in_tokens"], worker["out_tokens"]),
            passed=final["passed"], score=final["score"],
            latency_ms=sum(s["latency_ms"] for s in steps),
        )
        self._record(receipt)
        self._asks[receipt["id"]] = {"persona": persona_id, "task": task, "role": role,
                                     "task_class": cls["task_class"], "policy_ceiling": policy_ceiling,
                                     "worker": worker}
        out = {**base, "answer_kind": answer_kind, "status": "ok", "answer": answer, "receipt": receipt}
        if fallback_reason and answer_kind == "fallback":
            out["fallback_reason"] = fallback_reason
        if compare:
            out["compare"] = self._compare(task, answer, sample, worker)
        return out

    def _plan(self, cell: dict) -> dict[str, Any]:
        return {k: cell[k] for k in ("floor", "ceiling", "effort", "execution", "verify")} | {
            "max_output": self.policy["defaults"].get("max_output", 4096)}

    def _needs(self, cls: dict, sample: dict | None, task: str) -> str:
        if sample and sample.get("fail_issues"):
            return sample["min_tier"]
        if sample:
            return sample["difficulty"] or "medium"
        # Free text: most tasks are answerable at their difficulty; about 1 in 8 needs one tier more.
        bump = int(hashlib.sha256(task.encode()).hexdigest(), 16) % 8 == 0
        return TIERS[min(2, tier_index(cls["difficulty"]) + bump)]

    def _step(self, role: str, model: dict, effort: str, tin: int, tout: int, latency: int) -> dict[str, Any]:
        return {"role": role, "model": model["name"], "tier": model["tier"], "effort": effort,
                "in_tokens": tin, "out_tokens": tout, "cost_usd": self._price(model, tin, tout), "latency_ms": latency}

    def _sim_cascade(self, cell, start, needs, sample, steps, utility, answer, task) -> list[dict]:
        attempts: list[dict] = []
        tier = start
        tin = 220 + _tok(task)
        tout = int(_tok(answer) * EFFORT_OUT[cell["effort"]] * (traffic.THINKING_OUT if cell["execution"] == "thinking" else 1))
        while True:
            m = self._model(tier)
            if cell["execution"] == "cascade":
                steps.append(self._step("thinker", utility, "low", 180 + _tok(task), 46, 540))
            steps.append(self._step("worker", m, cell["effort"], tin, tout, 400 + tout * (3 if tier == "large" else 2)))
            if not cell["verify"]:
                attempts.append({"model": m["name"], "tier": tier, "effort": cell["effort"], "passed": None,
                                 "score": None, "issues": "", "error": ""})
                break
            ok = tier_index(tier) >= tier_index(needs)
            steps.append(self._step("verifier", utility, "low", tin + tout, 16 if ok else 72, 380 if ok else 700))
            issue = "" if ok else ((sample or {}).get("fail_issues", {}).get(tier)
                                   or f"Verifier: this {needs}-difficulty task needs a stronger model than {m['name']}.")
            score = ((sample or {}).get("pass_score") or 90) if ok else 45
            attempts.append({"model": m["name"], "tier": tier, "effort": cell["effort"], "passed": ok,
                             "score": score, "issues": issue, "error": ""})
            if ok or tier == cell["ceiling"]:
                break
            tier = TIERS[tier_index(tier) + 1]
        return attempts

    def _sim_answer(self, persona: dict, cls: dict, cell: dict, start: str) -> str:
        m = self._model(start)
        return (f"**Simulated answer.** The demo is running offline, so no model was called. "
                f"In live mode, {persona['name']}'s request would be answered by **{m['name']}** at "
                f"{cell['effort']} effort ({cell['execution'].replace('thinking', 'extended thinking')}).\n\n"
                f"Everything else on this page is real: the request was classified as "
                f"`{cls['task_class']}`, checked against policy v{self.version} for the "
                f"`{persona['role']}` role, routed, and priced on the receipt.\n\n"
                f"To get real answers, set `OPENROUTER_API_KEY` and run `switchllm demo --live`.")

    def _compare(self, task: str, answer: str, sample: dict | None, worker: dict) -> dict[str, Any]:
        baseline = self._by_name(self.pool["baseline"])
        if self.live:
            try:
                text, tin, tout, lat = self.live.chat(baseline["model_id"], [{"role": "user", "content": task}], "high")
                scores = self._live_scores(task, answer, text)
                return {"answer": text, "model": baseline["name"], "effort": "high",
                        "cost_usd": self._price(baseline, tin, tout), "latency_ms": lat,
                        "in_tokens": tin, "out_tokens": tout, "scores": scores}
            except Exception as e:  # noqa: BLE001
                return {"error": f"{type(e).__name__}: {e}"}
        c = (sample or {}).get("compare") or {}
        text = c.get("answer") or "*Simulated frontier answer: the demo is offline, so only the cost is modelled.*"
        tin, tout = worker["in_tokens"], int(worker["out_tokens"] / EFFORT_OUT[worker["effort"]] * EFFORT_OUT["high"])
        return {"answer": text, "model": baseline["name"], "effort": "high",
                "cost_usd": self._price(baseline, tin, tout), "latency_ms": 2000 + tout * 12,
                "in_tokens": tin, "out_tokens": tout, "scores": c.get("scores")}

    # -- live -----------------------------------------------------------------------

    def _live_classify(self, task: str, utility: dict) -> tuple[dict, dict]:
        classes = ", ".join(known_classes(self.policy))
        prompt = (f"Classify the request into exactly one task class from: {classes}.\n"
                  "Rate difficulty small (routine), medium, or large (expert-level).\n"
                  'Reply with JSON only: {"task_class": "...", "difficulty": "...", "confidence": 0.0}\n\n'
                  f"Request:\n{task}")
        text, tin, tout, lat = self.live.chat(utility["model_id"], [{"role": "user", "content": prompt}], "low", 200)
        d = parse_json(text)
        if d.get("task_class") not in known_classes(self.policy) or d.get("difficulty") not in TIERS:
            raise ValueError("classifier returned an unknown label")
        cls = {"task_class": d["task_class"], "difficulty": d["difficulty"],
               "confidence": float(d.get("confidence", 0.8)), "source": "model"}
        return cls, self._step("classifier", utility, "low", tin, tout, lat)

    def _live_cascade(self, task, cell, start, steps, utility) -> tuple[str, list[dict]]:
        attempts: list[dict] = []
        tier = start
        effort = cell["effort"] if cell["execution"] != "thinking" else "high"
        while True:
            m = self._model(tier)
            text, tin, tout, lat = self.live.chat(m["model_id"], [{"role": "user", "content": task}], effort,
                                                  self.policy["defaults"].get("max_output", 4096))
            steps.append(self._step("worker", m, cell["effort"], tin, tout, lat))
            if not cell["verify"]:
                attempts.append({"model": m["name"], "tier": tier, "effort": cell["effort"], "passed": None,
                                 "score": None, "issues": "", "error": ""})
                return text, attempts
            verdict, vstep = self._live_verify(task, text, utility)
            steps.append(vstep)
            attempts.append({"model": m["name"], "tier": tier, "effort": cell["effort"], "passed": verdict["pass"],
                             "score": verdict["score"], "issues": verdict["issues"], "error": ""})
            if verdict["pass"] or tier == cell["ceiling"]:
                return text, attempts
            tier = TIERS[tier_index(tier) + 1]

    def _live_verify(self, task: str, answer: str, utility: dict) -> tuple[dict, dict]:
        prompt = ("You are a strict reviewer. Check the answer for correctness and completeness.\n"
                  'Reply with JSON only: {"pass": true|false, "score": 0-100, "issues": "one sentence or empty"}\n\n'
                  f"Request:\n{task}\n\nAnswer:\n{answer}")
        text, tin, tout, lat = self.live.chat(utility["model_id"], [{"role": "user", "content": prompt}], "low", 300)
        d = parse_json(text)
        verdict = {"pass": bool(d.get("pass")), "score": int(d.get("score", 0)), "issues": str(d.get("issues", ""))}
        return verdict, self._step("verifier", utility, "low", tin, tout, lat)

    def _live_scores(self, task: str, routed: str, frontier: str) -> dict[str, int]:
        utility = self._by_name(self.pool["utility"])
        return {"route": self._live_verify(task, routed, utility)[0]["score"],
                "frontier": self._live_verify(task, frontier, utility)[0]["score"]}

    # -- receipts -------------------------------------------------------------------

    def _receipt_base(self, persona: dict, role: str, cls: dict, task: str) -> dict[str, Any]:
        self._counter += 1
        rid = hashlib.sha256(f"{self._counter}|{task}|{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        return {"id": rid, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tenant": "demo",
                "user": persona["name"], "role": role, "task_class": cls["task_class"],
                "difficulty": cls["difficulty"], "classifier_source": cls["source"], "escalations": 0,
                "attempts": [], "passed": None, "score": None, "latency_ms": 0,
                "task_preview": task[:117] + "…" if len(task) > 120 else task,
                "pool": self.pool["name"], "mode": "live" if self.live else "simulated"}

    def _record(self, receipt: dict) -> None:
        self.session.append(receipt)

    # -- feedback and overrides -----------------------------------------------------------

    def _receipt(self, receipt_id: str) -> dict:
        rec = next((r for r in self.session if r["id"] == receipt_id), None)
        if rec is None or receipt_id not in self._asks:
            raise ValueError("unknown or expired request; route it again")
        return rec

    def _override_option(self, receipt_id: str) -> dict[str, Any] | None:
        """The next tier up for a disliked answer, and whether it needs a manager."""
        rec, asked = self._receipt(receipt_id), self._asks[receipt_id]
        if rec["final_tier"] not in TIERS or rec["final_tier"] == TIERS[-1]:
            return None
        tier = TIERS[tier_index(rec["final_tier"]) + 1]
        model = self._model(tier)
        role = asked["role"]
        spent = self._spent_mtd().get(role, 0.0)
        cap = budget_for(self.policy, role)
        pace = pacing.pace(spent, cap, self.today)
        label = self.role_labels.get(role, role)
        if tier_index(tier) > tier_index(asked["policy_ceiling"]):
            why = f"{model['name']} is above what policy allows {label} for this task"
        elif spent >= cap > 0:
            why = f"{label} have used their ${cap:,.2f} monthly budget"
        elif pace["action"] != "none":
            why = f"{label} are on pace to exceed their ${cap:,.2f} cap"
        else:
            why = ""
        w = asked["worker"]
        return {"tier": tier, "model": model["name"], "needs_approval": bool(why),
                "why": why or f"within the {label} policy",
                "est_cost_usd": self._price(model, w["in_tokens"], w["out_tokens"])}

    def feedback(self, receipt_id: str, liked: bool, comment: str = "") -> dict[str, Any]:
        rec = self._receipt(receipt_id)
        rec["feedback"] = {"liked": bool(liked), "comment": comment[:500]}
        option = None if liked else self._override_option(receipt_id)
        return {"receipt_id": receipt_id, "liked": bool(liked), "option": option,
                "message": "" if liked or option else "This answer already came from the strongest model."}

    def request_override(self, receipt_id: str, reason: str = "") -> dict[str, Any]:
        option = self._override_option(receipt_id)
        if option is None:
            raise ValueError("this answer already came from the strongest model")
        asked, rec = self._asks[receipt_id], self._receipt(receipt_id)
        if not option["needs_approval"]:
            run = self.ask(asked["persona"], asked["task"], force_tier=option["tier"],
                           override={"kind": "self", "by": rec["user"], "reason": reason, "from": receipt_id})
            return {"status": "done", "run": run}
        if any(a["receipt_id"] == receipt_id and a["status"] == "pending" for a in self.approvals):
            raise ValueError("a manager is already reviewing this request")
        self._counter += 1
        approval = {"id": hashlib.sha256(f"appr|{self._counter}|{receipt_id}".encode()).hexdigest()[:10],
                    "receipt_id": receipt_id, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "user": rec["user"], "persona": asked["persona"], "role": asked["role"],
                    "task_class": asked["task_class"], "task": rec["task_preview"],
                    "from_model": rec["final_model"], "from_tier": rec["final_tier"],
                    "to_model": option["model"], "to_tier": option["tier"], "why": option["why"],
                    "est_cost_usd": option["est_cost_usd"], "reason": reason[:500], "status": "pending"}
        self.approvals.append(approval)
        return {"status": "pending", "approval": approval}

    def approvals_view(self) -> list[dict[str, Any]]:
        return list(reversed(self.approvals))

    def decide(self, approval_id: str, approve: bool, approver: str = "Manager") -> dict[str, Any]:
        approval = next((a for a in self.approvals if a["id"] == approval_id), None)
        if approval is None:
            raise ValueError("unknown approval request")
        if approval["status"] != "pending":
            raise ValueError(f"already {approval['status']}")
        approval.update(status="approved" if approve else "denied", decided_by=approver,
                        decided_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        out: dict[str, Any] = {"approval": approval}
        if approve:
            asked = self._asks[approval["receipt_id"]]
            out["run"] = self.ask(asked["persona"], asked["task"], force_tier=approval["to_tier"],
                                  override={"kind": "manager", "by": approver, "reason": approval["reason"],
                                            "from": approval["receipt_id"], "approval": approval_id})
            approval["run_receipt"] = out["run"]["receipt"]["id"]
        return out

    # -- policy -----------------------------------------------------------------------

    def _matrix(self, policy: dict) -> dict[str, Any]:
        spent = self._spent_mtd(policy)
        rows = []
        for role in ROLE_ORDER:
            r = policy["roles"].get(role, {})
            rows.append({"role": role, "description": r.get("description", ""),
                         "budget_usd_month": budget_for(policy, role), "label": self.role_labels.get(role, role),
                         "spent_mtd": round(spent.get(role, 0.0), 2),
                         "cells": {c: resolve(policy, role, c) for c in MATRIX_CLASSES}})
        return {"classes": MATRIX_CLASSES, "rows": rows}

    def policy_view(self) -> dict[str, Any]:
        return {"version": self.version, "yaml": to_yaml(self.policy), "matrix": self._matrix(self.policy),
                "history": self.history, "class_labels": {c: c.replace("_", " ") for c in MATRIX_CLASSES}}

    def preview(self, instruction: str) -> dict[str, Any]:
        parsed = parse_instruction(instruction, self.policy)
        out = {"instruction": instruction, **parsed}
        if not parsed["ok"]:
            return out
        after = merge_patch(self.policy, parsed["patch"])
        problems = [f"{role}: budget must be positive" for role, r in after["roles"].items()
                    if r.get("budget_usd_month", 1) <= 0]
        if problems:
            return {**out, "ok": False, "problems": problems}
        routes = []
        for role in ROLE_ORDER:
            for c in known_classes(after):
                b, a = resolve(self.policy, role, c), resolve(after, role, c)
                if b != a:
                    routes.append({"role": role, "task_class": c, "before": b, "after": a})
        budgets = [{"role": role, "before": budget_for(self.policy, role), "after": budget_for(after, role)}
                   for role in ROLE_ORDER if budget_for(self.policy, role) != budget_for(after, role)]
        before_rows, after_rows = self._replay(self.policy), self._replay(after)
        by_role: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
        for req, b, a in zip(self.requests, before_rows, after_rows):
            acc = by_role[req.role]
            acc[0] += b["cost"]
            acc[1] += a["cost"]
            acc[2] += int(a["blocked"]) - int(b["blocked"])
        tb, ta = sum(v[0] for v in by_role.values()), sum(v[1] for v in by_role.values())
        return {**out, "diff": yaml_diff(self.policy, after), "impact": {"routes": routes, "budgets": budgets},
                "projection": {"before": round(tb, 2), "after": round(ta, 2), "delta": round(ta - tb, 2),
                               "delta_pct": round(100 * (ta - tb) / tb, 1) if tb else 0.0,
                               "requests": len(self.requests),
                               "roles": [{"role": r, "label": self.role_labels.get(r, r), "before": round(v[0], 2),
                                          "after": round(v[1], 2), "blocked_delta": v[2]}
                                         for r, v in by_role.items() if abs(v[0] - v[1]) >= 0.005 or v[2]]}}

    def apply(self, instruction: str) -> dict[str, Any]:
        pv = self.preview(instruction)
        if not pv["ok"]:
            raise ValueError("couldn't turn that into a policy change: " + "; ".join(pv.get("problems") or pv["unparsed"] or ["nothing recognised"]))
        self.policy = merge_patch(self.policy, pv["patch"])
        self.version += 1
        self.history.append({"version": self.version, "instruction": instruction, "actions": pv["actions"]})
        return {"version": self.version, "policy": self.policy_view(), "dashboard": self.dashboard()}

    # -- savings --------------------------------------------------------------------------

    def _replay(self, policy: dict) -> list[dict]:
        key = (self.today, self.seed, self.ladder, to_yaml(policy))
        with _CACHE_LOCK:
            if key in _REPLAYS:
                _REPLAYS.move_to_end(key)
                return _REPLAYS[key]
        rows = traffic.replay(policy, self.pool, self.requests)
        with _CACHE_LOCK:
            _REPLAYS[key] = rows
            while len(_REPLAYS) > _REPLAY_CACHE_SIZE:
                _REPLAYS.popitem(last=False)
        return rows

    def _spent_mtd(self, policy: dict | None = None) -> dict[str, float]:
        rows = self._replay(policy or self.policy)
        spent: dict[str, float] = defaultdict(float)
        for req, row in zip(self.requests, rows):
            d = self.days[req.day]
            if (d.year, d.month) == (self.today.year, self.today.month):
                spent[req.role] += row["cost"]
        for r in self.session:
            spent[r["role"]] += r["cost_usd"]
        return spent

    def dashboard(self) -> dict[str, Any]:
        rows = self._replay(self.policy)
        spent = self._spent_mtd()
        day_cost, day_base, day_n = [0.0] * len(self.days), [0.0] * len(self.days), [0] * len(self.days)
        roles: dict[str, dict] = {r: {"requests": 0, "cost": 0.0, "baseline": 0.0, "escalated": 0, "blocked": 0,
                                      "tiers": {t: 0 for t in TIERS}} for r in traffic.HEADCOUNT}
        classes: dict[str, dict] = defaultdict(lambda: {"requests": 0, "cost": 0.0, "baseline": 0.0,
                                                         "tiers": {t: 0 for t in TIERS}})
        verified = passed = 0
        for req, row in zip(self.requests, rows):
            day_cost[req.day] += row["cost"]
            day_base[req.day] += row["baseline"]
            day_n[req.day] += 1
            for acc in (roles[req.role], classes[req.task_class]):
                acc["requests"] += 1
                acc["cost"] += row["cost"]
                acc["baseline"] += row["baseline"]
                if row["tier"]:
                    acc["tiers"][row["tier"]] += 1
            roles[req.role]["escalated"] += int(row["escalations"] > 0)
            roles[req.role]["blocked"] += int(row["blocked"])
            if row["tier"]:
                verified += 1
                passed += int(tier_index(row["tier"]) >= tier_index(req.needs))
        for r in self.session:  # requests sent from the Ask view
            acc = roles[r["role"]]
            acc["requests"] += 1
            acc["cost"] += r["cost_usd"]
            acc["baseline"] += r["baseline_cost_usd"]
            if r["final_tier"] in TIERS:
                acc["tiers"][r["final_tier"]] += 1
            acc["blocked"] += int(r["status"] == "blocked")
            acc["escalated"] += int(r.get("escalations", 0) > 0)
            day_cost[-1] += r["cost_usd"]
            day_base[-1] += r["baseline_cost_usd"]
            day_n[-1] += 1

        series, cc, cb = [], 0.0, 0.0
        for i, d in enumerate(self.days):
            cc += day_cost[i]
            cb += day_base[i]
            series.append({"date": d.isoformat(), "cost": round(cc, 2), "baseline": round(cb, 2),
                           "day_cost": round(day_cost[i], 2), "day_baseline": round(day_base[i], 2), "requests": day_n[i]})
        total = sum(v["requests"] for v in roles.values())
        cost = sum(v["cost"] for v in roles.values())
        base = sum(v["baseline"] for v in roles.values())
        escalated = sum(v["escalated"] for v in roles.values())
        blocked = sum(v["blocked"] for v in roles.values())
        employees = sum(traffic.HEADCOUNT.values())
        pct = lambda c, b: round(100 * (1 - c / b), 1) if b else 0.0  # noqa: E731
        return {
            "window": {"from": self.days[0].isoformat(), "to": self.days[-1].isoformat(), "days": len(self.days),
                       "month": self.today.strftime("%Y-%m")},
            "employees": employees,
            "kpis": {"requests": total, "cost": round(cost, 2), "baseline": round(base, 2),
                     "saved": round(base - cost, 2), "saved_pct": pct(cost, base),
                     "escalation_rate": round(100 * escalated / total, 1) if total else 0.0,
                     "escalated": escalated, "blocked": blocked,
                     "verified_pass_rate": round(100 * passed / verified, 1) if verified else 0.0,
                     "per_employee_saved": round((base - cost) / employees, 2),
                     "liked": sum(1 for r in self.session if (r.get("feedback") or {}).get("liked") is True),
                     "disliked": sum(1 for r in self.session if (r.get("feedback") or {}).get("liked") is False),
                     "overrides": sum(1 for r in self.session if r.get("override")),
                     "pending_approvals": sum(1 for a in self.approvals if a["status"] == "pending")},
            "series": series,
            "roles": sorted(({"role": r, "label": self.role_labels.get(r, r), "headcount": traffic.HEADCOUNT[r],
                              "requests": v["requests"], "cost": round(v["cost"], 2), "baseline": round(v["baseline"], 2),
                              "saved_pct": pct(v["cost"], v["baseline"]), "tiers": v["tiers"],
                              "escalated": v["escalated"], "blocked": v["blocked"],
                              "budget": budget_for(self.policy, r), "spent_mtd": round(spent.get(r, 0.0), 2)}
                             for r, v in roles.items()), key=lambda x: -x["cost"]),
            "classes": sorted(({"task_class": c, "requests": v["requests"], "cost": round(v["cost"], 2),
                                "baseline": round(v["baseline"], 2), "saved_pct": pct(v["cost"], v["baseline"]),
                                "tiers": v["tiers"],
                                "floor": self.policy["task_classes"].get(c, {}).get("floor")}
                               for c, v in classes.items()), key=lambda x: -x["cost"])[:8],
            "tiers": {t: sum(v["tiers"][t] for v in roles.values()) for t in TIERS},
            "recent": [{"id": r["id"], "ts": r["ts"], "user": r["user"], "role": r["role"],
                        "task_class": r["task_class"], "status": r["status"], "model": r["final_model"],
                        "tier": r["final_tier"], "effort": r["plan"]["effort"], "execution": r["plan"]["execution"],
                        "escalations": r.get("escalations", 0), "cost": r["cost_usd"], "baseline": r["baseline_cost_usd"],
                        "saved_pct": pct(r["cost_usd"], r["baseline_cost_usd"]), "task": r["task_preview"],
                        "override": (r.get("override") or {}).get("kind")}
                       for r in reversed(self.session[-12:])],
            "pool": {"name": self.pool["name"], "baseline": self.pool["baseline"]},
            "policy_version": self.version,
        }

    # -- static export ------------------------------------------------------------------

    def export(self) -> dict[str, Any]:
        """Everything the page needs to replay offline, for every model ladder."""
        live, self.live = self.live, None  # exports are always reproducible simulations
        ladder = self.ladder
        try:
            out: dict[str, Any] = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "ladders": {}}
            for name in LADDERS:
                self.reset(name)
                runs = {}
                for s in self.scenarios:
                    for cmp in (1, 0):
                        runs[f"{s['persona']}|{s['id']}|{cmp}"] = self.ask(s["persona"], s["task"], bool(cmp))
                # Each sample was asked twice (with and without compare); keep one receipt per request.
                seen: dict[tuple, dict] = {}
                for r in self.session:
                    seen[(r["user"], r["task_preview"])] = r
                self.session = list(seen.values())
                out["ladders"][name] = {"meta": self.meta(), "runs": runs,
                                        "previews": {x: self.preview(x) for x in self.policy_examples},
                                        "policy": self.policy_view(), "dashboard": self.dashboard()}
            return out
        finally:
            self.live = live
            self.reset(ladder)
