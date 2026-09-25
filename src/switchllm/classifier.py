"""Cheap task classifier: task class, difficulty and whether tools are needed.

This is the heuristic router's front half. It is deliberately transparent --
every decision carries the signals that produced it -- and is the component
the learned router replaces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .catalog import approx_tokens

_CLASS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("legal", re.compile(
        r"\b(contracts?|clauses?|indemnif\w*|liabilit\w*|compliance|regulat\w*|gdpr|nda|terms of service)\b")),
    ("code", re.compile(
        r"```|\b(def|function|bugs?|stack ?trace|traceback|refactor\w*|compiles?|regex|sql|python|"
        r"javascript|typescript|endpoints?|unit tests?|debug\w*)\b")),
    ("math", re.compile(
        r"\b(prove|proof|integral|derivative|equations?|theorem|probability|calculate|solve for)\b")),
    ("summarize", re.compile(r"\b(summari[sz]e|summary|tl;?dr|recap|key points|condense)\b")),
    ("analysis", re.compile(
        r"\b(analy[sz]e|analysis|compare|evaluate|trade-?offs?|strategy|forecast|assess|"
        r"pros and cons|root cause)\b")),
    ("writing", re.compile(r"\b(draft|write|rewrite|emails?|memo|blog|proofread|tone)\b")),
]

_BASE_DIFFICULTY = {
    "chat": 1, "summarize": 1, "writing": 2, "analysis": 3, "code": 3, "math": 3, "legal": 3,
}
_BASE_OUTPUT_TOKENS = {
    "chat": 250, "summarize": 300, "writing": 500, "analysis": 700, "code": 800, "math": 500, "legal": 700,
}

_HARD = re.compile(
    r"\b(prove|optimi[sz]e|architect\w*|design an?|from scratch|edge cases|rigorous\w*|in depth|"
    r"step[- ]by[- ]step|multi-?step|concurren\w*|distributed|novel|race conditions?)\b")
_EASY = re.compile(r"\b(quick(ly)?|simple|brief(ly)?|one[- ]line|short|typo|tl;?dr)\b")
_TOOLS = re.compile(
    r"\b(search the web|browse|look up|latest news|fetch|run the (tests|code)|across the (repo|codebase)|"
    r"in our (repo|drive|crm|wiki))\b")


@dataclass(frozen=True)
class TaskProfile:
    task_class: str
    difficulty: int  # 1 (trivial) .. 5 (frontier-hard)
    needs_tools: bool
    input_tokens: int
    base_output_tokens: int
    signals: list[str] = field(default_factory=list)


def classify(prompt: str) -> TaskProfile:
    text = prompt.lower()
    signals: list[str] = []

    scores = {name: len(pat.findall(text)) for name, pat in _CLASS_PATTERNS}
    best = max(scores.values())
    # Ties resolve to the earlier (higher-stakes) class in _CLASS_PATTERNS.
    task_class = next((n for n, _ in _CLASS_PATTERNS if scores[n] == best), "chat") if best else "chat"
    signals.append(f"class={task_class}")

    difficulty = _BASE_DIFFICULTY[task_class]
    hard = {m.group(0) for m in _HARD.finditer(text)}
    easy = {m.group(0) for m in _EASY.finditer(text)}
    if hard:
        difficulty += min(2, len(hard))
        signals.append("hard:" + ",".join(sorted(hard)))
    if easy:
        difficulty -= 1
        signals.append("easy:" + ",".join(sorted(easy)))
    tokens = approx_tokens(prompt)
    if tokens > 1500:
        difficulty += 1
        signals.append(f"long input ({tokens} tok)")
    if text.count("?") >= 3:
        difficulty += 1
        signals.append("multi-part question")
    difficulty = max(1, min(5, difficulty))

    needs_tools = bool(_TOOLS.search(text))
    if needs_tools:
        signals.append("needs tools")

    return TaskProfile(
        task_class=task_class,
        difficulty=difficulty,
        needs_tools=needs_tools,
        input_tokens=tokens,
        base_output_tokens=_BASE_OUTPUT_TOKENS[task_class],
        signals=signals,
    )
