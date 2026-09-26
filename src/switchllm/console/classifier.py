"""Keyword task classifier for the console (the live mode uses a model instead)."""

from __future__ import annotations

import re

from .policy import TIERS, TYPICAL_DIFFICULTY

# Ordered: earlier, higher-stakes classes win ties.
_RULES: list[tuple[str, str]] = [
    ("medical_or_safety", r"\b(diagnos\w*|dosage|symptom\w*|patient|safety incident)\b"),
    ("legal_clause_analysis", r"\b(clause|liabilit\w*|indemnif\w*|contract|warrant(y|ies)|governing law)\b"),
    ("compliance_check", r"\b(complian\w*|gdpr|hipaa|sox|kyc|aml|regulat\w*)\b"),
    ("financial_analysis", r"\b(valuation|ebitda|enterprise value|dcf|equity value|multiple|balance sheet)\b"),
    ("code_review", r"\b(review this|security issues?|vulnerab\w*|code review|sql injection)\b|cur\.execute"),
    ("code_debug", r"\b(bug|traceback|stack ?trace|exception|error:|debug\w*|fix (this|the) code)\b"),
    ("code_generate", r"\b(write a (python|javascript|typescript|sql|go|rust|java)?\s*(function|script|class|query)|implement|unit tests?)\b"),
    ("code_explain", r"\b(what does (this|the) (regex|code|function)|explain (this|the) (code|regex|function)|regex)\b"),
    ("architecture_design", r"\b(architecture|system design|design a (system|service)|scalab\w*)\b"),
    ("quantitative_reasoning", r"\b(sharpe|ratio|calculate|compute|how much|percent\w*|pro-?rated|refund|volatility|return(ed|s)?)\b"),
    ("math_proof", r"\b(prove|proof|theorem|lemma)\b"),
    ("strategic_analysis", r"\b(strategy|strategic|market entry|competitive position\w*|should we)\b"),
    ("research_synthesis", r"\b(literature|research|studies|synthesi[sz]e)\b"),
    ("customer_escalation", r"\b(social media|lawyer|dispute|furious|angry|escalat\w*|unless)\b"),
    ("customer_reply", r"\b(customer|reply to|charged|invoice|refund request|ticket)\b"),
    ("summarize", r"\b(summari[sz]e|summary|tl;?dr|recap|key points|in (one|two) sentences?)\b"),
    ("draft_email", r"\b(email|e-mail)\b"),
    ("translate", r"\b(translate|translation|in (spanish|french|german|hindi|japanese))\b"),
    ("rewrite_edit", r"\b(rewrite|proofread|edit this|rephrase|tone)\b"),
    ("data_extraction", r"\b(extract|pull out|parse)\b"),
    ("classify_label", r"\b(classify|categori[sz]e|label)\b"),
    ("brainstorm", r"\b(brainstorm|ideas for|suggest (some|a few))\b"),
    ("draft_document", r"\b(draft|write (a|an) (memo|report|proposal|document|plan))\b"),
    ("qa_factual", r"^(what|who|when|where|which) (is|are|was|were)\b"),
]
_COMPILED = [(name, re.compile(p, re.I)) for name, p in _RULES]
_HARD = re.compile(r"\b(step by step|rigorous\w*|in depth|edge cases|optimi[sz]e|trade-?offs?|stress[- ]test)\b", re.I)
_EASY = re.compile(r"\b(quick(ly)?|brief(ly)?|one (line|sentence)|short|simple)\b", re.I)


def classify(text: str) -> dict:
    scores = {name: len(p.findall(text)) for name, p in _COMPILED}
    best = max(scores.values())
    task_class = next((n for n, _ in _RULES if scores[n] == best), "general_chat") if best else "general_chat"
    d = TIERS.index(TYPICAL_DIFFICULTY.get(task_class, "medium"))
    if _HARD.search(text) or len(text) > 2500:
        d += 1
    if _EASY.search(text):
        d -= 1
    confidence = 0.55 if not best else min(0.9, 0.62 + 0.1 * best)
    return {"task_class": task_class, "difficulty": TIERS[max(0, min(2, d))],
            "confidence": round(confidence, 2), "source": "rules"}
