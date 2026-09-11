"""
Escalation gate: plain rules with structured reason codes.

AUTO only when ALL hold: known intent, model_score above threshold,
retrieval max-similarity above threshold, no risk-keyword hit, grounding
validator passed. Anything else -> HUMAN with a reason_code the eval can
aggregate (WEAK_EVIDENCE / LOW_SCORE / HIGH_RISK / GROUNDING_FAIL /
UNKNOWN_INTENT).

Thresholds are tuned on a calibration split, never on the final test split.
"""
from __future__ import annotations

from src import config


def decide(intent: str, model_score: float, max_sim: float,
           risk_hit: str | None, grounding_passed: bool,
           sim_thr: float | None = None,
           conf_thr: float | None = None,
           agreement: float | None = None,
           agree_thr: float = 2 / 3) -> dict:
    sim_thr = config.SIM_THRESHOLD_ESCALATE if sim_thr is None else sim_thr
    conf_thr = config.LOW_CONFIDENCE_ESCALATE if conf_thr is None else conf_thr
    if intent == "other_unclear":
        return {"decision": "human", "reason_code": "UNKNOWN_INTENT",
                "reason": "message does not fit a defined intent"}
    if risk_hit:
        return {"decision": "human", "reason_code": "HIGH_RISK",
                "reason": f"matched risk phrase: {risk_hit!r}"}
    if model_score < conf_thr:
        return {"decision": "human", "reason_code": "LOW_SCORE",
                "reason": f"model_score {model_score:.2f} below {conf_thr:.2f}"}
    if max_sim < sim_thr:
        return {"decision": "human", "reason_code": "WEAK_EVIDENCE",
                "reason": f"max similarity {max_sim:.2f} below {sim_thr:.2f}; no precedent"}
    if agreement is not None and agreement < agree_thr:
        return {"decision": "human", "reason_code": "WEAK_EVIDENCE",
                "reason": f"precedent agrees {agreement:.2f} < {agree_thr:.2f} "
                          f"with predicted intent; likely misclassified"}
    if not grounding_passed:
        return {"decision": "human", "reason_code": "GROUNDING_FAIL",
                "reason": "draft failed grounding validation"}
    return {"decision": "auto", "reason_code": "OK_AUTO",
            "reason": "confident intent, precedent found, low risk, grounded draft"}


def risk_scan(text: str) -> str | None:
    """Return a reason string if the message must escalate, else None.

    Checks (in priority order):
    1. Ultra-short text — too ambiguous to auto-handle safely
    2. Regex PII patterns (SSN, credit card, phone, email, password, license)
    3. Keyword list (legal threats, fraud, human demand, broken promises, sarcasm markers)
    """
    t = text or ""
    t_lower = t.lower()

    # 1. Ultra-short: < N non-whitespace chars can't be reliably intent-classified
    non_ws = len(t.replace(" ", "").replace("\t", "").replace("\n", ""))
    if non_ws < config.SHORT_TEXT_ESCALATE_CHARS:
        return f"ambiguous: ultra-short message ({non_ws} chars)"

    # 2. PII regex — always escalate regardless of intent/confidence
    for pattern, label in config._PII_PATTERNS:
        if pattern.search(t):
            return f"PII detected: {label}"

    # 3. Keywords
    for kw in config.ESCALATION_KEYWORDS:
        if kw.lower() in t_lower:
            return kw

    return None
