"""Tests for the decision/escalation path, grounding validator, judge parsing,
taxonomy schema, frozen split, and leakage invariants.

These are the modules a skeptical reviewer will poke: the gate decides who
gets an automated reply, so its logic must be pinned, not emergent.
"""
from __future__ import annotations

import json
from pathlib import Path

from src import config
from src.decide.gate import decide, risk_scan
from src.draft.reply import validate_grounding
from src.llm.gemini import extract_json

CASES = [
    {"case_id": 0, "customer_message": "where is my order",
     "brand_reply": "Sorry! Please DM your order number so we can help."},
    {"case_id": 1, "customer_message": "need a refund",
     "brand_reply": "Refunds are issued in 3-5 business days after we receive the item."},
]


def test_gate_unknown_intent_always_escalates():
    d = decide("other_unclear", 0.99, 0.99, None, True)
    assert d["decision"] == "human" and d["reason_code"] == "UNKNOWN_INTENT"


def test_gate_risk_overrides_everything():
    d = decide("refund_return", 0.99, 0.99, "lawyer", True)
    assert d["decision"] == "human" and d["reason_code"] == "HIGH_RISK"
    assert "reason" in d and d["reason"]


def test_gate_low_score_escalates():
    d = decide("delivery_delay", 0.10, 0.99, None, True, conf_thr=0.6)
    assert d["reason_code"] == "LOW_SCORE"


def test_gate_weak_similarity_escalates():
    d = decide("delivery_delay", 0.99, 0.10, None, True, sim_thr=0.6)
    assert d["reason_code"] == "WEAK_EVIDENCE"


def test_gate_disagreement_escalates_even_when_confident():
    d = decide("refund_return", 0.95, 0.90, None, True, agreement=0.0)
    assert d["decision"] == "human" and d["reason_code"] == "WEAK_EVIDENCE"


def test_gate_full_agreement_autos():
    d = decide("refund_return", 0.95, 0.90, None, True, agreement=1.0)
    assert d["decision"] == "auto" and d["reason_code"] == "OK_AUTO"


def test_gate_grounding_fail_escalates():
    d = decide("refund_return", 0.95, 0.90, None, False, agreement=1.0)
    assert d["reason_code"] == "GROUNDING_FAIL"


def test_gate_every_decision_has_reason():
    import itertools
    for kw in itertools.product(
            ["delivery_delay", "other_unclear"], [0.0, 0.9], [0.0, 0.9],
            [None, "sue"], [True, False]):
        d = decide(kw[0], kw[1], kw[2], kw[3], kw[4], agreement=0.5)
        assert d["decision"] in ("auto", "human") and d["reason"], kw


def test_risk_scan_ignores_routine_refund_but_catches_fraud():
    assert risk_scan("where is my refund, it has been 20 days") is None
    assert risk_scan("I filed a fraud dispute with my bank") is not None
    assert risk_scan("I will have my lawyer contact you") is not None


def test_validator_passes_cited_evidence():
    g = validate_grounding("Please DM your order number so we can help.",
                           CASES, [0])
    assert g["passed"] is True


def test_validator_rejects_unknown_case_id():
    g = validate_grounding("Please DM your order number.", CASES, [99])
    assert g["passed"] is False


def test_validator_rejects_unsupported_promise():
    g = validate_grounding("Your refund will arrive in 3 days, guaranteed.",
                           CASES, [0])
    assert g["passed"] is False


def test_validator_accepts_verbatim_promise():
    g = validate_grounding("Refunds are issued in 3-5 business days after we receive the item.",
                           CASES, [1])
    assert g["passed"] is True


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": 1} thanks') == {"a": 1}
    assert extract_json('{"a": 1,}') == {"a": 1}


def test_extract_json_raises_on_garbage():
    try:
        extract_json("no json here at all!!!")
    except ValueError:
        return
    raise AssertionError("should have raised")


def test_intent_schema_matches_config():
    schema = json.loads((config.ROOT / "data" / "intent_schema.json").read_text(encoding="utf-8"))
    names = [i["name"] for i in schema["intents"]]
    assert set(names) == set(config.INTENTS), (set(names) ^ set(config.INTENTS))
    for i in schema["intents"]:
        for field in ("definition", "include", "exclude", "examples"):
            assert i[field], (i["name"], field)


def test_split_frozen_covers_golden_exactly():
    gold_ids = {json.loads(l)["id"] for l in
                open(config.DATA_GOLDEN / "golden.jsonl", encoding="utf-8")}
    split = json.loads((config.DATA_GOLDEN / "split.json").read_text(encoding="utf-8"))
    assert set(split["cal_ids"]).isdisjoint(split["test_ids"])
    assert set(split["cal_ids"]) | set(split["test_ids"]) == gold_ids
    assert len(split["cal_ids"]) == 50 and len(split["test_ids"]) == 100


def test_no_golden_text_in_retrieval_corpus():
    import pandas as pd
    gold_texts = {json.loads(l)["text"].lower().strip() for l in
                  open(config.DATA_GOLDEN / "golden.jsonl", encoding="utf-8")}
    cases = pd.read_parquet(config.DATA_SAMPLE / "retrieval" / "cases.parquet")
    overlap = gold_texts & {str(t).lower().strip() for t in cases["customer_message"].fillna("")}
    assert not overlap, f"{len(overlap)} golden texts in corpus"


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in fns:
        fn()
        print(f"  PASS  {name}")
    print(f"\n{len(fns)}/{len(fns)} pipeline tests passed")
