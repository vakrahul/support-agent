"""Interactive & diagnostic inspector for the AmazonHelp AI support agent.

Usage:
    # Inspect specific golden test cases:
    python scripts/demo.py --id amz-001   # Sarcasm / Precedent conflict -> ESCALATE
    python scripts/demo.py --id amz-000   # Fraud / Dispute claim -> ESCALATE (HIGH_RISK)
    python scripts/demo.py --id amz-003   # Defective item -> AUTO-HANDLE (OK_AUTO)
    python scripts/demo.py --id amz-007   # Vague query -> ESCALATE (UNKNOWN_INTENT)

    # Interactive sample testing:
    python scripts/demo.py --interactive

    # List recommended test cases across all intents:
    python scripts/demo.py --list
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Handle Windows terminal Unicode / emoji safely
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import config
from src.classify.classifier import LLMClassifier
from src.decide.gate import decide, risk_scan
from src.draft.reply import rag_draft, validate_grounding
from src.retrieve.index import Retriever
from src.llm.gemini import CacheMiss

GOLDEN_PATH = config.DATA_GOLDEN / "golden.jsonl"


def load_golden_lookup() -> dict[str, dict]:
    if not GOLDEN_PATH.exists():
        return {}
    return {json.loads(l)["id"]: json.loads(l) for l in open(GOLDEN_PATH, encoding="utf-8")}


def run_pipeline(text: str, retriever: Retriever, classifier: LLMClassifier, mode: str = "replay") -> dict:
    # 1. Model prediction
    preds = classifier.predict([text], mode=mode)
    intent, score, top3 = preds[0]

    # 2. Retrieval of historical precedents
    cases = retriever.search(text, top_k=3)
    sims = [c["similarity"] for c in cases]
    max_sim = max(sims, default=0.0)
    avg_sim = sum(sims) / max(len(sims), 1)

    from scripts.build_golden import RULES as _RULES
    def _rule_label(msg: str) -> str:
        t = msg or ""
        fired = [name for name, rx in _RULES if rx.search(t)]
        return fired[0] if fired else "other_unclear"

    for c in cases:
        c["precedent_intent"] = _rule_label(c["customer_message"])
        c["supports"] = (c["precedent_intent"] == intent)

    matching_hits = sum(1 for c in cases if c["supports"])
    total_hits = max(len(cases), 1)
    agreement = matching_hits / total_hits

    # Evidence status classification
    if max_sim < config.SIM_THRESHOLD_ESCALATE:
        evidence_status = "WEAK_SIMILARITY"
    elif agreement < (2 / 3):
        evidence_status = "CONFLICTED"
    else:
        evidence_status = "SUPPORTIVE"

    # 3. Grounded reply draft & code validation
    d = rag_draft(text, intent, cases, mode=mode)
    g = validate_grounding(d["draft"], cases, d["evidence_ids"])

    # 4. Risk phrase scan
    risk = risk_scan(text)

    # 5. Escalation gate policy
    sim_thr = config.SIM_THRESHOLD_ESCALATE
    conf_thr = config.LOW_CONFIDENCE_ESCALATE
    agree_thr = 2 / 3

    policy_checks = [
        ("Known Intent", intent != "other_unclear", f"intent={intent!r}"),
        ("Model Score", score >= conf_thr, f"{score:.2f} >= {conf_thr:.2f}"),
        ("Max Similarity", max_sim >= sim_thr, f"{max_sim:.3f} >= {sim_thr:.2f}"),
        ("Precedent Agreement", agreement >= agree_thr, f"{agreement:.2f} >= {agree_thr:.2f} ({matching_hits}/{total_hits})"),
        ("Grounding Passed", g["passed"], f"score={g['score']:.2f} ({g['reason']})"),
        ("No Risk Trigger", risk is None, f"trigger={risk!r}" if risk else "clean"),
    ]

    dec = decide(
        intent, score, max_sim, risk, g["passed"],
        sim_thr=sim_thr, conf_thr=conf_thr, agreement=agreement, agree_thr=agree_thr,
    )

    return {
        "text": text,
        "intent": intent,
        "score": score,
        "top3": top3,
        "max_sim": max_sim,
        "avg_sim": avg_sim,
        "matching_hits": matching_hits,
        "total_hits": total_hits,
        "agreement": agreement,
        "evidence_status": evidence_status,
        "retrieved_cases": cases,
        "draft": d["draft"],
        "evidence_ids": d["evidence_ids"],
        "grounding_passed": g["passed"],
        "grounding_score": g["score"],
        "grounding_reason": g["reason"],
        "risk": risk,
        "policy_checks": policy_checks,
        "decision": dec["decision"],
        "reason_code": dec["reason_code"],
        "reason": dec["reason"],
    }


def print_result(res: dict, sample_id: str | None = None):
    header = f"[{sample_id}]" if sample_id else ""
    print("\n" + "=" * 68)
    print(f"CUSTOMER QUERY {header}:")
    print(f"  \"{res['text']}\"")
    print("=" * 68)

    # 1. Model prediction
    print("\n1. MODEL PREDICTION")
    print(f"  Intent                : {res['intent']}")
    print(f"  Model score           : {res['score']:.2f}")
    print(f"  Confidence calibrated : NO (uncalibrated LLM score, not a probability)")
    print(f"  Top-3 Ranking         : {', '.join(res['top3'])}")

    # 2. Decision signals
    print("\n2. DECISION SIGNALS")
    print(f"  Intent agreement      : {res['agreement']:.2f} / 1.00 ({res['matching_hits']}/{res['total_hits']} precedents agree)")
    print(f"  Max retrieval sim     : {res['max_sim']:.3f}")
    print(f"  Avg retrieval sim     : {res['avg_sim']:.3f}")
    print(f"  Grounding score       : {res['grounding_score']:.2f} / 5.00 ({'PASS' if res['grounding_passed'] else 'FAIL - ' + res['grounding_reason']})")
    risk_disp = f"TRIGGERED ({res['risk']!r})" if res['risk'] else "NONE (no fraud, legal, distress keywords)"
    print(f"  Risk trigger          : {risk_disp}")
    print(f"  Evidence status       : {res['evidence_status']}")

    # 3. Policy check
    print("\n3. POLICY CHECK (AUTO-HANDLE vs ESCALATE)")
    all_passed = True
    for name, ok, detail in res["policy_checks"]:
        mark = "✓" if ok else "✗"
        if not ok:
            all_passed = False
            print(f"  [{mark}] {name:<21} : {detail}  <-- TRIPPED")
        else:
            print(f"  [{mark}] {name:<21} : {detail}")
    
    dec_tag = "AUTO-HANDLE" if res["decision"] == "auto" else "ESCALATE TO HUMAN"
    print(f"\n  => GATE DECISION       : [{dec_tag}]")
    print(f"  => REASON CODE         : {res['reason_code']}")
    print(f"  => EXPLANATION         : {res['reason']}")

    # 4. Retrieval evidence
    print(f"\n4. RETRIEVAL EVIDENCE (Top-{len(res['retrieved_cases'])} Historical Precedents)")
    for i, c in enumerate(res["retrieved_cases"], 1):
        sup_str = "YES [✓]" if c["supports"] else "NO [✗]"
        c_text = (c["customer_message"] or "").replace("\n", " ")
        b_reply = (c["brand_reply"] or "").replace("\n", " ")
        if len(c_text) > 78:
            c_text = c_text[:78] + "..."
        if len(b_reply) > 78:
            b_reply = b_reply[:78] + "..."
        print(f"  #{i}  similarity={c['similarity']:.3f} | intent={c['precedent_intent']} | supports predicted? {sup_str}")
        print(f"      Customer : \"{c_text}\"")
        print(f"      Reply    : \"{b_reply}\"")

    # 5. Drafted reply
    print("\n5. DRAFTED REPLY")
    print(f"  \"{res['draft']}\"")

    # 6. Final outcome summary card
    ground_str = "PASS" if res['grounding_passed'] else "FAIL"
    risk_str = res['risk'] if res['risk'] else "NONE"
    dec_upper = "AUTO-HANDLE" if res['decision'] == "auto" else "ESCALATE"
    ev_desc = f"{res['evidence_status']} ({res['agreement']:.2f} agreement)"

    print("\n" + "╔" + "═" * 66 + "╗")
    print(f"║ {'FINAL OUTCOME SUMMARY':<64} ║")
    print("╟" + "─" * 66 + "╢")
    print(f"║ Intent       : {res['intent']:<47} ║")
    print(f"║ Evidence     : {ev_desc:<47} ║")
    print(f"║ Grounding    : {ground_str:<47} ║")
    print(f"║ Risk         : {risk_str:<47} ║")
    print(f"║ Decision     : {dec_upper:<47} ║")
    print(f"║ Reason Code  : {res['reason_code']:<47} ║")
    print("╚" + "═" * 66 + "╝\n")


FEATURED_SAMPLES = [
    ("amz-000", "refund_return", "Fraud / blocked card claim (HIGH_RISK trigger -> Escalate)"),
    ("amz-001", "missing_parcel_tracking", "Sarcastic missing delivery (Agreement 0.00 -> Escalate)"),
    ("amz-002", "device_app_account", "Threatening account AZ claim (Risk trigger 'threat' -> Escalate)"),
    ("amz-003", "refund_return", "Defective item replacement (Good precedent -> Auto-handle)"),
    ("amz-004", "refund_return", "Cashback question on return (Narrowed policy -> Auto-handle)"),
    ("amz-005", "delivery_delay", "Courier status complaint (Delay precedent -> Auto-handle)"),
    ("amz-007", "other_unclear", "Vague query 'How many more days?' (UNKNOWN_INTENT -> Escalate)"),
    ("amz-009", "refund_return", "Order tracking stuck + 'fraud' tag (HIGH_RISK -> Escalate)"),
    ("amz-022", "device_app_account", "Device Echo/Kindle question (Device precedent -> Escalate)"),
    ("amz-067", "order_status_general", "Cancelled Prime Now order without reason (Status -> Escalate)"),
]


def main():
    parser = argparse.ArgumentParser(description="AmazonHelp AI Support Agent Inspector")
    parser.add_argument("query", nargs="?", type=str, help="Customer query text")
    parser.add_argument("--id", type=str, help="Golden sample ID (e.g. amz-000, amz-001, amz-003)")
    parser.add_argument("--sample-random", action="store_true", help="Pick a random golden test sample")
    parser.add_argument("--interactive", action="store_true", help="Run interactive inspection loop")
    parser.add_argument("--list", action="store_true", help="List recommended test cases across all intents")
    parser.add_argument("--live", action="store_true", help="Use live API calls rather than replay cache")
    args = parser.parse_args()

    if args.list:
        print("\n" + "=" * 78)
        print("RECOMMENDED GOLDEN TEST QUERIES FOR DEMO:")
        print("=" * 78)
        for sid, intent, desc in FEATURED_SAMPLES:
            print(f"  • python scripts/demo.py --id {sid:<8} | Intent: {intent:<22} | {desc}")
        print("=" * 78 + "\n")
        return 0

    mode = "live" if args.live else "replay"
    goldens = load_golden_lookup()

    if args.id:
        if args.id not in goldens:
            print(f"Error: Unknown sample ID {args.id!r}. Choose from amz-000 to amz-149.")
            return 1
        sample = goldens[args.id]
        query = sample["text"]
        sample_id = args.id
    elif args.sample_random:
        sample_id = random.choice(list(goldens.keys()))
        query = goldens[sample_id]["text"]
    elif args.query:
        query = args.query
        sample_id = None
    else:
        sample_id = "amz-001"
        query = goldens[sample_id]["text"]

    print(f"Initializing AmazonHelp Agent (mode={mode})...")
    retriever = Retriever()
    classifier = LLMClassifier()

    if args.interactive:
        print("\n" + "=" * 70)
        print("INTERACTIVE DEMO INSPECTOR (Type 'exit' to quit)")
        print("Tip: Enter any golden ID like 'amz-000', 'amz-003', or press Enter for next.")
        print("=" * 70)
        idx = 0
        while True:
            try:
                suggested_id, _, desc = FEATURED_SAMPLES[idx % len(FEATURED_SAMPLES)]
                user_input = input(f"\nEnter Sample ID or custom text (default: {suggested_id} - {desc}):\n> ").strip()
                if user_input.lower() in ("exit", "quit", "q"):
                    break
                if not user_input:
                    target_id = suggested_id
                    target_text = goldens[target_id]["text"]
                elif user_input in goldens:
                    target_id = user_input
                    target_text = goldens[target_id]["text"]
                else:
                    target_id = None
                    target_text = user_input

                res = run_pipeline(target_text, retriever, classifier, mode=mode)
                print_result(res, sample_id=target_id)
                idx += 1
            except CacheMiss as e:
                print(f"\n[CACHE MISS] Query not in replay cache: {e}")
                print("Run on a golden ID (e.g. amz-000) or pass --live with GEMINI_API_KEY.\n")
            except KeyboardInterrupt:
                break
        print("\nExiting demo.")
        return 0

    try:
        res = run_pipeline(query, retriever, classifier, mode=mode)
        print_result(res, sample_id=sample_id)
    except CacheMiss as e:
        print(f"\n[CACHE MISS] Query is not in the committed replay cache.\n{e}\n"
              "Tip: Run on a cached golden sample like: python scripts/demo.py --id amz-001\n"
              "Or set GEMINI_API_KEY and run with: python scripts/demo.py \"your query\" --live\n")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
