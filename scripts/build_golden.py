"""
Golden-set builder: stratified 150-example DRAFT for AmazonHelp.

What this is: a deterministic, rule-pre-labelled sampling sheet. Every label is
marked needs_review=true because rules are NOT ground truth -- a human must
verify each row before the numbers count. The report states this plainly
(single-annotator + rule-draft limitation).

Stratification: rule-intent buckets, targets 35/35/35/20/15/10 across the five
operational intents + other, with hard cases (multi-intent, very short,
complaint markers) oversampled into every bucket.

Rule priority (documented because overlaps are real, e.g. "delayed, want
refund"): refund_return > missing_parcel_tracking > device_app_account >
delivery_delay > order_status_general > other_unclear.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from src import config
from src.eda.brand_select import is_deflection
from src.ingest.language import is_english

SEED = config.SEED
N_TOTAL = 150
TARGETS = {
    "delivery_delay": 35,
    "missing_parcel_tracking": 35,
    "refund_return": 35,
    "device_app_account": 20,
    "order_status_general": 15,
    "other_unclear": 10,
}

RULES: list[tuple[str, re.Pattern]] = [
    ("refund_return", re.compile(
        r"refund|money\s*back|\breturn\b|returned|wrong item|double charg|charged twice|"
        r"overcharg|compensation|cashback|reimburse|chargeback|fraud|unauthori", re.I)),
    ("missing_parcel_tracking", re.compile(
        r"(delivered|handed|left|arrived).{0,40}(not|never|no |n't|missing|without)|"
        r"never received|not received.{0,30}deliver|tracking|stuck|carrier|\bups\b|\busps\b|"
        r"door.{0,30}(rain|outside|no |empty)|parcel.{0,30}(missing|lost|gone|no )|"
        r"stolen|lost (my |the )?(package|parcel|order)", re.I)),
    ("device_app_account", re.compile(
        r"alexa|\becho\b|fire\s*stick|kindle|prime video|prime music|\bapp\b|"
        r"account|log\s*in|login|password|linked|linking|prime benefits|gift card", re.I)),
    ("delivery_delay", re.compile(
        r"delay|late|not arrived|hasn'?t arrived|haven'?t|still waiting|reschedul|"
        r"postponed|expedit|prime.{0,30}(days|date|late)|behind schedule|"
        r"when.{0,20}(arrive|deliver|come)", re.I)),
    ("order_status_general", re.compile(
        r"order\s*#?\s*\d|where.{0,20}order|status|track (my|the) order|cancel.*order|"
        r"order.*cancel", re.I)),
]

HARD_RE = re.compile(
    r"\b(and also|as well as|too? .{0,20}(refund|deliver)|wtf|pathetic|disgusting|"
    r"boycott|lawsuit|lawyer|sue|fraud|never again|worst)\b|\?\s*\w+\?|.{0,10}\?.{0,10}\?",
    re.I,
)

MUST_INCLUDE = {
    "delivery_delay": ["acknowledge the delay", "ask for order details or DM to investigate"],
    "missing_parcel_tracking": ["acknowledge non-receipt", "ask for tracking/address details to investigate"],
    "refund_return": ["acknowledge the refund/return request", "state the next step without inventing a date"],
    "device_app_account": ["acknowledge the device/account issue", "ask for device/account details or give a first step"],
    "order_status_general": ["acknowledge the request", "ask for the order number"],
    "other_unclear": ["ask a clarifying question"],
}
MUST_NOT = ["a specific refund arrival date", "a compensation amount",
            "a guaranteed delivery date", "account-specific facts not in evidence"]


def rule_intent(text: str) -> tuple[str, int]:
    """Return (intent, n_rules_fired). Priority order = RULES order."""
    t = text or ""
    fired = [name for name, rx in RULES if rx.search(t)]
    if not fired:
        return "other_unclear", 0
    return fired[0], len(fired)


def draft_escalation(text: str, intent: str) -> tuple[str, str, str]:
    t = (text or "").lower()
    if intent == "other_unclear":
        return "escalate", "UNKNOWN_INTENT", "too vague or out of taxonomy to auto-handle"
    for kw in config.ESCALATION_KEYWORDS:
        if kw.lower() in t:
            return "escalate", "HIGH_RISK", f"matched risk phrase: {kw!r}"
    if len(text or "") < 40:
        return "escalate", "LOW_CONFIDENCE", "message too short to classify reliably"
    return "auto_handle", "OK_AUTO", "routine case with precedent"


def main() -> int:
    df = pd.read_parquet(config.DATA_SAMPLE / "pairs_AmazonHelp.parquet")
    en = df[df["customer_message"].map(is_english)
            & df["brand_reply"].map(is_english)].copy().reset_index(drop=True)
    en["rule_intent"], en["n_fired"] = zip(*en["customer_message"].map(rule_intent))
    en["hard"] = en["customer_message"].map(lambda t: bool(HARD_RE.search(t or "")))
    rng = __import__("numpy").random.default_rng(SEED)

    picked: list[pd.DataFrame] = []
    used_idx: set[int] = set()
    for intent, n in TARGETS.items():
        pool = en[(en["rule_intent"] == intent) & (~en.index.isin(used_idx))]
        hard = pool[pool["hard"]]
        n_hard = min(len(hard), max(n // 3, 5))
        take_hard = hard.sample(n_hard, random_state=SEED) if n_hard else hard.iloc[0:0]
        rest = pool.drop(take_hard.index)
        take_rest = rest.sample(min(n - len(take_hard), len(rest)), random_state=SEED)
        take = pd.concat([take_hard, take_rest])
        used_idx |= set(take.index)
        picked.append(take)
    gold = pd.concat(picked).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    gold = gold.iloc[:N_TOTAL]

    out_path = config.DATA_GOLDEN / "golden.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in gold.iterrows():
            intent, _ = rule_intent(row["customer_message"])
            esc, code, reason = draft_escalation(row["customer_message"], intent)
            rec = {
                "id": f"amz-{i:03d}",
                "text": row["customer_message"],
                "brand_reply_ref": row["brand_reply"],
                "intent_draft": intent,
                "intent": intent,  # human overwrites after review
                "escalate_draft": esc,
                "escalate": esc,  # human overwrites after review
                "reason_code_draft": code,
                "reason_draft": reason,
                "must_include": MUST_INCLUDE[intent],
                "must_not_promise": MUST_NOT,
                "multi_rule_fired": int(row["n_fired"]),
                "hard_case": bool(row["hard"]),
                "double_label": bool(i % 5 == 0),  # 30 flagged for second annotator
                "needs_review": True,
                "customer_tweet_id": str(row.get("customer_tweet_id", "")),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    note = config.DATA_GOLDEN / "GOLDEN_NOTE.md"
    dist = gold["rule_intent"].value_counts().to_dict()
    note.write_text(
        "# Golden set note (AmazonHelp, n=150 DRAFT)\n\n"
        "## Sampling\n"
        "- Source: `data/sample/pairs_AmazonHelp.parquet` (fresh cleaning), English-only both sides.\n"
        "- Stratified by rule-intent buckets with targets "
        + json.dumps(TARGETS) + f"; realised {json.dumps(dist)}.\n"
        "- Hard cases (multi-intent, very short, complaint/sarcasm markers) oversampled "
        f"to ~1/3 per bucket; realised hard={int(gold['hard'].sum())}.\n"
        "- 30 rows (every 5th) flagged `double_label=true` for a second annotator.\n\n"
        "## Labelling\n"
        "- `intent`/`escalate` are RULE PRE-LABELS (`needs_review=true`). Priority: "
        "refund_return > missing_parcel_tracking > device_app_account > delivery_delay > "
        "order_status_general > other_unclear. A human must verify every row; unverified "
        "numbers are draft-only.\n"
        "- `must_include`/`must_not_promise` encode the brand playbook; the reply judge "
        "scores against these, not against a single reference reply (many valid replies exist).\n"
        "- Escalation pre-labels: `other_unclear` always escalates; ESCALATION_KEYWORDS hits "
        "escalate with HIGH_RISK; <40 chars escalate with LOW_CONFIDENCE.\n\n"
        "## Known limits\n"
        "- Single annotator + rule draft: label noise bounds how much of any model gap is real.\n"
        "- One brand / one time window (Oct 2017 tweets): will not generalise untouched.\n",
        encoding="utf-8",
    )
    print(f"[golden] wrote {len(gold)} rows -> {out_path}")
    print(f"[golden] dist={dist} hard={int(gold['hard'].sum())} double_label={int((gold.index % 5 == 0).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
