"""
Reply drafting: canned baseline, retrieval-only baseline, grounded RAG system.

The RAG drafter may only use patterns present in the retrieved resolutions and
must return which case(s) it used. Promise verbs (refund dates, amounts,
guaranteed delivery) are forbidden unless verbatim in the evidence -- the
validator enforces this, not the prompt alone.
"""
from __future__ import annotations

import re

from src import config

CANNED = ("Sorry to hear about the trouble with your order. "
          "Please DM us your order number so we can look into this for you.")

DRAFT_SCHEMA = "draft-v1"


def canned_draft(text: str) -> dict:
    return {"draft": CANNED, "evidence_ids": [], "grounding": 0.0}


def retrieval_only_draft(cases: list[dict]) -> dict:
    if not cases:
        return canned_draft("")
    return {"draft": cases[0]["brand_reply"],
            "evidence_ids": [cases[0]["case_id"]], "grounding": 1.0}


def _prompt(text: str, intent: str, cases: list[dict]) -> str:
    ev = "\n".join(
        f"[case {c['case_id']}] customer: {c['customer_message'][:300]}\n"
        f"[case {c['case_id']}] AmazonHelp reply: {c['brand_reply'][:300]}"
        for c in cases
    )
    return (
        "Draft an AmazonHelp support reply to this customer message. Intent: "
        f"{intent} ({config.INTENTS.get(intent, '')}).\n"
        "Rules: acknowledge the issue; ask for order/tracking details or DM to "
        "investigate; copy the ACTION pattern from the past resolutions below; "
        "NEVER state a refund date, compensation amount, or guaranteed delivery "
        "date unless it appears verbatim in the evidence.\n\n"
        f"Customer: {text!r}\n\nPast resolutions:\n{ev}\n\n"
        'Reply ONLY valid JSON: {"draft": "<reply <=70 words>", "used_ids": [<case ids>]}'
    )


RESOLUTION_URLS = {
    "delivery_delay": "https://www.amazon.com/your-orders",
    "missing_parcel_tracking": "https://www.amazon.com/your-orders",
    "refund_return": "https://www.amazon.com/returns",
    "device_app_account": "https://www.amazon.com/your-account",
    "order_status_general": "https://www.amazon.com/your-orders",
    "other_unclear": "https://www.amazon.com/help",
}


TWITTER_DM_URL = "https://twitter.com/messages/compose?recipient_id=AmazonHelp"


def inject_resolution_url(draft: str, intent: str, customer_text: str = "") -> str:
    url = RESOLUTION_URLS.get(intent, "https://www.amazon.com/help")

    # If the draft accidentally copied the customer's t.co link as a destination, replace it
    if customer_text:
        tco_matches = re.findall(r"https?://t\.co/\w+", customer_text)
        for m in tco_matches:
            if m in draft:
                draft = draft.replace(m, url)

    if "<URL>" in draft:
        draft = draft.replace("<URL>", url)

    # Ensure actionable support URL is always present
    if "amazon.com" not in draft:
        draft = draft.rstrip()
        if draft.endswith((".", "!", "?")):
            draft = f"{draft} For assistance: {url} | Twitter DM: {TWITTER_DM_URL}"
        else:
            draft = f"{draft} at {url} or via Twitter DM: {TWITTER_DM_URL}"
    elif TWITTER_DM_URL not in draft and (
        "dm" in draft.lower() or "t.co" in customer_text or "twitter" in customer_text.lower()
    ):
        draft = draft.rstrip()
        if not draft.endswith((".", "!", "?")):
            draft += "."
        draft = f"{draft} Twitter DM: {TWITTER_DM_URL}"

    return draft


def rag_draft(text: str, intent: str, cases: list[dict], mode: str | None = None) -> dict:
    from src.llm.gemini import complete_json
    try:
        d = complete_json(
            _prompt(text, intent, cases), model=config.GENERATOR_MODEL,
            temperature=config.GEN_TEMPERATURE, schema=DRAFT_SCHEMA, mode=mode,
            default={"draft": CANNED, "used_ids": []},
        )
        draft = str(d.get("draft", CANNED))[:600]
        draft = inject_resolution_url(draft, intent, customer_text=text)
        used = [int(i) for i in (d.get("used_ids") or [])
                if isinstance(i, int)]
        valid = {c["case_id"] for c in cases}
        used = [i for i in used if i in valid][:3]
        return {"draft": draft, "evidence_ids": used,
                "grounding": 1.0 if used else 0.0}
    except Exception:
        return canned_draft(text)


FORBIDDEN_CLAIM_RE = re.compile(
    r"(refund(ed| will arrive| within \d)|guarantee(d)?|compensation of|"
    r"\$\s*\d|\b\d+\s*(business\s*)?days.{0,20}refund|delivered by \w+day)",
    re.I,
)


def validate_grounding(draft: str, cases: list[dict], evidence_ids: list[int]) -> dict:
    """Code verification, not LLM trust: every cited id must exist, and
    forbidden promise patterns must either be absent or verbatim in evidence."""
    valid = {c["case_id"] for c in cases}
    cited_ok = bool(evidence_ids) and all(i in valid for i in evidence_ids)
    ev_text = " ".join(c["brand_reply"] for c in cases
                       if c["case_id"] in (evidence_ids or []))
    m = FORBIDDEN_CLAIM_RE.search(draft or "")
    claim_ok = True
    if m:
        claim_ok = m.group(0).lower() in ev_text.lower()
    passed = bool(cited_ok and claim_ok)
    reason = ("ok" if passed else
              ("no valid evidence cited" if not cited_ok else "unsupported promise claim"))
    return {"passed": passed, "reason": reason, "score": 4.5 if passed else 1.5}
