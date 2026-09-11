"""
Generate a small SYNTHETIC dataset in the exact twcs.csv schema.

Why this exists: the real Kaggle dump is ~500MB and cannot be committed, but
CI, tests, and a reviewer on a fresh clone still need something to run against.
This fixture makes the whole pipeline exercisable end-to-end in under a second
and pins the thread-reconstruction logic against a KNOWN-correct answer.

It is a test fixture only -- no reported metric is ever computed from it.
Run:  python -m tests.make_fixture
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "twcs_sample.csv"

COLUMNS = [
    "tweet_id", "author_id", "inbound", "created_at", "text",
    "response_tweet_id", "in_response_to_tweet_id",
]

# Two contrasting brands, on purpose:
#   ResolveCo  -> answers in-thread (a GOOD grounding corpus)
#   DeflectCo  -> pushes everyone to DM (a BAD grounding corpus)
# This lets us assert that the brand-selection EDA actually prefers ResolveCo.
CUSTOMER_MSGS = [
    "my order hasnt arrived and its been 9 days, whats going on?",
    "app keeps crashing every time i open the payments tab",
    "i was charged twice for the same subscription this month",
    "how do i change the email address on my account?",
    "your delivery driver left my parcel in the rain, its ruined",
    "cant log in, it says my password is wrong but its not",
    "when does the new update roll out for android?",
    "i want to cancel my account, this service is the worst",
]
RESOLVE_REPLIES = [
    "Sorry about that! Orders past 7 days qualify for a free reship - we've "
    "queued one for you, arriving in 2-3 days.",
    "Thanks for flagging. That crash is fixed in v4.2.1 - update from the Play "
    "Store and clear the app cache, that resolves it.",
    "We see the duplicate charge. It's been reversed and should land back in "
    "3-5 business days.",
    "You can change it under Settings > Account > Email, then confirm via the "
    "link we send you.",
    "That's not okay. We've logged a claim and a replacement ships today at no "
    "cost to you.",
    "Try a password reset from the login screen - if the email doesn't arrive "
    "in 5 minutes, check spam, it usually lands there.",
    "The Android update rolls out region by region through next week - you'll "
    "get an in-app prompt when it's live for you.",
    "Sorry to hear that. You can cancel any time under Settings > Billing, and "
    "we'll refund the unused portion of this month.",
]
DEFLECT_REPLIES = [
    "Sorry to hear this! Please DM us your order number so we can look into it.",
    "We'd like to help - can you send us a DM with more details?",
    "Please follow us and DM us your account email and we'll take a look.",
    "Apologies! Send us a direct message and we'll investigate.",
    "Thanks for reaching out, please DM us so we can assist further.",
]
FOLLOWUPS = ["thanks, that worked!", "still not fixed", "ok will try that", ""]


def generate(path: Path = FIXTURE, n_threads: int = 120, seed: int = 7) -> Path:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    tid = 1000

    for i in range(n_threads):
        brand, replies = (
            ("ResolveCo", RESOLVE_REPLIES) if i % 3 else ("DeflectCo", DEFLECT_REPLIES)
        )
        cust_id = str(100000 + i)
        ts = f"Tue Oct {1 + (i % 28):02d} 12:00:00 +0000 2017"

        cust_tid, brand_tid = str(tid), str(tid + 1)
        tid += 2

        rows.append({
            "tweet_id": cust_tid, "author_id": cust_id, "inbound": "True",
            "created_at": ts, "text": f"@{brand} {rng.choice(CUSTOMER_MSGS)}",
            "response_tweet_id": brand_tid, "in_response_to_tweet_id": "",
        })
        rows.append({
            "tweet_id": brand_tid, "author_id": brand, "inbound": "False",
            "created_at": ts, "text": f"@{cust_id} {rng.choice(replies)}",
            "response_tweet_id": "", "in_response_to_tweet_id": cust_tid,
        })

        fu = rng.choice(FOLLOWUPS)
        if fu:
            fu_tid = str(tid); tid += 1
            rows[-1]["response_tweet_id"] = fu_tid
            rows.append({
                "tweet_id": fu_tid, "author_id": cust_id, "inbound": "True",
                "created_at": ts, "text": f"@{brand} {fu}",
                "response_tweet_id": "", "in_response_to_tweet_id": brand_tid,
            })

    # Deliberate noise, because the real data is noisy and the loader must cope:
    rows.append({  # orphan reply -> parent missing
        "tweet_id": "99991", "author_id": "ResolveCo", "inbound": "False",
        "created_at": "Tue Oct 01 12:00:00 +0000 2017", "text": "@1 we're on it",
        "response_tweet_id": "", "in_response_to_tweet_id": "88888",
    })
    rows.append({  # too-short customer tweet -> must be filtered out
        "tweet_id": "99992", "author_id": "100999", "inbound": "True",
        "created_at": "Tue Oct 01 12:00:00 +0000 2017", "text": "ok",
        "response_tweet_id": "99993", "in_response_to_tweet_id": "",
    })
    rows.append({
        "tweet_id": "99993", "author_id": "ResolveCo", "inbound": "False",
        "created_at": "Tue Oct 01 12:00:00 +0000 2017", "text": "@100999 glad to help!",
        "response_tweet_id": "", "in_response_to_tweet_id": "99992",
    })

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


if __name__ == "__main__":
    p = generate()
    print(f"wrote fixture -> {p}")
