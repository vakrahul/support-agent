"""
Intent classifiers: trivial baseline, simple baseline, LLM system.

- MajorityClassifier: always predicts the majority weak-label class. The floor.
- TfidfLogReg: TF-IDF + LogisticRegression trained on ~2k WEAK (rule) labels.
  Stated transparently: it has never seen a human label, so the golden set is a
  fair held-out test for it too.
- LLMClassifier: Gemini flash few-shot with the frozen taxonomy definitions.
  Returns (intent, model_score, top3). model_score is an uncalibrated verbalised
  score -- the report never calls it a probability.

All three share one interface: predict(texts) -> list[(intent, score, top3)].
"""
from __future__ import annotations

from collections import Counter

from src import config

LABELS = [k for k in config.INTENTS if k != "other_unclear"] + ["other_unclear"]


class MajorityClassifier:
    name = "majority"

    def __init__(self, majority: str = "delivery_delay"):
        self.majority = majority

    def fit(self, texts, labels) -> "MajorityClassifier":
        if labels:
            self.majority = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, texts: list[str]):
        return [(self.majority, 1.0, [self.majority]) for _ in texts]


class TfidfLogReg:
    """Simple baseline. sklearn lives ONLY here so the eval path stays light."""

    name = "tfidf-logreg"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        self._pipe = make_pipeline(
            TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2),
            LogisticRegression(max_iter=500, C=2.0),
        )
        self._classes: list[str] = []

    def fit(self, texts, labels) -> "TfidfLogReg":
        self._pipe.fit(texts, labels)
        self._classes = list(self._pipe.classes_)
        return self

    def predict(self, texts: list[str]):
        import numpy as np
        proba = self._pipe.predict_proba(texts)
        out = []
        for row in proba:
            order = np.argsort(-row)
            top = [self._classes[i] for i in order[:3]]
            out.append((top[0], float(row[order[0]]), top))
        return out


CLASSIFY_SCHEMA = "intent-v2-fewshot"

FEWSHOTS: list[tuple[str, str]] = [
    ("delivery_delay", "Hi - another Amazon prime delivery hasn't arrived when it was meant to. Please advise"),
    ("delivery_delay", "Pathetic..Again ur delivery is delayed and this tym it is not even half way mark"),
    ("missing_parcel_tracking", "I ordered pigeon gas stove updated it delivered but I haven't received it yet"),
    ("missing_parcel_tracking", "So claims that I was handed my delivery today, I wasn't even in the house and there's no parcel"),
    ("refund_return", "poor refund process. not yet received refund after 20 days"),
    ("refund_return", "I was charged twice for my subscription this month"),
    ("device_app_account", "I cannot own any Alexa device because it won't link to my account"),
    ("device_app_account", "Why would Belgian customers pay for Prime Video if we cannot order the Fire TV Stick"),
    ("order_status_general", "Billing address is same as Shipping address. Could you please check I'm not able to change"),
    ("order_status_general", "I didn't want to cancel the order, but had to because payment failed, kindly help"),
    ("other_unclear", "AmazonGlobal Expedited shipping"),
    ("other_unclear", "One was shipped via AMZL US and the other UPS."),
]


def _prompt(text: str) -> str:
    defs = "\n".join(f"- {k}: {v}" for k, v in config.INTENTS.items())
    shots = "\n".join(f'Message: {t!r}\n{{"intent": "{k}", "model_score": 0.9, "alt": []}}'
                      for k, t in FEWSHOTS)
    return (
        "Classify this AmazonHelp customer message into exactly one intent.\n"
        f"Definitions:\n{defs}\n\n"
        "Priority for overlaps: refund_return > missing_parcel_tracking > "
        "device_app_account > delivery_delay > order_status_general > other_unclear.\n"
        "Vague fragments with no clear issue are other_unclear. "
        "'Delivered but not received' is missing_parcel_tracking, not delivery_delay. "
        "Unauthorized charges and cashback owed are refund_return.\n\n"
        f"Examples:\n{shots}\n\n"
        f"Message: {text!r}\n\n"
        'Reply with ONLY valid JSON, no other text: {"intent": "<one key above>", '
        '"model_score": <0-1 number>, "alt": ["<second>", "<third>"]}'
    )


class LLMClassifier:
    """System classifier. Every call goes through the replay cache."""

    name = "llm"

    def predict(self, texts: list[str], mode: str | None = None):
        from src.llm.gemini import TRANSPORT_FAILURES, CacheMiss, complete_json
        import sys
        out = []
        for i, t in enumerate(texts):
            if i and i % 25 == 0:
                print(f"  [classify] {i}/{len(texts)}", flush=True)
            try:
                d = complete_json(
                    _prompt(t), model=config.GENERATOR_MODEL,
                    temperature=0.0, schema=CLASSIFY_SCHEMA, mode=mode,
                    default={"intent": "other_unclear", "model_score": 0.0, "alt": []},
                )
                intent = d.get("intent", "other_unclear")
                if intent not in config.INTENTS:
                    intent = "other_unclear"
                score = float(d.get("model_score", 0.0))
                alt = [a for a in (d.get("alt") or []) if a in config.INTENTS][:2]
                top3 = [intent] + [a for a in alt if a != intent]
            except CacheMiss:
                raise
            except Exception as e:  # transport failure: fail safe (escalate) and count it
                TRANSPORT_FAILURES.append({"text": (t or "")[:120], "error": str(e)[:200]})
                intent, score, top3 = "other_unclear", 0.0, ["other_unclear"]
            out.append((intent, score, top3))
        return out
