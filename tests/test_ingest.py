"""Pin thread reconstruction against the synthetic fixture, where we know the
correct answer by construction.

The fixture deliberately contains an orphan reply, a too-short tweet, and
follow-ups, because the real Kaggle dump contains all three.
"""
from __future__ import annotations

from pathlib import Path

from src.ingest.threads import build_pairs, clean_text, is_brand_author, load_raw
from tests.make_fixture import FIXTURE, generate


def _pairs():
    if not FIXTURE.exists():
        generate()
    return build_pairs(load_raw(FIXTURE))


def test_clean_text_strips_noise_keeps_signal():
    raw = "@AppleSupport my app CRASHED!! see https://t.co/abc123 &amp; @115712 help"
    out = clean_text(raw)
    assert "http" not in out and "@" not in out
    assert "CRASHED!!" in out          # punctuation/caps kept -- sentiment signal
    assert "&" in out                  # html entity decoded
    assert "  " not in out             # whitespace collapsed


def test_clean_text_handles_non_strings():
    assert clean_text(None) == "" and clean_text(float("nan")) == ""


def test_brand_detection():
    assert is_brand_author("AppleSupport") is True
    assert is_brand_author("115712") is False   # numeric == customer


def test_pairs_are_customer_then_brand():
    p = _pairs()
    assert len(p) > 0
    assert set(p["brand"].unique()) <= {"ResolveCo", "DeflectCo"}
    # every customer_id is numeric, every brand is not
    assert p["customer_id"].map(str.isdigit).all()
    assert not p["brand"].map(str.isdigit).any()


def test_orphan_reply_is_dropped():
    # tweet 99991 replies to a parent that does not exist
    p = _pairs()
    assert "99991" not in set(p["brand_tweet_id"])


def test_short_customer_message_is_filtered():
    # tweet 99992 is just "ok" -> below the 12-char floor
    p = _pairs()
    assert "99992" not in set(p["customer_tweet_id"])


def test_followups_are_attached_when_present():
    p = _pairs()
    assert "customer_followup" in p.columns
    assert (p["customer_followup"].str.len() > 0).any()


def test_no_duplicate_pairs():
    p = _pairs()
    assert not p.duplicated(subset=["customer_tweet_id", "brand_tweet_id"]).any()


def test_empty_input_returns_empty_frame_not_crash():
    import pandas as pd
    out = build_pairs(pd.DataFrame())
    assert len(out) == 0 and "customer_message" in out.columns


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    p = _pairs()
    print(f"\n{len(fns)}/{len(fns)} ingest tests passed")
    print(f"pairs built from fixture: {len(p)}")
    print(p[["brand", "customer_message", "brand_reply"]].head(3).to_string())
