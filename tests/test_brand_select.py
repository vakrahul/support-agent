"""Brand-selection tests.

The fixture contains two brands built to be opposites:
  ResolveCo -- answers in-thread   (good grounding corpus)
  DeflectCo -- always punts to DM  (bad grounding corpus)
If the selection logic cannot tell them apart, the whole premise of the report
is broken -- so this is the most load-bearing test in the repo.
"""
from __future__ import annotations

from src.eda.brand_select import (
    is_deflection,
    is_grateful,
    is_specific,
    is_still_broken,
    profile_brands,
    to_markdown,
)
from src.ingest.threads import build_pairs, load_raw
from tests.make_fixture import FIXTURE, generate


def _profiles():
    if not FIXTURE.exists():
        generate()
    return profile_brands(build_pairs(load_raw(FIXTURE)), min_pairs=5)


def test_deflection_detector_catches_real_phrasings():
    for t in [
        "Sorry to hear that! Please DM us your order number.",
        "Can you send us a direct message with details?",
        "Please follow us and DM us your account email",
        "Kindly email us at support so we can help",
        "Please fill out this form and we'll take a look",
    ]:
        assert is_deflection(t), t


def test_deflection_detector_ignores_real_resolutions():
    for t in [
        "That crash is fixed in v4.2.1 - update from the Play Store.",
        "Your duplicate charge has been reversed, 3-5 business days.",
        "Change it under Settings > Account > Email.",
    ]:
        assert not is_deflection(t), t


# ---------------------------------------------------------------------------
# Regression tests for the under-firing bug.
#
# Every string below is a REAL brand reply from twcs.csv (handles removed, URLs
# replaced with the <URL> token exactly as clean_text produces). The first
# version of the detector missed all of them, which is how AmazonHelp scored a
# 96.9% "resolution rate" -- a number that was wrong, not merely imprecise.
# ---------------------------------------------------------------------------
DEFLECTIONS_MISSED_BY_V1 = [
    "Kindly drop in your details here: <URL> and I'll reach out to you.",
    "Please contact our support team for assistance: <URL>",
    "As you have filled the details, you will receive an email communication here: <URL>",
    "We would need to get you into a private secured link to further assist.",
    "Drop us a note here <URL> and we'll go from there.",
    "Please select your preferred language here <URL>",
    "We've received your details and shall reach out to you with an update soon.",
    "I've passed this on to our team, someone will be in touch shortly.",
    "You can reach us here <URL> or chat here <URL>",
    "Please get in touch so we can look into this for you.",
]


def test_url_stripping_no_longer_hides_deflections():
    """The bug was self-inflicted: clean_text deleted the URL before detection.

    "contact our support team here: https://..." became "...here:" -- the tell
    was removed by our own pipeline. clean_text now emits a <URL> token.
    """
    from src.ingest.threads import clean_text

    raw = "Please contact our support team for assistance: https://t.co/abc123"
    cleaned = clean_text(raw)
    assert "<URL>" in cleaned and "http" not in cleaned
    assert is_deflection(cleaned)


def test_detector_catches_phrasings_that_v1_missed():
    missed = [t for t in DEFLECTIONS_MISSED_BY_V1 if not is_deflection(t)]
    assert not missed, f"still missing {len(missed)}/{len(DEFLECTIONS_MISSED_BY_V1)}: {missed}"


def test_self_service_links_are_resolutions_not_deflections():
    """A link to the FIX is groundable material; a link to a CHANNEL is not.

    Mining found "the steps here" (442), "more info here" (307) and "be found
    here" (251) at real volume, so a blanket "contains a link => deflection"
    rule would have discarded genuine self-service answers.
    """
    for t in [
        "The steps to reset your password are here: <URL>",
        "More info here: <URL>",
        "Full instructions can be found here <URL>",
        "Have a look at this article <URL> which walks through it.",
    ]:
        assert not is_deflection(t), t


def test_self_service_overrides_a_channel_mention():
    # Both signals present: still a resolution, because the answer was given.
    t = "Here's the guide that fixes it <URL> - if it persists, contact us here <URL>"
    assert not is_deflection(t)


def test_agent_signature_is_stripped():
    from src.ingest.threads import clean_text

    assert clean_text("We've refunded that for you. ^KC") == "We've refunded that for you."


def test_multipart_replies_are_flagged():
    from src.ingest.threads import is_multipart

    assert is_multipart("...continued below (2/3)")
    assert is_multipart("1/2 First, open Settings")
    assert not is_multipart("Refunded in 3-5 business days")


def test_specificity_detector():
    assert is_specific("update to v4.2.1 and restart the app")
    assert is_specific("refunded within 3-5 business days")
    assert not is_specific("we are sorry for the trouble caused")


def test_followup_sentiment_detectors():
    assert is_grateful("thanks, that worked!")
    assert is_still_broken("still not fixed")
    assert not is_grateful("still not fixed")


def test_resolver_brand_beats_deflector():
    p = _profiles()
    assert len(p) == 2
    top = p.iloc[0]
    assert top["brand"] == "ResolveCo", f"picked {top['brand']} — selection logic is broken"

    resolve = p[p["brand"] == "ResolveCo"].iloc[0]
    deflect = p[p["brand"] == "DeflectCo"].iloc[0]
    assert resolve["resolution_rate"] > 0.9
    assert deflect["deflection_rate"] > 0.9
    assert resolve["groundable_pairs"] > deflect["groundable_pairs"]
    assert resolve["specificity_rate"] > deflect["specificity_rate"]


def test_min_pairs_filter_applies():
    p = profile_brands(build_pairs(load_raw(FIXTURE)), min_pairs=10_000)
    assert p.empty


def test_markdown_renders():
    md = to_markdown(_profiles())
    assert "ResolveCo" in md and md.startswith("| brand")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} brand-selection tests passed\n")
    print(to_markdown(_profiles()))
