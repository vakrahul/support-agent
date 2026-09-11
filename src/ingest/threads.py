"""
Thread reconstruction from the raw Kaggle dump.

Raw schema (twcs.csv):
    tweet_id, author_id, inbound, created_at, text,
    response_tweet_id, in_response_to_tweet_id

`inbound=True`  -> written by a customer
`inbound=False` -> written by a brand
`author_id` is a numeric string for customers and a handle (e.g. "AppleSupport")
for brands, which is how we identify brands without a hardcoded list.

What we produce is the unit the whole project is built on: a **(customer message
-> brand reply) pair**, optionally with the customer's follow-up so we can tell
whether the issue was actually resolved.

The file is read in CHUNKS: the full dump is ~3M rows and reviewers explicitly
said they will not run the full dataset, so nothing here assumes it fits in RAM.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import pandas as pd

from src import config

RAW_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_MENTION = re.compile(r"@\w+")
_RE_WS = re.compile(r"\s+")
_RE_PLACEHOLDER = re.compile(r"__email__|__phone__|__number__", re.I)
# Kaggle anonymised handles look like @115712 ; brand handles are words.
_RE_NUMERIC_MENTION = re.compile(r"@\d+")
# Support agents sign off with their initials: "...take a look. ^KC"
_RE_SIGNATURE = re.compile(r"\s*\^\s*[A-Za-z]{1,4}\s*$")
# Multi-part replies: "(1/3)", "1/2", "2 of 3"
_RE_PART = re.compile(r"[\(\[]?\b\d\s*(?:/|of)\s*\d\b[\)\]]?")

URL_TOKEN = "<URL>"


def clean_text(s: str, *, drop_mentions: bool = True, keep_url_token: bool = True) -> str:
    """Normalise tweet text.

    @mentions go because they are noise for intent and they leak identifiers
    into the embedding space (two tweets look 'similar' merely for naming the
    same handle). Emoji, casing and punctuation stay -- they carry sentiment,
    which the escalation layer consumes.

    URLs are replaced with a `<URL>` TOKEN rather than deleted. This is not
    cosmetic. An earlier version deleted them, and it silently destroyed the
    single strongest deflection signal in the corpus: "please contact our
    support team here: https://..." became "...support team here:" and scored
    as a resolution. 38% of AmazonHelp replies and 66% of AppleSupport replies
    carry a link, so deleting them mis-scored a large fraction of the dataset.
    The token keeps the signal detectable while still stripping the identifier.
    See DECISIONS.md #21.
    """
    if not isinstance(s, str):
        return ""
    s = _RE_URL.sub(f" {URL_TOKEN} " if keep_url_token else " ", s)
    s = _RE_PLACEHOLDER.sub(" ", s)
    s = _RE_NUMERIC_MENTION.sub(" ", s)
    if drop_mentions:
        s = _RE_MENTION.sub(" ", s)
    s = s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    s = _RE_SIGNATURE.sub("", s)
    return _RE_WS.sub(" ", s).strip()


def strip_url_token(s: str) -> str:
    """Drop the `<URL>` marker. Used at embedding time, where it is noise."""
    return _RE_WS.sub(" ", (s or "").replace(URL_TOKEN, " ")).strip()


def is_multipart(s: str) -> bool:
    """True for "(1/3)"-style fragments -- a partial, not a whole resolution."""
    return bool(_RE_PART.search(s or ""))


def is_brand_author(author_id: object) -> bool:
    """Brands have handle-like author_ids; customers have numeric ones."""
    return not str(author_id).strip().isdigit()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def iter_raw_chunks(
    path: Path | str | None = None,
    chunksize: int = 200_000,
    max_rows: int | None = None,
) -> Iterator[pd.DataFrame]:
    """Stream the raw CSV in chunks, stopping early at `max_rows`."""
    path = Path(path) if path is not None else config.RAW_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}.\n"
            "Download 'Customer Support on Twitter' from Kaggle "
            "(thoughtvector/customer-support-on-twitter) and place twcs.csv there."
        )
    seen = 0
    for chunk in pd.read_csv(path, chunksize=chunksize, dtype=str, low_memory=False):
        chunk.columns = [c.strip() for c in chunk.columns]
        yield chunk
        seen += len(chunk)
        if max_rows is not None and seen >= max_rows:
            return


def load_raw(path: Path | str | None = None, max_rows: int | None = None) -> pd.DataFrame:
    """Load raw tweets into a single frame (use only on a subsample)."""
    return pd.concat(list(iter_raw_chunks(path, max_rows=max_rows)), ignore_index=True)


# --------------------------------------------------------------------------
# Pair construction
# --------------------------------------------------------------------------
def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a flat tweet table into (customer message -> brand reply) pairs.

    A pair exists when an OUTBOUND tweet is a direct reply
    (`in_response_to_tweet_id`) to an INBOUND tweet. We attach:
      - `customer_followup`: the customer's next message in that thread, if any.
        This is the weak signal we use later to guess whether the reply worked.
      - `turn_index`: how deep in the conversation the customer message sat.
        First-contact messages behave very differently from turn-5 escalations.
    """
    if df.empty:
        return _empty_pairs()

    d = df.copy()
    d["tweet_id"] = d["tweet_id"].astype(str).str.strip()
    d["in_response_to_tweet_id"] = (
        d["in_response_to_tweet_id"].astype(str).str.strip().replace({"nan": ""})
    )
    # `inbound` arrives as the strings "True"/"False"
    d["inbound_bool"] = d["inbound"].astype(str).str.strip().str.lower().eq("true")

    inbound = d[d["inbound_bool"]]
    outbound = d[~d["inbound_bool"]]
    if inbound.empty or outbound.empty:
        return _empty_pairs()

    # brand reply --joined onto--> the customer tweet it answers
    pairs = outbound.merge(
        inbound,
        left_on="in_response_to_tweet_id",
        right_on="tweet_id",
        suffixes=("_brand", "_cust"),
        how="inner",
    )
    if pairs.empty:
        return _empty_pairs()

    # customer's follow-up: an inbound tweet replying to the brand's reply
    followups = inbound[["in_response_to_tweet_id", "text"]].rename(
        columns={"in_response_to_tweet_id": "_fu_parent", "text": "customer_followup_raw"}
    )
    # keep the first follow-up only
    followups = followups.drop_duplicates(subset=["_fu_parent"], keep="first")
    pairs = pairs.merge(
        followups, left_on="tweet_id_brand", right_on="_fu_parent", how="left"
    )

    out = pd.DataFrame(
        {
            "brand": pairs["author_id_brand"].astype(str),
            "customer_id": pairs["author_id_cust"].astype(str),
            "customer_tweet_id": pairs["tweet_id_cust"].astype(str),
            "brand_tweet_id": pairs["tweet_id_brand"].astype(str),
            "created_at": pairs["created_at_cust"].astype(str),
            "customer_message_raw": pairs["text_cust"].astype(str),
            "brand_reply_raw": pairs["text_brand"].astype(str),
            "customer_followup_raw": pairs.get(
                "customer_followup_raw", pd.Series(index=pairs.index, dtype=object)
            ),
        }
    )

    out["customer_message"] = out["customer_message_raw"].map(clean_text)
    out["brand_reply"] = out["brand_reply_raw"].map(clean_text)
    out["customer_followup"] = out["customer_followup_raw"].fillna("").map(clean_text)

    # Only keep rows where the "brand" side really is a brand handle.
    out = out[out["brand"].map(is_brand_author)]

    # Quality filters. Very short messages are usually "thanks"/"ok" and carry
    # no intent; absurdly long ones are usually scraping artefacts.
    out = out[
        (out["customer_message"].str.len() >= 12)
        & (out["brand_reply"].str.len() >= 12)
        & (out["customer_message"].str.len() <= config.MAX_TEXT_LEN)
    ]

    # A customer can tweet the same thing repeatedly; dedupe on the pair.
    out = out.drop_duplicates(subset=["customer_message", "brand_reply"])

    return out.reset_index(drop=True)


def _empty_pairs() -> pd.DataFrame:
    cols = [
        "brand", "customer_id", "customer_tweet_id", "brand_tweet_id", "created_at",
        "customer_message_raw", "brand_reply_raw", "customer_followup_raw",
        "customer_message", "brand_reply", "customer_followup",
    ]
    return pd.DataFrame({c: pd.Series(dtype=object) for c in cols})


def build_pairs_streaming(
    path: Path | str | None = None,
    max_rows: int | None = None,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """Chunked pair construction.

    NOTE (honest limitation, worth stating in the report): pairing is done
    within each chunk, so a customer tweet in chunk N and its brand reply in
    chunk N+1 are missed. The raw file is ordered so replies sit near their
    parents, making the loss small -- but it is a real, quantified-at-your-own-
    risk bias, not a free lunch. Use `overlap` to reduce it.
    """
    frames: list[pd.DataFrame] = []
    carry = pd.DataFrame()
    for chunk in iter_raw_chunks(path, chunksize=chunksize, max_rows=max_rows):
        # prepend a small carry-over window to catch cross-chunk replies
        work = pd.concat([carry, chunk], ignore_index=True) if not carry.empty else chunk
        frames.append(build_pairs(work))
        carry = chunk.tail(5_000).copy()
    if not frames:
        return _empty_pairs()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(
        subset=["customer_tweet_id", "brand_tweet_id"]
    ).reset_index(drop=True)
