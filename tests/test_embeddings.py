"""Tests for the pluggable embedder.

The bar for the numpy fallback is NOT "as good as a transformer". It is:
retrieval surfaces a past ticket from the right family, deterministically.

A note on what this fallback structurally cannot do, discovered while writing
these tests and worth carrying into the report: TF-IDF+LSA is a bag-of-words
method, so two tickets sharing no vocabulary ("my order hasn't arrived" vs
"where is my parcel") have cosine similarity of exactly 0 unless the corpus
contains bridging documents that use both words. On a small corpus it can even
rank an unrelated ticket higher purely for sharing one token. Real support
corpora do contain those bridges (thousands of customers, overlapping phrasing),
which is why the corpus below is written to look like one -- and why
`sentence-transformers` remains the production path.

We therefore assert an aggregate retrieval metric (precision@1) rather than
strict pairwise dominance, which is both more honest and closer to what the
system is actually judged on.
"""
from __future__ import annotations

import numpy as np

from src.retrieve.embeddings import (
    TfidfSvdEmbedder,
    cosine_sim,
    get_embedder,
    tokenize,
)

# Four intent families, five tickets each, with the vocabulary overlap a real
# support corpus has (order/parcel/delivered/arrived co-occur across tickets).
CORPUS = [
    # 0-4  delivery
    "my order hasnt arrived and its been 9 days",
    "where is my order it has not been delivered yet",
    "order still not delivered after two weeks",
    "the parcel from my order never arrived",
    "my parcel has not been delivered and tracking is stuck",
    # 5-9  app crash
    "the app keeps crashing when i open payments",
    "app crashes every time i tap the payment tab",
    "your app force closes on the payment screen",
    "the app crashed again after the latest update",
    "app keeps freezing and crashing on android",
    # 10-14  billing
    "i was charged twice for my subscription",
    "double charged this month for my subscription",
    "billing charged me two times for the same subscription",
    "i have been overcharged on my subscription this month",
    "why was i charged twice for the subscription i want a refund",
    # 15-19  account
    "how do i change the email on my account",
    "want to update my account email address",
    "need to change the email address on my account settings",
    "how can i update the email linked to my account",
    "changing my account email address is not working",
]
GROUPS = {"delivery": range(0, 5), "app": range(5, 10),
          "billing": range(10, 15), "account": range(15, 20)}
GROUP_OF = {i: name for name, idxs in GROUPS.items() for i in idxs}


def _fitted(dim: int = 16):
    return TfidfSvdEmbedder(dim=dim, min_df=1).fit(CORPUS)


def test_tokenize_drops_stopwords_and_shorts():
    toks = tokenize("My order HAS not arrived and it is a big problem")
    assert "order" in toks and "arrived" in toks and "problem" in toks
    assert "the" not in toks and "and" not in toks and "is" not in toks


def test_output_shape_and_normalisation():
    emb = _fitted()
    V = emb.encode(CORPUS)
    assert V.shape == (len(CORPUS), emb.dim)
    norms = np.linalg.norm(V, axis=1)
    assert np.all(np.isclose(norms, 1.0, atol=1e-4) | np.isclose(norms, 0.0))


def test_deterministic_across_fits():
    a = TfidfSvdEmbedder(dim=16, min_df=1, seed=1).fit(CORPUS).encode(CORPUS)
    b = TfidfSvdEmbedder(dim=16, min_df=1, seed=1).fit(CORPUS).encode(CORPUS)
    assert np.allclose(a, b), "embedder must be reproducible for the repro promise"


def _precision_at_1(V: np.ndarray) -> float:
    """Share of tickets whose nearest neighbour is from the same intent family.

    This is the metric that actually predicts retrieval usefulness, and it is
    the one reported for the fallback in the README.
    """
    S = cosine_sim(V, V)
    np.fill_diagonal(S, -np.inf)
    hits = sum(GROUP_OF[int(np.argmax(S[i]))] == GROUP_OF[i] for i in range(len(CORPUS)))
    return hits / len(CORPUS)


def test_precision_at_1_is_usable():
    p1 = _precision_at_1(_fitted().encode(CORPUS))
    assert p1 >= 0.80, f"precision@1 fell to {p1:.2f} — retrieval would be unreliable"


def test_within_group_similarity_beats_between_group():
    V = _fitted().encode(CORPUS)
    S = cosine_sim(V, V)
    within, between = [], []
    for i in range(len(CORPUS)):
        for j in range(i + 1, len(CORPUS)):
            (within if GROUP_OF[i] == GROUP_OF[j] else between).append(S[i, j])
    assert np.mean(within) > 2 * np.mean(between), (
        f"within={np.mean(within):.3f} between={np.mean(between):.3f} — "
        "the embedding space carries no usable intent structure"
    )


def test_query_retrieves_right_family():
    emb = _fitted()
    V = emb.encode(CORPUS)
    for query, expected in [
        ("my parcel was never delivered", "delivery"),
        ("the app force closes on payment", "app"),
        ("charged twice for subscription", "billing"),
        ("update the email on my account", "account"),
    ]:
        best = int(np.argmax(cosine_sim(emb.encode([query]), V)[0]))
        assert GROUP_OF[best] == expected, (
            f"{query!r} -> [{best}] {CORPUS[best]!r} ({GROUP_OF[best]}), expected {expected}"
        )


def test_unseen_vocabulary_degrades_to_zero_vector():
    # All-OOV text yields a zero vector rather than crashing. Downstream this
    # gives max-similarity 0, which correctly trips the escalation threshold --
    # a useful property, not merely a safe one.
    v = _fitted().encode(["zzzz qqqq wwww"])
    assert np.isclose(np.linalg.norm(v[0]), 0.0)


def test_empty_and_none_text_safe():
    v = _fitted().encode(["", "   "])
    assert v.shape[0] == 2 and np.all(np.isfinite(v))


def test_encode_before_fit_raises():
    try:
        TfidfSvdEmbedder().encode(["x"])
        raise AssertionError("should have raised")
    except RuntimeError:
        pass


def test_factory_falls_back_without_sentence_transformers():
    e = get_embedder("auto", dim=8, min_df=1)
    assert e.name in ("sentence-transformers", "tfidf-svd")


def test_dim_is_clamped_for_tiny_corpora():
    e = TfidfSvdEmbedder(dim=512, min_df=1).fit(CORPUS[:3])
    assert e.dim <= 3 and e.encode(CORPUS[:3]).shape[1] == e.dim


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} embedding tests passed")

    emb = _fitted()
    V = emb.encode(CORPUS)
    print(f"\nfallback retrieval quality on this corpus: "
          f"precision@1 = {_precision_at_1(V):.2f}")
    q = "app force closes on payments"
    sims = cosine_sim(emb.encode([q]), V)[0]
    print(f"query: {q!r}")
    for i in np.argsort(-sims)[:3]:
        print(f"  {sims[i]:.3f}  [{GROUP_OF[int(i)]}]  {CORPUS[i]}")
