"""Tests for intent discovery.

The clustering only earns its place if it recovers structure a human would
recognise. So the core test builds a corpus with four KNOWN intent families and
checks k-means finds them -- measured with cluster purity, which is the honest
way to score a clustering against known labels.
"""
from __future__ import annotations

import numpy as np

from src.retrieve.embeddings import TfidfSvdEmbedder
from src.taxonomy.cluster import (
    choose_k,
    format_k_curve,
    distinctive_terms,
    kmeans,
    representative_examples,
    silhouette_score,
    summarise_clusters,
    to_markdown,
)
from tests.test_embeddings import CORPUS, GROUP_OF, GROUPS


def _embedded(dim: int = 12):
    emb = TfidfSvdEmbedder(dim=dim, min_df=1).fit(CORPUS)
    return emb.encode(CORPUS)


def _purity(labels: np.ndarray) -> float:
    """Share of messages whose cluster is dominated by their true family."""
    total = 0
    for cid in np.unique(labels):
        members = [GROUP_OF[i] for i in np.where(labels == cid)[0]]
        if members:
            total += max(members.count(g) for g in set(members))
    return total / len(labels)


# ------------------------------------------------------------------ k-means
def test_kmeans_is_deterministic():
    X = _embedded()
    a, _, ia = kmeans(X, 4, seed=7)
    b, _, ib = kmeans(X, 4, seed=7)
    assert np.array_equal(a, b) and np.isclose(ia, ib)


def test_kmeans_recovers_known_intent_families():
    X = _embedded()
    labels, centroids, _ = kmeans(X, 4, seed=42, n_init=10)
    assert centroids.shape[0] == 4
    p = _purity(labels)
    assert p >= 0.80, f"cluster purity {p:.2f} — clustering is not recovering intents"


def test_inertia_decreases_with_more_clusters():
    X = _embedded()
    _, _, i2 = kmeans(X, 2, seed=1)
    _, _, i6 = kmeans(X, 6, seed=1)
    assert i6 < i2


def test_kmeans_handles_k_larger_than_n():
    X = _embedded()[:3]
    labels, centroids, _ = kmeans(X, 10, seed=1)
    assert centroids.shape[0] <= 3 and len(labels) == 3


def test_no_empty_clusters_returned():
    X = _embedded()
    labels, centroids, _ = kmeans(X, 4, seed=3, n_init=5)
    assert len(np.unique(labels)) == centroids.shape[0]


# --------------------------------------------------------------- choosing k
def test_silhouette_higher_for_true_structure():
    X = _embedded()
    good, _, _ = kmeans(X, 4, seed=42, n_init=10)
    rng = np.random.default_rng(0)
    random_labels = rng.integers(0, 4, size=len(CORPUS))
    assert silhouette_score(X, good) > silhouette_score(X, random_labels)


def test_choose_k_returns_scores_for_every_candidate():
    X = _embedded()
    k, scores = choose_k(X, k_range=range(2, 7))
    assert 2 <= k <= 6
    assert [s["k"] for s in scores] == [2, 3, 4, 5, 6]
    assert all("silhouette" in s and "adjusted" in s for s in scores)


def test_unpenalised_silhouette_runs_away_to_max_k():
    """Documents WHY the simplicity bias and operational band exist.

    On a corpus whose true structure is 4 families, raw silhouette selects the
    largest k on offer -- it rises monotonically because splitting any cluster
    increases average separation. Any claim that "silhouette chose k" would be
    a fabricated justification, so this behaviour is pinned by a test rather
    than left as folklore.
    """
    X = _embedded()
    k_raw, scores = choose_k(X, k_range=range(2, 9), simplicity_bias=0.0)
    assert k_raw == 8, f"expected runaway to max k, got {k_raw}"
    sils = [s["silhouette"] for s in scores]
    assert sils[-1] > sils[2], "silhouette should still be climbing at high k"


def test_simplicity_bias_recovers_true_k():
    # With the default price-per-intent the proposal lands at or near truth (4),
    # instead of the 8 that raw silhouette gives.
    X = _embedded()
    k, _ = choose_k(X, k_range=range(2, 9))
    assert 4 <= k <= 5, f"proposed k={k}, expected to land near the true k=4"


def test_simplicity_bias_prefers_fewer_clusters():
    X = _embedded()
    k_low, _ = choose_k(X, k_range=range(2, 9), simplicity_bias=0.20)
    k_high, _ = choose_k(X, k_range=range(2, 9), simplicity_bias=0.0)
    assert k_low <= k_high


def test_k_curve_is_auditable():
    X = _embedded()
    k, scores = choose_k(X, k_range=range(3, 7))
    curve = format_k_curve(scores, k)
    assert "silhouette" in curve and "inertia drop" in curve and "proposed" in curve


# ------------------------------------------------------- interpretability
def test_distinctive_terms_are_cluster_specific_not_corpus_wide():
    X = _embedded()
    labels, _, _ = kmeans(X, 4, seed=42, n_init=10)
    # find the cluster dominated by billing tickets
    billing_cid = None
    for cid in np.unique(labels):
        members = [GROUP_OF[i] for i in np.where(labels == cid)[0]]
        if members and max(set(members), key=members.count) == "billing":
            billing_cid = cid
            break
    assert billing_cid is not None
    terms = distinctive_terms(CORPUS, labels, billing_cid, min_count=2)
    assert any(t in terms for t in ("charged", "subscription", "twice", "billing")), terms


def test_representative_examples_come_from_the_cluster():
    X = _embedded()
    labels, centroids, _ = kmeans(X, 4, seed=42, n_init=10)
    for cid in np.unique(labels):
        ex = representative_examples(CORPUS, X, labels, centroids[cid], cid, n=3)
        assert 0 < len(ex) <= 3
        assert all(e in CORPUS for e in ex)


def test_summaries_cover_every_message():
    X = _embedded()
    labels, centroids, _ = kmeans(X, 4, seed=42, n_init=10)
    s = summarise_clusters(CORPUS, X, labels, centroids)
    assert sum(c.size for c in s) == len(CORPUS)
    assert np.isclose(sum(c.share for c in s), 1.0)
    assert [c.size for c in s] == sorted([c.size for c in s], reverse=True)


def test_markdown_renders():
    X = _embedded()
    labels, centroids, _ = kmeans(X, 4, seed=42, n_init=10)
    md = to_markdown(summarise_clusters(CORPUS, X, labels, centroids))
    assert "cluster" in md and "distinctive terms" in md


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} taxonomy tests passed")

    X = _embedded()
    labels, centroids, _ = kmeans(X, 4, seed=42, n_init=10)
    print(f"\ncluster purity vs known families: {_purity(labels):.2f}")
    k, scores = choose_k(X, k_range=range(2, 8))
    print(f"proposed k = {k}")
    print(format_k_curve(scores, k))
    print()
    print(to_markdown(summarise_clusters(CORPUS, X, labels, centroids, n_examples=2)))
