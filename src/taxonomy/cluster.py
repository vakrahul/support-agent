"""
Intent discovery: cluster the customer messages, then name the clusters.

The brief asks for intents "that you define from the data". Two ways to do that
badly: invent a taxonomy from intuition and pretend it came from the data, or
dump the clusters straight in as intents. We do neither.

The process is:
  1. embed customer messages
  2. k-means over a range of k, pick k by silhouette (with a deliberate bias
     toward FEWER clusters -- see `choose_k`)
  3. for each cluster, surface the distinctive terms and the messages nearest
     the centroid
  4. a human (me) reads those and writes the intent name + definition
  5. the named taxonomy is frozen into config and never re-derived

Step 4 is manual on purpose. Cluster != intent: k-means will happily split
"where is my order" into three clusters by tense and merge two genuinely
different problems that share vocabulary. The clustering is a *proposal*; the
judgement about what an agent should route differently is human.

k-means is implemented here rather than imported so the eval path keeps its
numpy-only dependency footprint, matching the decision made for metrics.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np

from src import config
from src.retrieve.embeddings import tokenize


# --------------------------------------------------------------------------
# k-means (numpy)
# --------------------------------------------------------------------------
def kmeans_plusplus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: spreads initial centroids out, which makes the result
    far less sensitive to the random seed than naive random init."""
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=X.dtype)
    centers[0] = X[rng.integers(n)]
    closest = ((X - centers[0]) ** 2).sum(axis=1)
    for i in range(1, k):
        total = closest.sum()
        if total <= 0:                     # all points identical
            centers[i] = X[rng.integers(n)]
        else:
            centers[i] = X[rng.choice(n, p=closest / total)]
        closest = np.minimum(closest, ((X - centers[i]) ** 2).sum(axis=1))
    return centers


def kmeans(
    X: np.ndarray,
    k: int,
    *,
    seed: int = config.SEED,
    n_init: int = 5,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Lloyd's algorithm with k-means++ init, best of `n_init` restarts.

    Returns (labels, centroids, inertia). Deterministic for a given seed.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = X.shape[0]
    k = max(1, min(k, n))
    best: tuple[np.ndarray, np.ndarray, float] | None = None

    for run in range(n_init):
        rng = np.random.default_rng(seed + run)
        C = kmeans_plusplus_init(X, k, rng)
        labels = np.zeros(n, dtype=int)
        prev = np.inf
        for _ in range(max_iter):
            d = _sq_dists(X, C)
            labels = np.argmin(d, axis=1)
            inertia = float(d[np.arange(n), labels].sum())
            for j in range(k):
                m = labels == j
                if m.any():
                    C[j] = X[m].mean(axis=0)
                else:
                    C[j] = X[rng.integers(n)]   # revive an empty cluster
            if abs(prev - inertia) <= tol * max(prev, 1.0):
                break
            prev = inertia
        if best is None or inertia < best[2]:
            best = (labels.copy(), C.copy(), inertia)

    assert best is not None
    return best


def _sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Pairwise squared euclidean distances, ||x||² - 2x·c + ||c||²."""
    return np.maximum(
        (X**2).sum(1)[:, None] - 2 * X @ C.T + (C**2).sum(1)[None, :], 0.0
    )


def silhouette_score(X: np.ndarray, labels: np.ndarray, sample: int = 1500,
                     seed: int = config.SEED) -> float:
    """Mean silhouette: (b - a) / max(a, b), in [-1, 1]; higher is better.

    Subsampled because it is O(n²). A score near 0 means the clusters overlap --
    which for support tickets is common and honest, not a bug to hide.
    """
    n = X.shape[0]
    if n < 3 or len(set(labels.tolist())) < 2:
        return 0.0
    if n > sample:
        idx = np.random.default_rng(seed).choice(n, sample, replace=False)
        X, labels = X[idx], labels[idx]
        n = sample

    D = np.sqrt(np.maximum(_sq_dists(X, X), 0.0))
    uniq = np.unique(labels)
    sil = np.zeros(n)
    for i in range(n):
        own = labels == labels[i]
        own_n = own.sum() - 1
        if own_n <= 0:
            sil[i] = 0.0
            continue
        a = D[i, own].sum() / own_n
        b = min(D[i, labels == c].mean() for c in uniq if c != labels[i])
        sil[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(sil.mean())


def choose_k(
    X: np.ndarray,
    k_range: Sequence[int] = range(4, 10),
    *,
    seed: int = config.SEED,
    simplicity_bias: float = 0.02,
) -> tuple[int, list[dict]]:
    """Propose k. Two guards, because silhouette alone is not trustworthy here.

    MEASURED BEHAVIOUR (tests/test_taxonomy.py): on a corpus with four known
    intent families, unpenalised silhouette selects the largest k offered --
    it rises monotonically and never finds the true k=4. That is typical for
    short-text clustering: splitting any cluster raises average separation.
    Reporting "silhouette picked k" would therefore be a fake justification.

    So k is constrained two ways instead:

    1. `k_range` is an OPERATIONAL band, not a search over all possibilities.
       Below ~4 the intents are too coarse to route on; above ~9 a 200-example
       golden set leaves ~20 examples per class (per-class F1 becomes noise) and
       a support team cannot maintain that many distinct reply playbooks.

    2. `simplicity_bias` is the price in silhouette demanded before accepting
       one more intent. At the default, recovering one extra cluster must
       improve separation by 0.02 to be worth it. It is a stated preference,
       deliberately visible rather than buried.

    The return value is a PROPOSAL. `scores` carries the full curve plus the
    inertia-drop ratio so a human can see the elbow and overrule it -- which is
    the intended workflow, since cluster != intent.
    """
    scores: list[dict] = []
    prev_inertia: float | None = None
    for k in k_range:
        labels, _, inertia = kmeans(X, k, seed=seed)
        s = silhouette_score(X, labels, seed=seed)
        # relative inertia drop vs the previous k -- the elbow diagnostic
        drop = (prev_inertia - inertia) / prev_inertia if prev_inertia else float("nan")
        prev_inertia = inertia
        scores.append({
            "k": k, "silhouette": s, "inertia": inertia,
            "inertia_drop": drop,
            "adjusted": s - simplicity_bias * k,
        })
    best = max(scores, key=lambda r: r["adjusted"])
    return int(best["k"]), scores


def format_k_curve(scores: Sequence[dict], chosen: int) -> str:
    """Render the selection curve so the choice is auditable, not asserted."""
    lines = ["| k | silhouette | inertia drop | adjusted | |",
             "|---|---|---|---|---|"]
    for s in scores:
        drop = "—" if s["inertia_drop"] != s["inertia_drop"] else f"{s['inertia_drop']:.1%}"
        mark = " **← proposed**" if s["k"] == chosen else ""
        lines.append(
            f"| {s['k']} | {s['silhouette']:+.3f} | {drop} | {s['adjusted']:+.3f} |{mark} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Cluster interpretation
# --------------------------------------------------------------------------
@dataclass
class ClusterSummary:
    cluster_id: int
    size: int
    share: float
    top_terms: list[str]
    examples: list[str]
    proposed_name: str = ""
    definition: str = ""
    notes: str = ""


def distinctive_terms(texts: Sequence[str], labels: np.ndarray, cluster_id: int,
                      top_n: int = 12, min_count: int = 3) -> list[str]:
    """Terms over-represented in this cluster relative to the whole corpus.

    Raw frequency would return "order" for every cluster of an e-commerce brand.
    A log-ratio of in-cluster to global rate surfaces what makes the cluster
    *different*, which is what a human needs in order to name it.
    """
    in_c = Counter()
    total = Counter()
    n_in = 0
    for t, lab in zip(texts, labels):
        toks = set(tokenize(t))
        total.update(toks)
        if lab == cluster_id:
            in_c.update(toks)
            n_in += 1
    if n_in == 0:
        return []
    n_all = len(texts)
    scored = []
    for term, c in in_c.items():
        if c < min_count:
            continue
        p_in = c / n_in
        p_all = total[term] / n_all
        scored.append((np.log((p_in + 1e-6) / (p_all + 1e-6)) * np.log1p(c), term))
    scored.sort(reverse=True)
    return [t for _, t in scored[:top_n]]


def representative_examples(texts: Sequence[str], X: np.ndarray, labels: np.ndarray,
                            centroid: np.ndarray, cluster_id: int,
                            n: int = 5) -> list[str]:
    """Messages nearest the centroid -- the cluster's 'typical' members."""
    idx = np.where(labels == cluster_id)[0]
    if len(idx) == 0:
        return []
    d = ((X[idx] - centroid) ** 2).sum(axis=1)
    return [texts[i] for i in idx[np.argsort(d)[:n]]]


def summarise_clusters(texts: Sequence[str], X: np.ndarray, labels: np.ndarray,
                       centroids: np.ndarray, n_examples: int = 5) -> list[ClusterSummary]:
    out: list[ClusterSummary] = []
    n = len(texts)
    for cid in range(centroids.shape[0]):
        size = int((labels == cid).sum())
        if size == 0:
            continue
        out.append(ClusterSummary(
            cluster_id=cid,
            size=size,
            share=size / n if n else 0.0,
            top_terms=distinctive_terms(texts, labels, cid),
            examples=representative_examples(texts, X, labels, centroids[cid], cid, n_examples),
        ))
    return sorted(out, key=lambda c: -c.size)


def to_markdown(summaries: Sequence[ClusterSummary]) -> str:
    lines = []
    for c in summaries:
        lines.append(f"### cluster {c.cluster_id}  —  {c.size:,} msgs ({c.share:.1%})")
        lines.append(f"**distinctive terms:** {', '.join(c.top_terms) or '—'}\n")
        for ex in c.examples:
            lines.append(f"  - {ex}")
        lines.append("")
    return "\n".join(lines)


def save_draft_taxonomy(summaries: Sequence[ClusterSummary], path=None) -> str:
    """Write a draft for MANUAL naming.

    Left intentionally unnamed: the whole point is that a human reads the
    clusters and decides what an agent should route differently.
    """
    path = path or (config.DATA_SAMPLE / "taxonomy_draft.json")
    payload = {
        "_instructions": (
            "Fill in proposed_name / definition for each cluster, MERGE clusters that "
            "should share one intent, and DELETE any that are noise. Then paste the "
            "result into INTENTS in src/config.py. Clusters are a proposal, not a taxonomy."
        ),
        "clusters": [asdict(c) for c in summaries],
    }
    Pathish = type(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return str(path)
