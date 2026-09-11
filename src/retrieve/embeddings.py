"""
Pluggable text embeddings.

Two implementations behind one interface:

  SentenceTransformerEmbedder -- the production path. Semantic, pretrained,
      handles paraphrase ("can't log in" ~ "unable to sign in") which is
      exactly what retrieval needs.

  TfidfSvdEmbedder -- a pure-numpy LSA fallback (TF-IDF -> randomized SVD ->
      L2 normalise). No torch, no downloads, fully deterministic.

Why bother with the fallback? Because `sentence-transformers` drags in torch
(~2GB) and downloads model weights on first use. A reviewer on a fresh clone,
behind a proxy, or on a laptop without patience would then be unable to run the
pipeline at all. The fallback keeps every stage executable with numpy alone --
the retrieval quality is lower, and the report says so rather than pretending
the two are equivalent.

Both return L2-normalised vectors, so cosine similarity is a plain dot product.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Protocol, Sequence

import numpy as np

from src import config

_TOKEN_RE = re.compile(r"[a-z0-9']+")

# Function words carry no intent signal but dominate TF-IDF counts.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "cant",
    "could", "did", "do", "does", "doing", "dont", "for", "from", "had", "has",
    "have", "he", "her", "hers", "him", "his", "how", "i", "if", "im", "in", "is",
    "it", "its", "ive", "me", "my", "no", "not", "of", "on", "or", "our", "out",
    "she", "so", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "to", "too", "up", "us", "was", "we", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "would", "you", "your", "youre",
    "am", "been", "being", "just", "get", "got", "now", "still", "any", "all",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower())
            if len(t) > 2 and t not in STOPWORDS]


class Embedder(Protocol):
    dim: int
    name: str

    def fit(self, texts: Sequence[str]) -> "Embedder": ...
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


# --------------------------------------------------------------------------
# Pure-numpy LSA
# --------------------------------------------------------------------------
class TfidfSvdEmbedder:
    """TF-IDF -> randomized truncated SVD -> L2 normalise.

    Deterministic given a seed. `fit` learns the vocabulary and IDF weights from
    the corpus, so it must be fit on the SAME corpus that will be indexed --
    otherwise unseen query terms silently drop to zero.
    """

    name = "tfidf-svd"

    def __init__(self, dim: int = 256, max_features: int = 2000,
                 min_df: int = 2, seed: int = config.SEED):
        self.dim = dim
        self.max_features = max_features
        self.min_df = min_df
        self.seed = seed
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.components: np.ndarray | None = None   # (dim, vocab)

    # -- fitting ----------------------------------------------------------
    def fit(self, texts: Sequence[str]) -> "TfidfSvdEmbedder":
        docs = [tokenize(t) for t in texts]
        df = Counter()
        for d in docs:
            df.update(set(d))
        # drop hapax terms (noise) then keep the most frequent
        kept = [(t, c) for t, c in df.items() if c >= self.min_df]
        kept.sort(key=lambda kv: (-kv[1], kv[0]))  # deterministic tie-break
        kept = kept[: self.max_features]
        self.vocab = {t: i for i, (t, _) in enumerate(kept)}

        n = max(len(docs), 1)
        self.idf = np.ones(len(self.vocab), dtype=np.float32)
        for t, i in self.vocab.items():
            self.idf[i] = math.log((1 + n) / (1 + df[t])) + 1.0

        X = self._tfidf(docs)                      # (n_docs, vocab)
        k = min(self.dim, min(X.shape) - 1) if min(X.shape) > 1 else 1
        self.components = _randomized_svd_components(X, k, seed=self.seed)
        self.dim = self.components.shape[0]
        return self

    def _tfidf(self, docs: Sequence[Sequence[str]]) -> np.ndarray:
        X = np.zeros((len(docs), max(len(self.vocab), 1)), dtype=np.float32)
        for r, d in enumerate(docs):
            if not d:
                continue
            counts = Counter(t for t in d if t in self.vocab)
            if not counts:
                continue
            total = sum(counts.values())
            for t, c in counts.items():
                X[r, self.vocab[t]] = (c / total) * self.idf[self.vocab[t]]
        return X

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.components is None:
            raise RuntimeError("call fit() before encode()")
        X = self._tfidf([tokenize(t) for t in texts])
        return _l2_normalize(X @ self.components.T)


def _randomized_svd_components(X: np.ndarray, k: int, seed: int = 42,
                               n_oversample: int = 10, n_iter: int = 2) -> np.ndarray:
    """Top-k right singular vectors via randomized SVD (Halko et al., 2011).

    Cheaper and more memory-friendly than a full SVD, and numpy-only.
    Reference: Halko, Martinsson & Tropp, "Finding Structure with Randomness".
    """
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape
    k = max(1, min(k, n_features, max(n_samples - 1, 1)))
    p = min(k + n_oversample, n_features)

    Omega = rng.standard_normal((n_features, p)).astype(np.float32)
    Y = X @ Omega
    Q, _ = np.linalg.qr(Y)
    for _ in range(n_iter):                 # power iterations sharpen the range
        Q, _ = np.linalg.qr(X.T @ Q)
        Q, _ = np.linalg.qr(X @ Q)
    B = Q.T @ X                             # (p, n_features)
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    return np.ascontiguousarray(Vt[:k]).astype(np.float32)


def _l2_normalize(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (M / norms).astype(np.float32)


# --------------------------------------------------------------------------
# sentence-transformers
# --------------------------------------------------------------------------
class SentenceTransformerEmbedder:
    """Production embedder. Lazy import so the module never hard-requires torch."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = config.EMBED_MODEL):
        self.model_name = model_name
        self.dim = config.EMBED_DIM
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy
            self._model = SentenceTransformer(self.model_name)
            self.dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def fit(self, texts: Sequence[str]) -> "SentenceTransformerEmbedder":
        self._ensure()   # pretrained: nothing to learn from the corpus
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        m = self._ensure()
        v = m.encode(list(texts), convert_to_numpy=True,
                     show_progress_bar=False, normalize_embeddings=True)
        return v.astype(np.float32)


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
def get_embedder(kind: str = "auto", **kwargs) -> Embedder:
    """`auto` prefers sentence-transformers and degrades to LSA if unavailable.

    The degradation is announced, not silent -- a quietly weaker retriever would
    make the reported numbers uninterpretable.
    """
    if kind in ("auto", "st", "sentence-transformers"):
        try:
            import sentence_transformers  # noqa: F401
            st_kwargs = {k: v for k, v in kwargs.items() if k in ("model_name",)}
            return SentenceTransformerEmbedder(**st_kwargs)
        except ImportError:
            if kind != "auto":
                raise
            print("[embeddings] sentence-transformers unavailable -> "
                  "falling back to TF-IDF+SVD (lower retrieval quality; "
                  "noted in the report)")
    return TfidfSvdEmbedder(**kwargs)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between row-normalised matrices == dot product."""
    return a @ b.T
