"""
Retrieval over resolved AmazonHelp cases.

Backend: Qdrant -- embedded local storage by default (no Docker, no account,
reviewer-safe), Qdrant Cloud when QDRANT_URL + QDRANT_API_KEY are set (see
config bare-.env loader). Falls back to a numpy cosine index if qdrant-client
is unavailable, so the eval path never hard-requires it.

Corpus = English + NON-deflection pairs: deflections are handoffs, not
resolutions, and would teach the drafter to deflect (DECISIONS #20).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.eda.brand_select import is_deflection
from src.ingest.language import is_english
from src.retrieve.embeddings import get_embedder

INDEX_DIR = config.DATA_SAMPLE / "retrieval"


def golden_exclusions() -> tuple[set[str], set[str]]:
    """Golden tweet-ids AND exact texts. Ids miss repeated messages sent as
    separate tweets (same text, new id, different reply); texts miss nothing
    but could over-exclude genuine duplicates -- both applied, counts logged."""
    import json as _json
    import re as _re
    ids: set[str] = set()
    texts: set[str] = set()
    gp = config.DATA_GOLDEN / "golden.jsonl"
    if gp.exists():
        for line in gp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = _json.loads(line)
            if r.get("customer_tweet_id"):
                ids.add(str(r["customer_tweet_id"]))
            t = _re.sub(r"\s+", " ", (r.get("text") or "").lower().strip())
            if t:
                texts.add(t)
    return ids, texts


def _corpus(pairs_path: Path) -> pd.DataFrame:
    import re as _re2
    df = pd.read_parquet(pairs_path)
    # DECONTAMINATION: golden rows must never be retrievable at eval time.
    gids, gtexts = golden_exclusions()
    n0 = len(df)
    if gids and "customer_tweet_id" in df.columns:
        df = df[~df["customer_tweet_id"].astype(str).isin(gids)].copy()
    n1 = len(df)
    norm = df["customer_message"].fillna("").str.lower().str.strip().str.replace(
        r"\s+", " ", regex=True)
    df = df[~norm.isin(gtexts)].copy()
    n2 = len(df)
    en = df[df["customer_message"].map(is_english)
            & df["brand_reply"].map(is_english)].copy()
    n_def = int(en["brand_reply"].map(is_deflection).sum())
    res = en[~en["brand_reply"].map(is_deflection)].copy().reset_index(drop=True)
    print(f"[retrieve] pairs={n0} (excluded {n0 - n1} by golden id, "
          f"{n1 - n2} by golden text) en={len(en)} deflected={n_def} "
          f"corpus={len(res)}")
    return res


def _embed(texts: list[str]):
    emb = get_embedder("auto")
    emb.fit(texts)
    V = np.asarray(emb.encode(texts), dtype=np.float32)
    return emb, V


def _qdrant_client():
    from qdrant_client import QdrantClient
    import os
    # Cloud only on explicit opt-in: the free-tier write path is flaky from
    # here (observed WriteTimeout on batched upserts) and the eval must not
    # depend on it. Set QDRANT_BACKEND=cloud to use Qdrant Cloud.
    if (os.getenv("QDRANT_BACKEND", "") == "cloud"
            and config.QDRANT_URL and config.QDRANT_API_KEY):
        print(f"[retrieve] qdrant CLOUD ({config.QDRANT_URL[:30]}...)")
        return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY,
                            timeout=120)
    print(f"[retrieve] qdrant embedded ({config.QDRANT_PATH})")
    return QdrantClient(path=config.QDRANT_PATH)


def build_index(pairs_path: Path | None = None, out_dir: Path | None = None,
                use_qdrant: bool = True) -> Path:
    out_dir = out_dir or INDEX_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = pairs_path or (config.DATA_SAMPLE / "pairs_AmazonHelp.parquet")
    res = _corpus(pairs_path)
    texts = res["customer_message"].fillna("").astype(str).tolist()
    emb, V = _embed(texts)
    dim = int(V.shape[1])

    res[["customer_message", "brand_reply"]].to_parquet(out_dir / "cases.parquet", index=False)
    (out_dir / "embedder.txt").write_text(emb.name, encoding="utf-8")

    if use_qdrant:
        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams
            client = _qdrant_client()
            try:
                client.delete_collection(collection_name=config.QDRANT_COLLECTION)
            except Exception:
                pass  # fresh storage: nothing to delete
            client.create_collection(
                collection_name=config.QDRANT_COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            norms = np.linalg.norm(V, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            Vn = (V / norms).astype(np.float32)
            pts = [PointStruct(id=int(i), vector=Vn[i].tolist(), payload={})
                   for i in range(len(res))]
            for s in range(0, len(pts), 256):
                client.upsert(collection_name=config.QDRANT_COLLECTION,
                              points=pts[s:s + 256])
            (out_dir / "backend.txt").write_text("embedded", encoding="utf-8")
            # Post-condition: storage must exactly match cases.parquet.
            # (Local-mode collection info lies about counts; scroll is truth.)
            got, _ = client.scroll(collection_name=config.QDRANT_COLLECTION,
                                   limit=1, with_vectors=False)
            n_stored = client.count(config.QDRANT_COLLECTION, exact=True).count
            assert n_stored == len(res), (n_stored, len(res))
            print(f"[retrieve] qdrant upserted {len(pts)} points "
                  f"-> {config.QDRANT_COLLECTION} (verified {n_stored})")
            return out_dir
        except ImportError:
            print("[retrieve] qdrant-client missing -> numpy fallback")
    np.save(out_dir / "vectors.npy", V)
    (out_dir / "backend.txt").write_text("numpy", encoding="utf-8")
    print(f"[retrieve] numpy index saved -> {out_dir}")
    return out_dir


class Retriever:
    """Same interface regardless of backend: search(query, top_k) -> cases."""

    def __init__(self, index_dir: Path | None = None):
        self.index_dir = index_dir or INDEX_DIR
        self.cases = pd.read_parquet(self.index_dir / "cases.parquet")
        self._emb = get_embedder("auto")
        self._emb.fit(self.cases["customer_message"].fillna("").astype(str).tolist())
        self._backend = "numpy"
        self._client = None
        self.vectors = None
        backend_file = self.index_dir / "backend.txt"
        want = backend_file.read_text(encoding="utf-8").strip() if backend_file.exists() else "numpy"
        if want in ("embedded", "cloud"):
            try:
                self._client = _qdrant_client()
                self._backend = want
            except Exception as e:  # pragma: no cover
                print(f"[retrieve] qdrant unavailable ({e}) -> numpy fallback")

    def _get_vectors(self) -> np.ndarray:
        if self.vectors is None:
            vec_path = self.index_dir / "vectors.npy"
            if not vec_path.exists():
                # Qdrant lock was held by another process — build numpy cache on the fly
                print("[retrieve] vectors.npy missing, building numpy fallback index...")
                texts = self.cases["customer_message"].fillna("").astype(str).tolist()
                V = np.asarray(self._emb.encode(texts), dtype=np.float32)
                np.save(vec_path, V)
                print(f"[retrieve] numpy fallback saved -> {vec_path}")
            V = np.load(vec_path)
            norms = np.linalg.norm(V, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.vectors = (V / norms).astype(np.float32)
        return self.vectors

    def _query_vec(self, query: str) -> np.ndarray:
        q = np.asarray(self._emb.encode([query or ""]), dtype=np.float32)[0]
        n = float(np.linalg.norm(q))
        return (q / n).astype(np.float32) if n else q

    def search(self, query: str, top_k: int = 3):
        if self._client is not None:
            try:
                qvec = self._query_vec(query).tolist()
                hits = None
                if hasattr(self._client, "search"):
                    hits = self._client.search(
                        collection_name=config.QDRANT_COLLECTION,
                        query_vector=qvec, limit=top_k)
                elif hasattr(self._client, "query_points"):
                    res = self._client.query_points(
                        collection_name=config.QDRANT_COLLECTION,
                        query=qvec, limit=top_k)
                    hits = getattr(res, "points", res)
                
                if hits:
                    out = []
                    for h in hits:
                        i = int(h.id)
                        if 0 <= i < len(self.cases):  # skip stale ids, never crash
                            out.append({
                                "case_id": i,
                                "customer_message": str(self.cases.iloc[i]["customer_message"]),
                                "brand_reply": str(self.cases.iloc[i]["brand_reply"]),
                                "similarity": float(h.score)})
                    if out:
                        return out
            except Exception as e:
                # Fall through to numpy cosine search
                pass

        q = self._query_vec(query)
        V = self._get_vectors()
        sims = V @ q
        idx = np.argsort(-sims)[:top_k]
        return [{"case_id": int(i),
                 "customer_message": str(self.cases.iloc[int(i)]["customer_message"]),
                 "brand_reply": str(self.cases.iloc[int(i)]["brand_reply"]),
                 "similarity": float(sims[int(i)])} for i in idx]
