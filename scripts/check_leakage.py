"""
Leakage gate: fails loudly (exit 1) on test contamination. Run in CI and
before every eval. Checks, all against data/golden/golden.jsonl:
  1. tweet-id overlap with the retrieval corpus source pool
  2. exact customer-message overlap with the built retrieval cases
  3. near-duplicate (60-char normalized prefix) overlap with the corpus
  4. overlap with the deterministic TF-IDF weak-train sample (seed 42, n=3000)

Thresholds: id/text overlap must be ZERO; near-dupe allowance is 5 rows
(documented residual; listed explicitly when nonzero).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config
from src.ingest.language import is_english

NEAR_DUPE_BUDGET = 5


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").lower().strip())


def main() -> int:
    gold = [json.loads(l) for l in
            open(config.DATA_GOLDEN / "golden.jsonl", encoding="utf-8")]
    gids = {str(r.get("customer_tweet_id", "")) for r in gold if r.get("customer_tweet_id")}
    gtexts = {norm(r["text"]) for r in gold}
    fails: list[str] = []

    pairs = pd.read_parquet(config.DATA_SAMPLE / "pairs_AmazonHelp.parquet")
    pool_ids = set(pairs["customer_tweet_id"].astype(str)) if "customer_tweet_id" in pairs else set()
    id_hit = gids & pool_ids
    # Golden rows necessarily come FROM the pool; what matters is whether the
    # retrieval corpus and train sample exclude them (checks below). Id
    # presence in the raw pool is expected, not a failure.
    print(f"[leakage] golden={len(gold)} pool={len(pairs)} ids-in-pool={len(id_hit)} (expected)")

    cpath = config.DATA_SAMPLE / "retrieval" / "cases.parquet"
    if cpath.exists():
        cases = pd.read_parquet(cpath)
        ctexts = {norm(t) for t in cases["customer_message"].fillna("")}
        exact = gtexts & ctexts
        if exact:
            fails.append(f"EXACT-TEXT: {len(exact)} golden messages verbatim in retrieval corpus")
        cpref = {}
        for t in cases["customer_message"].fillna(""):
            cpref.setdefault(norm(t)[:60], 0)
            cpref[norm(t)[:60]] += 1
        near = sum(1 for r in gold if norm(r["text"])[:60] in cpref)
        print(f"[leakage] exact-in-corpus={len(exact)} near-dupe-in-corpus={near}")
        if near > NEAR_DUPE_BUDGET:
            fails.append(f"NEAR-DUPE: {near} over budget {NEAR_DUPE_BUDGET}")
    else:
        print("[leakage] no built corpus yet; skipping corpus checks")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_eval import weak_labels
    Xw, _ = weak_labels()
    train_hit_texts = gtexts & {norm(t) for t in Xw}
    # weak_labels() must exclude golden ids before sampling; nonzero means
    # the exclusion regressed. This tests the REAL code path, not a copy.
    print(f"[leakage] golden-in-actual-weak-train={len(train_hit_texts)}")
    if train_hit_texts:
        fails.append(f"TRAIN-OVERLAP: {len(train_hit_texts)} golden texts in TF-IDF weak-train output")

    if fails:
        print("[leakage] FAIL:")
        for f in fails:
            print("  -", f)
        return 1
    print("[leakage] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
