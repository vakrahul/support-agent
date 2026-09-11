# Data — pipeline stages and filtering decisions

`data/twcs/twcs.csv` (raw Kaggle dump, ~3M rows, gitignored — obtain from
thoughtvector/customer-support-on-twitter)
→ `src/ingest/threads.py` (chunked pair reconstruction, 5k-row overlap window;
  cleaning: @mentions stripped, URLs → `<URL>` token, agent signatures cut,
  length filters 12–400 chars, pair dedup)
→ 510,927 pairs / 108 brands (1.2M-row scan)
→ brand filter AmazonHelp (85,291 pairs)
→ `data/sample/pairs_AmazonHelp.parquet` (6,000-row working subsample, seed 42)
→ English-only both sides (4,860) → minus deflections (1,425) → minus 101
  golden-contaminated rows (ids + exact texts)
→ retrieval corpus 3,334 cases (`data/sample/retrieval/cases.parquet`)
→ golden 150 (`data/golden/golden.jsonl`), split 50/100 (`split.json`)
→ TF-IDF weak-train pool: same 6,000 minus golden (sampled 3,000, seed 42)

Dropped-row accounting per stage prints in build logs. No stage silently
discards: counts are logged (`[retrieve]`, `[leakage]`, `[eda]` lines).
