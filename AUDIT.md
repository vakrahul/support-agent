# AUDIT — repo vs the Hiver take-home contract (2026-09-10)

Method: full repo inspection + measured checks (leakage overlap, cache/model
tags, saved metrics). No code changed. Verdicts: PASS / PARTIAL / FAIL.
Priority: P0 = submission blocker, P1 = important, P2 = polish.

## What already works (keep, don't rewrite)

- Thread reconstruction + cleaning + language filter, pinned by tests.
- Brand selection by measured resolution density (strongest part of the repo).
- Metrics hand-implemented + tested (acc, macro/weighted F1, binary, κ, ρ, bootstrap CI).
- Replay cache (hard CacheMiss, prompt-versioned keys), Qdrant embedded, local embeddings.
- Golden 150 stratified with 4 logged adjudication passes + GOLDEN_NOTE.md.
- Three systems end-to-end on locked test-100 with saved outputs; gate with
  reason codes; grounding validator; report + misleading section exist.

## P0 — submission blockers

**P0-1. Retrieval leakage: 100/150 golden messages verbatim in the corpus.**
Golden was sampled from the same 6k subsample that feeds the index, and the
golden rows' own pairs (same text → same reference reply) were never removed.
Top-1 self-hits observed (sim=1.0). Inflates similarity, agreement, draft
quality. Contract §6 + §20 violated. Fix: decontaminate corpus by tweet_id,
rebuild index, add `scripts/check_leakage.py` (exact + near-dupe + id overlap)
that fails loudly, re-run affected rows.
**P0-2. Baseline leakage: 108/150 golden texts inside B1's weak-train sample.**
TF-IDF sampled 3,000 rows from the same pool with no exclusion. B1 partly
trained on test. Fix with P0-1: exclude golden ids from weak-label sampling.
**P0-3. Test set was never frozen.** Label passes 1–4 reshuffled the seeded
cal/test split 3 times, so earlier reported numbers refer to different test
sets. No committed split artifact exists. Fix: commit `data/golden/split.json`
(cal/test ids), make `run_eval.py` load it, never re-derive. (Matches
reviewer change #1; enables change #3 "improve without touching frozen test".)
**P0-4. Human agreement n=16 paired, contract wants 40–60.** κ≈0 on n=16 is
directional noise. Fix: hand-score to ≥50 paired (reviewer change #2) and keep
human scores primary. No API needed — pure reading work.
**P0-5. No leakage tests, no escalation-policy tests, no response-schema
tests.** Contract §28. `tests/` covers metrics/ingest/brand/taxonomy/embeddings/
llm-cache only. gate.py, reply.py, judge.py have zero tests.

## P1 — important (weakens trust if missing)

- **P1-1. No retrieval evaluation (§13).** No Recall@K / Precision@K at 1/3/5,
  no intent-consistency of retrieved sets. Proxies (max_sim, agreement) exist
  but unreported as retrieval metrics.
- **P1-2. No ablation (§21).** LLM±retrieval, intent on/off unmeasured. The
  "why is the architecture better" claim is currently asserted, and B1's
  0.644 vs 0.676 makes this the obvious reviewer question.
- **P1-3. No CIs on headlines (§3, §11).** `bootstrap_ci` exists, never used
  in report/README. Unsafe 0.13 (n≈29 autos) without an interval is the exact
  sin §35 warns about.
- **P1-4. Weighted F1 + confusion matrix not saved (§11, §32).** Computed
  ad-hoc, absent from `eval_results.json`. Result table incomplete.
- **P1-5. No cost/latency (§27).** Zero token/latency/cost tracking anywhere.
- **P1-6. Failure analysis lacks frequencies (§22).** Report §5 has 5 modes +
  examples but no counts, no expected-vs-actual structure.
- **P1-7. B0 inconsistent (§9).** Code hardcodes `delivery_delay`; measured
  majority of the weak-label pool is `other_unclear`. Rename/document as
  "most-frequent golden intent" or switch to measured majority.
- **P1-8. Missing docs:** `docs/brand_selection.md` (§4), `docs/intent_taxonomy.md`
  + `data/intent_schema.json` (§7 — taxonomy lives only in `config.py` +
  prose), `docs/golden_set.md` (exists as GOLDEN_NOTE.md — rename/keep both
  with pointer), `docs/judge_validation.md` (rubric only in code prompt, §14/15).
- **P1-9. Missing `.env.example` (§29).** Secrets handling otherwise OK
  (.env gitignored — repo has no git yet, verify on first push; keys never
  printed in code/logs).
- **P1-10. README/report contradictions (reviewer change #5).** Status
  checklist shows taxonomy/golden/classifier/retrieval/escalation/judge/report
  unchecked (all done); duplicated intro paragraph; stale "Phase 4 lands"
  text; generator/judge described as flash/pro but pinned lite/3.5-flash;
  `data/raw/` path wrong (real: `data/twcs/`); "`make eval` ~2 min" unverified;
  "25 tests" count stale.
- **P1-11. DECISIONS.md has 28 entries; contract wants 10–15 (§30).** Trim to
  15 max with WHAT/WHY/alternatives/tradeoff each; move overflow to report
  appendix or GOLDEN_NOTE.
- **P1-12. Golden auto-labelling (§8).** Rule-drafted then 4 human passes is
  defensible but must be stated up front in `docs/golden_set.md`, with
  per-example `labeler`/`notes` fields (currently only `annotator` string).
- **P1-13. Judge blindness (§15).** Judge prompt doesn't identify system
  origin — OK — but undocumented; B0's identical canned text shares one cache
  key (fine) — undocumented. State both.

## P2 — polish

- Temporal vs random split unjustified (§6 — random stratified used; add one
  paragraph justifying or acknowledge).
- No run manifest (§26: timestamp/model/prompt-version/seed per experiment;
  partially covered by cache keys).
- `outputs/` vs contract's `results/`, `golden.jsonl` vs `golden_set.jsonl`
  naming (§36 — cosmetic; adapt-by-reference, don't churn).
- No architecture diagram file (§33 — README ASCII exists; export one PNG).
- No `REVIEW.md` skeptical audit (§38).
- Near-dupe leakage minor: 5/150 share a 60-char prefix with another row.
- Report is ~4 pages of 6 allowed — room for the curve table (§: reviewer
  change #4: coverage-vs-unsafe as central result, currently only endpoints).

## Recommended order (maps to reviewer changes #1–#5)

1. Freeze taxonomy + split + decontaminated corpus (P0-1/2/3) — one rebuild.
2. Human scoring to ≥50 paired (P0-4).
3. Weak-class work using cal-only signals, frozen test untouched.
4. Coverage-vs-unsafe curve as headline + CIs + full result table (P1-3/4, P2).
5. README/report/DECISIONS cleanup + missing docs + tests (P1-5…13, P0-5).
6. Ablation + retrieval metrics + cost/latency (P1-1/2/5).
7. `REVIEW.md` + final replay verify.
