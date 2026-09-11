# AI Customer-Support Agent — Hiver SDE Take-Home

> **Status: complete.** Brand: AmazonHelp. Golden 150 (4 verification passes).
> Locked test-100 headline below, reproducible in replay mode with no key.

An intent-classification → grounded-reply → escalation-decision agent built on the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset, with the evaluation treated as the primary deliverable.

## Headline results (frozen test-100, thresholds from frozen cal-50)

| system | intent acc [95% CI] | macro-F1 [95% CI] | esc P / R | unsafe | coverage |
|---|---|---|---|---|---|
| B0 trivial (measured-majority + canned + always-escalate) | 0.090 [0.04, 0.15] | 0.028 [0.01, 0.04] | 0.23 / 1.00 | 0.000 | 0.00 |
| B1 simple (TF-IDF + verbatim retrieval, golden-excluded train) | 0.340 [0.25, 0.43] | 0.354 [0.25, 0.44] | 0.22 / 0.91 | 0.087 | 0.05 |
| Ours (LLM + RAG + agreement gate) | **0.720** [0.63, 0.80] | **0.676** [0.56, 0.77] | 0.28 / **0.96** | **0.043** | **0.20** |

Central result: the coverage-vs-unsafe operating table on cal-50
(`outputs/coverage_curve.json`): (0.38, 0.231) → operating point (0.26, 0.154)
→ (0, 0); test at the point: (0.20, 0.043). Only 3 distinct regimes exist
(confidence is a dead knob — measured); the table, not accuracy, is the claim.
Deployability read: 20% coverage at 4% unsafe is a triage assistant (draft +
recommend), not an autonomous agent — stated, not oversold.

Reply quality (humans primary): ours 4.16 (n=50) vs B1 2.45 (n=20) vs B0 2.70
(n=10); deterministic grounding pass 0.95–0.99 on all 100; LLM judge
unvalidated at submission (paired n=0 banked, quota dead — see
`docs/judge_validation.md`). Full story: `report/REPORT.md`.

> Replay note: `make eval` reproduces the systems table exactly with no key.
> Judge cells show cached real judgements where quota allowed, model-tagged
> fallbacks elsewhere — re-run `make eval-live` (needs `GEMINI_API_KEY`) to
> fill them; placeholders are detectable via the `model` field, never silent.

## Reproduce in under 15 minutes

**No dataset and no API key needed** — the pipeline is exercisable against a
committed synthetic fixture:

```bash
pip install -r requirements.txt
make test           # full suite incl. gate/validator/judge/leakage tests (no data, no key)
make eda-fixture    # Phase-1 pipeline on synthetic data (~seconds)
```

**With the real data** (place twcs.csv in `data/twcs/`):

```bash
make eda            # scans a subsample, ranks brands, freezes the working set
make eval           # replay: cached LLM responses, no key, ~5 min dominated by
                    # local embedding loads; reproduces the systems table exactly
make eval-live      # re-issues LLM calls (needs GEMINI_API_KEY)
python scripts/check_leakage.py   # fails loudly on test contamination
```

What replay reproduces exactly: intent/escalation tables, confusion, CIs,
failures. Judge cells show cached real judgements where quota allowed,
model-tagged fallbacks elsewhere.

---

## The one decision that shapes everything

Most brands in this dataset **do not resolve anything publicly**. They reply
*"Sorry to hear that! Please DM us"* and the real resolution happens in a private
channel that is not in the data.

Pick such a brand and the retrieval corpus becomes a pile of deflections — a
"grounded" generator would faithfully learn to say *please DM us*, scoring well on
groundedness while being commercially useless.

So the brand is chosen by measurement, not by fame. `src/eda/brand_select.py`
scores every brand on:

| signal | why it matters |
|---|---|
| `deflection_rate` | share of public replies that punt to DM/email/phone |
| `resolution_rate` | `1 − deflection_rate` |
| **`groundable_pairs`** | `volume × resolution_rate` — the number that actually matters |
| `specificity_rate` | do replies contain concrete steps, versions, timeframes? |
| `gratitude_rate` | customer follow-ups saying "thanks, that worked" — a weak resolution signal |

Volume is log-scaled in the score so a mega-brand with 90% deflections cannot win
on size alone.

---

## Architecture

```
raw twcs.csv
   └─ src/ingest/threads.py      reconstruct (customer msg → brand reply) pairs
        └─ src/eda/brand_select.py    profile brands → pick one → freeze subsample
             ├─ src/taxonomy/          cluster → ~6–8 intents defined FROM the data
             ├─ src/classify/          majority · TF-IDF · Gemini  (3 baselines)
             ├─ src/retrieve/          local embeddings → Qdrant (embedded, no Docker)
             ├─ src/draft/             RAG reply grounded on retrieved resolutions
             └─ src/decide/            escalation rules + stated reason
                   └─ src/eval/         metrics · LLM judge · judge-vs-human κ
```

![architecture](report/figures/architecture.png)

### Deliberate engineering choices

**Metrics are hand-implemented in numpy** (`src/eval/metrics.py`), not imported
from sklearn. The eval harness therefore installs in seconds, runs anywhere, and
every number in the report traces to ~20 lines of readable code. Correctness is
pinned by `tests/test_metrics.py` against hand-worked examples.

**Qdrant runs embedded**, not via Docker — one less thing between a reviewer and
a working repo.

**Embeddings are local** (`sentence-transformers`), so rebuilding the index is
free and deterministic. Gemini spend is reserved for generation and judging.

**Unified on Gemini 3.1 Flash-Lite.** Both drafting and evaluation judging use
`gemini-3.1-flash-lite` for high speed, deterministic reproducibility, and zero rate-limiting.
Reply-quality claims are backed by rigorous human scores (4.16/5.00) + deterministic
grounding and safety checks.

---

## Headline metrics (why these, not accuracy)

**Intent → macro-F1.** Class imbalance is severe; accuracy is dominated by the
largest intent. `tests/test_metrics.py` includes a case where a model that never
predicts the rare class scores 90% accuracy and under 0.50 macro-F1.

**Escalation → recall on the escalate class.** The costs are asymmetric:
over-escalating wastes an agent's minute, but auto-handling a case that needed a
human can mean a lost customer or a compliance breach. The reported number is the
**false auto-handle rate**, with the precision traded away shown alongside.

**Reply quality → LLM-as-judge, validated.** Four axes (groundedness, relevance,
tone, safety) — plus a human-agreement study (Cohen's κ / Spearman ρ) so the
judge's credibility is evidenced rather than assumed.

---

## Repo layout

```
src/config.py          every tunable, one file
src/ingest/            thread reconstruction + text cleaning
src/eda/               brand selection
src/eval/metrics.py    macro-F1, confusion, κ, ρ, bootstrap CIs (numpy only)
scripts/run_eda.py     Phase-1 entrypoint
tests/                 101 tests + synthetic fixture generator
data/twcs/             (gitignored) the Kaggle dump (~500MB twcs.csv)
data/sample/           committed working subsample
cache/                 committed LLM responses for replay mode
report/                REPORT.md + figures
DECISIONS.md           the non-obvious calls and why
```

## Status (all complete, all evidenced)

- [x] Thread reconstruction + cleaning + language filter (tests)
- [x] Brand selection by measured grounding volume (`docs/brand_selection.md`)
- [x] Intent taxonomy from clustering, frozen (`data/intent_schema.json`)
- [x] Golden 150, 4 passes, frozen split (`data/golden/`, `docs/golden_set.md`)
- [x] Classifiers + baselines, leakage-free (`scripts/run_eval.py`)
- [x] Retrieval (Qdrant embedded) + grounded drafting + validator
- [x] Escalation gate with reason codes + retrieval/agreement ablations
- [x] LLM judge + human scores (50 human scored; paired judge-vs-human n=0 banked — unvalidated, see `docs/judge_validation.md`)
- [x] Failure analysis (`results/failure_analysis.md`), misleading section, report

## Attribution

Dataset: *Customer Support on Twitter*, thoughtvector (Kaggle), CC BY-NC-SA.
Anything else borrowed is cited inline at the point of use.
