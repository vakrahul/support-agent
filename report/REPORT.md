# Report: an AI support agent for AmazonHelp that knows when to shut up

## 1. Problem framing: what "good" means

Good = **safe automation, not maximum automation**. For AmazonHelp, a wrong
auto-handled reply (invented refund date, wrong troubleshooting, missed fraud)
costs far more than an unnecessary escalation costs an agent's minute. So the
headline result is not accuracy but the **coverage-vs-unsafe curve**: how much
routine work we auto-handle at each level of false auto-handling of
must-escalate cases.

The agent does three things per message: (1) classify into 5 operational
intents + `other_unclear`, (2) draft a reply grounded ONLY in retrieved past
AmazonHelp resolutions, (3) emit auto/human + a machine-readable reason code.

**Non-goals:** multi-turn dialogue memory; fine-tuning; multilingual handling
(English-only by explicit filter); all-brand coverage (one brand, deep); live
posting (draft/approve only); Qdrant Cloud in the eval path.

## 2. Data & brand selection

Primary: `twcs.csv` (~3M tweets). Rebuilt (customer → brand reply) pairs with
follow-ups: 510,927 pairs across 108 brands from a 1.2M-row scan. Working set:
6,000 AmazonHelp pairs. Chose **AmazonHelp** (85k pairs, largest absolute
grounding pool) with eyes open: rank 14 on resolution *rate* (70.1%, 29.9%
deflections, 19% non-English) but #1 on groundable volume (48.6k). Table:
`data/sample/brand_profiles.csv`; chart: `report/figures/brand_selection.png`;
method: `docs/brand_selection.md`.

Key trap found and measured: most big brands "resolve" by replying "please DM
us" — a grounding corpus of deflections teaches the generator to deflect while
scoring perfectly on groundedness. We filter deflections (URL-aware,
self-service override, data-mined patterns) OUT of the retrieval corpus, and
score deflection ONLY over English pairs (Spanish "DM us" matches no English
regex; without this AmazonHelp scored a bogus 96.9% resolution rate).

Gratitude check failed honestly: customers thank deflections slightly MORE
(+compliant "thanks, will DM"), lift ≈ 0 even after cleaning. Gratitude is
politeness, not ground truth — kept as diagnostic only.

Leakage control (§6 of the build): golden ids AND exact texts are excluded from
both the retrieval corpus (3,334 cases) and the TF-IDF weak-train pool;
`scripts/check_leakage.py` fails loudly (exact 0, near-dupe 0, train 0 —
verified). The audit caught the pre-fix state leaking 100/150 golden messages
into the corpus and 108/150 into B1's training pool; B1's macro-F1 fell
0.644 → 0.354 after cleaning. That collapse is reported, not hidden.

## 3. System

`message → intent (Gemini 3.1-flash-lite, few-shot, JSON) → Qdrant embedded
retrieval (top-3 of 3,334 English non-deflection cases) → risk rules → RAG draft
→ code grounding validator (cited ids must exist; promise patterns must be
verbatim in evidence) → gate`. Gate autos ONLY on: known intent + score +
similarity + **intent-evidence agreement ≥2/3** + low risk + validator pass.
`model_score` is an uncalibrated score, never a probability (measured: wrong
intents average 0.93 — the confidence threshold is empirically dead weight;
agreement does the real filtering). Retrieval intent-consistency: 0.47 / 0.63 /
0.74 at K=1/3/5; self-hits 0.

## 4. Results vs baselines (frozen test-100; thresholds from frozen cal-50)

| system | intent acc [95% CI] | macro-F1 [95% CI] | esc P / R | unsafe | coverage |
|---|---|---|---|---|---|
| B0 trivial (measured-majority `other_unclear` + canned + always-escalate) | 0.090 [0.04, 0.15] | 0.028 [0.01, 0.04] | 0.23 / 1.00 | 0.000 | 0.00 |
| B1 simple (TF-IDF LogReg + verbatim retrieval, golden-excluded train) | 0.340 [0.25, 0.43] | 0.354 [0.25, 0.44] | 0.22 / 0.91 | 0.087 | 0.05 |
| Ours (LLM + RAG + agreement gate) | **0.720** [0.63, 0.80] | **0.676** [0.56, 0.77] | 0.28 / **0.96** | **0.043** | **0.20** |

Weighted F1: 0.015 / 0.371 / 0.711. Confusion matrices saved in
`outputs/eval_results.json` (ours: only miss pattern of note is delay↔missing,
9 cases). Escalation confusion (ours): 22 caught / 58 over-escalated / 1 missed
/ 19 correctly auto-handled. Reason codes: WEAK_EVIDENCE 61, OK_AUTO 20,
UNKNOWN_INTENT 13, HIGH_RISK 5, GROUNDING_FAIL 1 — the gate over-escalates
by design (precision 0.28 is the price of recall 0.96).

Per-class (ours): refund F1 0.83 (n=27), device 0.84 (n=11), delay 0.75 (n=27),
other 0.64 (n=9), missing 0.53 (n=19), status 0.46 (n=7). Status/other CIs span
±0.2 — unmeasurable at this n, stated not hidden.

Central result — coverage-vs-unsafe on cal-50 (full table
`outputs/coverage_curve.json`): (0.38 cov, 0.231 unsafe) → operating point
(0.26, 0.154) → (0, 0). Test at the operating point: (0.20, 0.043). The table,
not the 0.72 accuracy, is the claim — 35 grid rows collapse to 3 distinct
operating regimes because the confidence knob is empirically dead (scores are
bimodal ~0/0.9), which is itself a finding. Gate ablation (no new calls,
contract-§18 definition throughout): dropping agreement → coverage 0.60 at
unsafe 0.261; dropping sim too → 0.78/0.391. Agreement is what makes
automation safe.

Reply quality (human-primary by design): ours 4.16 (n=50) vs B1 verbatim 2.45
(n=20) vs B0 canned 2.70 (n=10), same 1–5 groundedness rubric. Verbatim copies
score low because real brand replies are often fragments ("(3/3)" tails) or
wrong-context; canned is safe but generic. Deterministic checks on all 100:
grounding pass 0.95–0.99, evidence cited 0.99, promise-pattern violations 1
(a legit "$1 authorization hold" the regex over-fires on — §5 mode 3).
LLM judge (gemini-3.1-flash-lite, blinded,
structured JSON): quota-capped to n=1 banked judgement on current drafts, so
NO judge headline is reported. A pre-decontamination 16-pair pilot existed but
judged superseded drafts; its statistics are not quoted here (see
`outputs/judge_agreement.json`, which now contains only banked facts).
Reply-quality claims rest on the 50 human scores + deterministic checks above.
Claiming otherwise would be fabrication.

Cost/latency: 436 cached calls, ~244k tokens, $0 actual spend (free tier;
≈$0.07 at paid list). Retrieval p50 32ms / p95 39ms local; LLM classify
~2–4s, drafts similar (observed ranges).

## 5. Failure analysis (top 5, 29 test failures, all real)

`results/failure_analysis.md` has each with frequency, expected vs actual, real
example, why, hypothesis, fix. Summary:
1. **Delay↔missing boundary (9/29, 31%)** — late vs lost share vocabulary
   (amz-012, 082, 005…); rules, LLM, and annotators all split here. Fix:
   "failed attempt" third state.
2. **Cancelled orders over-trigger refund (4/29)** — amz-067/111/023/134:
   model over-applies the refund-priority rule where gold wants status. Fix:
   refund only on explicit money words.
3. **Noisy rants punted to other (5/29)** — amz-128/031/096/098/088: safe
   direction (all escalated) but intent recall suffers; few-shots lack rants.
4. **Tracking/returns misrouted (4/29)** — amz-119/028/125/009: carrier and
   return-to-sender vocabulary sits equidistant from three intents.
5. **Residual (7/29)** — genuinely ambiguous amz-142 ("when is my echo
   coming?", device vs delivery); validator regex over-fire on amz-051 "$1
   authorization hold"; one escalation miss amz-102 "No tracking???" (12 chars
   auto-handled — gate needs the golden length floor ported in).

## 6. What is misleading about my headline number? (mandatory)

- Macro-F1 0.676: delay+refund are 54% of test; status (n=7)/other (n=9) F1s
  are noise (CI table above). The per-class row, not the headline, is honest.
- Unsafe 0.043 = 1 missed of 23 must-escalate; 95% CI [0.00, 0.15]. The 5%
  target was aspirational; the curve is the claim.
- Flash-lite outputs vary run-to-run even at temperature 0: a live re-run
  measured cal coverage 0.40 vs replay-pinned 0.26. Replay cache pins exact
  numbers; live numbers wobble. Reported numbers are the pinned ones.
- Golden: 4 adjudication passes by ONE annotator; delay↔missing labels moved
  twice. Some "model errors" are annotator noise (30 rows flagged for a second
  annotator; pending). Test set is frozen now — earlier numbers in git history
  refer to shifted splits and are superseded.
- The B1 collapse (0.644 → 0.354) cuts both ways: it proves the leakage audit
  mattered, but it also means our margin over "simple" partly reflects B1's
  training-data discipline, not just modeling.
- Judge shares the generator's vendor; paired measurement banked at n=0
  (quota re-probed dead; pilot invalidated by decontamination and unquoted).
  There is deliberately no judge headline. Never invent one from this report.
- One brand, Oct-2017 window, English-only, n=100: generalises nowhere
  untouched. Coverage 0.20 means 4 of 5 messages still need a human.

## 7. One more week

1. Second annotator: 30 flagged double-labels + paired judge set to n≥40 when
   quota resets → noise-bounded metrics and a real κ.
2. "Failed attempt" intent; port the 40-char floor + payment-complexity and
   batch-question patterns into the gate (kills the known unsafe autos).
3. Promise-regex → NLI-vs-cited-span validator (kills the $1 over-fire).
4. Bigger cal split (100) + frozen re-run; Lite run-to-run variance study.
5. Shadow-mode approve/edit UI → active-learning loop from human edits.

## 8. Decision log

`DECISIONS.md` holds exactly 15 entries (WHAT/WHY/alternatives/tradeoff) —
the ones above, compressed. Overflow detail lives in GOLDEN_NOTE.md.
