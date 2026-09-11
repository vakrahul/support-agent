# Decision log (15 — WHAT / WHY / alternatives / tradeoff)

**1. Brand by groundable volume, deflections filtered out of the corpus.**
WHAT: rank brands on usable in-thread resolutions; drop DM-deflection replies
from retrieval. WHY: a deflection corpus teaches the generator to deflect while
scoring perfectly on groundedness. Alternative: most famous brand. Tradeoff:
AmazonHelp is rate-rank-14 but volume-rank-1 — messier, more honest.

**2. Metrics hand-implemented in numpy, not sklearn.**
WHAT: acc/macro-F1/confusion/κ/ρ/bootstrap in `src/eval/metrics.py`. WHY:
trivial install (15-min repro), every number traceable, conventions owned.
Alternative: sklearn. Tradeoff: re-implemented wheels, pinned by hand-worked tests.

**3. Curve headline, not accuracy.**
WHAT: coverage-vs-unsafe as the central result; macro-F1 supporting.
WHY: accuracy is class-imbalance theater; a 95%-accurate auto-handler of
billing disputes is worse than a 91% one that escalates. Alternative: single
headline number. Tradeoff: harder story, honest story.

**4. Escalation tuned for recall, reported as false auto-handle rate.**
WHAT: recall on must-escalate + precision traded away, both shown. WHY: costs
are asymmetric (lost customer vs wasted minute). Alternative: F1. Tradeoff:
precision 0.28 looks bad out of context — context is mandatory.

**5. Cross-model judge, agreement-first reporting.**
WHAT: generator 3.1-flash-lite vs judge 3.5-flash; reply claims rest on human
scores (50) + deterministic checks; paired judge measurement banked at n=0
(quota dead, pilot invalidated — unquoted). WHY: an unvalidated judge is
decoration; agreement evidence that doesn't exist must not be cited.
Alternative: quote the invalidated pilot decimals. Tradeoff: no judge
headline at all — the gap is visible.

**6. Local embeddings, embedded Qdrant, replay-by-default cache.**
WHAT: no embedding API, no Docker, no key needed to reproduce. WHY: every
external dependency is a reviewer failure mode; Cloud write-timeout measured.
Alternative: managed vector DB + live APIs. Tradeoff: local-only scale.

**7. Synthetic fixture + direct SDK calls, no agent framework.**
WHAT: committed twcs-schema fixture; Gemini SDK called directly with
prompt-versioned cache keys. WHY: framework prompt-assembly breaks cache
reasoning and live-review readability. Alternative: LangChain. Tradeoff:
re-implemented retry/parse (~40 tested lines).

**8. Cache miss in replay is a hard error.**
WHAT: never a silent fallback. WHY: a half-cached run printing a headline is
fabricated results. Alternative: graceful defaults. Tradeoff: reviewer with a
partial cache sees an error with exact fix instructions.

**9. Pluggable embeddings with numpy LSA fallback.**
WHAT: sentence-transformers preferred, TF-IDF→SVD fallback, degradation
announced. WHY: torch+weights download breaks fresh clones. Alternative:
transformers-only. Tradeoff: fallback retrieval is weaker, stated.

**10. k as operational band + simplicity bias, clusters as proposal.**
WHAT: k∈4–9 with explicit price-per-intent; human names/merges/deletes.
WHY: raw silhouette runs away to max-k on short text (pinned by test);
clusters split tense and merge distinct problems. Alternative: "silhouette
chose k". Tradeoff: human judgment in the loop, documented.

**11. Frozen 5+other taxonomy; retrieval judged by intent-consistency.**
WHAT: ~25 examples/class minimum; R@K consistency 0.47/0.63/0.74 at K=1/3/5.
WHY: 30 intents × 200 goldens = noise; pairwise-similarity assertions fail on
bag-of-words (measured 0.0 vs 0.138). Alternative: fine-grained taxonomy.
Tradeoff: delay↔missing boundary stays coarse (top failure mode).

**12. Stratified golden, 4 logged passes, then frozen.**
WHAT: rule draft → disagreement/boundary/policy adjudications → split.json
lock; no edits after. WHY: an unfrozen test set makes every number
uninterpretable (we reshuffled 3× before learning this). Alternative: label
once. Tradeoff: early numbers in history are superseded, stated.

**13. NaN-hardened detectors; narrowed escalation keywords.**
WHAT: type-guards (`x or ''` misses NaN); risk list cut after routine "refund"
fired on 25% of traffic → coverage 0. WHY: measured failures, not taste.
Alternative: broad keyword safety net. Tradeoff: 28 gold rows re-adjudicated;
narrower net documented as limitation.

**14. Intent-evidence agreement gate (≥2/3).**
WHAT: weak-rule-label top-3 cases; escalate on disagreement. WHY: verbalised
scores don't separate right/wrong (0.94 vs 0.93); agreement does (0.68 vs
0.17); killed all wrong-intent autos. Alternative: confidence threshold.
Tradeoff: coverage drops to 0.20 — the price is on the table.

**15. Model churn and quota handled as evidence.**
WHAT: 2.0-flash 404s → 3.1-flash-lite; dead judge models → 3.5-flash;
5-worker 429 storm (144 fallbacks) → paced retries; prompt-versioned keys
orphan stale cache. WHY: infra facts shape results; hiding them is gaming.
Alternative: silently switch and re-report. Tradeoff: report carries quota
scars — that's the point.
