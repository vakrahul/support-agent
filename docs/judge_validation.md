# Judge validation — why humans carry reply quality

Rubric (1–5, fixed before results): groundedness (claims in evidence),
relevance (issue + required points), tone (professional, brand-fit), safety
(no invented promises/dates/amounts). Scale meanings: 5 correct+grounded+
actionable; 4 strong, minor weakness; 3 acceptable generic; 2 materially weak
or wrong-context; 1 incorrect/harmful/unsupported. Prompt: `src/eval/judge.py`
(blind to system origin; must quote evidence spans).

Credibility evidence (`outputs/judge_agreement.json`, `human_reply_scores.json`):
- Human cross-system scores (primary): ours 4.16 (n=50) vs B1 verbatim 2.45
  (n=20) vs B0 canned 2.70 (n=10), same rubric, scored blind to judge output.
- LLM-vs-human paired: n=0 banked. A 16-pair pilot was measured against
  pre-decontamination drafts and invalidated by the (correct) corpus rebuild;
  its statistics are NOT quoted as results. Re-measurement blocked by
  free-tier quota (re-probed dead 2026-09-10); one real judgement (amz-022)
  banked post-rebuild.
- Deterministic checks on all 100 (no judge needed): grounding pass,
  evidence-citation rate, promise-pattern violations.

Verdict, stated in the report: with no banked paired sample, the judge is
implemented but unvalidated — reply-quality claims rest on humans +
deterministic checks. Free-tier quota (pro models 0, 2.5-flash ~20/day,
throttled 3.5-flash) is the binding constraint, documented — not worked
around by prompt-hacking the judge.

To close the gap (no fakes): `make human-pack` emits blank
`data/human_scoring/judge_pairs_template.csv` (100) +
`second_annotator_template.csv` (30); fill blind by hand, then
`make agreement` computes Spearman + kappa. Until then n stays 0.
