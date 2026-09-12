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
- LLM-vs-human paired: **n=50 banked** (live gemini-3.1-flash-lite judgements on
  the same test-100 drafts the human scored). Spearman ρ=0.173, binned
  κ=-0.11, raw agreement 0.36, mean|diff|=0.675, judge mean 4.20 vs human
  4.16. Reading: means agree, per-item agreement is poor — the judge ranks
  drafts weakly at best. Two causes, both stated: (1) restriction of range —
  RAG drafts cluster at 3-5, so a 4-vs-5 flip moves ranks wildly; (2) the
  judge was more generous on borderline drafts. Consequence: judge_avg
  4.314 (n=98) is reported as a cross-check on OURS drafts only; the
  human scores remain the primary reply-quality evidence, and the judge
  must not be used for fine-grained ranking decisions. A pre-decontamination
  16-pair pilot was invalidated by the corpus rebuild and is not quoted.
- Deterministic checks on all 100 (no judge needed): grounding pass,
  evidence-citation rate, promise-pattern violations.

Verdict, stated in the report: the judge is implemented, blinded, and now
**measured against a human on n=50 paired drafts — agreement is weak
(ρ=0.173, κ=-0.11)**, so human scores stay primary for every reply-quality
claim and the judge is used only as a secondary cross-check on ours-drafts
(judge_avg 4.314, n=98 banked). Free-tier quota shaped this: judgements were
banked in a paced live pass; baseline drafts remain un-judged (quota), stated
not hidden. Reproduce: `python scripts/bank_judges.py` (banks from cache,
no network) then `python scripts/pair_agreement.py`.
