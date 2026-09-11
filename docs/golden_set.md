# Golden set — sampling, labelling, freezes

Authoritative note: `data/golden/GOLDEN_NOTE.md` (sampling targets, all four
adjudication passes, label-freeze declaration). Split: `data/golden/split.json`
(cal-50/test-100, frozen 2026-09-10; `run_eval.py` loads it, never re-derives).
Schema per row: id, text, intent, escalate, reason fields, must_include,
must_not_promise, annotator trail, double_label flags (30 rows).

Method in brief: stratified by rule-intent buckets (35/35/35/20/15/10 targets),
hard cases oversampled (~1/3 per bucket: multi-intent, very short,
complaint/sarcasm markers), noise and typos kept. Labels are rule-drafted then
human-verified over 4 passes (18 + 5 + 2 fixes, 16 policy flips); single
annotator — the bound on how much of any model gap is real. Frozen: no further
edits without a versioned re-freeze.
