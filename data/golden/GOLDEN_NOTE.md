# Golden set note (AmazonHelp, n=150 DRAFT)

## Sampling
- Source: `data/sample/pairs_AmazonHelp.parquet` (fresh cleaning), English-only both sides.
- Stratified by rule-intent buckets with targets {"delivery_delay": 35, "missing_parcel_tracking": 35, "refund_return": 35, "device_app_account": 20, "order_status_general": 15, "other_unclear": 10}; realised {"refund_return": 35, "delivery_delay": 35, "missing_parcel_tracking": 35, "device_app_account": 20, "order_status_general": 15, "other_unclear": 10}.
- Hard cases (multi-intent, very short, complaint/sarcasm markers) oversampled to ~1/3 per bucket; realised hard=58.
- 30 rows (every 5th) flagged `double_label=true` for a second annotator.

## Labelling
- `intent`/`escalate` are RULE PRE-LABELS (`needs_review=true`). Priority: refund_return > missing_parcel_tracking > device_app_account > delivery_delay > order_status_general > other_unclear. A human must verify every row; unverified numbers are draft-only.
- `must_include`/`must_not_promise` encode the brand playbook; the reply judge scores against these, not against a single reference reply (many valid replies exist).
- Escalation pre-labels: `other_unclear` always escalates; ESCALATION_KEYWORDS hits escalate with HIGH_RISK; <40 chars escalate with LOW_CONFIDENCE.

## Known limits
- Single annotator + rule draft: label noise bounds how much of any model gap is real.
- One brand / one time window (Oct 2017 tweets): will not generalise untouched.

## Verification pass 1 (done)
- Second independent rule set agreed on 106/150 (70.7%); 44 disagreements adjudicated by reading each text against frozen definitions; 18 intent labels corrected (mostly delay-vs-missing boundary) + 1 escalation fix (phone number in text -> escalate, privacy).
- Realised distribution after pass: delivery_delay 42, refund_return 38, missing_parcel_tracking 29, device_app_account 18, other_unclear 12, order_status_general 11. Escalate rate 33.3%.
- Delay-vs-missing is the noisy boundary (late vs lost); expect classifier confusion there too.

## Pass-3/4 adjudication (2026-09-10)
- Pass-3: 5 boundary fixes from calibration confusion (017->delay, 040->refund, 067->order_status, 101->other+escalate, 147->other+escalate).
- Pass-4: escalation policy narrowed (routine refund/worst/disgusting no longer auto-escalate). 28 rows affected, each read on merits: 16 routine -> auto_handle, 12 kept escalate (CEO threat, legal-adjacent deception claims, conflicting resolution, large blocked amount, distress language, payment complexity, multi-issue). 2 intent fixes (010->other, 021->refund).
- Realised: escalate rate 33% -> 24%. Dist: delay 40, refund 41, missing 28, device 17, other 13, status 11.

## LABEL FREEZE (2026-09-10)
Labels and the cal(50)/test(100) split are FROZEN (data/golden/split.json). No further label edits: weak-class work must use calibration-side signals only, never the frozen test. Prior passes 1-4 remain history above; any future correction goes through a versioned re-freeze, never a silent edit.

## Intra-rater replication (2026-09-10, measurement only - labels stay frozen)
Blind re-label of the 30 double-flagged rows: self-agreement 25/30 = 0.83 (outputs/intra_rater.json). All 5 flips sit on known-ambiguous rows (delay<->missing x3 incl. both directions on the same boundary, vague-commentary x2). Implication: ~17% of gold is annotator-unstable, bounding intent accuracy below ~0.83 even for a perfect model; current 0.72 is closer to ceiling than it looks. Second human annotator still pending.
