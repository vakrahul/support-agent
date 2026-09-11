# Failure analysis — locked test-100, ours (29 failures: 28 intent, 1 escalation)

## Mode 1 — delay↔missing boundary confusion (9/29, 31%)
EXPECTED: missing_parcel_tracking (7×) / delivery_delay (2×).
ACTUAL: the neighbor intent, at confidence 0.85–0.98.
REAL EXAMPLES: amz-012 "- tracking number … Had to [be] delivered by 11 still
not delivered" → delay; amz-082 "fake 'we attempted delivery' message … sitting
on my porch" → delay; amz-005 "Courier guys update wrong status" → missing.
WHY: late vs lost share nearly all content words (delivered/delay/tracking);
single-label taxonomy forces a binary call on a continuum (failed attempt →
late → lost).
HYPOTHESIS: the boundary needs a third "failed delivery attempt" state or
carrier-attempt features.
FIX: add attempt-state intent; agreement gate already halves the damage
(wrong-intent autos went 14 → 0 on cal).

## Mode 2 — cancelled orders over-trigger refund (4/29)
EXPECTED: order_status_general. ACTUAL: refund_return (amz-067, 111, 023, 134).
REAL EXAMPLE: amz-067 "prime now order … cancelled.. no reason given" → refund.
WHY: "cancel" co-occurs with money language in few-shots; model over-applies
the refund-priority rule where gold wants status-first (no explicit money claim).
HYPOTHESIS: priority rule too coarse.
FIX: refund only on explicit money words (refund/money/charge/back), else status.

## Mode 3 — noisy rants punted to other_unclear (5/29)
EXPECTED: delay/device/missing. ACTUAL: other_unclear (amz-128 "Not at all.
Haven't added at all", amz-031 "has been shitty for everyone", amz-096,
amz-098 "Used the mobile app.", amz-088).
WHY: few-shots contain no noisy rants/fragments; model abstains where a human
sees an issue. Safe direction (all escalated), but intent recall suffers.
HYPOTHESIS: training distribution mismatch, not a reasoning failure.
FIX: add 3–4 noisy-rant few-shots; keep abstention for true fragments.

## Mode 4 — tracking questions and returns misrouted (4/29)
EXPECTED: missing (amz-119 carrier question, amz-028) / refund (amz-125
returned-to-sender). ACTUAL: order_status_general / refund_return swapped
(amz-119 → status, amz-125 → refund, amz-028 → status, amz-009 → missing).
WHY: carrier/return logistics vocabulary sits equidistant from three intents.
HYPOTHESIS: same root as Mode 1 — logistics sub-language underrepresented.
FIX: same as Mode 1 plus return-to-sender → missing rule.

## Mode 5 — residual: ambiguity + one policy gap (7/29)
- amz-142 "when is my echo coming?": Echo-the-device vs echo-delivery —
  genuinely ambiguous; either label defensible. Gold noise, not model error.
- amz-130/147 (other↔refund/status), amz-033/049: single-word or
  multi-clause edge cases, one each, no pattern.
- ESCALATION MISS (1): amz-102 "No tracking???" (12 chars) auto-handled with
  OK_AUTO. Gold escalates short messages. EXPECTED: human. ACTUAL: auto.
  WHY: gate has similarity, confidence, and agreement floors but no
  message-length floor. FIX: escalate messages under ~40 chars (already the
  golden rule — port it into gate.py).
