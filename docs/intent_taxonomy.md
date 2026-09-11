# Intent taxonomy — how it was made (AmazonHelp)

1. Filtered the 6,000-pair AmazonHelp subsample to English-only both sides
   (4,860 messages; ~19% non-English removed — Japanese/Spanish clusters would
   otherwise pollute the taxonomy).
2. Embedded customer messages (sentence-transformers, 384d), k-means over
   k=4–9 with silhouette + an explicit simplicity bias (0.02 per extra intent);
   rule proposed k=4, human overruled to k=6 after reading clusters: k=4 left a
   61% mega-cluster, k=6 split delay-apology from carrier-investigation
   (different routing), k=7+ added only URL-noise.
3. Per cluster: distinctive log-ratio terms + centroid-nearest examples
   (`data/sample/taxonomy_draft.json`). Human named/merged/deleted: the 3%
   URL-noise cluster deleted; heating/cheese Glasgow chatter (VirginTrains
   run) and equivalents merged, never kept as intents.
4. Frozen to 5 operational intents + `other_unclear` rejection bucket in
   `data/intent_schema.json` (2026-09-10). Rationale for ≤6: a 150-example
   golden set leaves ~25 examples per class; more intents = unmeasurable
   per-class F1 (order_status_general at n=11 is already borderline and a
   documented merge candidate).
5. Known weak boundary: delay↔missing (late vs lost). Priority rule
   refund > missing > device > delay > status > other breaks ties; the
   boundary still produced most annotator disagreements — owned, not hidden.

Full definitions, inclusions/exclusions, examples, and confusing neighbors:
`data/intent_schema.json` (machine-readable, validated by test).
