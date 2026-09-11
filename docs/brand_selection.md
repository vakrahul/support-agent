# Brand selection — AmazonHelp, by measurement

Scored 108 brands (510,927 pairs from a 1.2M-row scan) on volume, deflection
rate (English-only pairs), groundable volume, specificity, and gratitude.
Full table: `data/sample/brand_profiles.csv`. Chart:
`report/figures/brand_selection.png`. Code: `src/eda/brand_select.py`.

AmazonHelp: 85k pairs, resolution rate 70.1% (deflection 29.9%), english rate
81%, groundable ≈48.6k — rank 14 on rate, #1 on absolute grounding volume.
Picked on volume, not fame or metrics: the retrieval corpus needs tens of
thousands of in-thread resolutions, and rate-leaders (rail brands, ~95%) are
too small and too domain-narrow to demonstrate intent diversity. Rejected
AppleSupport/AmazonHelp-scale tech brands only where troubleshooting variance
would make grounding unmeasurable — stated trade-off, not a claim of
superiority.

Anti-gaming note: selection used rates, never golden labels or model scores.
