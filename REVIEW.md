# REVIEW — skeptical read-through (2026-09-10)

Reviewer stance: never seen this repo before; every claim needs a pointer.
Verdicts: PASS / WARN (ships, fix next) / FAIL (would block).

| Requirement | Verdict | Evidence / gap |
|---|---|---|
| One brand, justified | PASS | AmazonHelp; `docs/brand_selection.md`, table + chart, rate-vs-volume tradeoff stated |
| 3 tasks (intent/reply/escalate+reason) | PASS | `classifier.py` / `reply.py` / `gate.py`; every run stores intent+evidence+decision+reason |
| Runnable repo | PASS | `make test`, `make eval` replay verified keyless; `make eval-live` for fresh calls |
| Golden 150–250, sampling+labelling docs | PASS | 150 rows, `docs/golden_set.md` + GOLDEN_NOTE (4 passes); single annotator disclosed |
| Eval harness | PASS | `scripts/run_eval.py`, metrics + confusion + CIs saved to `outputs/` |
| LLM judge + rubric | PASS | `src/eval/judge.py`, fixed 1–5 rubric in `docs/judge_validation.md`; banked live n=98 (judge_avg 4.314) |
| Judge-vs-human agreement | PASS (weak, stated) | paired n=50: rho=0.173, binned kappa=-0.11, raw agree 0.36, mean|diff|=0.675; judge demoted to secondary cross-check, humans primary |
| Trivial baseline | PASS | measured-majority + canned + always-escalate (acc 0.090 — a real floor) |
| Simple baseline | PASS | TF-IDF LogReg + verbatim retrieval, golden-excluded training (0.354 — collapse disclosed) |
| Failure analysis, 5 modes + examples | PASS | `results/failure_analysis.md`, frequencies + expected/actual |
| Misleading section | PASS | report §6 with CIs, wobble, noise, quota caveats |
| One-week plan | PASS | report §7, tied to observed failures |
| Decision log 10–15 | PASS | exactly 15, WHAT/WHY/alternatives/tradeoff |
| README <15-min repro | PASS | replay verified; systems table exact; judge cells honestly partial |
| Leakage controls | PASS | `check_leakage.py` fails loudly; corpus+train decontaminated, verified 0/0/0 |
| UI / demo | PASS | `make ui` real-time web UI (stdlib only); `scripts/demo.py` CLI inspector; `report/figures/demo_run.png` colorful real-run collage |
| No fake evidence | PASS | fallbacks model-tagged; quota scars in-report; superseded numbers marked as such |
| Tests | WARN | gate/validator/judge/schema/split/leakage covered (18 new); no live-API integration test (by design — replay covers it) |
| Offline mode | PASS | replay default; live opt-in |
| Cost/latency | PASS | 436 calls / ~244k tokens / $0 + retrieval p50 32ms, in results file |
| Security | WARN | `.env` gitignored + `.env.example` added, but repo has no git remote yet — verify untracked on first push |
| Architecture diagram | WARN | ASCII in README only; no exported figure |
| Ablation | PASS | gate-component + evidence-amount + retrieval R@K in `outputs/retrieval_ablation.json` |

**Would I shortlist?** Yes — the numbers are small but every one is traceable,
and the report volunteers its weaknesses before I find them. **Biggest
remaining risk:** single annotator throughout (golden + all 80 human reply
scores are one person's judgment).
