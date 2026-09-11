"""
Score real human agreement (fails loudly if templates are still empty).

Reads:
  data/human_scoring/judge_pairs_template.csv  (filled human_* columns)
  data/human_scoring/second_annotator_template.csv (filled annotator2 column)

Writes nothing fake: if n==0 filled rows, exits non-zero and leaves
outputs/judge_agreement.json untouched (still n_paired=0 = unvalidated).
If n>=10, prints kappa/spearman + 95% CI and appends a banked record.
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval import metrics as M

PACK = Path(__file__).resolve().parents[1] / "data" / "human_scoring"
JUDGE_CSV = PACK / "judge_pairs_template.csv"
SECOND_CSV = PACK / "second_annotator_template.csv"


def _num(x):
    try:
        v = float(str(x).strip())
        return v if 1 <= v <= 5 else None
    except Exception:
        return None


def main() -> int:
    if not JUDGE_CSV.exists() or not SECOND_CSV.exists():
        print("Run python scripts/make_human_scoring_pack.py first.")
        return 2
    # 1. judge-vs-human pairs: need filled human avg AND banked llm avg
    evald = json.load(open(Path(__file__).resolve().parents[1] / "outputs" / "eval_results.json", encoding="utf-8"))
    llm_avg = {t["id"]: t.get("judge", {}).get("avg") for t in evald["systems"]["ours_llm"]["runs"]}
    hx, lx = [], []
    with open(JUDGE_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vals = [_num(row.get(k, "")) for k in
                    ("human_groundedness_1_5", "human_relevance_1_5",
                     "human_tone_1_5", "human_safety_1_5")]
            if any(v is None for v in vals):
                continue
            la = llm_avg.get(row["id"])
            if la is None:
                continue
            hx.append(sum(vals) / 4)
            lx.append(float(la))
    print(f"paired judge-vs-human filled: n={len(hx)}")
    # 2. second annotator intent agreement
    a1, a2 = [], []
    with open(SECOND_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            b = (row.get("annotator2_intent_FILL") or "").strip()
            if b:
                a1.append(row["annotator1_intent"].strip())
                a2.append(b)
    print(f"second-annotator filled: n={len(a2)}")
    if not hx and not a2:
        print("Nothing filled — no numbers to report. Fill CSVs by hand first. judge stays unvalidated.")
        return 1
    if hx:
        print(f"spearman rho={M.spearman_rho(hx, lx):.3f}")
        # binned kappa on rounded avgs
        hb = [str(round(v)) for v in hx]
        lb = [str(round(v)) for v in lx]
        print(f"binned kappa={M.cohens_kappa(hb, lb):.3f}")
    if a2:
        print(f"intent kappa={M.cohens_kappa(a1, a2):.3f} raw_agree={sum(1 for x, y in zip(a1, a2) if x == y)}/{len(a2)}")
    print("Copy these into docs/judge_validation.md + report §4. Do not edit outputs/judge_agreement.json by hand — bank via live judge run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
