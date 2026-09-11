"""
Make human scoring packs (templates only — no fake scores).

Generates:
  data/human_scoring/judge_pairs_template.csv   — ours test-100 drafts for blind human scoring
  data/human_scoring/second_annotator_template.csv — 30 double_label rows for intent re-label

Fill the *_human columns by hand, then run scripts/score_agreement.py.
Never invent scores to fill them.
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config

GOLDEN = config.DATA_GOLDEN / "golden.jsonl"
EVAL = config.ROOT / "outputs" / "eval_results.json"
OUTDIR = config.ROOT / "data" / "human_scoring"


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    gold = {json.loads(l)["id"]: json.loads(l) for l in open(GOLDEN, encoding="utf-8")}
    evald = json.load(open(EVAL, encoding="utf-8"))
    ours = {t["id"]: t for t in evald["systems"]["ours_llm"]["runs"]}

    p1 = OUTDIR / "judge_pairs_template.csv"
    with open(p1, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "customer_text", "draft", "evidence_ids",
                    "must_include", "must_not_promise",
                    "human_groundedness_1_5", "human_relevance_1_5",
                    "human_tone_1_5", "human_safety_1_5",
                    "llm_judge_avg_if_any"])
        for _id, t in ours.items():
            g = gold[_id]
            j = t.get("judge", {})
            w.writerow([_id, g["text"][:500], t["draft"][:500],
                        ";".join(map(str, t.get("evidence_ids", []))),
                        "|".join(g.get("must_include", [])),
                        "|".join(g.get("must_not_promise", [])),
                        "", "", "", "",
                        j.get("avg", "")])
    doubles = [r for r in gold.values() if r.get("double_label")]
    p2 = OUTDIR / "second_annotator_template.csv"
    with open(p2, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "customer_text", "annotator1_intent",
                    "annotator2_intent_FILL", "agree_FILL"])
        for r in doubles:
            w.writerow([r["id"], r["text"][:500], r["intent"], "", ""])
    print(f"wrote {p1} ({len(ours)} rows, all human columns EMPTY)")
    print(f"wrote {p2} ({len(doubles)} rows, annotator2 EMPTY)")
    print("Next: fill by hand (blind to system), then python scripts/score_agreement.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
