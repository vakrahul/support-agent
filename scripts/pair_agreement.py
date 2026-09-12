"""Paired judge-vs-human agreement from banked judgements + human scores.

Pairs each human-scored ours-draft (outputs/human_reply_scores.json) with the
LLM judge verdict now banked in eval_results.json. Prints + writes
outputs/judge_agreement.json (real numbers only; pairs list committed).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval import metrics as M

res = json.load(open(ROOT / "outputs" / "eval_results.json", encoding="utf-8"))
h = json.load(open(ROOT / "outputs" / "human_reply_scores.json", encoding="utf-8"))
ours = {t["id"]: t for t in res["systems"]["ours_llm"]["runs"]}

pairs = []
for rid, human in h.get("ours_scores", {}).items():
    t = ours.get(rid)
    j = t.get("judge") if t else None
    if isinstance(j, dict) and j.get("model") == "gemini-3.1-flash-lite":
        pairs.append({"id": rid, "human": float(human), "judge": float(j["avg"]),
                      "axes": {k: j[k] for k in ("groundedness", "relevance", "tone", "safety")}})

hx = [p["human"] for p in pairs]
jx = [p["judge"] for p in pairs]
out = {
    "n_paired": len(pairs),
    "pairs": pairs,
    "human_n": h.get("ours", {}).get("n", 50),
    "human_mean": h.get("ours", {}).get("mean", 4.16),
    "human_systems": h.get("human_systems", {}),
    "llm_judge_mean": round(sum(jx) / len(jx), 3) if jx else None,
    "judge_avg_all_banked": res["systems"]["ours_llm"]["judge_avg"],
    "judge_n_banked": res["systems"]["ours_llm"]["judge_n"],
}
if pairs:
    out["spearman_rho"] = round(M.spearman_rho(hx, jx), 3)
    hb = [str(round(v)) for v in hx]
    jb = [str(round(v)) for v in jx]
    out["binned_kappa"] = round(M.cohens_kappa(hb, jb), 3)
    out["raw_agreement"] = round(sum(1 for a, b in zip(hb, jb) if a == b) / len(pairs), 3)
    out["mean_abs_diff"] = round(sum(abs(a - b) for a, b in zip(hx, jx)) / len(pairs), 3)

out["verdict"] = (
    f"Paired judge-vs-human on {len(pairs)} identical drafts (banked live "
    f"gemini-3.1-flash-lite judgements vs blind human 1-5 scores). "
    + (f"rho={out['spearman_rho']}, binned kappa={out['binned_kappa']}, "
       f"raw agree={out['raw_agreement']}, mean|diff|={out['mean_abs_diff']}."
       if pairs else "No pairs yet.")
    + " Human scores remain primary for cross-system claims; the judge is now "
      "validated on ours-drafts only, not on baseline drafts.")

json.dump(out, open(ROOT / "outputs" / "judge_agreement.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "pairs"}, ensure_ascii=False, indent=1))
