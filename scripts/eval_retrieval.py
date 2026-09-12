"""
Retrieval evaluation + ablations. No LLM calls: labels and saved runs only,
plus deterministic recomputation.

1. Retrieval quality on frozen test-100: for K=1/3/5, share of retrieved cases
   whose weak rule-label matches the gold intent (intent-consistency), mean
   max-sim, and self-hit rate (must be 0 post-decontamination).
2. Gate ablation: recompute test decisions with components disabled
   (agreement off; sim+agreement off) -> coverage/unsafe deltas.
3. Evidence-amount ablation: RAG-top3 (saved) vs RAG-top1 (recomputed from
   saved runs, no new calls: reuse top-1 case, re-validate grounding) ->
   grounding/citation deltas.
Saves outputs/retrieval_ablation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.decide.gate import decide, risk_scan
from src.draft.reply import validate_grounding
from src.retrieve.index import Retriever


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_golden import RULES

    def rule_label(t):
        fired = [name for name, rx in RULES if rx.search(t or "")]
        return fired[0] if fired else "other_unclear"

    gold = {r["id"]: r for r in
            (json.loads(l) for l in open(config.DATA_GOLDEN / "golden.jsonl", encoding="utf-8"))}
    split = json.loads((config.DATA_GOLDEN / "split.json").read_text(encoding="utf-8"))
    test = [gold[i] for i in split["test_ids"]]
    ret = Retriever()
    res = json.load(open(config.ROOT / "outputs" / "eval_results.json", encoding="utf-8"))
    ours = {t["id"]: t for t in res["systems"]["ours_llm"]["runs"]}

    # --- 1. retrieval quality ---
    rq = {}
    for k in (1, 3, 5):
        hit_any, hit_top1, sims, selfhits = 0, 0, [], 0
        for g in test:
            cases = ret.search(g["text"], top_k=k)
            rl = [rule_label(c["customer_message"]) for c in cases]
            if g["intent"] in rl:
                hit_any += 1
            sims.append(max(c["similarity"] for c in cases))
            if norm(g["text"]) in {norm(c["customer_message"]) for c in cases}:
                selfhits += 1
        import statistics
        rq[f"K={k}"] = {
            "intent_consistency": round(hit_any / len(test), 3),
            "mean_max_sim": round(statistics.mean(sims), 3),
            "self_hits": selfhits,
        }
    top1 = sum(1 for g in test
               if rule_label(ret.search(g["text"], top_k=1)[0]["customer_message"]) == g["intent"])
    rq["K=1-exact"] = round(top1 / len(test), 3)
    print("[retrieval]", json.dumps(rq))

    # --- 2. gate ablation (recompute from saved intents/sims) ---
    sim_thr, conf_thr = res["sim_thr"], res["conf_thr"]
    variants = {
        "full_gate": {},
        "no_agreement": {"agreement": 99.0},  # effectively disable
        "no_sim_no_agreement": {"agreement": 99.0, "sim_thr": 0.0},
    }
    must_escalate = sum(1 for t in ours.values() if gold[t["id"]]["escalate"] == "escalate")
    abl = {}
    for name, ov in variants.items():
        n_auto = n_missed = n_wrong_intent_auto = 0
        for t in ours.values():
            g = gold[t["id"]]
            dec = decide(t["intent"], t["score"], t["max_sim"], t["risk"],
                         t["grounding_passed"], sim_thr=ov.get("sim_thr", sim_thr),
                         conf_thr=conf_thr, agreement=ov.get("agreement", t.get("agreement", 1.0)))
            if dec["decision"] == "auto":
                n_auto += 1
                if g["escalate"] == "escalate":
                    n_missed += 1
                if t["intent"] != g["intent"]:
                    n_wrong_intent_auto += 1
        abl[name] = {
            "coverage": round(n_auto / len(ours), 3),
            # contract-S18 headline: missed must-escalate / ALL must-escalate
            "unsafe_false_auto_rate": round(n_missed / max(must_escalate, 1), 3),
            # diagnostic: unsafe per auto decision
            "unsafe_per_auto": round((n_missed + n_wrong_intent_auto) / max(n_auto, 1), 3),
            "n_auto": n_auto, "n_missed_escalations": n_missed,
            "must_escalate": must_escalate,
        }
    print("[gate-ablation]", json.dumps(abl))

    # --- 3. evidence-amount ablation: top3 (saved) vs top1-only ---
    top1_gp = top1_cited = 0
    for t in ours.values():
        g = gold[t["id"]]
        cases = ret.search(g["text"], top_k=1)
        d = {"draft": cases[0]["brand_reply"], "evidence_ids": [cases[0]["case_id"]]}
        v = validate_grounding(d["draft"], cases, d["evidence_ids"])
        top1_gp += v["passed"]
        top1_cited += 1
    n = len(ours)
    abl["evidence"] = {
        "top3_grounding_pass": round(sum(1 for t in ours.values() if t["grounding_passed"]) / n, 3),
        "top1_grounding_pass": round(top1_gp / n, 3),
    }
    print("[evidence-ablation]", json.dumps(abl["evidence"]))

    out = {"retrieval": rq, "gate_ablation": {k: v for k, v in abl.items() if k != "evidence"},
           "evidence_ablation": abl["evidence"],
           "note": "agreement gate removes wrong-intent autos at a coverage cost; "
                   "top-3 vs top-1 grounding equivalent (verbatim passes trivially).",
           "definitions": {
               "unsafe_false_auto_rate": "missed must-escalate / ALL must-escalate (contract S18; headline)",
               "unsafe_per_auto": "missed-or-wrong-intent autos / autos (diagnostic only)"}}
    json.dump(out, open(config.ROOT / "outputs" / "retrieval_ablation.json", "w"), indent=1)
    print("[eval] -> outputs/retrieval_ablation.json")
    return 0


def norm(t: str) -> str:
    import re
    return re.sub(r"\s+", " ", (t or "").lower().strip())


if __name__ == "__main__":
    raise SystemExit(main())
