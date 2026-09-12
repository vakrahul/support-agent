"""Bank judge verdicts from the committed cache into eval_results.json.

Resume-safe, no network: for every un-judged run, rebuild the judge prompt,
look it up in cache/llm_cache.jsonl; if present, attach the real verdict.
Never fabricates: absent cache entries stay un-judged (model tag absent).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config
from src.eval.judge import _prompt
from src.llm.gemini import get_cache, make_key
from src.llm.gemini import extract_json

OUT = ROOT / "outputs" / "eval_results.json"


def main() -> int:
    results = json.load(open(OUT, encoding="utf-8"))
    gold = {r["id"]: r for r in
            (json.loads(l) for l in open(config.DATA_GOLDEN / "golden.jsonl", encoding="utf-8"))}
    cache = get_cache()

    from src.retrieve.index import Retriever
    ret = Retriever()
    # rebuild evidence lookup the same way run_eval._judge_all does
    order = [t["id"] for t in results["systems"]["ours_llm"]["runs"]]

    banked = skipped = 0
    jobs = [("ours_llm", t) for t in results["systems"]["ours_llm"]["runs"]]
    for name in ("B0_trivial", "B1_tfidf"):
        jobs += [(name, t) for t in results["systems"][name]["runs"][:30]]

    for name, t in jobs:
        j = t.get("judge")
        if isinstance(j, dict) and j.get("model") == config.JUDGE_MODEL:
            continue  # already banked
        g = gold[t["id"]]
        cases = ret.search(g["text"], top_k=3)
        ev = [c["brand_reply"] for c in cases if c["case_id"] in t["evidence_ids"]]
        prompt = _prompt(g["text"], t["draft"], ev, g["must_include"], g["must_not_promise"])
        key = make_key(prompt, config.JUDGE_MODEL, config.JUDGE_TEMPERATURE, "judge-v1")
        resp = cache.get(key)
        if resp is None:
            skipped += 1
            continue
        try:
            d = extract_json(resp)
        except Exception:
            skipped += 1
            continue
        out = {k: max(1, min(5, int(d.get(k, 3)))) for k in
               ("groundedness", "relevance", "tone", "safety")}
        out["quotes"] = [str(q) for q in (d.get("quotes") or [])][:4]
        out["missing"] = [str(m) for m in (d.get("missing") or [])][:5]
        out["avg"] = round(sum(out[k] for k in
                          ("groundedness", "relevance", "tone", "safety")) / 4, 2)
        out["model"] = config.JUDGE_MODEL
        t["judge"] = out
        banked += 1

    for name in ("B0_trivial", "B1_tfidf", "ours_llm"):
        js = [t["judge"]["avg"] for t in results["systems"][name]["runs"] if "judge" in t]
        results["systems"][name]["judge_avg"] = round(sum(js) / len(js), 3) if js else 0.0
        results["systems"][name]["judge_n"] = len(js)
        print(f"[{name}] judge_avg={results['systems'][name]['judge_avg']} (n={len(js)})")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"banked={banked} skipped(no cache)={skipped} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
