"""
Headline eval: intent, escalation, reply quality for B0 / B1 / ours.

    LLM_MODE=replay python scripts/run_eval.py          # $0, ~2 min, cached
    LLM_MODE=live python scripts/run_eval.py            # record cache (needs key)
    python scripts/run_eval.py --calibration-only       # tune thresholds only

Splits golden 150 -> 50 calibration / 100 locked test (stratified, seed 42).
Thresholds are tuned on calibration ONLY; every headline number is test-only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config
from src.classify.classifier import LABELS, LLMClassifier, MajorityClassifier, TfidfLogReg
from src.decide.gate import decide, risk_scan
from src.draft.reply import CANNED, canned_draft, rag_draft, retrieval_only_draft, validate_grounding
from src.eval import metrics as M
from src.eval.judge import judge
from src.retrieve.index import INDEX_DIR, Retriever, build_index
from src.ingest.language import is_english

GOLDEN = config.DATA_GOLDEN / "golden.jsonl"
OUT_DIR = config.ROOT / "outputs"


def load_golden():
    rows = [json.loads(l) for l in open(GOLDEN, encoding="utf-8")]
    return rows


def stratified_split(rows, n_cal=50, seed=config.SEED):
    """Derive a split (used once, then frozen). Runtime path loads
    data/golden/split.json instead -- the test set must never move again."""
    import numpy as np
    rng = np.random.default_rng(seed)
    cal, test = [], []
    by_intent: dict[str, list] = {}
    for r in rows:
        by_intent.setdefault(r["intent"], []).append(r)
    for intent, rs in by_intent.items():
        rs = list(rs)
        rng.shuffle(rs)
        n_c = max(1, round(len(rs) * n_cal / len(rows)))
        cal += rs[:n_c]
        test += rs[n_c:]
    rng.shuffle(cal)
    rng.shuffle(test)
    return cal, test


def load_split(rows):
    """Load the FROZEN split. Fails loudly if an id is missing or labels
    moved rows across the boundary (labels are frozen too)."""
    import json as _json
    sp = config.DATA_GOLDEN / "split.json"
    if not sp.exists():
        print("[eval] WARNING: no frozen split; deriving (commit split.json!)")
        return stratified_split(rows)
    s = _json.loads(sp.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in rows}
    missing = [i for i in s["cal_ids"] + s["test_ids"] if i not in by_id]
    if missing:
        raise SystemExit(f"[eval] split.json references {len(missing)} ids "
                         f"absent from golden: {missing[:5]}")
    return [by_id[i] for i in s["cal_ids"]], [by_id[i] for i in s["test_ids"]]


def predict_fill(clf, rows, mode, tries=3):
    """Classify with retries: transient Gemini failures default safe (escalate)
    but must not silently shrink the measured set. Retries fill cache gaps."""
    from src.llm.gemini import TRANSPORT_FAILURES
    preds = clf.predict([r["text"] for r in rows], mode=mode)
    for _ in range(tries - 1):
        missing = [i for i, (p, s, _) in enumerate(preds)
                   if p == "other_unclear" and s == 0.0]
        if not missing or mode != "live":
            break
        redo = clf.predict([rows[i]["text"] for i in missing], mode=mode)
        for i, pr in zip(missing, redo):
            preds[i] = pr
    return preds


def weak_labels(n=3000):
    """TF-IDF training data: rule-labelled English AmazonHelp messages,
    golden ids EXCLUDED (baseline must not train on test)."""
    import re
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_golden import RULES
    df = pd.read_parquet(config.DATA_SAMPLE / "pairs_AmazonHelp.parquet")
    from src.retrieve.index import golden_exclusions
    gids, gtexts = golden_exclusions()
    if "customer_tweet_id" in df.columns:
        df = df[~df["customer_tweet_id"].astype(str).isin(gids)]
    norm = df["customer_message"].fillna("").str.lower().str.strip().str.replace(
        r"\s+", " ", regex=True)
    df = df[~norm.isin(gtexts)]
    en = df[df["customer_message"].map(is_english)].sample(
        n, random_state=config.SEED)["customer_message"].fillna("").tolist()
    X, y = [], []
    for t in en:
        fired = [name for name, rx in RULES if rx.search(t or "")]
        X.append(t)
        y.append(fired[0] if fired else "other_unclear")
    return X, y


def run_system(rows, clf_name, clf, retriever, draft_kind, mode,
               sim_thr, conf_thr, intent_cache=None):
    texts = [r["text"] for r in rows]
    if clf_name == "b0":
        # Trivial baseline = most-frequent weak-label class (measured on the
        # unlabeled pool, no golden peeking) + canned reply + always escalate.
        preds = clf.predict(texts)
    elif clf_name == "llm_cached":
        assert intent_cache is not None
        preds = [intent_cache[r["id"]] for r in rows]
    elif clf_name == "llm":
        preds = predict_fill(clf, rows, mode)
    else:
        preds = clf.predict(texts)
    out = []
    from build_golden import RULES as _RULES

    def _rule_label(t):
        fired = [name for name, rx in _RULES if rx.search(t or "")]
        return fired[0] if fired else "other_unclear"

    for ri, (r, (intent, score, top3)) in enumerate(zip(rows, preds)):
        if draft_kind == "rag" and ri and ri % 25 == 0:
            print(f"  [draft] {ri}/{len(rows)}", flush=True)
        cases = retriever.search(r["text"], top_k=3)
        max_sim = max([c["similarity"] for c in cases], default=0.0)
        agreement = (sum(1 for c in cases
                         if _rule_label(c["customer_message"]) == intent)
                     / max(len(cases), 1))
        if draft_kind == "canned":
            d = canned_draft(r["text"])
            gpass, gscore = True, 0.0
        elif draft_kind == "retrieval":
            d = retrieval_only_draft(cases)
            g = validate_grounding(d["draft"], cases, d["evidence_ids"])
            gpass, gscore = g["passed"], g["score"]
        else:
            d = rag_draft(r["text"], intent, cases, mode=mode)
            g = validate_grounding(d["draft"], cases, d["evidence_ids"])
            gpass, gscore = g["passed"], g["score"]
        risk = risk_scan(r["text"])
        if draft_kind == "canned":
            dec = {"decision": "human", "reason_code": "ALWAYS_ESCALATE",
                   "reason": "trivial baseline escalates everything"}
        else:
            dec = decide(intent, score, max_sim, risk, gpass,
                         sim_thr=sim_thr, conf_thr=conf_thr,
                         agreement=agreement)
        out.append({"id": r["id"], "intent": intent, "score": round(score, 3),
                    "top3": top3, "max_sim": round(max_sim, 3),
                    "agreement": round(agreement, 3),
                    "draft": d["draft"], "evidence_ids": d["evidence_ids"],
                    "grounding_passed": gpass, "risk": risk,
                    "decision": dec["decision"], "reason_code": dec["reason_code"],
                    "reason": dec["reason"]})
    return out


def _judge_all(results, test, by_id, retriever, mode, workers=1, delay=6.0):
    """Judge ours (all test) + baselines (30 each). Threaded; resume-safe:
    entries already judged by this model are kept, so aborted runs continue."""
    from concurrent.futures import ThreadPoolExecutor
    from src import config as _cfg

    def _needs_judge(t):
        j = t.get("judge")
        return not isinstance(j, dict) or j.get("model") != _cfg.JUDGE_MODEL

    jobs = []
    for t in results["systems"]["ours_llm"]["runs"]:
        if _needs_judge(t):
            jobs.append(("ours_llm", t))
    for name in ("B0_trivial", "B1_tfidf"):
        for t in results["systems"][name]["runs"][:30]:
            if _needs_judge(t):
                jobs.append((name, t))
    print(f"[judge] {len(jobs)} calls to make ({workers} workers)", flush=True)
    if not jobs:
        return

    done = [0]

    def _one(job):
        _, t = job
        g = by_id[t["id"]]
        ev = [c["brand_reply"] for c in retriever.search(g["text"], top_k=3)
              if c["case_id"] in t["evidence_ids"]]
        t["judge"] = judge(g["text"], t["draft"], ev, g["must_include"],
                           g["must_not_promise"], mode=mode)
        done[0] += 1
        if done[0] % 10 == 0:
            print(f"  [judge] {done[0]}/{len(jobs)}", flush=True)
        if mode == "live":
            time.sleep(delay)
        return True

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(_one, jobs))
    print(f"  [judge] {done[0]}/{len(jobs)} done", flush=True)


def _finalize(results, test):
    systems = ("B0_trivial", "B1_tfidf", "ours_llm")
    # Carry over real judgements from a previous file: a --skip-judge rerun
    # must never destroy banked judge entries (quota makes them expensive).
    from src import config as _cfg2
    old_path = OUT_DIR / "eval_results.json"
    if old_path.exists():
        try:
            old = json.load(open(old_path, encoding="utf-8"))
            for name in systems:
                if name not in old.get("systems", {}):
                    continue
                old_j = {t["id"]: t.get("judge") for t in old["systems"][name]["runs"]}
                for t in results["systems"][name]["runs"]:
                    if "judge" not in t and isinstance(old_j.get(t["id"]), dict) \
                            and old_j[t["id"]].get("model") == _cfg2.JUDGE_MODEL:
                        t["judge"] = old_j[t["id"]]
        except Exception as e:  # never let bookkeeping break the save
            print(f"[eval] judge carryover skipped ({e})")
    for name in systems:
        js = [t["judge"]["avg"] for t in results["systems"][name]["runs"] if "judge" in t]
        results["systems"][name]["judge_avg"] = sum(js) / len(js) if js else 0.0
        results["systems"][name]["judge_n"] = len(js)
        print(f"[{name}] judge_avg={results['systems'][name]['judge_avg']:.2f} (n={len(js)})")

    # failures: test misses for ours (runs and test share saved order)
    by_id = {r["id"]: r for r in test}
    fails = []
    for t in results["systems"]["ours_llm"]["runs"]:
        g = by_id[t["id"]]
        if t["intent"] != g["intent"] or (
                t["decision"] == "auto" and g["escalate"] == "escalate"):
            fails.append({"id": g["id"], "text": g["text"][:200],
                          "gold_intent": g["intent"], "pred_intent": t["intent"],
                          "gold_esc": g["escalate"], "decision": t["decision"],
                          "reason_code": t["reason_code"]})
    results["failures"] = fails
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"[eval] failures={len(fails)} -> outputs/eval_results.json")


def _write_curve(curve, sim_thr, conf_thr):
    """Persist the cal-50 coverage-vs-unsafe sweep — the report's central table.
    Written on every run so the committed JSON can never go stale again."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    distinct = {}
    for row in curve:
        key = (row["coverage"], row["unsafe"], row["esc_recall"])
        distinct.setdefault(key, row)
    payload = {
        "operating_point": {"sim": sim_thr, "conf": conf_thr},
        "curve": curve,
        "distinct_regimes": [
            {"coverage": c, "unsafe": u, "esc_recall": e}
            for (c, u, e), row in distinct.items()],
        "reading": "sweep over sim x conf on frozen cal-50; many grid cells "
                   "collapse because model_score is bimodal (the conf knob is "
                   "empirically dead); test-100 runs at the operating point.",
    }
    with open(OUT_DIR / "coverage_curve.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"[eval] curve ({len(distinct)} distinct regimes) -> outputs/coverage_curve.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration-only", action="store_true")
    ap.add_argument("--skip-judge", action="store_true",
                    help="run systems only; judge in a second pass")
    ap.add_argument("--judge-only", action="store_true",
                    help="judge existing outputs/eval_results.json (resume-safe)")
    ap.add_argument("--judge-workers", type=int, default=1,
                    help="parallel judge calls (1 is quota-safe on free tier)")
    ap.add_argument("--judge-delay", type=float, default=6.0,
                    help="seconds between judge calls (free-tier pacing)")
    ap.add_argument("--rebuild-index", action="store_true",
                    help="rebuild the Qdrant index (slow); default reuses it")
    ap.add_argument("--sim-thr", type=float, default=None)
    ap.add_argument("--conf-thr", type=float, default=None)
    ap.add_argument("--replay", action="store_true", default=True,
                    help="replay mode using committed cache (default, zero key needed)")
    ap.add_argument("--live", action="store_true",
                    help="live mode calling Gemini API (needs GEMINI_API_KEY)")
    args = ap.parse_args()
    mode = "live" if args.live else (os.getenv("LLM_MODE") or "replay")

    if args.judge_only:
        # Fast path: no calibration, no systems recompute, no TF-IDF train.
        # Everything needed is already in outputs/eval_results.json.
        with open(OUT_DIR / "eval_results.json", encoding="utf-8") as f:
            results = json.load(f)
        order = [t["id"] for t in results["systems"]["ours_llm"]["runs"]]
        gold_by_id = {r["id"]: r for r in load_golden()}
        test = [gold_by_id[i] for i in order]
        by_id = gold_by_id
        if args.rebuild_index or not (INDEX_DIR / "cases.parquet").exists():
            build_index()
        retriever = Retriever()
        _judge_all(results, test, by_id, retriever, mode, args.judge_workers,
                   args.judge_delay)
        _finalize(results, test)
        return 0

    rows = load_golden()
    cal, test = load_split(rows)
    print(f"[eval] golden={len(rows)} cal={len(cal)} test={len(test)} mode={mode} (split frozen)")

    if args.rebuild_index or not (INDEX_DIR / "cases.parquet").exists():
        build_index()
    retriever = Retriever()

    Xw, yw = weak_labels()
    maj = MajorityClassifier().fit(Xw, yw)
    print(f"[eval] majority class = {maj.majority}")
    tfidf = TfidfLogReg().fit(Xw, yw)
    llm = LLMClassifier()

    # threshold sweep on calibration with the LLM intents (cached after the
    # first live run) so the operating point transfers to the real system.
    # The full sweep grid is saved as the coverage-vs-unsafe curve (the report's
    # central result) — regenerating outputs/coverage_curve.json on every run.
    if args.sim_thr is None:
        best, best_cov = None, -1.0
        cal_llm = run_system(cal, "llm", llm, retriever, "retrieval", mode,
                             sim_thr=0.0, conf_thr=0.0)
        cal_intents = {t["id"]: (t["intent"], t["score"], [t["intent"]]) for t in cal_llm}
        cal_gold_escalate = {r["id"]: r["escalate"] for r in cal}
        curve = []
        for sim in (0.30, 0.40, 0.45, 0.50, 0.60):
            for conf in (0.40, 0.50, 0.60, 0.70):
                tr = run_system(cal, "llm_cached", None, retriever, "retrieval",
                                mode, sim_thr=sim, conf_thr=conf,
                                intent_cache=cal_intents)
                yt = ["escalate" if r["escalate"] == "escalate" else "auto_handle" for r in cal]
                yp = [("escalate" if t["decision"] == "human" else "auto_handle") for t in tr]
                br = M.binary_report(yt, yp)
                cov = sum(1 for t in tr if t["decision"] == "auto") / len(tr)
                curve.append({"sim": sim, "conf": conf,
                              "coverage": round(cov, 3),
                              "unsafe": round(br.false_auto_handle_rate, 3),
                              "esc_recall": round(br.recall, 3)})
                if br.false_auto_handle_rate <= 0.05 and cov > best_cov:
                    best, best_cov = (sim, conf, br), cov
        if best is None:
            tr = run_system(cal, "llm_cached", None, retriever, "retrieval",
                            mode, sim_thr=0.60, conf_thr=0.70,
                            intent_cache=cal_intents)
            yt = ["escalate" if r["escalate"] == "escalate" else "auto_handle" for r in cal]
            yp = [("escalate" if t["decision"] == "human" else "auto_handle") for t in tr]
            br = M.binary_report(yt, yp)
            best, best_cov = (0.60, 0.70, br), sum(
                1 for t in tr if t["decision"] == "auto") / len(tr)
            print("[eval] no operating point met unsafe<=5%; taking most conservative")
        sim_thr, conf_thr = best[0], best[1]
        _write_curve(curve, sim_thr, conf_thr)
        print(f"[eval] tuned on cal: sim>={sim_thr} conf>={conf_thr} "
              f"coverage={best_cov:.2f} unsafe={best[2].false_auto_handle_rate:.3f}")
    else:
        sim_thr, conf_thr = args.sim_thr, args.conf_thr
    if args.calibration_only:
        return 0

    systems = {
        "B0_trivial": ("b0", maj, "canned"),
        "B1_tfidf": ("tfidf", tfidf, "retrieval"),
        "ours_llm": ("llm", llm, "rag"),
    }
    results: dict = {"sim_thr": sim_thr, "conf_thr": conf_thr, "systems": {}}
    for name, (ckind, clf, dkind) in systems.items():
        runs = run_system(test, ckind, clf or maj, retriever, dkind, mode,
                          sim_thr=sim_thr, conf_thr=conf_thr)
        yt = [r["intent"] for r in test]
        yp = [t["intent"] for t in runs]
        irep = M.classification_report(yt, yp, labels=LABELS)
        ye = ["escalate" if r["escalate"] == "escalate" else "auto_handle" for r in test]
        pe = [("escalate" if t["decision"] == "human" else "auto_handle") for t in runs]
        erep = M.binary_report(ye, pe)
        cov = sum(1 for t in runs if t["decision"] == "auto") / len(runs)
        results["systems"][name] = {
            "runs": runs, "accuracy": irep.accuracy, "macro_f1": irep.macro_f1,
            "esc_precision": erep.precision, "esc_recall": erep.recall,
            "false_auto": erep.false_auto_handle_rate, "coverage": cov,
            "reason_codes": dict(Counter(t["reason_code"] for t in runs)),
        }
        print(f"[{name}] intent acc={irep.accuracy:.3f} macroF1={irep.macro_f1:.3f} | "
              f"esc P={erep.precision:.3f} R={erep.recall:.3f} unsafe={erep.false_auto_handle_rate:.3f} "
              f"coverage={cov:.2f}")

    # judge: ours on all test, baselines on 30-stratum subsample.
    # Skipped with --skip-judge (judge later via --judge-only, resume-safe).
    by_id = {r["id"]: r for r in test}
    if not args.skip_judge:
        _judge_all(results, test, by_id, retriever, mode, args.judge_workers,
                   args.judge_delay)
    _finalize(results, test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
