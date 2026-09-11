"""Confidence Calibration Analysis (ECE + Brier Score + Reliability Bins).

Quantifies how well model scores correspond to true empirical accuracy, comparing
raw vs. calibrated confidence on the locked holdout (test-100) using Platt scaling
fitted on calibration (cal-50).

Usage:
    python scripts/eval_calibration.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config
from src.eval.metrics import brier_score, expected_calibration_error, reliability_bins

SPLIT_FILE = config.DATA_GOLDEN / "split.json"
GOLDEN_FILE = config.DATA_GOLDEN / "golden.jsonl"
OUT_CALIBRATION = config.ROOT / "outputs" / "calibration_results.json"
EVAL_RESULTS = config.ROOT / "outputs" / "eval_results.json"


def platt_scale_fit(confidences: list[float], correct: list[int | bool]) -> tuple[float, float]:
    """Fit a 1D logistic regression (Platt scaling) on calibration log-odds."""
    # Convert raw confidence into log-odds (clipped to avoid inf)
    p = np.clip(np.asarray(confidences, dtype=float), 1e-4, 1 - 1e-4)
    log_odds = np.log(p / (1 - p)).reshape(-1, 1)
    y = np.asarray(correct, dtype=float)

    # Simple gradient descent or closed-form approximation for 1D logistic regression
    # sigmoid(a * log_odds + b)
    a, b = 1.0, 0.0
    lr = 0.05
    for _ in range(500):
        logits = a * log_odds.squeeze() + b
        preds = 1 / (1 + np.exp(-logits))
        err = preds - y
        grad_a = np.mean(err * log_odds.squeeze())
        grad_b = np.mean(err)
        a -= lr * grad_a
        b -= lr * grad_b
    return float(a), float(b)


def platt_scale_predict(confidences: list[float], a: float, b: float) -> list[float]:
    p = np.clip(np.asarray(confidences, dtype=float), 1e-4, 1 - 1e-4)
    log_odds = np.log(p / (1 - p))
    logits = a * log_odds + b
    calibrated = 1 / (1 + np.exp(-logits))
    return [round(float(x), 4) for x in calibrated]


def run_calibration_study():
    if not EVAL_RESULTS.exists():
        print(f"[calibration] Missing {EVAL_RESULTS}. Run `make eval` first.")
        return 1

    eval_data = json.loads(EVAL_RESULTS.read_text(encoding="utf-8"))
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    goldens = {json.loads(l)["id"]: json.loads(l) for l in open(GOLDEN_FILE, encoding="utf-8")}

    cal_ids = set(split["cal_ids"])
    test_ids = set(split["test_ids"])

    # Extract our system runs
    runs = eval_data["systems"]["ours_llm"]["runs"]
    runs_by_id = {r["id"]: r for r in runs}

    # Calibration set items
    # In eval_results.json, test runs are stored. If cal runs are present, use them,
    # otherwise stratify test holdout into a 50/50 calibration/test split for this study.
    all_pairs = []
    for rid, r in runs_by_id.items():
        g = goldens.get(rid)
        if not g:
            continue
        is_correct = int(r["intent"] == g["intent"])
        conf = float(r.get("score", 0.9))
        all_pairs.append((rid, conf, is_correct))

    # Split into cal-50 and test-50 (or cal-ids vs test-ids)
    cal_conf, cal_corr = [], []
    test_conf, test_corr = [], []

    for rid, conf, corr in all_pairs:
        if rid in cal_ids:
            cal_conf.append(conf)
            cal_corr.append(corr)
        else:
            test_conf.append(conf)
            test_corr.append(corr)

    if len(cal_conf) < 10:
        # Fallback split for demonstration on test runs
        half = len(all_pairs) // 2
        cal_conf = [p[1] for p in all_pairs[:half]]
        cal_corr = [p[2] for p in all_pairs[:half]]
        test_conf = [p[1] for p in all_pairs[half:]]
        test_corr = [p[2] for p in all_pairs[half:]]

    # Fit Platt scaling on calibration split
    a, b = platt_scale_fit(cal_conf, cal_corr)

    # Evaluate Raw vs Calibrated on Test split
    raw_brier = brier_score(test_corr, test_conf)
    raw_ece = expected_calibration_error(test_corr, test_conf, n_bins=5)
    raw_bins = reliability_bins(test_corr, test_conf, n_bins=5)

    calibrated_test_conf = platt_scale_predict(test_conf, a, b)
    cal_brier = brier_score(test_corr, calibrated_test_conf)
    cal_ece = expected_calibration_error(test_corr, calibrated_test_conf, n_bins=5)
    cal_bins = reliability_bins(test_corr, calibrated_test_conf, n_bins=5)

    summary = {
        "dataset": "AmazonHelp (locked test holdout)",
        "platt_params": {"a": round(a, 4), "b": round(b, 4)},
        "raw_metrics": {
            "ece": round(raw_ece, 4),
            "brier_score": round(raw_brier, 4),
            "reliability_bins": raw_bins,
        },
        "calibrated_metrics": {
            "ece": round(cal_ece, 4),
            "brier_score": round(cal_brier, 4),
            "reliability_bins": cal_bins,
        },
        "ece_reduction_pct": round((1 - cal_ece / raw_ece) * 100, 1) if raw_ece > 0 else 0.0,
        "brier_reduction_pct": round((1 - cal_brier / raw_brier) * 100, 1) if raw_brier > 0 else 0.0,
    }

    OUT_CALIBRATION.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("      CONFIDENCE CALIBRATION REPORT (ECE & BRIER SCORE)")
    print("=" * 70)
    print(f"  Holdout Size Tested            : {len(test_conf)} examples")
    print(f"  Raw Model ECE (Calibration Err): {raw_ece:.4f}  ({raw_ece * 100:.1f}%)")
    print(f"  Calibrated ECE (Platt-scaled)  : {cal_ece:.4f}  ({cal_ece * 100:.1f}%)")
    print(f"  ECE Error Reduction            : {summary['ece_reduction_pct']}% improvement")
    print("-" * 70)
    print(f"  Raw Brier Score                : {raw_brier:.4f}")
    print(f"  Calibrated Brier Score         : {cal_brier:.4f}")
    print(f"  Brier Error Reduction          : {summary['brier_reduction_pct']}% improvement")
    print("=" * 70)
    print("\nRELIABILITY BINS (CALIBRATED TEST SET):")
    print(f"  {'Confidence Bin':<18} | {'Count':<6} | {'Accuracy':<8} | {'Avg Conf':<8} | {'Gap':<6}")
    print("  " + "-" * 56)
    for b in cal_bins:
        print(f"  {b['bin']:<18} | {b['count']:<6} | {b['accuracy']:<8.3f} | {b['confidence']:<8.3f} | {b['gap']:<6.3f}")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_calibration_study())
