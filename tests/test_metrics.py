"""Pin the metric implementations against hand-worked examples.

If these ever fail, every number in the report is suspect -- which is exactly
why they exist. Run: python -m pytest tests/ -q   (or: python tests/test_metrics.py)
"""
from __future__ import annotations

import numpy as np

from src.eval.metrics import (
    accuracy,
    binary_report,
    bootstrap_ci,
    classification_report,
    cohens_kappa,
    confusion_matrix,
    spearman_rho,
)


def test_confusion_matrix_orientation():
    # rows = TRUE, cols = PRED. Getting this backwards silently flips
    # precision and recall in the report, so assert it explicitly.
    y_true = ["a", "a", "b"]
    y_pred = ["a", "b", "b"]
    cm = confusion_matrix(y_true, y_pred, ["a", "b"])
    assert cm.tolist() == [[1, 1], [0, 1]]


def test_accuracy_basic():
    assert accuracy(["a", "b", "c"], ["a", "b", "x"]) == 2 / 3
    assert accuracy([], []) == 0.0


def test_classification_report_hand_worked():
    # class a: tp=1 fp=0 fn=1 -> p=1.00 r=0.50 f1=0.667
    # class b: tp=1 fp=1 fn=0 -> p=0.50 r=1.00 f1=0.667
    rep = classification_report(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
    a, b = rep.per_class
    assert np.isclose(a.precision, 1.0) and np.isclose(a.recall, 0.5)
    assert np.isclose(b.precision, 0.5) and np.isclose(b.recall, 1.0)
    assert np.isclose(rep.macro_f1, 2 / 3, atol=1e-6)
    assert np.isclose(rep.accuracy, 2 / 3)


def test_macro_f1_punishes_ignoring_rare_class():
    # THE headline argument: a model that never predicts the rare class still
    # scores high accuracy but is exposed by macro-F1.
    y_true = ["big"] * 90 + ["rare"] * 10
    y_pred = ["big"] * 100
    rep = classification_report(y_true, y_pred, ["big", "rare"])
    assert np.isclose(rep.accuracy, 0.90)
    assert rep.macro_f1 < 0.50  # ~0.47


def test_binary_report_and_false_auto_handle():
    y_true = ["escalate", "escalate", "auto_handle", "auto_handle"]
    y_pred = ["escalate", "auto_handle", "auto_handle", "escalate"]
    r = binary_report(y_true, y_pred, "escalate")
    assert (r.tp, r.fp, r.fn, r.tn) == (1, 1, 1, 1)
    assert np.isclose(r.precision, 0.5) and np.isclose(r.recall, 0.5)
    # one of two must-escalate cases was wrongly auto-handled
    assert np.isclose(r.false_auto_handle_rate, 0.5)


def test_kappa_perfect_and_chance():
    assert np.isclose(cohens_kappa(["a", "b", "a"], ["a", "b", "a"]), 1.0)
    # both annotators always say the same single class -> no information,
    # raw agreement 100% but kappa is undefined/0 by our convention
    assert cohens_kappa(["a"] * 10, ["a"] * 10) == 1.0
    # systematic disagreement scores negative
    assert cohens_kappa(["a", "a", "b", "b"], ["b", "b", "a", "a"]) < 0


def test_kappa_below_raw_agreement_under_imbalance():
    # The whole point of kappa: 90% raw agreement, near-zero real signal.
    a = ["auto"] * 90 + ["esc"] * 10
    b = ["auto"] * 95 + ["esc"] * 5
    raw = sum(x == y for x, y in zip(a, b)) / len(a)
    k = cohens_kappa(a, b)
    assert raw > 0.9
    assert k < raw


def test_spearman_monotonic():
    assert np.isclose(spearman_rho([1, 2, 3, 4], [2, 4, 6, 8]), 1.0)
    assert np.isclose(spearman_rho([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
    # ties handled by average ranks
    assert abs(spearman_rho([1, 1, 2, 2], [1, 2, 1, 2])) < 1e-9


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    vals = [3, 4, 5, 4, 3, 5, 4, 4, 3, 5]
    lo, hi = bootstrap_ci(vals, n_boot=500, seed=1)
    assert lo <= float(np.mean(vals)) <= hi
    assert (lo, hi) == bootstrap_ci(vals, n_boot=500, seed=1)  # reproducible


def test_brier_and_ece():
    from src.eval.metrics import brier_score, expected_calibration_error, reliability_bins
    # Perfect calibration: prob matches outcome
    y_t = [1, 1, 0, 0]
    y_p = [1.0, 1.0, 0.0, 0.0]
    assert brier_score(y_t, y_p) == 0.0
    assert expected_calibration_error(y_t, y_p, n_bins=2) == 0.0

    # Overconfident wrong predictions: prob 1.0 on 0
    y_bad = [0.9, 0.9, 0.9, 0.9]
    y_act = [0, 0, 0, 1]
    assert brier_score(y_act, y_bad) > 0.5
    assert expected_calibration_error(y_act, y_bad, n_bins=5) > 0.5
    bins = reliability_bins(y_act, y_bad, n_bins=5)
    assert len(bins) == 5



if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} metric tests passed")
