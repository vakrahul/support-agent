"""
Evaluation metrics, implemented directly on numpy.

Why not sklearn? Three reasons, all defensible in the live review:
  1. The headline eval must run anywhere in seconds with a trivial install --
     that protects the "<15 minutes to reproduce" promise.
  2. Every number in the report is then traceable to ~20 lines of code in this
     file, rather than to a library call.
  3. Confusion-matrix conventions (label ordering, zero-division handling) are
     where silent reporting bugs hide. Owning them means we can explain them.

Correctness is pinned by tests/test_metrics.py, which checks these against
hand-worked examples.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# Multi-class
# --------------------------------------------------------------------------


def confusion_matrix(y_true, y_pred, labels: list[str]) -> np.ndarray:
    """Rows = true label, columns = predicted label (the usual convention)."""
    idx = {lab: i for i, lab in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t], idx[p]] += 1
    return m


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = list(y_true), list(y_pred)
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


@dataclass
class PerClass:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class ClassificationReport:
    labels: list[str]
    per_class: list[PerClass]
    accuracy: float
    macro_f1: float
    weighted_f1: float
    confusion: np.ndarray = field(repr=False)

    def to_markdown(self) -> str:
        head = "| intent | precision | recall | f1 | support |\n|---|---|---|---|---|\n"
        rows = "".join(
            f"| {c.label} | {c.precision:.3f} | {c.recall:.3f} | {c.f1:.3f} | {c.support} |\n"
            for c in self.per_class
        )
        tail = (
            f"\n**accuracy** {self.accuracy:.3f} · "
            f"**macro-F1** {self.macro_f1:.3f} · "
            f"**weighted-F1** {self.weighted_f1:.3f}\n"
        )
        return head + rows + tail

    def confusion_markdown(self) -> str:
        hdr = "| true \\ pred | " + " | ".join(self.labels) + " |\n"
        sep = "|" + "---|" * (len(self.labels) + 1) + "\n"
        body = ""
        for i, lab in enumerate(self.labels):
            body += f"| {lab} | " + " | ".join(str(v) for v in self.confusion[i]) + " |\n"
        return hdr + sep + body


def classification_report(y_true, y_pred, labels: list[str] | None = None) -> ClassificationReport:
    """Per-class P/R/F1 plus accuracy, macro-F1 and weighted-F1.

    We report MACRO-F1 as the headline rather than accuracy: support-class
    imbalance in this dataset is severe, so accuracy is dominated by the
    largest intent and flatters the model. Macro-F1 weights every intent
    equally, which matches what a support team actually cares about -- the
    rare, expensive intents.
    """
    y_true, y_pred = list(y_true), list(y_pred)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels)
    per_class: list[PerClass] = []
    for i, lab in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class.append(PerClass(lab, prec, rec, f1, support))

    total = sum(c.support for c in per_class)
    macro = float(np.mean([c.f1 for c in per_class])) if per_class else 0.0
    weighted = (
        float(sum(c.f1 * c.support for c in per_class) / total) if total else 0.0
    )
    return ClassificationReport(
        labels=labels,
        per_class=per_class,
        accuracy=accuracy(y_true, y_pred),
        macro_f1=macro,
        weighted_f1=weighted,
        confusion=cm,
    )


# --------------------------------------------------------------------------
# Binary -- the escalation decision
# --------------------------------------------------------------------------
@dataclass
class BinaryReport:
    positive_label: str
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def false_auto_handle_rate(self) -> float:
        """Fraction of must-escalate cases we wrongly auto-handled.

        This is THE number for this task. Escalation is cost-asymmetric:
        over-escalating wastes an agent's minute, but auto-handling a case that
        needed a human can mean a lost customer or a compliance problem.
        Equivalent to (1 - recall) on the escalate class.
        """
        denom = self.tp + self.fn
        return self.fn / denom if denom else 0.0

    def to_markdown(self) -> str:
        return (
            f"| metric | value |\n|---|---|\n"
            f"| precision (`{self.positive_label}`) | {self.precision:.3f} |\n"
            f"| **recall (`{self.positive_label}`)** | **{self.recall:.3f}** |\n"
            f"| f1 | {self.f1:.3f} |\n"
            f"| missed escalations (FN) | {self.fn} |\n"
            f"| over-escalations (FP) | {self.fp} |\n"
            f"| false auto-handle rate | {self.false_auto_handle_rate:.3f} |\n"
        )


def binary_report(y_true, y_pred, positive_label: str = "escalate") -> BinaryReport:
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        t_pos, p_pos = (t == positive_label), (p == positive_label)
        if t_pos and p_pos:
            tp += 1
        elif not t_pos and p_pos:
            fp += 1
        elif t_pos and not p_pos:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return BinaryReport(positive_label, prec, rec, f1, tp, fp, fn, tn)


# --------------------------------------------------------------------------
# Agreement -- used to validate the LLM judge against human labels
# --------------------------------------------------------------------------
def cohens_kappa(a, b, labels: list[str] | None = None) -> float:
    """Cohen's kappa: agreement corrected for agreement-by-chance.

    Raw agreement is misleading when one class dominates -- two annotators who
    both always say "auto_handle" agree 90% of the time while carrying zero
    information. Kappa subtracts that baseline.
    Rule of thumb: <0.20 poor, 0.21-0.40 fair, 0.41-0.60 moderate,
    0.61-0.80 substantial, >0.80 near-perfect.
    """
    a, b = list(a), list(b)
    if not a:
        return 0.0
    if labels is None:
        labels = sorted(set(a) | set(b))
    cm = confusion_matrix(a, b, labels)
    n = cm.sum()
    if n == 0:
        return 0.0
    p_obs = np.trace(cm) / n
    p_exp = float((cm.sum(axis=0) * cm.sum(axis=1)).sum()) / (n * n)
    if abs(1 - p_exp) < 1e-12:
        return 1.0 if abs(p_obs - 1.0) < 1e-12 else 0.0
    return float((p_obs - p_exp) / (1 - p_exp))


def spearman_rho(x, y) -> float:
    """Rank correlation -- for comparing 1-5 judge scores against human scores.

    Kappa treats a 1-vs-5 disagreement the same as 4-vs-5, which is wrong for
    ordinal quality ratings; Spearman respects the ordering.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2:
        return 0.0
    rx, ry = _rankdata(x), _rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else 0.0


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared (matches scipy.stats.rankdata default)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average tied ranks
    _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


def bootstrap_ci(
    values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    """Percentile bootstrap CI for a mean.

    With a 150-250 example golden set, a headline difference of a couple of
    points is often inside the noise. Reporting an interval instead of a bare
    point estimate is the honest move -- and it feeds the report's
    "what is misleading about my headline number?" section.
    """
    v = np.asarray(list(values), dtype=float)
    if len(v) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return (
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


# --------------------------------------------------------------------------
# Confidence Calibration
# --------------------------------------------------------------------------
def brier_score(y_true: list[int | bool], y_prob: list[float]) -> float:
    """Mean squared error between predicted probabilities and binary outcomes.
    
    Lower is better. A model predicting 0.5 for everything scores 0.25 on a 50/50 split.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    if len(yt) == 0:
        return 0.0
    return float(np.mean((yp - yt) ** 2))


def expected_calibration_error(y_true: list[int | bool], y_prob: list[float], n_bins: int = 5) -> float:
    """Expected Calibration Error (ECE) across equal-width confidence bins.
    
    Weighted average of |accuracy - confidence| per bin. Lower is better.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    n = len(yt)
    if n == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        idx = (yp >= low) & (yp <= high if i == n_bins - 1 else yp < high)
        bin_count = int(np.sum(idx))
        if bin_count > 0:
            bin_acc = float(np.mean(yt[idx]))
            bin_conf = float(np.mean(yp[idx]))
            ece += (bin_count / n) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_bins(y_true: list[int | bool], y_prob: list[float], n_bins: int = 5) -> list[dict]:
    """Breakdown of confidence bins for plotting or inspecting reliability diagrams."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    n = len(yt)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        idx = (yp >= low) & (yp <= high if i == n_bins - 1 else yp < high)
        count = int(np.sum(idx))
        acc = float(np.mean(yt[idx])) if count > 0 else 0.0
        conf = float(np.mean(yp[idx])) if count > 0 else (low + high) / 2
        out.append({
            "bin": f"[{low:.2f}-{high:.2f}]",
            "count": count,
            "accuracy": round(acc, 3),
            "confidence": round(conf, 3),
            "gap": round(abs(acc - conf), 3) if count > 0 else 0.0,
        })
    return out

