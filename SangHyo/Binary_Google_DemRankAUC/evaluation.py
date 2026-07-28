"""Subject-level metrics, uncertainty, and curve artifacts.

Primary metric is ROC-AUC.  Secondary metrics -- PR-AUC, Dem recall, F1,
balanced accuracy, MCC, specificity -- are reported alongside, and every
threshold-dependent one is computed at a threshold chosen on *inner* OOF data
(see ``engine.py``), never on the scores being reported.

Uncertainty is reported twice because the two numbers answer different
questions.  The spread across repeats is split noise: how much the estimate
moves when the folds are redrawn.  The subject bootstrap CI is sampling noise:
how much it would move with a different sample of patients.  With 12 positives
the second dominates and is far wider than any difference between models here,
which is why this folder reports paired bootstrap differences rather than
declaring a winner from point estimates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PRIMARY_METRIC = "roc_auc"


def subject_metrics(y: np.ndarray, score: np.ndarray, *, threshold: float | None = None) -> dict:
    """All reported metrics for one set of subject-level scores."""

    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if y.shape != score.shape:
        raise ValueError("y and score must have the same shape")

    out: dict[str, float | int | dict] = {
        "n": int(y.size),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
    }
    if len(np.unique(y)) < 2:
        out.update({"roc_auc": float("nan"), "pr_auc": float("nan")})
        return out

    out["roc_auc"] = float(roc_auc_score(y, score))
    out["pr_auc"] = float(average_precision_score(y, score))
    out["pr_auc_baseline"] = float(y.mean())

    if threshold is None:
        return out
    out.update(metrics_from_predictions(y, (score >= threshold).astype(np.int64)))
    out["threshold"] = float(threshold)
    return out


def metrics_from_predictions(y: np.ndarray, predicted: np.ndarray) -> dict:
    """Threshold-dependent metrics from binary predictions that already exist.

    Used for the nested operating point, where each fold applied *its own*
    inner-derived threshold to its own test rows.  Pooling those decisions is
    the honest way to report recall/F1/MCC: re-thresholding a pooled score
    vector would quietly replace the nested threshold with a global one.
    """

    y = np.asarray(y, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    tp = int(np.sum((y == 1) & (predicted == 1)))
    tn = int(np.sum((y == 0) & (predicted == 0)))
    fp = int(np.sum((y == 0) & (predicted == 1)))
    fn = int(np.sum((y == 1) & (predicted == 0)))
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    out: dict[str, float | dict] = {}
    out.update(
        {
            "dem_recall": float(recall),
            "specificity": float(specificity),
            "precision": float(precision),
            "f1": float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
            "balanced_accuracy": float(0.5 * (recall + specificity)),
            "accuracy": float((tp + tn) / y.size),
            "mcc": float(matthews_corrcoef(y, predicted)) if len(np.unique(predicted)) > 1 else 0.0,
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }
    )
    return out


def youden_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Operating point from inner OOF only.  Never fitted on reported scores."""

    y = np.asarray(y, dtype=np.int64)
    if len(np.unique(y)) < 2:
        return float(np.median(score))
    fpr, tpr, thresholds = roc_curve(y, score)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def specificity_threshold(y: np.ndarray, score: np.ndarray, target: float = 0.95) -> float:
    """Threshold achieving a target specificity on inner OOF data."""

    y = np.asarray(y, dtype=np.int64)
    negatives = np.asarray(score, dtype=np.float64)[y == 0]
    if negatives.size == 0:
        return float(np.median(score))
    return float(np.quantile(negatives, target))


def bootstrap_auc(y: np.ndarray, score: np.ndarray, *, n_boot: int = 4000,
                  seed: int = 0) -> dict:
    """Percentile bootstrap over *subjects* for the ROC-AUC point estimate."""

    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = y.size
    values: list[float] = []
    for _ in range(int(n_boot)):
        index = rng.integers(0, n, size=n)
        if len(np.unique(y[index])) < 2:
            continue
        values.append(float(roc_auc_score(y[index], score[index])))
    if not values:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    values_array = np.asarray(values)
    return {
        "point": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan"),
        "lo": float(np.percentile(values_array, 2.5)),
        "hi": float(np.percentile(values_array, 97.5)),
        "n_boot": int(values_array.size),
    }


def paired_bootstrap_delta(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, *,
                           n_boot: int = 4000, seed: int = 0) -> dict:
    """CI for ``AUC(a) - AUC(b)`` resampling the same subjects for both.

    A CI containing zero means the comparison does not support declaring a
    winner, no matter what the point estimates look like.  At 12 positives this
    is the normal outcome and is reported as such.
    """

    y = np.asarray(y, dtype=np.int64)
    score_a = np.asarray(score_a, dtype=np.float64)
    score_b = np.asarray(score_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = y.size
    deltas: list[float] = []
    for _ in range(int(n_boot)):
        index = rng.integers(0, n, size=n)
        if len(np.unique(y[index])) < 2:
            continue
        deltas.append(
            float(roc_auc_score(y[index], score_a[index]))
            - float(roc_auc_score(y[index], score_b[index]))
        )
    if not deltas:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "excludes_zero": False, "n_boot": 0}
    deltas_array = np.asarray(deltas)
    lo, hi = (float(np.percentile(deltas_array, 2.5)), float(np.percentile(deltas_array, 97.5)))
    return {
        "delta": float(roc_auc_score(y, score_a) - roc_auc_score(y, score_b)),
        "lo": lo,
        "hi": hi,
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "n_boot": int(deltas_array.size),
    }


def curve_points(y: np.ndarray, score: np.ndarray) -> dict:
    """ROC and PR curve coordinates, saved as data so plots are reproducible."""

    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    fpr, tpr, roc_thresholds = roc_curve(y, score)
    precision, recall, pr_thresholds = precision_recall_curve(y, score)
    return {
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                "thresholds": np.nan_to_num(roc_thresholds, posinf=0.0).tolist()},
        "pr": {"precision": precision.tolist(), "recall": recall.tolist(),
               "thresholds": pr_thresholds.tolist()},
    }


def save_curves(y: np.ndarray, score: np.ndarray, output_dir: Path, stem: str,
                *, title: str = "") -> list[str]:
    """Write ROC and PR curves as CSV always, and as PNG when matplotlib exists."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    points = curve_points(y, score)
    roc_csv = output_dir / f"{stem}_roc_curve.csv"
    pr_csv = output_dir / f"{stem}_pr_curve.csv"
    _write_csv(roc_csv, ["fpr", "tpr"], [points["roc"]["fpr"], points["roc"]["tpr"]])
    _write_csv(pr_csv, ["recall", "precision"], [points["pr"]["recall"], points["pr"]["precision"]])
    written += [str(roc_csv), str(pr_csv)]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - headless without matplotlib
        return written

    auc = roc_auc_score(y, score) if len(np.unique(y)) > 1 else float("nan")
    ap = average_precision_score(y, score) if len(np.unique(y)) > 1 else float("nan")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(points["roc"]["fpr"], points["roc"]["tpr"], lw=2)
    axes[0].plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate",
                title=f"ROC (AUC = {auc:.3f})")
    axes[1].plot(points["pr"]["recall"], points["pr"]["precision"], lw=2)
    axes[1].axhline(float(np.mean(y)), ls="--", c="grey", lw=1)
    axes[1].set(xlabel="Recall (Dem)", ylabel="Precision",
                title=f"Precision-Recall (AP = {ap:.3f}, prevalence = {np.mean(y):.3f})")
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    png = output_dir / f"{stem}_curves.png"
    figure.savefig(png, dpi=140)
    plt.close(figure)
    written.append(str(png))
    return written


def _write_csv(path: Path, header: list[str], columns: list[list[float]]) -> None:
    rows = ["\t".join(header).replace("\t", ",")]
    for values in zip(*columns):
        rows.append(",".join(f"{float(v):.6f}" for v in values))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def summarize_repeats(per_repeat: list[float]) -> dict:
    values = np.asarray([v for v in per_repeat if np.isfinite(v)], dtype=np.float64)
    if values.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"),
                "max": float("nan"), "n": 0}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
        "n": int(values.size),
    }


def write_json(path: Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return str(value)


__all__ = [
    "PRIMARY_METRIC", "bootstrap_auc", "curve_points", "paired_bootstrap_delta",
    "save_curves", "specificity_threshold", "subject_metrics", "summarize_repeats",
    "write_json", "youden_threshold",
]
