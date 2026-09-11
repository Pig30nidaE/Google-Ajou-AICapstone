"""Metrics (Table 1 columns + auxiliaries) and the pre-defined aggregation rules (prompt §19)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

PAPER_METRICS = ["accuracy", "roc_auc", "precision_macro", "recall_macro", "f1_macro"]
AUX_METRICS = ["precision_pos", "recall_pos_sensitivity", "specificity", "balanced_accuracy", "pr_auc"]
ALL_METRICS = PAPER_METRICS + AUX_METRICS


def compute_metrics(y_true: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred = np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, score) if len(np.unique(y_true)) == 2 else np.nan,
        "precision_macro": precision_score(y_true, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "precision_pos": precision_score(y_true, pred, pos_label=1, zero_division=0),
        "recall_pos_sensitivity": recall_score(y_true, pred, pos_label=1, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "pr_auc": average_precision_score(y_true, score) if len(np.unique(y_true)) == 2 else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_test": int(len(y_true)), "n_test_pos": int(y_true.sum()),
    }
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


def _t_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (float("nan"), float("nan"))
    m, sd = v.mean(), v.std(ddof=1)
    h = stats.t.ppf(1 - alpha / 2, len(v) - 1) * sd / np.sqrt(len(v))
    return (float(m - h), float(m + h))


def aggregate(fold_df: pd.DataFrame, metrics: list[str] = ALL_METRICS) -> dict:
    """Pre-defined statistics for one (experiment, stage):

    primary  = per-repeat mean over folds (paper Figure 3 definition), then mean / SD / 95% t-CI across repeats.
    pooled   = all fold scores pooled (n_repeats x n_folds), mean / SD  (sensitivity only).
    """
    out = {}
    per_rep = fold_df.groupby("repeat")[metrics].mean()
    for m in metrics:
        rep_vals = per_rep[m].to_numpy(dtype=float)
        pooled = fold_df[m].to_numpy(dtype=float)
        lo, hi = _t_ci(rep_vals)
        out[m] = {
            "primary_repeat_mean": {
                "mean": float(np.nanmean(rep_vals)),
                "sd_across_repeats": float(np.nanstd(rep_vals, ddof=1)) if len(rep_vals) > 1 else None,
                "ci95_low": lo if len(rep_vals) > 1 else None, "ci95_high": hi if len(rep_vals) > 1 else None,
                "n_repeats": int(len(rep_vals)), "per_repeat": [float(x) for x in rep_vals],
            },
            "sensitivity_pooled_folds": {
                "mean": float(np.nanmean(pooled)), "sd": float(np.nanstd(pooled, ddof=1)) if len(pooled) > 1 else None,
                "n_fold_scores": int(len(pooled)),
            },
        }
    return out


def pooled_oof_auc(oof: pd.DataFrame) -> dict:
    """Per-repeat AUC over concatenated out-of-fold predictions (sensitivity: single AUC per repeat)."""
    res = {}
    for rep, g in oof.groupby("repeat"):
        res[int(rep)] = float(roc_auc_score(g["y"], g["score"])) if g["y"].nunique() == 2 else float("nan")
    vals = np.array(list(res.values()), dtype=float)
    return {"per_repeat": res, "mean": float(np.nanmean(vals)), "sd": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else None}
