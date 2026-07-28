"""Training-only EDA.

Runs on the 141 Training people and nothing else.  These numbers are printed
for orientation; they must not be used to pick features or thresholds, because
nothing computed over the whole Training set is fold-local.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_cohort(y: np.ndarray, person_ids: np.ndarray, diagnoses: pd.Series | None = None) -> dict:
    counts = {"CN": int(np.count_nonzero(y == 0)), "MCI_DEM": int(np.count_nonzero(y == 1))}
    out = {
        "n_persons": int(len(y)),
        "class_counts": counts,
        "positive_rate": float(counts["MCI_DEM"] / len(y)) if len(y) else float("nan"),
        "all_cn_baseline_accuracy": float(counts["CN"] / len(y)) if len(y) else float("nan"),
        "n_unique_person_ids": int(len(set(map(str, person_ids)))),
    }
    if diagnoses is not None:
        out["diagnosis_counts"] = {k: int(v) for k, v in diagnoses.value_counts().items()}
    return out


def missingness_report(X: pd.DataFrame, top: int = 15) -> dict:
    fraction = X.isna().mean().sort_values(ascending=False)
    return {
        "overall_missing_fraction": float(X.isna().to_numpy().mean()),
        "n_columns_any_missing": int((fraction > 0).sum()),
        "worst_columns": {k: round(float(v), 4) for k, v in fraction.head(top).items()},
    }


def univariate_screen(X: pd.DataFrame, y: np.ndarray, top: int = 20) -> dict:
    """Rank-biserial |AUC-0.5| per feature. Diagnostic only -- never a selector."""

    from .metrics import roc_auc

    scores: dict[str, float] = {}
    for column in X.columns:
        values = pd.to_numeric(X[column], errors="coerce").to_numpy(dtype=np.float64)
        mask = np.isfinite(values)
        if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
            continue
        scores[column] = abs(roc_auc(y[mask], values[mask]) - 0.5)

    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:top]
    return {
        "n_scored": len(scores),
        "top_absolute_auc_gap": {k: round(float(v), 4) for k, v in ordered},
        "note": "Computed on all Training people; diagnostic only, not used for selection.",
    }


def run_eda(X: pd.DataFrame, y: np.ndarray, person_ids: np.ndarray,
            diagnoses: pd.Series | None = None) -> dict:
    return {
        "cohort": summarize_cohort(y, person_ids, diagnoses),
        "missingness": missingness_report(X),
        "univariate": univariate_screen(X, y),
        "n_features": int(X.shape[1]),
    }
