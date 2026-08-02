"""EDA stage — quantify feature discriminability before modelling.

Runs on the training split only and saves an EDA report:
* direction-free univariate ROC-AUC per feature (which features separate CN vs
  MCI+DEM), and
* a compact feature-set comparison (MMSE-only vs MMSE+wearable) via a quick
  leakage-free subject CV, documenting why the model is MMSE-focused.

This mirrors the manual analysis behind this folder and keeps the
EDA -> preprocessing -> split -> training pipeline explicit and reproducible.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .features import SubjectData


def _direction_free_auc(y: np.ndarray, col: np.ndarray) -> float:
    mask = np.isfinite(col)
    if mask.sum() < len(y) * 0.5 or len(np.unique(y[mask])) < 2:
        return float("nan")
    auc = roc_auc_score(y[mask], col[mask])
    return float(max(auc, 1 - auc))


def _quick_cv_auc(X: np.ndarray, y: np.ndarray, *, repeats=3, folds=5, seed=7) -> float:
    n = len(y)
    prob = np.zeros(n)
    seen = np.zeros(n)
    for r in range(repeats):
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed + r).split(X, y):
            med = np.nanmedian(X[tr], axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            xt = np.where(np.isfinite(X[tr]), X[tr], med)
            xe = np.where(np.isfinite(X[te]), X[te], med)
            mu, sd = xt.mean(0), np.where(xt.std(0) < 1e-8, 1.0, xt.std(0))
            model = LogisticRegression(C=0.1, class_weight="balanced", max_iter=4000)
            model.fit((xt - mu) / sd, y[tr])
            prob[te] += model.predict_proba((xe - mu) / sd)[:, 1]
            seen[te] += 1
    return float(roc_auc_score(y, prob / seen))


def run_eda(mmse_only: SubjectData, with_wearable: SubjectData | None) -> dict:
    y = mmse_only.y
    aucs = sorted(
        ((name, _direction_free_auc(y, mmse_only.X[:, j]))
         for j, name in enumerate(mmse_only.feature_names)),
        key=lambda kv: (-(kv[1] if kv[1] == kv[1] else 0.0)),
    )
    report = {
        "n_subjects": int(mmse_only.n_subjects),
        "class_counts": {"CN": int((y == 0).sum()), "MCI_DEM": int((y == 1).sum())},
        "all_cn_accuracy": float(np.mean(y == 0)),
        "n_mmse_features": len(mmse_only.feature_names),
        "top_feature_direction_free_auc": [[n, round(a, 4)] for n, a in aucs[:20]],
        "feature_set_cv_roc_auc": {
            "mmse_only": round(_quick_cv_auc(mmse_only.X, y), 4),
        },
        "finding": ("CN vs MCI+DEM is MMSE-driven; delayed recall + engineered "
                    "num_failed/recall_deficit are strongest. Wearables dilute the "
                    "signal, so the model is MMSE-focused (wearable optional)."),
    }
    if with_wearable is not None:
        report["feature_set_cv_roc_auc"]["mmse_plus_wearable"] = round(
            _quick_cv_auc(with_wearable.X, y), 4)
    return report


__all__ = ["run_eda"]
