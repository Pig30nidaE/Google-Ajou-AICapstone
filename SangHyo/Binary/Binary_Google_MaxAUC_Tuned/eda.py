"""EDA stage: what is actually in the feature matrix before any tuning starts.

Produces four things the modelling stage (and the write-up) depends on:

1. per-feature direction-free univariate ROC-AUC and missingness,
2. a feature-*block* comparison (MMSE / wearable / both) under one fixed cheap
   classifier, which is what justifies keeping wearables in the pool at all,
3. an ablation of the adherence features flagged in ``features.SUSPECT_FEATURES``
   -- if dropping them barely moves AUC they are harmless, and if they move it a
   lot they need to be treated as a possible protocol artifact rather than a win,
4. a correlation-redundancy summary, which motivates the ``corr_threshold``
   knob in the search space.

Everything here runs on the training split only.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .engine import direction_free_auc, safe_auc
from .features import SUSPECT_FEATURES, SubjectData
from .numeric import column_median, impute


def quick_cv_auc(X: np.ndarray, y: np.ndarray, *, repeats: int = 3, folds: int = 5,
                 seed: int = 7, C: float = 0.1) -> float:
    """One fixed, cheap, leakage-free pipeline used only to compare feature sets."""

    if X.shape[1] == 0:
        return 0.5
    total = np.zeros(len(y))
    seen = np.zeros(len(y))
    for r in range(repeats):
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed + r).split(X, y):
            median = column_median(X[tr])
            xt = impute(X[tr], median)
            xe = impute(X[te], median)
            mu = xt.mean(0)
            sd = np.where(xt.std(0) < 1e-8, 1.0, xt.std(0))
            model = LogisticRegression(C=C, class_weight="balanced", max_iter=6000)
            model.fit((xt - mu) / sd, y[tr])
            total[te] += model.predict_proba((xe - mu) / sd)[:, 1]
            seen[te] += 1
    return safe_auc(y, total / np.where(seen == 0, 1, seen))


def _block(data: SubjectData, prefix: str) -> np.ndarray:
    cols = [i for i, n in enumerate(data.feature_names) if n.startswith(prefix)]
    return data.X[:, cols] if cols else np.empty((data.n_subjects, 0))


def _redundancy(X: np.ndarray, threshold: float = 0.95) -> int:
    finite_mask = np.isfinite(X)
    keep = [j for j in range(X.shape[1])
            if finite_mask[:, j].sum() >= 2 and np.std(X[finite_mask[:, j], j]) > 1e-10]
    if len(keep) < 2:
        return 0
    sub = np.where(finite_mask[:, keep], X[:, keep], np.nan)
    filled = np.where(np.isfinite(sub), sub, np.nanmedian(sub, axis=0))
    corr = np.corrcoef(filled, rowvar=False)
    upper = np.triu(np.abs(corr), k=1)
    return int(np.sum(upper >= threshold))


def run_eda(data: SubjectData) -> dict:
    y = data.y
    aucs = [(name, direction_free_auc(y, data.X[:, j]))
            for j, name in enumerate(data.feature_names)]
    aucs.sort(key=lambda kv: -kv[1])
    missing = {name: float(np.mean(~np.isfinite(data.X[:, j])))
               for j, name in enumerate(data.feature_names)}

    mmse_block = _block(data, "mmse_")
    wear_cols = [i for i, n in enumerate(data.feature_names) if n.startswith("w_")]
    wear_block = data.X[:, wear_cols] if wear_cols else np.empty((data.n_subjects, 0))

    suspect_present = [n for n in SUSPECT_FEATURES if n in data.feature_names]
    without_suspect = data.drop(suspect_present) if suspect_present else data

    report = {
        "n_subjects": int(data.n_subjects),
        "n_features": int(data.n_features),
        "class_counts": {"CN": int((y == 0).sum()), "MCI_DEM": int((y == 1).sum())},
        "all_cn_accuracy": float(np.mean(y == 0)),
        "n_mmse_features": int(mmse_block.shape[1]),
        "n_wearable_features": int(wear_block.shape[1]),
        "top_feature_direction_free_auc": [[n, round(a, 4)] for n, a in aucs[:30]],
        "worst_feature_direction_free_auc": [[n, round(a, 4)] for n, a in aucs[-10:]],
        "features_with_missing": {n: round(v, 3) for n, v in sorted(
            missing.items(), key=lambda kv: -kv[1]) if v > 0}
        ,
        "highly_correlated_pairs_at_0.95": _redundancy(data.X, 0.95),
        "feature_block_cv_roc_auc": {
            "mmse_only": round(quick_cv_auc(mmse_block, y), 4),
            "wearable_only": round(quick_cv_auc(wear_block, y), 4),
            "all_features": round(quick_cv_auc(data.X, y), 4),
        },
        "suspect_feature_ablation": {
            "suspect_features": suspect_present,
            "univariate_auc": {n: round(direction_free_auc(y, data.X[:, list(data.feature_names).index(n)]), 4)
                               for n in suspect_present},
            "all_features_cv_roc_auc": round(quick_cv_auc(data.X, y), 4),
            "without_suspect_cv_roc_auc": round(quick_cv_auc(without_suspect.X, y), 4),
            "note": ("A large gap here means the adherence/wear-time features are doing "
                     "real work -- which may be genuine (disengagement) or a protocol "
                     "artifact. Re-run with MAXAUC_DROP_SUSPECT=1 to get the conservative "
                     "number."),
        },
        "note": ("Direction-free univariate AUC: max(auc, 1-auc), so 0.5 means no "
                 "separation regardless of sign. The block comparison uses one fixed "
                 "logistic regression, not the tuned pipeline, so it only ranks feature "
                 "sets -- it is not the headline metric."),
    }
    return report


__all__ = ["quick_cv_auc", "run_eda"]
