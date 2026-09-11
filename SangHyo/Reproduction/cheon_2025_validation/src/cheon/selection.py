"""SHAP-importance ranking and forward-selection path (paper §3.2(2), p.165).

The functions here never see anything except the arrays passed in. The *caller* decides the scope:
  R : ranking on the whole dataset, path evaluated with the outer folds        (paper-faithful, A15)
  G : ranking on the training fold only, k fixed at the paper's 40             (A17)
  N : ranking on the inner-training part, path evaluated on inner validation   (A19)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import warnings

import shap

warnings.filterwarnings("ignore", message=".*TreeExplainer shap values output has changed.*")
from sklearn.metrics import roc_auc_score

from .model import make_lgbm


def shap_mean_abs(model, X: np.ndarray) -> np.ndarray:
    """Mean |SHAP| per feature (positive class) via TreeExplainer (A14)."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):  # older shap: [class0, class1]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:  # (n, p, 2)
        sv = sv[:, :, 1]
    if sv.shape != X.shape:
        raise RuntimeError(f"unexpected SHAP shape {sv.shape} for X {X.shape}")
    return np.abs(sv).mean(axis=0)


def rank_features(X_fit: np.ndarray, y_fit: np.ndarray, feature_names: list[str], params: dict, seed: int,
                  n_threads: int, X_explain: np.ndarray | None = None) -> pd.DataFrame:
    """Fit LightGBM(params) on (X_fit, y_fit), explain on X_explain (default: X_fit), return ranked importances."""
    model = make_lgbm(params, seed, n_threads).fit(X_fit, y_fit)
    imp = shap_mean_abs(model, X_fit if X_explain is None else X_explain)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": imp})
    df = df.sort_values(["mean_abs_shap", "feature"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def forward_path(X: np.ndarray, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]], rankings,
                 feature_names: list[str], params: dict, seed: int, n_threads: int, max_k: int | None = None,
                 log=None) -> pd.DataFrame:
    """For each fold f and k = 1..max_k: fit LightGBM(params) on top-k of rankings[f] using the fold's train rows,
    score the fold's test rows -> DataFrame[fold, k, auc].

    `rankings` is either one list of feature names (used for every fold; R) or a list of per-fold lists (G, N).
    """
    p = len(feature_names)
    max_k = p if max_k is None else min(max_k, p)
    name_to_col = {n: i for i, n in enumerate(feature_names)}
    per_fold = rankings if (len(rankings) == len(folds) and isinstance(rankings[0], (list, tuple, np.ndarray))
                            and not isinstance(rankings[0], str)) else [rankings] * len(folds)
    rows = []
    for f, (tr, te) in enumerate(folds):
        cols = [name_to_col[n] for n in per_fold[f]]
        for k in range(1, max_k + 1):
            sel = cols[:k]
            m = make_lgbm(params, seed, n_threads).fit(X[np.ix_(tr, sel)], y[tr])
            auc = roc_auc_score(y[te], m.predict_proba(X[np.ix_(te, sel)])[:, 1])
            rows.append({"fold": f, "k": k, "auc": float(auc)})
        if log:
            log(f"    forward path fold {f}: done k=1..{max_k}")
    return pd.DataFrame(rows)


def best_k(path: pd.DataFrame) -> tuple[int, float]:
    """Global argmax of the fold-mean AUC over k; ties -> smallest k (A16)."""
    mean = path.groupby("k")["auc"].mean()
    k = int(mean.idxmax())
    return k, float(mean.loc[k])
