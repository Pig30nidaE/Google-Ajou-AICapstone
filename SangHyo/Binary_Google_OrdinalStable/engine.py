"""Nested CV for the ablation, with the two selection fixes built in.

The evaluation harness is deliberately identical in shape to
``Binary_Google_MaxAUC_Tuned`` -- subject-level outer folds, every decision made
on outer-train only, the same pooled-OOF ROC-AUC as the headline -- so the number
this folder produces is directly comparable to that run's **0.7172** and to
``Binary_MMSE_MaxAUC``'s **0.7657**.  Only three things change:

1. **Selection**: ``stability`` replaces the per-fold "tune ``top_k``" step whose
   choices never converged (see ``selection.py``).
2. **Config choice**: instead of ``argmax`` over the inner grid, predictions are
   averaged over the **top-M** configurations (``top_m_average``).  Picking the
   single inner winner is itself an estimate with variance; averaging the near-
   ties keeps the good region of the space and discards the coin-flip between
   configurations that were statistically indistinguishable.
3. **Budget**: a small fixed grid, with the compute going into repeats. The
   previous run showed that a large grid buys inner-fold AUC (+0.053 over outer)
   rather than real performance.

``optimism`` is computed exactly as before, because a tuned number without it is
not interpretable.
"""

from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold

from SangHyo.Binary_Google_MaxAUC_Tuned.engine import (
    binary_metrics,
    blend_scores,
    optimize_blend,
    safe_auc,
    select_features,
)

from .learners import make_model
from .selection import stability_select

# Small fixed grid: the knobs that matter, nothing more.
GRIDS = {
    "logreg": [{"C": c, "l1_ratio": 0.0} for c in (0.03, 0.1, 0.3, 1.0)],
    "svm": [{"C": c} for c in (0.3, 1.0, 3.0)],
    "ydf_gbt": [{"num_trees": 300, "max_depth": d, "shrinkage": 0.05,
                 "min_examples": 8, "l2_regularization": 2.0} for d in (2, 3, 4)],
    "ydf_rf": [{"num_trees": 500, "max_depth": d, "min_examples": 5,
                "num_candidate_attributes_ratio": 0.5} for d in (4, 6)],
    "ydf_gbt_oblique": [{"num_trees": 300, "max_depth": d, "shrinkage": 0.05,
                         "min_examples": 8, "l2_regularization": 2.0} for d in (2, 3, 4)],
}


def choose_columns(X, y, mode: str, *, top_k: int, seed: int):
    """Fold-internal feature selection. ``mode`` is 'fold_topk' or 'stability'."""

    if mode == "stability":
        cols, _freq = stability_select(X, y, top_k=top_k, seed=seed)
        return cols
    return select_features(X, y, top_k=top_k, corr_threshold=0.95)


def _inner_oof(X, y, severity, cols, kind, params, strategy, *, folds, seed):
    prob = np.zeros(len(y))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in splitter.split(X, y):
        model = make_model(kind, params, strategy=strategy, seed=seed)
        model.fit(X[np.ix_(tr, cols)], y[tr], severity[tr])
        prob[te] = model.predict_score(X[np.ix_(te, cols)])
    return prob


def tune_kind(X, y, severity, cols, kind, strategy, *, folds, seed, top_m):
    """Score the small grid by inner AUC, then average the best ``top_m`` configs."""

    scored = []
    for params in GRIDS[kind]:
        oof = _inner_oof(X, y, severity, cols, kind, params, strategy, folds=folds, seed=seed)
        scored.append((safe_auc(y, oof), params, oof))
    scored.sort(key=lambda item: -item[0])
    keep = scored[:max(1, min(top_m, len(scored)))]
    blended = np.mean([item[2] for item in keep], axis=0)
    return {"auc": safe_auc(y, blended), "params_list": [item[1] for item in keep],
            "oof": blended, "best_single_auc": scored[0][0]}


def fit_pool(X, y, severity, cols, kinds, strategy, *, folds, seed, auc_gate, top_m):
    tuned = {k: tune_kind(X, y, severity, cols, k, strategy, folds=folds, seed=seed, top_m=top_m)
             for k in kinds}
    eligible = [k for k in kinds if tuned[k]["auc"] >= auc_gate]
    mode = "gated_blend"
    if not eligible:
        eligible = [max(kinds, key=lambda k: tuned[k]["auc"])]
        mode = f"fallback_best:{eligible[0]}"
    matrix = np.column_stack([tuned[k]["oof"] for k in eligible])
    weights, blend_auc = optimize_blend(matrix, y, rng=np.random.default_rng(seed))
    return {"tuned": tuned, "eligible": eligible, "weights": weights,
            "inner_auc": blend_auc, "mode": mode}


def _predict_pool(X_tr, y_tr, sev_tr, X_te, cols, pool, strategy, seed):
    scores = []
    for kind in pool["eligible"]:
        per_config = []
        for params in pool["tuned"][kind]["params_list"]:
            model = make_model(kind, params, strategy=strategy, seed=seed)
            model.fit(X_tr[:, cols], y_tr, sev_tr)
            per_config.append(model.predict_score(X_te[:, cols]))
        scores.append(np.mean(per_config, axis=0))
    return blend_scores(np.column_stack(scores), pool["weights"])


def nested_cv(data, kinds: Sequence[str], *, strategy: str, selection: str, repeats: int,
              outer_k: int, inner_k: int, top_k: int, top_m: int, auc_gate: float = 0.55,
              seed: int = 20260728, log: Callable[[str], None] = print) -> dict:
    X, y, severity = data.X, data.y, data.severity
    n = len(y)
    score_sum = np.zeros(n)
    seen = np.zeros(n)
    fold_aucs: list[float] = []
    inner_minus_outer: list[float] = []
    selected_counts = np.zeros(X.shape[1])
    n_selected: list[int] = []
    started = time.monotonic()

    for repeat in range(repeats):
        outer = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=seed + repeat)
        for fold, (tr, te) in enumerate(outer.split(np.arange(n), y)):
            fold_seed = seed + 100 * repeat + fold
            cols = choose_columns(X[tr], y[tr], selection, top_k=top_k, seed=fold_seed)
            selected_counts[cols] += 1
            n_selected.append(len(cols))
            pool = fit_pool(X[tr], y[tr], severity[tr], cols, kinds, strategy,
                            folds=inner_k, seed=fold_seed, auc_gate=auc_gate, top_m=top_m)
            score = _predict_pool(X[tr], y[tr], severity[tr], X[te], cols, pool,
                                  strategy, fold_seed)
            score_sum[te] += score
            seen[te] += 1
            fold_auc = safe_auc(y[te], score)
            fold_aucs.append(fold_auc)
            inner_minus_outer.append(pool["inner_auc"] - fold_auc)
        log(f"      repeat {repeat + 1}/{repeats}: pooled-so-far "
            f"{safe_auc(y, score_sum / np.where(seen == 0, 1, seen)):.4f} "
            f"[{time.monotonic() - started:.0f}s]")

    score = score_sum / np.where(seen == 0, 1, seen)
    prob = 1.0 / (1.0 + np.exp(-score))
    return {
        "pooled_oof_roc_auc": safe_auc(y, score),
        "mean_fold_roc_auc": float(np.mean(fold_aucs)),
        "std_fold_roc_auc": float(np.std(fold_aucs)),
        "mean_inner_minus_outer": float(np.mean(inner_minus_outer)),
        "oof_score": score.tolist(),
        "oof_prob": prob.tolist(),
        "mean_n_selected": float(np.mean(n_selected)),
        "std_n_selected": float(np.std(n_selected)),
        "selection_frequency_top": _top_features(selected_counts, data.feature_names,
                                                 len(fold_aucs)),
        "elapsed_seconds": time.monotonic() - started,
    }


def _top_features(counts: np.ndarray, feature_names, n_folds: int, top: int = 15) -> list:
    if n_folds == 0:
        return []
    order = np.argsort(-counts)[:top]
    return [[feature_names[int(j)], round(float(counts[j] / n_folds), 3)] for j in order]


def non_nested_reference(data, kinds, *, strategy: str, selection: str, inner_k: int,
                         top_k: int, top_m: int, auc_gate: float = 0.55,
                         seed: int = 20260728) -> dict:
    """Deliberately optimistic: select and tune on everything, report that same OOF."""

    cols = choose_columns(data.X, data.y, selection, top_k=top_k, seed=seed)
    pool = fit_pool(data.X, data.y, data.severity, cols, kinds, strategy,
                    folds=inner_k, seed=seed, auc_gate=auc_gate, top_m=top_m)
    matrix = np.column_stack([pool["tuned"][k]["oof"] for k in pool["eligible"]])
    return {"roc_auc": safe_auc(data.y, blend_scores(matrix, pool["weights"])),
            "eligible": list(pool["eligible"]),
            "weights": {k: float(w) for k, w in zip(pool["eligible"], pool["weights"])},
            "cols": cols.tolist(),
            "params_by_kind": {k: pool["tuned"][k]["params_list"] for k in pool["eligible"]}}


__all__ = ["GRIDS", "binary_metrics", "choose_columns", "fit_pool", "nested_cv",
           "non_nested_reference", "safe_auc"]
