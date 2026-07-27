"""Nested CV where *all* tuning happens strictly inside the outer fold.

The single most important property of this file
-----------------------------------------------
Hyperparameter search, feature selection and ensemble-weight fitting are three
separate opportunities to overfit the evaluation.  If any of them sees the data
you later score on, the reported ROC-AUC is inflated -- this is exactly the
mechanism that produced the paper's "0.9" in the sibling folders, only subtler.

So the structure is::

    for each outer fold (subject-level, leakage-free):
        train part -> feature selection    (re-done inside every inner fold)
                   -> hyperparameter search (scored by inner-fold OOF AUC)
                   -> ensemble weight search (fitted on inner-fold OOF probs)
        test part  -> touched exactly once, for the final blended prediction

``non_nested_reference`` deliberately does the *wrong* thing (tunes on all the
data, then reports OOF from that same tuning) so the run can quantify its own
selection bias as ``optimism = non_nested_auc - nested_auc``.  Reporting that
gap is the honest way to present a tuned number.
"""

from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .learners import make_learner
from .spaces import default_params, params_key, sample_params

# --------------------------------------------------------------- metrics -----


def binary_metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    metrics = {
        "accuracy": (tp + tn) / len(y) if len(y) else 0.0,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "impaired_recall": recall,
        "cn_specificity": specificity,
        "precision": precision,
        "f1": f1,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    metrics["roc_auc"] = (float(roc_auc_score(y, prob))
                          if prob is not None and len(np.unique(y)) == 2 else float("nan"))
    return metrics


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y)) < 2 or not np.all(np.isfinite(score)):
        finite = np.isfinite(score)
        if finite.sum() < 2 or len(np.unique(y[finite])) < 2:
            return 0.5
        return float(roc_auc_score(y[finite], score[finite]))
    return float(roc_auc_score(y, score))


def select_threshold(y: np.ndarray, prob: np.ndarray, objective: str = "balanced_accuracy") -> float:
    grid = np.unique(np.clip(np.round(prob, 3), 0.02, 0.98))
    candidates = np.concatenate([[0.5], grid]) if len(grid) else np.asarray([0.5])
    best_threshold, best_key = 0.5, (-1.0, -1.0)
    for threshold in candidates:
        m = binary_metrics(y, (prob >= threshold).astype(int))
        key = ((m["accuracy"], m["balanced_accuracy"]) if objective == "accuracy"
               else (m["balanced_accuracy"], m["accuracy"]))
        if key > best_key:
            best_key, best_threshold = key, float(threshold)
    return best_threshold


def select_threshold_specificity(y: np.ndarray, prob: np.ndarray, target: float) -> float:
    y = np.asarray(y)
    cn = np.asarray(prob)[y == 0]
    if len(cn) == 0:
        return 0.5
    for t in np.sort(np.unique(prob)):
        if float((cn < t).mean()) >= target:
            return float(t)
    return float(np.max(prob)) + 1e-6


def bootstrap_ci(y, prob, pred, *, n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    acc, bal, auc = [], [], []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        if len(np.unique(y[s])) < 2:
            continue
        m = binary_metrics(y[s], np.asarray(pred)[s], np.asarray(prob)[s])
        acc.append(m["accuracy"]); bal.append(m["balanced_accuracy"]); auc.append(m["roc_auc"])

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [float("nan")] * 2

    return {"accuracy": ci(acc), "balanced_accuracy": ci(bal), "roc_auc": ci(auc)}


# ----------------------------------------------------- feature selection -----


def direction_free_auc(y: np.ndarray, column: np.ndarray) -> float:
    mask = np.isfinite(column)
    if mask.sum() < max(8, 0.4 * len(y)) or len(np.unique(y[mask])) < 2:
        return 0.5
    auc = roc_auc_score(y[mask], column[mask])
    return float(max(auc, 1.0 - auc))


def select_features(X: np.ndarray, y: np.ndarray, *, top_k: int, corr_threshold: float,
                    min_finite_ratio: float = 0.5) -> np.ndarray:
    """Fold-internal selection: drop degenerate -> rank by AUC -> prune redundancy.

    Called with the *training* part of whichever fold is active, never with the
    part it will be scored on.
    """

    n_features = X.shape[1]
    finite_mask = np.isfinite(X)
    finite_ratio = finite_mask.mean(axis=0)
    finite_count = finite_mask.sum(axis=0)

    # Only columns with >=2 finite values have a defined spread; gating here
    # (instead of letting nanstd warn) keeps long Colab logs readable.
    spread = np.zeros(n_features)
    measurable = finite_count >= 2
    for j in np.where(measurable)[0]:
        spread[j] = np.std(X[finite_mask[:, j], j])
    usable = np.where((finite_ratio >= min_finite_ratio) & (spread > 1e-10))[0]
    if usable.size == 0:
        return np.arange(n_features)

    scores = np.array([direction_free_auc(y, X[:, j]) for j in usable])
    order = usable[np.argsort(-scores)]
    score_by_index = dict(zip(usable.tolist(), scores.tolist()))

    if corr_threshold < 1.0:
        kept: list[int] = []
        for j in order:
            redundant = False
            for k in kept:
                a, b = X[:, j], X[:, k]
                mask = np.isfinite(a) & np.isfinite(b)
                if mask.sum() < 8:
                    continue
                if np.std(a[mask]) < 1e-10 or np.std(b[mask]) < 1e-10:
                    continue
                if abs(np.corrcoef(a[mask], b[mask])[0, 1]) >= corr_threshold:
                    redundant = True
                    break
            if not redundant:
                kept.append(int(j))
        order = np.asarray(kept, dtype=int) if kept else order

    if top_k and top_k > 0:
        order = order[:top_k]
    if order.size == 0:
        order = np.asarray([int(max(score_by_index, key=score_by_index.get))])
    return np.sort(order)


# ---------------------------------------------------------------- tuning -----


def make_splits(y: np.ndarray, *, k: int, repeats: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Repeated stratified splits, shared by every learner kind so their
    out-of-fold prediction vectors stay aligned and therefore blendable."""

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    index = np.arange(len(y))
    for r in range(repeats):
        folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed + r)
        splits.extend((tr, te) for tr, te in folds.split(index, y))
    return splits


def oof_predict(X: np.ndarray, y: np.ndarray, kind: str, params: dict,
                splits: Sequence[tuple[np.ndarray, np.ndarray]], *, seed: int = 0) -> np.ndarray:
    """Out-of-fold probabilities, averaging repeats. Selection re-fit per fold."""

    total = np.zeros(len(y))
    seen = np.zeros(len(y))
    for tr, te in splits:
        cols = select_features(X[tr], y[tr], top_k=int(params.get("top_k", 0)),
                               corr_threshold=float(params.get("corr_threshold", 1.01)))
        learner = make_learner(kind, params, seed=seed)
        learner.fit(X[np.ix_(tr, cols)], y[tr])
        total[te] += learner.predict_proba(X[np.ix_(te, cols)])
        seen[te] += 1
    seen = np.where(seen == 0, 1, seen)
    return total / seen


def tune_kind(X: np.ndarray, y: np.ndarray, kind: str, *, budget: int, screen_splits,
              final_splits, rng: np.random.Generator, keep_fraction: float = 0.25,
              seed: int = 0, anchors: Sequence[dict] = ()) -> dict:
    """Two-stage successive-halving random search, scored by inner-fold AUC.

    Stage 1 screens every candidate on a cheap split set; stage 2 re-scores the
    survivors on the full (repeated) split set.  Only stage-2 OOF vectors are
    returned, so all kinds hand back predictions on identical splits.
    """

    seen_keys: set[str] = set()
    candidates: list[dict] = []
    for params in anchors:
        key = params_key(kind, params)
        if key not in seen_keys:
            seen_keys.add(key)
            candidates.append(dict(params))
    guard = 0
    while len(candidates) < budget and guard < budget * 20:
        guard += 1
        params = sample_params(kind, rng)
        key = params_key(kind, params)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        candidates.append(params)

    screened = []
    for params in candidates:
        oof = oof_predict(X, y, kind, params, screen_splits, seed=seed)
        screened.append((safe_auc(y, oof), params))
    screened.sort(key=lambda item: -item[0])

    n_keep = max(1, int(round(len(screened) * keep_fraction)))
    finalists = [params for _, params in screened[:n_keep]]

    best = {"auc": -1.0, "params": finalists[0], "oof": None}
    for params in finalists:
        oof = oof_predict(X, y, kind, params, final_splits, seed=seed)
        auc = safe_auc(y, oof)
        if auc > best["auc"]:
            best = {"auc": auc, "params": params, "oof": oof}
    if best["oof"] is None:
        best["oof"] = oof_predict(X, y, kind, best["params"], final_splits, seed=seed)
    best["n_candidates"] = len(candidates)
    best["screen_best_auc"] = float(screened[0][0]) if screened else float("nan")
    return best


# -------------------------------------------------------------- blending -----


def _logit(prob: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def blend_scores(prob_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted average in log-odds space.

    Log-odds averaging (rather than rank averaging) keeps an absolute, monotone
    score, so a threshold picked on training out-of-fold predictions still means
    something on new data -- rank averaging would make the score depend on the
    batch it was computed with.
    """

    return _logit(prob_matrix) @ np.asarray(weights, dtype=float)


def optimize_blend(prob_matrix: np.ndarray, y: np.ndarray, *, rng: np.random.Generator,
                   n_draws: int = 4000) -> tuple[np.ndarray, float]:
    """Weights over the simplex maximizing inner-fold AUC of the blended score."""

    n_models = prob_matrix.shape[1]
    if n_models == 1:
        return np.ones(1), safe_auc(y, prob_matrix[:, 0])

    candidates = [np.eye(n_models)[i] for i in range(n_models)]
    candidates.append(np.full(n_models, 1.0 / n_models))
    candidates.extend(rng.dirichlet(np.ones(n_models), size=n_draws))

    best_weights, best_auc = candidates[0], -1.0
    for weights in candidates:
        auc = safe_auc(y, blend_scores(prob_matrix, weights))
        if auc > best_auc:
            best_auc, best_weights = auc, np.asarray(weights, dtype=float)
    return best_weights / best_weights.sum(), float(best_auc)


# ------------------------------------------------------------- nested CV -----


def _fit_full(X: np.ndarray, y: np.ndarray, kind: str, params: dict, seed: int):
    cols = select_features(X, y, top_k=int(params.get("top_k", 0)),
                           corr_threshold=float(params.get("corr_threshold", 1.01)))
    learner = make_learner(kind, params, seed=seed)
    learner.fit(X[:, cols], y)
    return learner, cols


def tune_pool(X: np.ndarray, y: np.ndarray, kinds: Sequence[str], *, budgets: dict,
              inner_k: int, screen_repeats: int, final_repeats: int, auc_gate: float,
              rng: np.random.Generator, seed: int, log: Callable[[str], None]) -> dict:
    """Tune every kind, then fit blend weights on the aligned inner-OOF matrix."""

    screen_splits = make_splits(y, k=inner_k, repeats=screen_repeats, seed=seed + 11)
    final_splits = make_splits(y, k=inner_k, repeats=final_repeats, seed=seed + 101)

    tuned: dict[str, dict] = {}
    for kind in kinds:
        started = time.monotonic()
        # Seed the search with the middle-of-space defaults so a run of bad
        # random draws can never leave a kind worse off than untuned.
        tuned[kind] = tune_kind(X, y, kind, budget=int(budgets.get(kind, 24)),
                                screen_splits=screen_splits, final_splits=final_splits,
                                rng=rng, seed=seed, anchors=[default_params(kind)])
        log(f"      {kind:<18} inner AUC {tuned[kind]['auc']:.4f}  "
            f"({tuned[kind]['n_candidates']} cfgs, {time.monotonic() - started:.0f}s)")

    eligible = [k for k in kinds if tuned[k]["auc"] >= auc_gate]
    mode = "gated_blend"
    if not eligible:
        eligible = [max(kinds, key=lambda k: tuned[k]["auc"])]
        mode = f"fallback_best:{eligible[0]}"

    matrix = np.column_stack([tuned[k]["oof"] for k in eligible])
    weights, blend_auc = optimize_blend(matrix, y, rng=rng)
    return {"tuned": tuned, "eligible": eligible, "weights": weights,
            "inner_blend_auc": blend_auc, "mode": mode}


def nested_cv(X: np.ndarray, y: np.ndarray, kinds: Sequence[str], *, repeats: int, outer_k: int,
              inner_k: int, budgets: dict, screen_repeats: int, final_repeats: int,
              auc_gate: float = 0.55, seed: int = 20260727,
              deadline_seconds: float | None = None,
              log: Callable[[str], None] = print) -> dict:
    """Leakage-free outer CV; every tuning decision is made on outer-train only."""

    n = len(y)
    prob_sum = np.zeros(n)
    score_sum = np.zeros(n)
    seen = np.zeros(n)
    fold_records: list[dict] = []
    repeats_done = 0
    started = time.monotonic()

    for repeat in range(repeats):
        if deadline_seconds is not None and repeat > 0 and time.monotonic() - started > deadline_seconds:
            log(f"  [deadline] stopping after {repeats_done} complete repeat(s)")
            break
        outer = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=seed + repeat)
        for fold, (train_idx, test_idx) in enumerate(outer.split(np.arange(n), y)):
            fold_started = time.monotonic()
            log(f"  [repeat {repeat + 1}/{repeats} fold {fold + 1}/{outer_k}] "
                f"tuning on {len(train_idx)} subjects")
            rng = np.random.default_rng(seed + 1000 * repeat + fold)
            pool = tune_pool(X[train_idx], y[train_idx], kinds, budgets=budgets, inner_k=inner_k,
                             screen_repeats=screen_repeats, final_repeats=final_repeats,
                             auc_gate=auc_gate, rng=rng, seed=seed + 1000 * repeat + fold, log=log)

            test_probs = []
            for kind in pool["eligible"]:
                learner, cols = _fit_full(X[train_idx], y[train_idx], kind,
                                          pool["tuned"][kind]["params"], seed + fold)
                test_probs.append(learner.predict_proba(X[np.ix_(test_idx, cols)]))
            matrix = np.column_stack(test_probs)
            score = blend_scores(matrix, pool["weights"])
            prob = 1.0 / (1.0 + np.exp(-score))

            prob_sum[test_idx] += prob
            score_sum[test_idx] += score
            seen[test_idx] += 1
            fold_auc = safe_auc(y[test_idx], score)
            fold_records.append({
                "repeat": repeat, "fold": fold, "mode": pool["mode"],
                "eligible": list(pool["eligible"]),
                "weights": {k: float(w) for k, w in zip(pool["eligible"], pool["weights"])},
                "inner_blend_auc": pool["inner_blend_auc"],
                "inner_auc_by_kind": {k: float(v["auc"]) for k, v in pool["tuned"].items()},
                "best_params": {k: v["params"] for k, v in pool["tuned"].items()},
                "outer_test_auc": fold_auc,
                "seconds": time.monotonic() - fold_started,
            })
            log(f"      -> blend inner {pool['inner_blend_auc']:.4f} | "
                f"outer-test AUC {fold_auc:.4f} | {time.monotonic() - fold_started:.0f}s")
        repeats_done = repeat + 1

    if repeats_done == 0:
        raise RuntimeError("No outer repeat completed")
    if np.any(seen == 0):
        raise AssertionError("Every subject must be evaluated at least once")

    prob = prob_sum / seen
    score = score_sum / seen
    fold_aucs = [r["outer_test_auc"] for r in fold_records]
    return {
        "oof_prob": prob,
        "oof_score": score,
        "pooled_oof_roc_auc": safe_auc(y, score),
        "mean_fold_roc_auc": float(np.mean(fold_aucs)),
        "std_fold_roc_auc": float(np.std(fold_aucs)),
        "fold_records": fold_records,
        "repeats_completed": repeats_done,
        "elapsed_seconds": time.monotonic() - started,
    }


def non_nested_reference(X: np.ndarray, y: np.ndarray, kinds: Sequence[str], *, budgets: dict,
                         inner_k: int, screen_repeats: int, final_repeats: int,
                         auc_gate: float = 0.55, seed: int = 20260727,
                         log: Callable[[str], None] = print) -> dict:
    """Deliberately optimistic: tune on everything, report that same OOF.

    Its only purpose is the ``optimism`` diagnostic -- the gap against the nested
    estimate is how much a tuned-and-reported-on-the-same-data number would have
    lied.  Never use this as the headline metric.
    """

    rng = np.random.default_rng(seed + 7)
    pool = tune_pool(X, y, kinds, budgets=budgets, inner_k=inner_k, screen_repeats=screen_repeats,
                     final_repeats=final_repeats, auc_gate=auc_gate, rng=rng, seed=seed, log=log)
    matrix = np.column_stack([pool["tuned"][k]["oof"] for k in pool["eligible"]])
    score = blend_scores(matrix, pool["weights"])
    return {
        "roc_auc": safe_auc(y, score),
        "eligible": list(pool["eligible"]),
        "weights": {k: float(w) for k, w in zip(pool["eligible"], pool["weights"])},
        "auc_by_kind": {k: float(v["auc"]) for k, v in pool["tuned"].items()},
        "best_params": {k: v["params"] for k, v in pool["tuned"].items()},
        "pool": pool,
    }


__all__ = ["binary_metrics", "blend_scores", "bootstrap_ci", "direction_free_auc", "make_splits",
           "nested_cv", "non_nested_reference", "oof_predict", "optimize_blend", "safe_auc",
           "select_features", "select_threshold", "select_threshold_specificity", "tune_kind",
           "tune_pool"]
