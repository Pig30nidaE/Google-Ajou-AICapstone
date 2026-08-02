"""Nested, person-level cross-validation and the frozen final pipeline.

Fold structure
--------------
::

    outer StratifiedKFold over PEOPLE (one row per person)
      |
      |-- outer-training fold ------------------------------------------+
      |     fold-local median / variance filter                         |
      |     YDF importance ranking            (training fold only)      |
      |     forward selection, scored on inner CV of the training fold  |
      |     random search,     scored on inner CV of the training fold  |
      |     inner OOF predictions -> stacking meta-learner + threshold  |
      |     SMOTE on inner-train / outer-train only                     |
      +-----------------------------------------------------------------+
      |
      +-- outer-validation fold: scored once, never fitted on

Because every person contributes exactly one row, a person cannot appear on
both sides of a split -- the failure that inflated the source paper's numbers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .metrics import (
    bootstrap_interval, classification_metrics, jaccard_stability, roc_auc,
    youden_threshold,
)
from .models import ALL_KINDS, SOFT_VOTING_WEIGHTS, GoogleTreeModel, default_params
from .preprocessing import (
    FoldPreprocessor, forward_select, random_search, rank_features, smote_resample,
)

#: Ensemble variants reported side by side.
VARIANTS = ("soft_voting", "stacking")


@dataclass
class PipelineConfig:
    outer_k: int = 5
    inner_k: int = 5
    repeats: int = 3
    max_candidates: int = 40          # report: top-40 SHAP features screened
    max_selected: int = 15            # report: Optimal K = 15
    search_budget: int = 12           # report: 30 Optuna trials (nested here)
    use_search: bool = True
    smote_kind: str = "borderline"    # 'borderline' | 'plain' | 'none'
    kinds: tuple[str, ...] = ALL_KINDS
    seed: int = 20260728
    deadline_seconds: float | None = None
    bootstrap_resamples: int = 2000

    def to_dict(self) -> dict:
        out = asdict(self)
        out["kinds"] = list(self.kinds)
        return out


class DeadlineReached(RuntimeError):
    """Raised when the soft time budget is exhausted at a repeat boundary."""


# ------------------------------------------------------------------ helpers ---
def _resample(X: np.ndarray, y: np.ndarray, config: PipelineConfig, seed: int):
    if config.smote_kind == "none":
        return X, y
    return smote_resample(X, y, kind=config.smote_kind, seed=seed)


def _safe_splits(y: np.ndarray, requested: int) -> int:
    """Never ask for more folds than the minority class can populate."""

    minority = int(min(np.count_nonzero(y == 0), np.count_nonzero(y == 1)))
    return max(2, min(requested, minority))


def _stack_meta(train_matrix: np.ndarray, y: np.ndarray, seed: int):
    """Logistic-regression meta-learner over base-model probabilities.

    Four inputs and an intercept; the combiner is deliberately tiny so the
    ensemble's behaviour stays dominated by the YDF base models.
    """

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    if len(np.unique(y)) < 2:
        return None
    return model.fit(np.nan_to_num(train_matrix, nan=0.5), y)


# ------------------------------------------------------------- fold pipeline --
@dataclass
class FoldArtifacts:
    """Everything a fitted fold needs in order to score new people."""

    preprocessor: FoldPreprocessor
    selected: list[int]
    params: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    meta: object = None
    thresholds: dict = field(default_factory=dict)
    inner_auc: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)


def fit_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: PipelineConfig,
    seed: int,
) -> FoldArtifacts:
    """Fit the complete pipeline on one training fold. Touches no held-out row."""

    pre = FoldPreprocessor().fit(X_train, y_train)
    # SMOTE interpolates, so it needs complete rows; when it is disabled the
    # trees keep raw NaN and use YDF's native missing-value splits instead.
    impute = config.smote_kind != "none"
    X_kept = pre.transform(X_train, impute=impute)

    # -- feature ranking + forward selection (training fold only) -------------
    ranking = rank_features(X_kept, y_train, seed=seed)
    selected, trace = forward_select(
        X_kept, y_train, ranking,
        max_candidates=config.max_candidates, max_selected=config.max_selected,
        inner_splits=_safe_splits(y_train, config.inner_k), seed=seed,
    )
    X_sel = X_kept[:, selected]

    # -- per-learner hyperparameters (training fold only) ---------------------
    params: dict[str, dict] = {}
    inner_auc: dict[str, float] = {}
    for kind in config.kinds:
        if config.use_search:
            found = random_search(
                X_sel, y_train, kind, budget=config.search_budget,
                inner_splits=_safe_splits(y_train, min(3, config.inner_k)), seed=seed,
            )
            params[kind] = found["params"]
            inner_auc[kind] = found["inner_auc"]
        else:
            params[kind] = default_params(kind)

    # -- inner OOF: feeds the stacking meta-learner and the threshold ---------
    splits = _safe_splits(y_train, config.inner_k)
    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    inner_oof = {kind: np.full(len(y_train), np.nan) for kind in config.kinds}
    for inner_train, inner_test in splitter.split(np.zeros(len(y_train)), y_train):
        if len(np.unique(y_train[inner_train])) < 2:
            continue
        X_fit, y_fit = _resample(X_sel[inner_train], y_train[inner_train], config, seed)
        for kind in config.kinds:
            model = GoogleTreeModel(kind, params[kind], seed=seed).fit(X_fit, y_fit)
            inner_oof[kind][inner_test] = model.predict_proba(X_sel[inner_test])

    for kind in config.kinds:
        column = inner_oof[kind]
        mask = ~np.isnan(column)
        inner_auc.setdefault(kind, roc_auc(y_train[mask], column[mask]) if mask.any() else 0.5)

    # -- final base models on the whole training fold -------------------------
    X_fit, y_fit = _resample(X_sel, y_train, config, seed)
    models = {kind: GoogleTreeModel(kind, params[kind], seed=seed).fit(X_fit, y_fit)
              for kind in config.kinds}

    # -- combiners, both calibrated on inner OOF only -------------------------
    inner_matrix = np.column_stack([inner_oof[kind] for kind in config.kinds])
    meta = _stack_meta(inner_matrix, y_train, seed)

    soft_inner = _soft_vote(inner_oof, config.kinds)
    thresholds = {"soft_voting": youden_threshold(y_train, np.nan_to_num(soft_inner, nan=0.5))}
    if meta is not None:
        stack_inner = meta.predict_proba(np.nan_to_num(inner_matrix, nan=0.5))[:, 1]
        thresholds["stacking"] = youden_threshold(y_train, stack_inner)
    else:
        thresholds["stacking"] = thresholds["soft_voting"]

    return FoldArtifacts(preprocessor=pre, selected=selected, params=params,
                         models=models, meta=meta, thresholds=thresholds,
                         inner_auc=inner_auc, trace=trace)


def _soft_vote(scores: dict[str, np.ndarray], kinds) -> np.ndarray:
    """Report's fixed-weight soft voting (.40/.20/.20/.20), renormalised."""

    total = sum(SOFT_VOTING_WEIGHTS[k] for k in kinds)
    stacked = np.zeros_like(next(iter(scores.values())), dtype=np.float64)
    for kind in kinds:
        stacked += (SOFT_VOTING_WEIGHTS[kind] / total) * np.nan_to_num(scores[kind], nan=0.5)
    return stacked


def predict_fold(artifacts: FoldArtifacts, X: np.ndarray, config: PipelineConfig) -> dict:
    """Score new people with a fitted fold. Returns per-model and ensemble scores."""

    impute = config.smote_kind != "none"
    X_kept = artifacts.preprocessor.transform(X, impute=impute)
    X_sel = X_kept[:, artifacts.selected]

    per_model = {kind: model.predict_proba(X_sel) for kind, model in artifacts.models.items()}
    kinds = tuple(artifacts.models)
    out = dict(per_model)
    out["soft_voting"] = _soft_vote(per_model, kinds)

    if artifacts.meta is not None:
        matrix = np.column_stack([per_model[k] for k in kinds])
        out["stacking"] = artifacts.meta.predict_proba(np.nan_to_num(matrix, nan=0.5))[:, 1]
    else:
        out["stacking"] = out["soft_voting"]
    return out


# --------------------------------------------------------------- nested loop --
def nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    config: PipelineConfig,
    feature_names: list[str],
    *,
    verbose: bool = True,
) -> dict:
    """Repeated nested CV. Returns OOF scores, metrics and stability diagnostics."""

    started = time.time()
    n = len(y)
    accumulated = {name: [] for name in (*config.kinds, *VARIANTS)}
    margins = {name: [] for name in VARIANTS}
    selections: list[list[int]] = []
    selected_names: list[list[str]] = []
    fold_records: list[dict] = []
    completed_repeats = 0

    for repeat in range(config.repeats):
        if config.deadline_seconds and (time.time() - started) > config.deadline_seconds:
            if verbose:
                print(f"[train] soft deadline reached; stopping after "
                      f"{completed_repeats} complete repeat(s).")
            break

        seed = config.seed + 1000 * repeat
        splitter = StratifiedKFold(n_splits=_safe_splits(y, config.outer_k),
                                   shuffle=True, random_state=seed)
        repeat_scores = {name: np.full(n, np.nan) for name in (*config.kinds, *VARIANTS)}
        repeat_margins = {name: np.full(n, np.nan) for name in VARIANTS}

        for fold_index, (train_idx, test_idx) in enumerate(
            splitter.split(np.zeros(n), y)
        ):
            fold_seed = seed + fold_index
            artifacts = fit_fold(X[train_idx], y[train_idx], config, fold_seed)
            predictions = predict_fold(artifacts, X[test_idx], config)

            for name, values in predictions.items():
                repeat_scores[name][test_idx] = values
            for variant in VARIANTS:
                # Margin against the fold's own threshold, so folds with
                # different operating points remain comparable when pooled.
                repeat_margins[variant][test_idx] = (
                    predictions[variant] - artifacts.thresholds[variant]
                )

            selections.append(list(artifacts.selected))
            selected_names.append([feature_names[i] for i in artifacts.selected])
            fold_records.append({
                "repeat": repeat, "fold": fold_index,
                "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
                "n_selected": len(artifacts.selected),
                "thresholds": {k: round(float(v), 4) for k, v in artifacts.thresholds.items()},
                "inner_auc": {k: round(float(v), 4) for k, v in artifacts.inner_auc.items()},
            })
            if verbose:
                print(f"[train] repeat {repeat} fold {fold_index}: "
                      f"{len(artifacts.selected)} features, "
                      f"inner AUC {artifacts.inner_auc}")

        for name in repeat_scores:
            accumulated[name].append(repeat_scores[name])
        for variant in VARIANTS:
            margins[variant].append(repeat_margins[variant])
        completed_repeats += 1

    if completed_repeats == 0:
        raise DeadlineReached("No repeat completed within the time budget.")

    # Average the per-repeat OOF scores; every entry is out-of-fold in its repeat.
    oof = {name: np.nanmean(np.vstack(values), axis=0) for name, values in accumulated.items()}
    oof_margin = {name: np.nanmean(np.vstack(values), axis=0) for name, values in margins.items()}

    results = {
        "config": config.to_dict(),
        "completed_repeats": completed_repeats,
        "n_persons": int(n),
        "n_mci_dem": int(np.count_nonzero(y == 1)),
        "folds": fold_records,
        "feature_stability": jaccard_stability(selections),
        "selected_feature_names": selected_names,
        "metrics": {},
        "oof_scores": {name: values.tolist() for name, values in oof.items()},
    }

    for name, values in oof.items():
        threshold = 0.0 if name in VARIANTS else 0.5
        score = oof_margin[name] if name in VARIANTS else values
        metrics = classification_metrics(y, score, threshold)
        metrics["bootstrap"] = bootstrap_interval(
            y, score, threshold, n_resamples=config.bootstrap_resamples, seed=config.seed
        )
        results["metrics"][name] = metrics

    # Most-frequently selected features across all folds, for the report.
    tally: dict[str, int] = {}
    for names in selected_names:
        for name in names:
            tally[name] = tally.get(name, 0) + 1
    results["feature_frequency"] = dict(
        sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))[:40]
    )
    return results


# --------------------------------------------------------- final frozen model --
def fit_final(X: np.ndarray, y: np.ndarray, config: PipelineConfig, feature_names: list[str]):
    """Refit the entire pipeline on all Training people, for the frozen forecast."""

    artifacts = fit_fold(X, y, config, config.seed)
    summary = {
        "n_selected": len(artifacts.selected),
        "selected_features": [feature_names[i] for i in artifacts.selected],
        "thresholds": {k: float(v) for k, v in artifacts.thresholds.items()},
        "inner_auc": {k: float(v) for k, v in artifacts.inner_auc.items()},
        "params": artifacts.params,
    }
    return artifacts, summary
