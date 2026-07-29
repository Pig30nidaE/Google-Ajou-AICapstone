"""Cross-validation engine: fold execution, screening, and nested selection.

Two evaluation modes, reported separately and labelled explicitly
----------------------------------------------------------------
``screen_model`` / ``run_repeated_cv``
    Repeated subject-level CV of **one fixed model with fixed hyperparameters**.
    Nothing is selected inside the loop, so the pooled OOF AUC is an honest
    estimate *for that model*.  What it does not account for is that a human then
    reads the table and picks the best row -- that is arm-selection bias, and it
    is why these numbers are labelled ``non-nested (fixed configuration)`` and
    are not the headline.

``nested_selection_cv``
    Model choice, blend weights and the operating threshold are all decided on
    inner folds of the outer-training block.  The outer-test subjects influence
    nothing, so the pooled OOF AUC absorbs the selection cost and is the headline.

Both use the *same* fold objects, so every comparison is paired on identical
splits (``evaluation_guidance``: 모델 간 비교에는 동일한 split을 사용).

Fold-local scope
----------------
:func:`fold_fit_predict` is the only place a model is fitted.  It re-derives the
preprocessor, the feature subset and the resampling inside every call, and asks
:class:`FoldPreprocessor` to verify -- via the row fingerprint -- that the
transform applied to the test rows was fitted on this fold's training rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable, Sequence

import numpy as np

from .ensemble import Blender, best_blender, rank_normalize, safe_auc
from .evaluation import (
    metrics_from_predictions,
    subject_metrics,
    summarize_repeats,
    youden_threshold,
)
from .models import Model, ModelSpec, build_model
from .preprocessing import FoldPreprocessor, resample, rows_fingerprint, select_features
from .splits import Fold, assert_both_classes, assert_no_subject_overlap, inner_folds


class RuntimeBudgetExceeded(TimeoutError):
    """Raised when a run would pass its declared wall-clock limit."""


@dataclass
class Budget:
    deadline: float | None = None

    def check(self, where: str) -> None:
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise RuntimeBudgetExceeded(f"Runtime budget exhausted during {where}")


@dataclass
class FoldOutcome:
    scores: np.ndarray
    n_features_used: int
    selected: np.ndarray
    model: Model | None = None


def fold_fit_predict(X: np.ndarray, y: np.ndarray, train_index: np.ndarray,
                     test_index: np.ndarray, spec: ModelSpec, *, seed: int,
                     resampler: str = "class_weight", top_k: int = 0,
                     corr_threshold: float = 0.95, keep_model: bool = False,
                     row_map: np.ndarray | None = None) -> FoldOutcome:
    """Fit one model on one fold's training rows and score its test rows.

    Order matters and is fixed here: split -> preprocess (fit on train) ->
    select features (fit on train) -> resample (train only) -> fit -> score.
    Nothing downstream of the split ever sees ``test_index`` rows except the
    final ``transform`` + ``score_samples`` pair.
    """

    assert_no_subject_overlap([f"i{i}" for i in train_index], [f"i{i}" for i in test_index])
    y_train = np.asarray(y)[train_index]
    assert_both_classes(y_train, where="fold_fit_predict train")

    fingerprint = rows_fingerprint(train_index)
    preprocessor = FoldPreprocessor().fit(X[train_index], train_index)
    X_train = preprocessor.transform(X[train_index], expect_fingerprint=fingerprint)
    X_test = preprocessor.transform(X[test_index], expect_fingerprint=fingerprint)

    selected = select_features(X_train, y_train, top_k=top_k, corr_threshold=corr_threshold)
    X_train, X_test = X_train[:, selected], X_test[:, selected]

    # ``row_map`` translates local positions into cohort rows, so a model that
    # needs external per-subject data (the sequence arm) addresses the right
    # subjects even when it runs on an inner split of an outer-training block.
    def _cohort_rows(index: np.ndarray) -> np.ndarray:
        index = np.asarray(index, dtype=np.int64)
        return index if row_map is None else np.asarray(row_map, dtype=np.int64)[index]

    X_fit, y_fit, rows_fit = _resample_with_rows(
        X_train, y_train, _cohort_rows(train_index), resampler, seed
    )
    model = spec.build(seed)
    model.fit(X_fit, y_fit, rows=rows_fit)
    scores = model.score_samples(X_test, rows=_cohort_rows(test_index))
    return FoldOutcome(scores=scores, n_features_used=int(selected.size), selected=selected,
                       model=model if keep_model else None)


def _resample_with_rows(X: np.ndarray, y: np.ndarray, rows: np.ndarray, kind: str,
                        seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Resample training rows, tracking which cohort row each sample came from.

    Synthetic samplers invent rows that correspond to no subject, so the row map
    is dropped in that case -- and the sequence arm, which needs it, therefore
    only runs under ``none``/``class_weight``/``random_over``.
    """

    if kind in ("none", "class_weight"):
        return X, y, np.asarray(rows, dtype=np.int64)
    X_resampled, y_resampled = resample(X, y, kind, seed=seed)
    if X_resampled.shape[0] == X.shape[0] and np.array_equal(y_resampled, y):
        return X_resampled, y_resampled, np.asarray(rows, dtype=np.int64)
    return X_resampled, y_resampled, None


def inner_oof_scores(X: np.ndarray, y: np.ndarray, spec: ModelSpec, *, inner_k: int,
                     seed: int, resampler: str, top_k: int, corr_threshold: float,
                     row_map: np.ndarray | None = None) -> np.ndarray:
    """Out-of-fold scores *within* an outer-training block.

    Returned scores are rank-normalised per inner fold before being pooled: the
    inner folds are separate fits whose raw score scales are not comparable, and
    pooling raw scores across them would corrupt the very ordering the blender is
    about to be fitted on.
    """

    scores = np.full(len(y), np.nan, dtype=np.float64)
    for train_index, test_index in inner_folds(y, n_splits=inner_k, seed=seed):
        outcome = fold_fit_predict(X, y, train_index, test_index, spec, seed=seed,
                                   resampler=resampler, top_k=top_k,
                                   corr_threshold=corr_threshold, row_map=row_map)
        scores[test_index] = rank_normalize(outcome.scores)
    if not np.isfinite(scores).all():  # pragma: no cover - inner_folds partitions
        scores = np.nan_to_num(scores, nan=float(np.nanmedian(scores)))
    return scores


# --------------------------------------------------------------- screening ---
@dataclass
class CVResult:
    label: str
    model: str
    block: str
    resampler: str
    params: dict
    oof_by_repeat: dict[int, np.ndarray]
    per_repeat_auc: list[float]
    per_fold_auc: list[dict]
    mean_features: float
    elapsed_seconds: float
    mode: str = "non-nested (fixed configuration)"

    @property
    def summary(self) -> dict:
        return summarize_repeats(self.per_repeat_auc)

    def mean_oof(self) -> np.ndarray:
        """Rank-average of the per-repeat OOF vectors.

        Each repeat is a full partition, so averaging their rank-normalised OOF
        scores gives one score per subject with the split noise reduced -- the
        vector used for bootstrap CIs and for paired comparisons.
        """

        stacked = np.column_stack([rank_normalize(v) for v in self.oof_by_repeat.values()])
        return stacked.mean(axis=1)


def run_repeated_cv(X: np.ndarray, y: np.ndarray, subject_ids: Sequence[str],
                    folds: Sequence[Fold], spec: ModelSpec, *, block: str,
                    resampler: str = "class_weight", top_k: int = 0,
                    corr_threshold: float = 0.95, seed: int = 0,
                    budget: Budget | None = None,
                    on_error: str = "raise") -> CVResult:
    """Repeated CV for one fixed configuration, on a fixed set of folds."""

    y = np.asarray(y, dtype=np.int64)
    subject_ids = np.asarray(subject_ids, dtype=str)
    budget = budget or Budget()
    started = time.monotonic()

    oof_by_repeat: dict[int, np.ndarray] = {}
    per_fold: list[dict] = []
    feature_counts: list[int] = []

    for fold in folds:
        budget.check(f"{spec.name}/{block}")
        assert_no_subject_overlap(subject_ids[fold.train_index], subject_ids[fold.test_index])
        try:
            outcome = fold_fit_predict(X, y, fold.train_index, fold.test_index, spec,
                                       seed=seed + fold.repeat, resampler=resampler,
                                       top_k=top_k, corr_threshold=corr_threshold)
        except RuntimeBudgetExceeded:
            raise
        except Exception as error:
            if on_error == "raise":
                raise
            # A learner that cannot fit this fold is recorded, not fatal: one
            # broken optional dependency should not end a six-hour sweep.
            per_fold.append({"fold": fold.key, "error": f"{type(error).__name__}: {error}"})
            continue
        store = oof_by_repeat.setdefault(fold.repeat, np.full(len(y), np.nan))
        store[fold.test_index] = rank_normalize(outcome.scores)
        feature_counts.append(outcome.n_features_used)
        per_fold.append(
            {
                "fold": fold.key,
                "repeat": fold.repeat,
                "n_test": int(fold.test_index.size),
                "n_test_positive": int(y[fold.test_index].sum()),
                "roc_auc": safe_auc(y[fold.test_index], outcome.scores),
                "n_features": outcome.n_features_used,
            }
        )

    complete = {r: v for r, v in oof_by_repeat.items() if np.isfinite(v).all()}
    per_repeat = [safe_auc(y, v) for v in complete.values()]
    return CVResult(
        label=f"{spec.name}|{block}|{resampler}",
        model=spec.name,
        block=block,
        resampler=resampler,
        params=dict(spec.params),
        oof_by_repeat=complete,
        per_repeat_auc=per_repeat,
        per_fold_auc=per_fold,
        mean_features=float(np.mean(feature_counts)) if feature_counts else 0.0,
        elapsed_seconds=float(time.monotonic() - started),
    )


# -------------------------------------------------------- nested selection ---
@dataclass
class NestedResult:
    label: str
    block: str
    candidates: list[str]
    oof_by_repeat: dict[int, np.ndarray]
    per_repeat_auc: list[float]
    per_fold_auc: list[dict]
    predictions_by_repeat: dict[int, np.ndarray] = field(default_factory=dict)
    chosen_combiner: dict[str, int] = field(default_factory=dict)
    chosen_members: dict[str, int] = field(default_factory=dict)
    thresholds: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    mode: str = "nested (model, weights and threshold chosen inside each outer fold)"

    @property
    def summary(self) -> dict:
        return summarize_repeats(self.per_repeat_auc)

    def mean_oof(self) -> np.ndarray:
        stacked = np.column_stack([rank_normalize(v) for v in self.oof_by_repeat.values()])
        return stacked.mean(axis=1)


def nested_selection_cv(X: np.ndarray, y: np.ndarray, subject_ids: Sequence[str],
                        folds: Sequence[Fold],
                        specs: Sequence[ModelSpec | Sequence[ModelSpec]], *, block: str,
                        inner_k: int = 4, resampler: str = "class_weight", top_k: int = 0,
                        corr_threshold: float = 0.95, seed: int = 0,
                        budget: Budget | None = None,
                        log: Callable[[str], None] = lambda _msg: None) -> NestedResult:
    """Nested CV where the *whole* selection procedure runs inside each fold.

    Per outer fold:
      1. build inner OOF scores for every candidate on the outer-training block
      2. choose the combination rule that maximises inner OOF AUC
      3. choose the operating threshold on the blended inner OOF scores
      4. refit every candidate on the full outer-training block
      5. blend their outer-test scores with the rule chosen in step 2

    Steps 1-3 never touch an outer-test row, so the pooled OOF is not optimistic
    about the selection itself.  It is still an OOF estimate on the same 174
    subjects, not an external validation, and is reported as such.
    """

    y = np.asarray(y, dtype=np.int64)
    subject_ids = np.asarray(subject_ids, dtype=str)
    budget = budget or Budget()
    started = time.monotonic()

    oof_by_repeat: dict[int, np.ndarray] = {}
    predictions_by_repeat: dict[int, np.ndarray] = {}
    per_fold: list[dict] = []
    combiner_counts: dict[str, int] = {}
    member_counts: dict[str, int] = {}
    thresholds: list[float] = []

    for fold in folds:
        budget.check(f"nested/{block}")
        assert_no_subject_overlap(subject_ids[fold.train_index], subject_ids[fold.test_index])
        X_train_block = X[fold.train_index]
        y_train_block = y[fold.train_index]
        assert_both_classes(y_train_block, where=f"{fold.key} outer-train")
        fold_seed = seed + fold.repeat

        inner_matrix: list[np.ndarray] = []
        usable: list[ModelSpec] = []
        for candidate in specs:
            variants = [candidate] if isinstance(candidate, ModelSpec) else list(candidate)
            if not variants:
                continue
            budget.check(f"nested/{block}/{variants[0].name}")

            def _inner(spec: ModelSpec, _rows=fold.train_index) -> np.ndarray:
                return inner_oof_scores(X_train_block, y_train_block, spec, inner_k=inner_k,
                                        seed=fold_seed, resampler=resampler, top_k=top_k,
                                        corr_threshold=corr_threshold, row_map=_rows)

            try:
                if len(variants) == 1:
                    chosen, scores = variants[0], _inner(variants[0])
                else:
                    # In-fold hyperparameter choice: the grid is searched on inner
                    # OOF scores of the outer-training block only.
                    from .tuning import select_spec_inner

                    chosen, scores, _ = select_spec_inner(variants, _inner, y_train_block)
                inner_matrix.append(scores)
                usable.append(chosen)
            except Exception as error:
                log(f"{fold.key}: candidate {variants[0].name} unusable "
                    f"({type(error).__name__}: {error})")
        if not usable:
            raise RuntimeError(f"{fold.key}: no candidate model could be fitted")

        inner_scores = np.column_stack(inner_matrix)
        blender = best_blender(inner_scores, y_train_block, seed=fold_seed)
        combiner_counts[blender.kind] = combiner_counts.get(blender.kind, 0) + 1
        for position in _blender_members(blender, len(usable)):
            member_counts[usable[position].name] = member_counts.get(usable[position].name, 0) + 1
        threshold = youden_threshold(y_train_block, blender.apply(inner_scores))
        thresholds.append(float(threshold))

        outer_matrix: list[np.ndarray] = []
        for spec in usable:
            outcome = fold_fit_predict(X, y, fold.train_index, fold.test_index, spec,
                                       seed=fold_seed, resampler=resampler, top_k=top_k,
                                       corr_threshold=corr_threshold)
            outer_matrix.append(rank_normalize(outcome.scores))
        blended = blender.apply(np.column_stack(outer_matrix))

        store = oof_by_repeat.setdefault(fold.repeat, np.full(len(y), np.nan))
        store[fold.test_index] = rank_normalize(blended)
        # The decision is made here, with this fold's own inner-derived
        # threshold, and stored.  Re-thresholding pooled scores later would
        # silently replace the nested operating point with a global one.
        decisions = predictions_by_repeat.setdefault(fold.repeat, np.full(len(y), -1, dtype=np.int64))
        decisions[fold.test_index] = (blended >= threshold).astype(np.int64)
        per_fold.append(
            {
                "fold": fold.key,
                "repeat": fold.repeat,
                "n_test": int(fold.test_index.size),
                "n_test_positive": int(y[fold.test_index].sum()),
                "roc_auc": safe_auc(y[fold.test_index], blended),
                "combiner": blender.kind,
                "inner_auc": float(blender.inner_auc),
                "threshold": float(threshold),
                "candidates": [s.name for s in usable],
            }
        )

    complete = {r: v for r, v in oof_by_repeat.items() if np.isfinite(v).all()}
    return NestedResult(
        label=f"nested|{block}",
        block=block,
        candidates=[
            (candidate.name if isinstance(candidate, ModelSpec) else list(candidate)[0].name)
            for candidate in specs
        ],
        oof_by_repeat=complete,
        per_repeat_auc=[safe_auc(y, v) for v in complete.values()],
        per_fold_auc=per_fold,
        predictions_by_repeat={r: v for r, v in predictions_by_repeat.items()
                               if r in complete and (v >= 0).all()},
        chosen_combiner=combiner_counts,
        chosen_members=member_counts,
        thresholds=thresholds,
        elapsed_seconds=float(time.monotonic() - started),
    )


def _blender_members(blender: Blender, n_models: int) -> list[int]:
    if blender.kind == "greedy":
        return sorted(set(blender.members))
    if blender.kind == "rank_weighted" and blender.weights is not None:
        return [int(i) for i in np.flatnonzero(blender.weights > 1e-3)]
    return list(range(n_models))


def nested_threshold_metrics(y: np.ndarray, result: NestedResult) -> dict:
    """Secondary metrics at the *nested* operating point.

    Each fold applied its own inner-derived threshold to its own test rows and
    the resulting decisions were stored.  This pools those decisions per repeat,
    scores each repeat, and reports the mean and SD across repeats -- so the
    reported recall/F1/MCC/specificity really are out-of-fold decisions from a
    threshold that never saw the row it was applied to.

    ROC-AUC is threshold-free and is unaffected by any of this; it remains the
    headline, and a better operating point never improves it.
    """

    y = np.asarray(y, dtype=np.int64)
    out: dict = {
        "roc_auc": safe_auc(y, result.mean_oof()),
        "pr_auc": float(subject_metrics(y, result.mean_oof())["pr_auc"]),
        "pr_auc_baseline": float(y.mean()),
        "threshold_source": "Youden's J on the blended inner OOF of each outer fold",
        "n_repeats_scored": 0,
    }
    if not result.predictions_by_repeat:
        return out

    per_repeat = [metrics_from_predictions(y, predictions)
                  for predictions in result.predictions_by_repeat.values()]
    scalar_keys = ("dem_recall", "specificity", "precision", "f1", "balanced_accuracy",
                   "accuracy", "mcc")
    for key in scalar_keys:
        values = np.array([m[key] for m in per_repeat], dtype=np.float64)
        out[key] = float(values.mean())
        out[f"{key}_std"] = float(values.std(ddof=0))
    out["mean_confusion"] = {
        part: float(np.mean([m["confusion"][part] for m in per_repeat]))
        for part in ("tn", "fp", "fn", "tp")
    }
    out["n_repeats_scored"] = len(per_repeat)
    return out


__all__ = [
    "Budget", "CVResult", "FoldOutcome", "NestedResult", "RuntimeBudgetExceeded",
    "fold_fit_predict", "inner_oof_scores", "nested_selection_cv",
    "nested_threshold_metrics", "run_repeated_cv",
]
