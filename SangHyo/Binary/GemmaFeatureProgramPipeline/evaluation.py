"""Leakage-resistant nested evaluation for the Gemma feature program.

All six arms use the same repeated, subject-level outer folds.  Diagnosis
(``CN``/``MCI``/``Dem``), rather than the binary target alone, is the
stratification variable so every fold preserves the clinically important
three-group composition.  Convex fusion weights are selected from inner OOF
predictions only; the outer-test rows participate solely in final prediction.

The reported 0.5 operating point is fixed on the inductive training-CDF scale.
No threshold is fitted, and no prediction vector is rank-normalised within an
inner-validation or outer-test batch.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .modeling import FittedBaseBlocks

__all__ = [
    "ARMS",
    "EvaluationResult",
    "binary_metrics",
    "bootstrap_auc",
    "paired_bootstrap_auc_difference",
    "evaluate_nested",
]


ARMS: tuple[str, ...] = (
    "mmse_only",
    "wearable_only",
    "program_only",
    "wearable_plus_program",
    "mmse_plus_wearable",
    "full",
)

PROGRAM_BETAS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)
WEARABLE_ALPHAS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)
FULL_AUX_VALUES: tuple[float, ...] = (0.0, 0.1, 0.2)
SELECTION_TOLERANCE = 0.005
FIXED_THRESHOLD = 0.5


def _json_safe(value: Any) -> Any:
    """Recursively convert NumPy/pandas values to strict JSON-compatible data."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


@dataclass
class EvaluationResult:
    """Evaluation artifacts ready for JSON/CSV serialization by the pipeline."""

    report: dict[str, Any]
    oof: pd.DataFrame
    split_registry: dict[str, Any]

    def to_serializable(self) -> dict[str, Any]:
        return _json_safe(
            {
                "report": self.report,
                "oof_records": self.oof.to_dict(orient="records"),
                "split_registry": self.split_registry,
            }
        )


def _confusion(y: np.ndarray, predicted: np.ndarray) -> dict[str, int]:
    return {
        "tn": int(np.sum((y == 0) & (predicted == 0))),
        "fp": int(np.sum((y == 0) & (predicted == 1))),
        "fn": int(np.sum((y == 1) & (predicted == 0))),
        "tp": int(np.sum((y == 1) & (predicted == 1))),
    }


def binary_metrics(
    y_true: Sequence[int],
    scores: Sequence[float],
    *,
    threshold: float = FIXED_THRESHOLD,
) -> dict[str, Any]:
    """Ranking and fixed-operating-point metrics for subject-level scores."""

    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.shape != score.shape:
        raise ValueError("y_true and scores must have the same shape")
    if not np.isfinite(score).all():
        raise ValueError("binary_metrics received non-finite scores")
    classes = np.unique(y)
    if not set(classes.tolist()).issubset({0, 1}):
        raise ValueError("binary_metrics expects binary labels 0/1")

    predicted = (score >= float(threshold)).astype(np.int64)
    confusion = _confusion(y, predicted)
    positives = confusion["tp"] + confusion["fn"]
    negatives = confusion["tn"] + confusion["fp"]
    recall = confusion["tp"] / positives if positives else None
    specificity = confusion["tn"] / negatives if negatives else None
    precision_denominator = confusion["tp"] + confusion["fp"]
    precision = (
        confusion["tp"] / precision_denominator if precision_denominator else None
    )
    if recall is not None and precision is not None and recall + precision > 0:
        f1 = 2.0 * recall * precision / (recall + precision)
    else:
        f1 = 0.0
    balanced = (
        0.5 * (recall + specificity)
        if recall is not None and specificity is not None
        else None
    )

    ranking_defined = len(classes) == 2
    mcc = (
        float(matthews_corrcoef(y, predicted))
        if y.size and len(np.unique(predicted)) > 1 and ranking_defined
        else 0.0
    )
    return _json_safe(
        {
            "n": int(y.size),
            "n_positive": int((y == 1).sum()),
            "n_negative": int((y == 0).sum()),
            "roc_auc": float(roc_auc_score(y, score)) if ranking_defined else None,
            "pr_auc": (
                float(average_precision_score(y, score)) if ranking_defined else None
            ),
            "pr_auc_prevalence": float(y.mean()) if y.size else None,
            "threshold": float(threshold),
            "recall_sensitivity": recall,
            "specificity": specificity,
            "precision": precision,
            "f1": f1,
            "balanced_accuracy": balanced,
            "mcc": mcc,
            "accuracy": float(np.mean(predicted == y)) if y.size else None,
            "all_negative_baseline_accuracy": (
                float(np.mean(y == 0)) if y.size else None
            ),
            "confusion": confusion,
        }
    )


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def bootstrap_auc(
    y_true: Sequence[int],
    scores: Sequence[float],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Percentile ROC-AUC interval from subject-level bootstrap resamples."""

    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.shape != score.shape:
        raise ValueError("bootstrap_auc inputs must have matching shapes")
    if not np.isfinite(score).all():
        raise ValueError("bootstrap_auc received non-finite scores")
    observed = _safe_auc(y, score)
    if int(n_bootstrap) <= 0:
        return {
            "roc_auc": float(observed),
            "ci95": None,
            "n_requested": int(n_bootstrap),
            "n_valid": 0,
            "resampling_unit": "subject",
        }

    rng = np.random.default_rng(int(seed))
    values: list[float] = []
    for _ in range(int(n_bootstrap)):
        index = rng.integers(0, y.size, size=y.size)
        if len(np.unique(y[index])) < 2:
            continue
        values.append(_safe_auc(y[index], score[index]))
    if not values:
        interval = None
    else:
        low, high = np.percentile(np.asarray(values), [2.5, 97.5])
        interval = [float(low), float(high)]
    return _json_safe(
        {
            "roc_auc": float(observed),
            "ci95": interval,
            "n_requested": int(n_bootstrap),
            "n_valid": len(values),
            "resampling_unit": "subject",
        }
    )


def paired_bootstrap_auc_difference(
    y_true: Sequence[int],
    first_scores: Sequence[float],
    second_scores: Sequence[float],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Paired percentile bootstrap over subjects for ``AUC(first)-AUC(second)``."""

    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    first = np.asarray(first_scores, dtype=np.float64).reshape(-1)
    second = np.asarray(second_scores, dtype=np.float64).reshape(-1)
    if y.shape != first.shape or y.shape != second.shape:
        raise ValueError("paired bootstrap inputs must have matching shapes")
    observed = _safe_auc(y, first) - _safe_auc(y, second)
    if int(n_bootstrap) <= 0:
        return {
            "observed_auc_difference": float(observed),
            "ci95": None,
            "n_requested": int(n_bootstrap),
            "n_valid": 0,
        }

    rng = np.random.default_rng(int(seed))
    values: list[float] = []
    for _ in range(int(n_bootstrap)):
        index = rng.integers(0, y.size, size=y.size)
        if len(np.unique(y[index])) < 2:
            continue
        values.append(
            _safe_auc(y[index], first[index]) - _safe_auc(y[index], second[index])
        )
    if not values:
        interval = None
    else:
        low, high = np.percentile(np.asarray(values), [2.5, 97.5])
        interval = [float(low), float(high)]
    return _json_safe(
        {
            "observed_auc_difference": float(observed),
            "ci95": interval,
            "n_requested": int(n_bootstrap),
            "n_valid": len(values),
            "resampling_unit": "subject",
            "paired": True,
        }
    )


def _diagnosis_counts(values: Sequence[str]) -> dict[str, int]:
    array = np.asarray(values, dtype=str)
    return {label: int((array == label).sum()) for label in ("CN", "MCI", "Dem")}


def _validate_data(data: Any, *, outer_folds: int, inner_folds: int) -> None:
    subject_ids = np.asarray(data.subject_ids, dtype=str)
    diagnosis = np.asarray(data.diagnosis, dtype=str)
    y = np.asarray(data.y, dtype=np.int64)
    n_subjects = subject_ids.size
    if diagnosis.shape != (n_subjects,) or y.shape != (n_subjects,):
        raise ValueError("subject_ids, diagnosis and y must be aligned one-dimensional arrays")
    if len(set(subject_ids.tolist())) != n_subjects:
        raise ValueError("subject_ids contains duplicates; subject-level CV is impossible")
    unknown = sorted(set(diagnosis.tolist()) - {"CN", "MCI", "Dem"})
    if unknown:
        raise ValueError(f"Unexpected diagnosis value(s): {unknown}")
    expected_y = (diagnosis != "CN").astype(np.int64)
    if not np.array_equal(y, expected_y):
        raise ValueError("Label contract changed: expected CN=0 and MCI/Dem=1")
    if not isinstance(data.mmse, pd.DataFrame) or not isinstance(data.wearable, pd.DataFrame):
        raise TypeError("data.mmse and data.wearable must be pandas DataFrames")
    if len(data.mmse) != n_subjects or len(data.wearable) != n_subjects:
        raise ValueError("Feature frames are not aligned with subject_ids")
    outer_counts = _diagnosis_counts(diagnosis)
    too_small = {key: value for key, value in outer_counts.items() if value < outer_folds}
    if too_small:
        raise ValueError(
            f"Three-stratum outer {outer_folds}-fold CV is impossible: {too_small}"
        )
    # An outer training set contains at least count-floor(count/k) members of
    # each stratum.  Fail before fitting if a requested inner plan cannot work.
    remaining_min = {
        label: count - int(np.ceil(count / outer_folds))
        for label, count in outer_counts.items()
    }
    inner_too_small = {
        key: value for key, value in remaining_min.items() if value < inner_folds
    }
    if inner_too_small:
        raise ValueError(
            f"Three-stratum inner {inner_folds}-fold CV is impossible: {inner_too_small}"
        )


def _candidate_auc(y: np.ndarray, score: np.ndarray) -> float:
    value = _safe_auc(y, score)
    return value if np.isfinite(value) else -np.inf


def _select_program_beta(
    y: np.ndarray,
    wearable: np.ndarray,
    program: np.ndarray,
) -> tuple[float, dict[str, float]]:
    aucs = {
        f"{beta:.1f}": _candidate_auc(
            y, (1.0 - beta) * wearable + beta * program
        )
        for beta in PROGRAM_BETAS
    }
    best = max(aucs.values())
    eligible = [
        beta
        for beta in PROGRAM_BETAS
        if aucs[f"{beta:.1f}"] >= best - SELECTION_TOLERANCE
    ]
    return float(min(eligible)), aucs


def _select_wearable_alpha(
    y: np.ndarray,
    mmse: np.ndarray,
    wearable: np.ndarray,
) -> tuple[float, dict[str, float]]:
    aucs = {
        f"{alpha:.1f}": _candidate_auc(
            y, (1.0 - alpha) * mmse + alpha * wearable
        )
        for alpha in WEARABLE_ALPHAS
    }
    best = max(aucs.values())
    eligible = [
        alpha
        for alpha in WEARABLE_ALPHAS
        if aucs[f"{alpha:.1f}"] >= best - SELECTION_TOLERANCE
    ]
    return float(min(eligible)), aucs


def _full_candidates() -> tuple[tuple[float, float], ...]:
    values: list[tuple[float, float]] = []
    for wearable_weight in FULL_AUX_VALUES:
        for program_weight in FULL_AUX_VALUES:
            if wearable_weight + program_weight <= 0.3 + 1e-12:
                values.append((float(wearable_weight), float(program_weight)))
    return tuple(values)


def _full_tie_key(weight: tuple[float, float]) -> tuple[float, float, float]:
    wearable_weight, program_weight = weight
    # First suppress the less certain LLM-program contribution; then minimise
    # total deviation from the MMSE baseline.
    return (
        float(program_weight),
        float(wearable_weight + program_weight),
        float(wearable_weight),
    )


def _select_full_weights(
    y: np.ndarray,
    mmse: np.ndarray,
    wearable: np.ndarray,
    program: np.ndarray,
) -> tuple[tuple[float, float], dict[str, float]]:
    aucs: dict[str, float] = {}
    for wearable_weight, program_weight in _full_candidates():
        mmse_weight = 1.0 - wearable_weight - program_weight
        score = (
            mmse_weight * mmse
            + wearable_weight * wearable
            + program_weight * program
        )
        key = f"wearable={wearable_weight:.1f},program={program_weight:.1f}"
        aucs[key] = _candidate_auc(y, score)
    best = max(aucs.values())
    eligible = [
        weight
        for weight in _full_candidates()
        if aucs[f"wearable={weight[0]:.1f},program={weight[1]:.1f}"]
        >= best - SELECTION_TOLERANCE
    ]
    return min(eligible, key=_full_tie_key), aucs


def _scores_for_arms(
    base: Mapping[str, np.ndarray],
    *,
    program_beta: float,
    wearable_alpha: float,
    full_weight: tuple[float, float],
) -> dict[str, np.ndarray]:
    mmse = np.asarray(base["mmse"], dtype=np.float64)
    wearable = np.asarray(base["wearable"], dtype=np.float64)
    program = np.asarray(base["program"], dtype=np.float64)
    wearable_weight, program_weight = full_weight
    return {
        "mmse_only": mmse,
        "wearable_only": wearable,
        "program_only": program,
        "wearable_plus_program": (
            (1.0 - program_beta) * wearable + program_beta * program
        ),
        "mmse_plus_wearable": (
            (1.0 - wearable_alpha) * mmse + wearable_alpha * wearable
        ),
        "full": (
            (1.0 - wearable_weight - program_weight) * mmse
            + wearable_weight * wearable
            + program_weight * program
        ),
    }


def _program_hash(program: Mapping[str, Any]) -> str:
    from .program_schema import canonical_json

    encoded = canonical_json(program)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _frequency_map(values: Sequence[Any], formatter: Callable[[Any], str]) -> dict[str, int]:
    counts = Counter(values)
    return {
        formatter(key): int(value)
        for key, value in sorted(counts.items(), key=lambda item: formatter(item[0]))
    }


def _modal_full_weight(weights: Sequence[tuple[float, float]]) -> tuple[float, float]:
    counts = Counter(weights)
    highest = max(counts.values())
    candidates = [weight for weight, count in counts.items() if count == highest]
    return min(candidates, key=_full_tie_key)


def _repeat_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return _json_safe(
        {
            "values": array.tolist(),
            "mean": float(array.mean()),
            "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            "min": float(array.min()),
            "max": float(array.max()),
        }
    )


def evaluate_nested(
    data: Any,
    program: Mapping[str, Any],
    *,
    outer_folds: int = 5,
    inner_folds: int = 4,
    repeats: int = 20,
    seed: int = 20260729,
    n_bootstrap: int = 4000,
    logger: Callable[[str], None] | None = None,
) -> EvaluationResult:
    """Evaluate the fixed six-arm design with nested, three-stratum CV.

    The repeat-wise mean AUC of ``full`` is the primary estimate.  The AUC of
    scores averaged subject-wise across repeats is secondary because the latter
    benefits from repeated-model averaging.  Neither is an external-test result.
    """

    outer_folds = int(outer_folds)
    inner_folds = int(inner_folds)
    repeats = int(repeats)
    if outer_folds < 2 or inner_folds < 2 or repeats < 1:
        raise ValueError("outer_folds/inner_folds must be >=2 and repeats must be >=1")
    _validate_data(data, outer_folds=outer_folds, inner_folds=inner_folds)
    log = logger if logger is not None else (lambda _message: None)

    subject_ids = np.asarray(data.subject_ids, dtype=str)
    diagnosis = np.asarray(data.diagnosis, dtype=str)
    y = np.asarray(data.y, dtype=np.int64)
    n_subjects = y.size
    fingerprint = str(getattr(data, "fingerprint", ""))

    predictions = {
        arm: np.full((repeats, n_subjects), np.nan, dtype=np.float64)
        for arm in ARMS
    }
    split_entries: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    selected_betas: list[float] = []
    selected_alphas: list[float] = []
    selected_full: list[tuple[float, float]] = []

    for repeat in range(repeats):
        outer_seed = int(seed) + repeat
        outer_splitter = StratifiedKFold(
            n_splits=outer_folds, shuffle=True, random_state=outer_seed
        )
        for fold, (outer_train, outer_test) in enumerate(
            outer_splitter.split(np.zeros(n_subjects), diagnosis)
        ):
            outer_train = np.asarray(outer_train, dtype=np.int64)
            outer_test = np.asarray(outer_test, dtype=np.int64)
            if np.intersect1d(outer_train, outer_test).size:
                raise AssertionError("Outer train/test subject overlap")
            split_id = f"r{repeat:02d}f{fold:02d}"
            inner_seed = int(seed) + 10_000 + repeat * 101 + fold
            train_diagnosis = diagnosis[outer_train]
            inner_splitter = StratifiedKFold(
                n_splits=inner_folds, shuffle=True, random_state=inner_seed
            )
            inner_base = {
                name: np.full(outer_train.size, np.nan, dtype=np.float64)
                for name in ("mmse", "wearable", "program")
            }
            inner_entries: list[dict[str, Any]] = []

            for inner_fold, (inner_fit_local, inner_validation_local) in enumerate(
                inner_splitter.split(np.zeros(outer_train.size), train_diagnosis)
            ):
                inner_fit_local = np.asarray(inner_fit_local, dtype=np.int64)
                inner_validation_local = np.asarray(
                    inner_validation_local, dtype=np.int64
                )
                inner_fit_global = outer_train[inner_fit_local]
                inner_validation_global = outer_train[inner_validation_local]
                if np.intersect1d(inner_fit_global, inner_validation_global).size:
                    raise AssertionError("Inner train/validation subject overlap")

                block_seed = inner_seed + inner_fold * 1_003
                fitted_inner = FittedBaseBlocks.fit(
                    data.mmse.iloc[inner_fit_global],
                    data.wearable.iloc[inner_fit_global],
                    y[inner_fit_global],
                    program,
                    seed=block_seed,
                )
                inner_prediction = fitted_inner.predict_base(
                    data.mmse.iloc[inner_validation_global],
                    data.wearable.iloc[inner_validation_global],
                )
                for name in inner_base:
                    inner_base[name][inner_validation_local] = inner_prediction[name]
                inner_entries.append(
                    {
                        "inner_fold": int(inner_fold),
                        "fit_indices": inner_fit_global.tolist(),
                        "validation_indices": inner_validation_global.tolist(),
                        "fit_diagnosis_counts": _diagnosis_counts(
                            diagnosis[inner_fit_global]
                        ),
                        "validation_diagnosis_counts": _diagnosis_counts(
                            diagnosis[inner_validation_global]
                        ),
                    }
                )

            for name, score in inner_base.items():
                if not np.isfinite(score).all():
                    missing = int((~np.isfinite(score)).sum())
                    raise AssertionError(
                        f"{split_id}: inner OOF block {name} left {missing} rows unscored"
                    )

            inner_y = y[outer_train]
            program_beta, beta_aucs = _select_program_beta(
                inner_y, inner_base["wearable"], inner_base["program"]
            )
            wearable_alpha, alpha_aucs = _select_wearable_alpha(
                inner_y, inner_base["mmse"], inner_base["wearable"]
            )
            full_weight, full_aucs = _select_full_weights(
                inner_y,
                inner_base["mmse"],
                inner_base["wearable"],
                inner_base["program"],
            )
            selected_betas.append(program_beta)
            selected_alphas.append(wearable_alpha)
            selected_full.append(full_weight)

            outer_model_seed = int(seed) + 1_000_000 + repeat * 101 + fold
            fitted_outer = FittedBaseBlocks.fit(
                data.mmse.iloc[outer_train],
                data.wearable.iloc[outer_train],
                y[outer_train],
                program,
                seed=outer_model_seed,
            )
            outer_base = fitted_outer.predict_base(
                data.mmse.iloc[outer_test],
                data.wearable.iloc[outer_test],
            )
            arm_scores = _scores_for_arms(
                outer_base,
                program_beta=program_beta,
                wearable_alpha=wearable_alpha,
                full_weight=full_weight,
            )
            for arm in ARMS:
                predictions[arm][repeat, outer_test] = arm_scores[arm]

            selection_record = {
                "split_id": split_id,
                "repeat": int(repeat),
                "fold": int(fold),
                "wearable_plus_program": {
                    "program_beta": float(program_beta),
                    "candidate_inner_auc": beta_aucs,
                },
                "mmse_plus_wearable": {
                    "wearable_alpha": float(wearable_alpha),
                    "candidate_inner_auc": alpha_aucs,
                },
                "full": {
                    "wearable_weight": float(full_weight[0]),
                    "program_weight": float(full_weight[1]),
                    "mmse_weight": float(1.0 - full_weight[0] - full_weight[1]),
                    "candidate_inner_auc": full_aucs,
                },
            }
            selection_records.append(selection_record)
            fold_metrics.append(
                {
                    "split_id": split_id,
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "arms": {
                        arm: binary_metrics(y[outer_test], arm_scores[arm])
                        for arm in ARMS
                    },
                }
            )
            split_entries.append(
                {
                    "split_id": split_id,
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "outer_seed": int(outer_seed),
                    "inner_seed": int(inner_seed),
                    "train_indices": outer_train.tolist(),
                    "test_indices": outer_test.tolist(),
                    "train_diagnosis_counts": _diagnosis_counts(
                        diagnosis[outer_train]
                    ),
                    "test_diagnosis_counts": _diagnosis_counts(diagnosis[outer_test]),
                    "inner_splits": inner_entries,
                }
            )
            log(
                f"[nested] {split_id}: beta={program_beta:.1f}, "
                f"alpha={wearable_alpha:.1f}, "
                f"full=(wearable={full_weight[0]:.1f}, program={full_weight[1]:.1f})"
            )

        for arm in ARMS:
            if not np.isfinite(predictions[arm][repeat]).all():
                missing = int((~np.isfinite(predictions[arm][repeat])).sum())
                raise AssertionError(
                    f"repeat {repeat} arm {arm} left {missing} outer-OOF rows unscored"
                )

    repeat_metrics: dict[str, list[dict[str, Any]]] = {}
    repeat_auc_summary: dict[str, dict[str, Any]] = {}
    subject_mean_scores: dict[str, np.ndarray] = {}
    subject_mean_metrics: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        metrics = [
            binary_metrics(y, predictions[arm][repeat]) for repeat in range(repeats)
        ]
        repeat_metrics[arm] = metrics
        auc_values = [float(item["roc_auc"]) for item in metrics]
        repeat_auc_summary[arm] = _repeat_summary(auc_values)
        subject_mean_scores[arm] = predictions[arm].mean(axis=0)
        subject_mean_metrics[arm] = binary_metrics(y, subject_mean_scores[arm])

    modal_full = _modal_full_weight(selected_full)
    mean_diagnosis_diagnostics: dict[str, Any] = {}
    for label in ("CN", "MCI", "Dem"):
        mask = diagnosis == label
        mean_diagnosis_diagnostics[label] = {
            "n": int(mask.sum()),
            "arms": {
                arm: {
                    "mean_score": float(subject_mean_scores[arm][mask].mean()),
                    "predicted_positive_rate_at_0.5": float(
                        np.mean(subject_mean_scores[arm][mask] >= FIXED_THRESHOLD)
                    ),
                }
                for arm in ARMS
            },
        }

    paired_bootstrap = {
        "full_minus_mmse_only": paired_bootstrap_auc_difference(
            y,
            subject_mean_scores["full"],
            subject_mean_scores["mmse_only"],
            n_bootstrap=n_bootstrap,
            seed=int(seed) + 2_000_001,
        ),
        "wearable_plus_program_minus_wearable_only": (
            paired_bootstrap_auc_difference(
                y,
                subject_mean_scores["wearable_plus_program"],
                subject_mean_scores["wearable_only"],
                n_bootstrap=n_bootstrap,
                seed=int(seed) + 2_000_002,
            )
        ),
    }
    absolute_bootstrap = {
        arm: bootstrap_auc(
            y,
            subject_mean_scores[arm],
            n_bootstrap=n_bootstrap,
            seed=int(seed) + 2_100_000 + arm_index,
        )
        for arm_index, arm in enumerate(ARMS)
    }

    oof_rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for subject_index in range(n_subjects):
            oof_rows.append(
                {
                    "subject_index": int(subject_index),
                    "diagnosis": diagnosis[subject_index],
                    "y_true": int(y[subject_index]),
                    "repeat": int(repeat),
                    **{
                        arm: float(predictions[arm][repeat, subject_index])
                        for arm in ARMS
                    },
                }
            )
    oof_frame = pd.DataFrame.from_records(oof_rows)

    selection_frequency = {
        "wearable_plus_program_program_beta": _frequency_map(
            selected_betas, lambda value: f"{float(value):.1f}"
        ),
        "mmse_plus_wearable_wearable_alpha": _frequency_map(
            selected_alphas, lambda value: f"{float(value):.1f}"
        ),
        "full": _frequency_map(
            selected_full,
            lambda value: (
                f"wearable={float(value[0]):.1f},program={float(value[1]):.1f}"
            ),
        ),
    }
    report = _json_safe(
        {
            "experiment": "GemmaFeatureProgramPipeline",
            "performance_target_guard": {
                "roc_auc": 0.92,
                "status": "design goal, not a claimed result",
            },
            "program_sha256": _program_hash(program),
            "cohort": {
                "fingerprint": fingerprint,
                "n_subjects": int(n_subjects),
                "diagnosis_counts": _diagnosis_counts(diagnosis),
                "positive_class": "MCI or Dem",
                "negative_class": "CN",
                "has_external_holdout": False,
            },
            "evaluation_contract": {
                "outer_split": (
                    f"repeated {outer_folds}-fold subject-level StratifiedKFold "
                    "stratified by CN/MCI/Dem"
                ),
                "inner_split": (
                    f"{inner_folds}-fold subject-level StratifiedKFold "
                    "inside each outer-training block"
                ),
                "repeats": int(repeats),
                "same_outer_folds_for_all_arms": True,
                "fold_local_preprocessing": True,
                "test_batch_rank_normalization": False,
                "score_scale": "raw decision score located in fitted training empirical CDF",
                "threshold": FIXED_THRESHOLD,
                "threshold_tuning": False,
                "selection_tolerance_auc": SELECTION_TOLERANCE,
                "weight_selection": "inner OOF only",
                "full_cohort_feature_selection": False,
            },
            "arms": list(ARMS),
            "weight_grids": {
                "wearable_plus_program_program_beta": list(PROGRAM_BETAS),
                "mmse_plus_wearable_wearable_alpha": list(WEARABLE_ALPHAS),
                "full_wearable_program_pairs": [
                    {"wearable": weight[0], "program": weight[1]}
                    for weight in _full_candidates()
                ],
            },
            "primary": {
                "arm": "full",
                "metric": "mean of repeat-wise full OOF ROC-AUC",
                **repeat_auc_summary["full"],
            },
            "repeat_auc_summary_by_arm": repeat_auc_summary,
            "repeat_metrics_by_arm": repeat_metrics,
            "secondary_subject_mean_metrics_by_arm": subject_mean_metrics,
            "diagnosis_diagnostics_on_subject_mean_scores": (
                mean_diagnosis_diagnostics
            ),
            "subject_mean_auc_subject_bootstrap_by_arm": absolute_bootstrap,
            "paired_subject_bootstrap": paired_bootstrap,
            "selection": {
                "frequency": selection_frequency,
                "modal_full_weight": {
                    "wearable_weight": float(modal_full[0]),
                    "program_weight": float(modal_full[1]),
                    "mmse_weight": float(1.0 - modal_full[0] - modal_full[1]),
                    "selection_rule": (
                        "highest outer-fold frequency; ties prefer lower program "
                        "weight then lower total auxiliary weight"
                    ),
                },
                "outer_fold_records": selection_records,
            },
            "outer_fold_metrics": fold_metrics,
            "interpretation_guard": (
                "All metrics are repeated nested OOF estimates on the available "
                "cohort, not external validation. Reaching 0.92 here would require "
                "confirmation on a genuinely untouched cohort."
            ),
        }
    )
    split_registry = _json_safe(
        {
            "seed": int(seed),
            "outer_folds": int(outer_folds),
            "inner_folds": int(inner_folds),
            "repeats": int(repeats),
            "stratification": "diagnosis: CN/MCI/Dem",
            "subject_level": True,
            "n_subjects": int(n_subjects),
            "splits": split_entries,
        }
    )
    return EvaluationResult(
        report=report,
        oof=oof_frame,
        split_registry=split_registry,
    )
