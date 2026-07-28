"""ROC-AUC-first metrics for subject-level binary experiments.

The functions in this module deliberately separate three quantities that are
easy to conflate:

* the ROC-AUC from each repeated OOF split;
* the mean and split-to-split standard deviation of those repeat AUCs; and
* the ROC-AUC obtained *after* averaging every subject's repeated OOF scores.

The last quantity is an ensemble estimand and is not the centre of the
repeat-level distribution.  Consequently, its subject-bootstrap confidence
interval must not be attached to the mean repeat AUC.

No function searches for a threshold.  Threshold-dependent metrics are
computed only when the caller supplies a fixed threshold.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)


DEFAULT_BOOTSTRAP_SEED = 20260728
DEFAULT_BOOTSTRAP_RESAMPLES = 5000


class MetricContractError(ValueError):
    """Raised when labels or model scores violate the metric contract."""


def _binary_target(y: Sequence[int] | np.ndarray) -> np.ndarray:
    """Return a finite 0/1 target containing both classes."""

    target = np.asarray(y)
    if target.ndim != 1:
        raise MetricContractError(
            f"y must be one-dimensional; received shape {target.shape}"
        )
    if target.size < 2:
        raise MetricContractError("y must contain at least two subjects")
    try:
        numeric = target.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise MetricContractError("y must contain numeric binary labels") from error
    if not np.isfinite(numeric).all():
        raise MetricContractError("y contains a missing or non-finite label")
    unique = np.unique(numeric)
    if not np.all(np.isin(unique, (0.0, 1.0))):
        raise MetricContractError(
            f"y must use CN=0 and MCI+Dem=1; observed {unique.tolist()}"
        )
    if unique.size != 2:
        raise MetricContractError("ROC-AUC requires both classes in y")
    return numeric.astype(np.int64, copy=False)


def validate_continuous_scores(
    y: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    score_name: str = "scores",
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one subject-level continuous score per binary target.

    Scores may be probabilities, logits, margins, or another finite monotone
    ranking score.  They are not restricted to ``[0, 1]``.  Hard predictions
    and other two-level outputs are rejected because evaluating them as a
    continuous ROC curve would misrepresent threshold performance as ranking
    performance.
    """

    target = _binary_target(y)
    raw = np.asarray(scores)
    if raw.ndim != 1:
        raise MetricContractError(
            f"{score_name} must be one-dimensional; received shape {raw.shape}"
        )
    if len(raw) != len(target):
        raise MetricContractError(
            f"{score_name} has {len(raw)} rows but y has {len(target)}"
        )
    try:
        values = raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise MetricContractError(f"{score_name} must be numeric") from error
    if not np.isfinite(values).all():
        raise MetricContractError(
            f"{score_name} contains a missing or non-finite value"
        )
    unique = np.unique(values)
    if unique.size < 3:
        raise MetricContractError(
            f"{score_name} has only {unique.size} distinct value(s); "
            "continuous ROC-AUC scores require at least three"
        )
    return target, values


def _bootstrap_contract(
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> tuple[int, float, int]:
    count = int(n_resamples)
    confidence = float(confidence_level)
    resolved_seed = int(seed)
    if count < 1:
        raise MetricContractError("n_resamples must be at least one")
    if not 0.0 < confidence < 1.0:
        raise MetricContractError("confidence_level must be strictly between 0 and 1")
    return count, confidence, resolved_seed


def _stratified_bootstrap_indices(
    target: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
):
    """Yield paired subject indices while preserving the two class counts."""

    negative = np.flatnonzero(target == 0)
    positive = np.flatnonzero(target == 1)
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        sampled_negative = rng.choice(negative, size=len(negative), replace=True)
        sampled_positive = rng.choice(positive, size=len(positive), replace=True)
        yield np.concatenate((sampled_negative, sampled_positive))


def stratified_subject_bootstrap_auc(
    y: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = 0.95,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    score_name: str = "scores",
) -> dict[str, Any]:
    """Percentile CI for subject ROC-AUC using within-class resampling."""

    target, values = validate_continuous_scores(
        y, scores, score_name=score_name
    )
    count, confidence, resolved_seed = _bootstrap_contract(
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
    bootstrap = np.empty(count, dtype=np.float64)
    for index, sampled in enumerate(
        _stratified_bootstrap_indices(
            target, n_resamples=count, seed=resolved_seed
        )
    ):
        bootstrap[index] = roc_auc_score(target[sampled], values[sampled])
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap, (alpha, 1.0 - alpha))
    standard_error = (
        float(np.std(bootstrap, ddof=1)) if count > 1 else 0.0
    )
    return {
        "estimand": "ROC-AUC of the supplied subject-level continuous scores",
        "method": "stratified subject bootstrap percentile interval",
        "point": float(roc_auc_score(target, values)),
        "lo": float(lower),
        "hi": float(upper),
        "confidence_level": confidence,
        "standard_error": standard_error,
        "n_resamples": count,
        "seed": resolved_seed,
        "class_counts_preserved": {
            "CN_0": int(np.sum(target == 0)),
            "MCI_Dem_1": int(np.sum(target == 1)),
        },
    }


def paired_bootstrap_auc_difference(
    y: Sequence[int] | np.ndarray,
    reference_scores: Sequence[float] | np.ndarray,
    candidate_scores: Sequence[float] | np.ndarray,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = 0.95,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    reference_name: str = "reference",
    candidate_name: str = "candidate",
) -> dict[str, Any]:
    """Paired CI for ``candidate ROC-AUC - reference ROC-AUC``.

    The same bootstrapped subject indices are applied to both score vectors.
    This preserves both subject pairing and class counts.
    """

    target, reference = validate_continuous_scores(
        y, reference_scores, score_name=f"{reference_name}_scores"
    )
    paired_target, candidate = validate_continuous_scores(
        y, candidate_scores, score_name=f"{candidate_name}_scores"
    )
    if not np.array_equal(target, paired_target):
        raise MetricContractError("Paired score vectors do not share the same target")
    count, confidence, resolved_seed = _bootstrap_contract(
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
    differences = np.empty(count, dtype=np.float64)
    for index, sampled in enumerate(
        _stratified_bootstrap_indices(
            target, n_resamples=count, seed=resolved_seed
        )
    ):
        candidate_auc = roc_auc_score(target[sampled], candidate[sampled])
        reference_auc = roc_auc_score(target[sampled], reference[sampled])
        differences[index] = candidate_auc - reference_auc
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(differences, (alpha, 1.0 - alpha))
    reference_auc = float(roc_auc_score(target, reference))
    candidate_auc = float(roc_auc_score(target, candidate))
    standard_error = (
        float(np.std(differences, ddof=1)) if count > 1 else 0.0
    )
    return {
        "estimand": f"ROC-AUC({candidate_name}) - ROC-AUC({reference_name})",
        "direction": "candidate_minus_reference",
        "reference_name": str(reference_name),
        "candidate_name": str(candidate_name),
        "reference_auc": reference_auc,
        "candidate_auc": candidate_auc,
        "point": float(candidate_auc - reference_auc),
        "lo": float(lower),
        "hi": float(upper),
        "confidence_level": confidence,
        "standard_error": standard_error,
        "n_resamples": count,
        "seed": resolved_seed,
        "resampling": "same stratified subject indices for both score vectors",
        "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
        "candidate_better_at_interval_level": bool(lower > 0.0),
    }


def _fixed_threshold_metrics(
    target: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    resolved_threshold = float(threshold)
    if not np.isfinite(resolved_threshold):
        raise MetricContractError("threshold must be finite")
    prediction = (scores >= resolved_threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(
        target, prediction, labels=(0, 1)
    ).ravel()
    recall = float(tp / (tp + fn))
    specificity = float(tn / (tn + fp))
    return {
        "policy": "evaluated at caller-supplied threshold; no threshold search",
        "threshold": resolved_threshold,
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(target, prediction)
        ),
        "precision_mci_dem": float(
            precision_score(target, prediction, zero_division=0)
        ),
        "recall_mci_dem": recall,
        "specificity_cn": specificity,
        "f1_mci_dem": float(f1_score(target, prediction, zero_division=0)),
        "confusion": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def evaluate_binary_scores(
    y: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    threshold: float | None = None,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = 0.95,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    score_name: str = "scores",
) -> dict[str, Any]:
    """Evaluate one frozen continuous score vector with ROC-AUC primary."""

    target, values = validate_continuous_scores(
        y, scores, score_name=score_name
    )
    baseline_accuracy = float(np.mean(target == 0))
    result: dict[str, Any] = {
        "primary": {
            "metric": "ROC-AUC",
            "roc_auc": float(roc_auc_score(target, values)),
            "bootstrap": stratified_subject_bootstrap_auc(
                target,
                values,
                n_resamples=n_resamples,
                confidence_level=confidence_level,
                seed=seed,
                score_name=score_name,
            ),
        },
        "secondary": {
            "pr_auc_average_precision": float(
                average_precision_score(target, values)
            ),
            "all_cn_baseline": {
                "accuracy": baseline_accuracy,
                "balanced_accuracy": 0.5,
                "recall_mci_dem": 0.0,
                "specificity_cn": 1.0,
                "counts": {
                    "tn": int(np.sum(target == 0)),
                    "fp": 0,
                    "fn": int(np.sum(target == 1)),
                    "tp": 0,
                },
            },
            "fixed_threshold": (
                None
                if threshold is None
                else _fixed_threshold_metrics(target, values, threshold)
            ),
        },
        "score_audit": {
            "name": str(score_name),
            "n_subjects": int(len(target)),
            "n_unique_scores": int(np.unique(values).size),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "score_type": "finite continuous ranking score",
            "threshold_selected_here": False,
        },
        "class_counts": {
            "CN_0": int(np.sum(target == 0)),
            "MCI_Dem_1": int(np.sum(target == 1)),
        },
    }
    if threshold is None:
        result["secondary"]["fixed_threshold_policy"] = (
            "not evaluated because the caller supplied no threshold"
        )
    return result


def summarize_repeated_oof(
    y: Sequence[int] | np.ndarray,
    repeated_scores: Sequence[Sequence[float]] | np.ndarray,
    *,
    threshold: float | None = None,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = 0.95,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    score_name: str = "repeated_oof_scores",
) -> dict[str, Any]:
    """Report repeat dispersion separately from averaged-score performance.

    ``repeated_scores`` must have shape ``[n_repeats, n_subjects]``.  Every row
    must be a complete OOF score vector: each subject's score in that row must
    come from a model that did not train on that subject.
    """

    target = _binary_target(y)
    try:
        matrix = np.asarray(repeated_scores, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise MetricContractError("repeated_scores must be a numeric matrix") from error
    if matrix.ndim != 2:
        raise MetricContractError(
            "repeated_scores must have shape [n_repeats, n_subjects]; "
            f"received {matrix.shape}"
        )
    if matrix.shape[0] < 1:
        raise MetricContractError("repeated_scores contains no repeat")
    if matrix.shape[1] != len(target):
        raise MetricContractError(
            f"repeated_scores has {matrix.shape[1]} subjects but y has {len(target)}"
        )

    repeat_auc: list[float] = []
    for repeat_index, row in enumerate(matrix):
        _, validated = validate_continuous_scores(
            target,
            row,
            score_name=f"{score_name}[repeat={repeat_index}]",
        )
        repeat_auc.append(float(roc_auc_score(target, validated)))

    subject_mean_scores = np.mean(matrix, axis=0)
    validate_continuous_scores(
        target,
        subject_mean_scores,
        score_name=f"{score_name}_subject_mean",
    )
    repeat_array = np.asarray(repeat_auc, dtype=np.float64)
    averaged_evaluation = evaluate_binary_scores(
        target,
        subject_mean_scores,
        threshold=threshold,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
        score_name=f"{score_name}_subject_mean",
    )
    return {
        "repeat_level_roc_auc": {
            "estimand": (
                "distribution of ROC-AUC across complete repeated OOF score vectors"
            ),
            "values": repeat_auc,
            "mean": float(np.mean(repeat_array)),
            "sd": (
                float(np.std(repeat_array, ddof=0))
                if len(repeat_array) > 1
                else 0.0
            ),
            "sd_ddof": 0,
            "n_repeats": int(len(repeat_array)),
            "bootstrap_ci_attached": False,
        },
        "subject_mean_repeated_oof": {
            "estimand": (
                "ROC-AUC after arithmetic-mean aggregation of each subject's "
                "scores across repeats"
            ),
            "score_aggregation": "arithmetic mean across repeats within subject",
            "evaluation": averaged_evaluation,
        },
        "separation_warning": (
            "The subject-mean ROC-AUC and its subject-bootstrap CI are a distinct "
            "ensemble estimand; they are not a confidence interval for the mean "
            "repeat ROC-AUC."
        ),
        "input_contract": {
            "unit": "subject",
            "matrix_shape": [
                int(matrix.shape[0]),
                int(matrix.shape[1]),
            ],
            "one_complete_oof_vector_per_repeat_required": True,
        },
    }


# Descriptive aliases retained to keep future training/report code readable.
bootstrap_auc_ci = stratified_subject_bootstrap_auc
paired_stratified_subject_bootstrap_auc_difference = (
    paired_bootstrap_auc_difference
)
repeated_oof_auc_summary = summarize_repeated_oof


__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "MetricContractError",
    "bootstrap_auc_ci",
    "evaluate_binary_scores",
    "paired_bootstrap_auc_difference",
    "paired_stratified_subject_bootstrap_auc_difference",
    "repeated_oof_auc_summary",
    "stratified_subject_bootstrap_auc",
    "summarize_repeated_oof",
    "validate_continuous_scores",
]
