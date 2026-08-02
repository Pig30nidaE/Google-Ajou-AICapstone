"""Cross-validated evaluation, metrics and result serialization.

Validation design (and why): a **non-nested repeated subject-level
StratifiedKFold**.  Nested CV exists to protect a *search*, and this stage runs
exactly one fixed configuration per model with no tuning, no feature selection
and no threshold fitting, so there is nothing for an inner loop to protect.
Repeats are used instead to quantify split noise, which is the dominant source
of variation at n=141 with 56 positives (see ``SangHyo/AGENTS.md`` 3-3).  If a
search is ever added, ``cv`` must become nested first - that is recorded as a
limitation in ``README_KO.md`` rather than silently assumed away.

Metrics: ROC-AUC (primary), PR-AUC, and at the fixed 0.5 threshold recall
(sensitivity), specificity, F1, balanced accuracy and MCC.  Accuracy alone is
never reported without the all-negative baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .features import DesignMatrix
from .guards import assert_disjoint_subjects, hash_subject_id
from .models import build_model
from .splits import SplitPlan

__all__ = [
    "ArmResult",
    "evaluate_arm",
    "binary_metrics",
    "write_json",
    "write_oof_csv",
]


def write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return destination


def binary_metrics(y_true: Sequence[int], scores: Sequence[float], *, threshold: float = 0.5) -> dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
        roc_auc_score,
    )

    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(p)
    if not finite.all():
        raise ValueError(f"{int((~finite).sum())} non-finite prediction(s)")
    if len(set(y.tolist())) < 2:
        return {"n": int(len(y)), "note": "single-class subset; ranking metrics undefined"}

    predicted = (p >= float(threshold)).astype(np.int64)
    true_positive = int(np.sum((predicted == 1) & (y == 1)))
    false_positive = int(np.sum((predicted == 1) & (y == 0)))
    true_negative = int(np.sum((predicted == 0) & (y == 0)))
    false_negative = int(np.sum((predicted == 0) & (y == 1)))
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    return {
        "n": int(len(y)),
        "n_positive": positives,
        "n_negative": negatives,
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "threshold": float(threshold),
        "recall_sensitivity": float(true_positive / positives) if positives else float("nan"),
        "specificity": float(true_negative / negatives) if negatives else float("nan"),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "mcc": float(matthews_corrcoef(y, predicted)) if len(set(predicted.tolist())) > 1 else 0.0,
        "accuracy": float(np.mean(predicted == y)),
        "all_negative_baseline_accuracy": float(negatives / len(y)),
        "confusion": {
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
    }


@dataclass
class ArmResult:
    arm_id: str
    feature_set: str
    mmse_mode: str
    model: str
    implementation: str
    n_features: int
    fold_records: list[dict[str, Any]] = field(default_factory=list)
    oof_by_repeat: dict[int, np.ndarray] = field(default_factory=dict)
    mean_oof: np.ndarray | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "feature_set": self.feature_set,
            "mmse_mode": self.mmse_mode,
            "model": self.model,
            "implementation": self.implementation,
            "n_features": self.n_features,
            "metrics": self.metrics,
            "folds": self.fold_records,
        }


def evaluate_arm(
    matrix: DesignMatrix,
    y: np.ndarray,
    plan: SplitPlan,
    *,
    model_name: str,
    model_params: Mapping[str, Any],
    seed: int,
    arm_id: str,
    logger=print,
) -> ArmResult:
    """Fit one fixed configuration over every fold of the shared split plan."""

    X = np.asarray(matrix.X, dtype=np.float64)
    target = np.asarray(y, dtype=np.int64)
    if X.shape[0] != target.shape[0] or X.shape[0] != plan.n_subjects:
        raise ValueError("Design matrix, target and split plan disagree on the subject count")

    _, spec = build_model(model_name, model_params, seed=seed)
    result = ArmResult(
        arm_id=arm_id,
        feature_set=matrix.feature_set,
        mmse_mode=matrix.mmse_mode,
        model=model_name,
        implementation=spec.implementation,
        n_features=matrix.n_features,
    )

    oof = {repeat: np.full(len(target), np.nan) for repeat in range(plan.n_repeats)}
    for record in plan.records:
        train_index, validation_index = record.train_indices, record.validation_indices
        assert_disjoint_subjects(
            matrix.subject_ids[train_index],
            matrix.subject_ids[validation_index],
            context=f"{arm_id}/{record.split_id}",
        )
        pipeline, _ = build_model(model_name, model_params, seed=seed + record.fold)
        pipeline.fit(X[train_index], target[train_index])
        scores = pipeline.predict_proba(X[validation_index])[:, 1]
        oof[record.repeat][validation_index] = scores
        fold_metrics = binary_metrics(target[validation_index], scores)
        result.fold_records.append(
            {
                "split_id": record.split_id,
                "repeat": record.repeat,
                "fold": record.fold,
                "n_train": int(len(train_index)),
                "n_validation": int(len(validation_index)),
                "roc_auc": fold_metrics.get("roc_auc"),
                "pr_auc": fold_metrics.get("pr_auc"),
            }
        )

    per_repeat_auc: list[float] = []
    for repeat, predictions in oof.items():
        if np.isnan(predictions).any():
            raise ValueError(f"{arm_id}: repeat {repeat} left {int(np.isnan(predictions).sum())} subjects unscored")
        per_repeat_auc.append(float(binary_metrics(target, predictions)["roc_auc"]))
    result.oof_by_repeat = oof
    result.mean_oof = np.mean(np.vstack([oof[repeat] for repeat in sorted(oof)]), axis=0)

    pooled = binary_metrics(target, result.mean_oof)
    fold_aucs = [
        record["roc_auc"] for record in result.fold_records if record["roc_auc"] is not None
    ]
    result.metrics = {
        **pooled,
        "roc_auc_repeat_mean": float(np.mean(per_repeat_auc)),
        "roc_auc_repeat_sd": float(np.std(per_repeat_auc, ddof=1)) if len(per_repeat_auc) > 1 else 0.0,
        "roc_auc_per_repeat": [round(value, 6) for value in per_repeat_auc],
        "roc_auc_fold_mean": float(np.mean(fold_aucs)) if fold_aucs else float("nan"),
        "roc_auc_fold_sd": float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0,
        "n_folds": len(result.fold_records),
    }
    logger(
        f"[eval] {arm_id}: pooled OOF ROC-AUC={result.metrics['roc_auc']:.4f} "
        f"(per-repeat {result.metrics['roc_auc_repeat_mean']:.4f} "
        f"+/- {result.metrics['roc_auc_repeat_sd']:.4f}), features={matrix.n_features}"
    )
    return result


def write_oof_csv(
    path: str | Path,
    *,
    subject_ids: Sequence[str],
    y: Sequence[int],
    results: Sequence[ArmResult],
    salt: str,
) -> Path:
    """Hashed subject-level OOF predictions for every arm (one column per arm)."""

    import pandas as pd

    frame = pd.DataFrame(
        {
            "subject_hash": [hash_subject_id(value, salt=salt) for value in subject_ids],
            "y_true": np.asarray(y, dtype=np.int64),
        }
    )
    for result in results:
        if result.mean_oof is None:
            continue
        frame[f"oof__{result.arm_id}"] = result.mean_oof
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def paired_arm_comparison(
    y: Sequence[int], reference: ArmResult, candidate: ArmResult
) -> dict[str, Any]:
    """Descriptive paired difference on identical subjects and identical splits.

    Reported as a difference with a subject-level bootstrap interval; it is a
    descriptive statistic, not a significance claim.
    """

    from sklearn.metrics import roc_auc_score

    target = np.asarray(y, dtype=np.int64)
    if reference.mean_oof is None or candidate.mean_oof is None:
        raise ValueError("Both arms need out-of-fold predictions")
    rng = np.random.default_rng(20260729)
    differences: list[float] = []
    indices = np.arange(len(target))
    for _ in range(2000):
        sample = rng.choice(indices, size=len(indices), replace=True)
        if len(set(target[sample].tolist())) < 2:
            continue
        differences.append(
            float(roc_auc_score(target[sample], candidate.mean_oof[sample]))
            - float(roc_auc_score(target[sample], reference.mean_oof[sample]))
        )
    observed = float(
        roc_auc_score(target, candidate.mean_oof) - roc_auc_score(target, reference.mean_oof)
    )
    low, high = (
        (float(np.percentile(differences, 2.5)), float(np.percentile(differences, 97.5)))
        if differences
        else (float("nan"), float("nan"))
    )
    return {
        "reference_arm": reference.arm_id,
        "candidate_arm": candidate.arm_id,
        "roc_auc_difference": observed,
        "bootstrap_95_interval": [low, high],
        "interval_contains_zero": bool(low <= 0.0 <= high),
        "n_bootstrap": len(differences),
        "interpretation": (
            "Descriptive paired difference on identical subjects and identical folds. "
            "An interval containing zero means no demonstrated improvement."
        ),
    }
