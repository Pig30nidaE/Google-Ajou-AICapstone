"""Reproducible repeated subject-level split registry."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .leakage import (
    LeakageError,
    assert_binary_fold,
    assert_disjoint_groups,
    assert_group_label_consistency,
    hash_subject_id,
)


@dataclass(frozen=True)
class FoldSplit:
    split_id: str
    repeat: int
    fold: int
    seed: int
    train_indices: np.ndarray
    validation_indices: np.ndarray


@dataclass(frozen=True)
class SplitPlan:
    """All folds for one repeated CV layer."""

    layer: str
    n_subjects: int
    n_splits: int
    n_repeats: int
    records: tuple[FoldSplit, ...]

    def records_for_repeat(self, repeat: int) -> tuple[FoldSplit, ...]:
        return tuple(record for record in self.records if record.repeat == repeat)


def _validate_requested_folds(y: np.ndarray, requested: int, context: str) -> int:
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=2)
    minority = int(counts.min())
    if minority < 2:
        raise LeakageError(
            f"{context}: at least two subjects per class are required; counts={counts.tolist()}"
        )
    if int(requested) > minority:
        raise LeakageError(
            f"{context}: {requested} folds exceed minority count {minority}; "
            "change CVConfig instead of silently changing the estimand"
        )
    if int(requested) < 2:
        raise LeakageError(f"{context}: at least two folds are required")
    return int(requested)


def build_repeated_group_plan(
    y: Sequence[int],
    groups: Sequence[str],
    *,
    n_splits: int,
    n_repeats: int,
    seed: int,
    minimum_positive_validation: int,
    layer: str,
) -> SplitPlan:
    """Build repeated StratifiedGroupKFold splits and assert every contract."""

    from sklearn.model_selection import StratifiedGroupKFold

    target = np.asarray(y, dtype=np.int64)
    subject_groups = np.asarray(groups, dtype=str)
    if target.shape != subject_groups.shape:
        raise LeakageError(f"{layer}: target/group arrays are not aligned")
    assert_group_label_consistency(subject_groups, target)
    folds = _validate_requested_folds(target, n_splits, layer)
    repeats = max(1, int(n_repeats))
    records: list[FoldSplit] = []
    dummy = np.zeros((len(target), 1), dtype=np.float32)
    for repeat in range(repeats):
        repeat_seed = int(seed + repeat * 1009)
        splitter = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=repeat_seed,
        )
        validation_count = np.zeros(len(target), dtype=np.int64)
        for fold, (train_index, validation_index) in enumerate(
            splitter.split(dummy, target, groups=subject_groups)
        ):
            context = f"{layer}/repeat={repeat}/fold={fold}"
            assert_disjoint_groups(
                subject_groups[train_index],
                subject_groups[validation_index],
                context=context,
            )
            assert_binary_fold(
                target[train_index],
                target[validation_index],
                minimum_positive_validation=minimum_positive_validation,
                context=context,
            )
            validation_count[validation_index] += 1
            records.append(
                FoldSplit(
                    split_id=f"{layer}_r{repeat:02d}_f{fold:02d}",
                    repeat=repeat,
                    fold=fold,
                    seed=repeat_seed + fold,
                    train_indices=np.asarray(train_index, dtype=np.int64),
                    validation_indices=np.asarray(validation_index, dtype=np.int64),
                )
            )
        if np.any(validation_count != 1):
            raise LeakageError(
                f"{layer}/repeat={repeat}: each subject must be validation exactly once"
            )
    return SplitPlan(
        layer=layer,
        n_subjects=len(target),
        n_splits=folds,
        n_repeats=repeats,
        records=tuple(records),
    )


def build_inner_plan(
    outer_train_indices: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    n_repeats: int,
    seed: int,
    minimum_positive_validation: int,
    layer: str,
) -> SplitPlan:
    """Create indices relative to one outer-training subset."""

    outer_train = np.asarray(outer_train_indices, dtype=np.int64)
    return build_repeated_group_plan(
        np.asarray(y)[outer_train],
        np.asarray(groups)[outer_train],
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        minimum_positive_validation=minimum_positive_validation,
        layer=layer,
    )


def split_plan_payload(
    plan: SplitPlan,
    *,
    subject_ids: Sequence[str],
    y: Sequence[int],
) -> dict:
    """Serialize without exposing source identifiers."""

    subjects = np.asarray(subject_ids, dtype=str)
    target = np.asarray(y, dtype=np.int64)
    if len(subjects) != plan.n_subjects or len(target) != plan.n_subjects:
        raise LeakageError("Split plan serialization arrays are misaligned")
    records = []
    for record in plan.records:
        train = record.train_indices
        validation = record.validation_indices
        records.append(
            {
                "split_id": record.split_id,
                "repeat": record.repeat,
                "fold": record.fold,
                "seed": record.seed,
                "train_subject_hashes": [
                    hash_subject_id(value) for value in subjects[train]
                ],
                "validation_subject_hashes": [
                    hash_subject_id(value) for value in subjects[validation]
                ],
                "train_class_counts": {
                    "negative": int(np.sum(target[train] == 0)),
                    "positive_dem": int(np.sum(target[train] == 1)),
                },
                "validation_class_counts": {
                    "negative": int(np.sum(target[validation] == 0)),
                    "positive_dem": int(np.sum(target[validation] == 1)),
                },
            }
        )
    return {
        "layer": plan.layer,
        "n_subjects": plan.n_subjects,
        "n_splits": plan.n_splits,
        "n_repeats": plan.n_repeats,
        "label_definition": {"negative": "CN+MCI", "positive": "Dem"},
        "records": records,
    }


def save_split_plan(
    path: str | Path,
    plan: SplitPlan,
    *,
    subject_ids: Sequence[str],
    y: Sequence[int],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            split_plan_payload(plan, subject_ids=subject_ids, y=y),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


__all__ = [
    "FoldSplit",
    "SplitPlan",
    "build_inner_plan",
    "build_repeated_group_plan",
    "save_split_plan",
    "split_plan_payload",
]
