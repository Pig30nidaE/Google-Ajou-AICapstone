"""Subject-level cross-validation plan, shared by every arm of the comparison.

The cohort has exactly one row per subject, so a repeated ``StratifiedKFold``
over rows *is* a subject-level split (see <evaluation_design>).  The subject
identity is still carried through and asserted disjoint on every fold, so the
contract stays valid if a future version switches to multi-row subjects.

The same plan object is reused for BASE vs BASE+Gemini and for
``mmse_mode=without`` vs ``with``; that is what makes the four arms comparable
(<data_leakage_rules> items 8 and 12).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .guards import LeakageError, assert_disjoint_subjects, hash_subject_id

__all__ = ["FoldSplit", "SplitPlan", "build_split_plan", "save_split_plan"]


@dataclass(frozen=True)
class FoldSplit:
    split_id: str
    repeat: int
    fold: int
    train_indices: np.ndarray
    validation_indices: np.ndarray


@dataclass(frozen=True)
class SplitPlan:
    n_subjects: int
    n_splits: int
    n_repeats: int
    seed: int
    records: tuple[FoldSplit, ...]
    plan_hash: str

    def for_repeat(self, repeat: int) -> tuple[FoldSplit, ...]:
        return tuple(record for record in self.records if record.repeat == repeat)


def build_split_plan(
    y: Sequence[int],
    subject_ids: Sequence[str],
    *,
    n_splits: int,
    n_repeats: int,
    seed: int,
    min_positive_per_validation_fold: int = 1,
) -> SplitPlan:
    from sklearn.model_selection import StratifiedKFold

    target = np.asarray(y, dtype=np.int64)
    subjects = np.asarray([str(value) for value in subject_ids], dtype=str)
    if target.shape != subjects.shape:
        raise LeakageError("Target and subject arrays are not aligned")
    if len(set(subjects.tolist())) != len(subjects):
        raise LeakageError("Subject identifiers must be unique for a subject-level split")

    counts = np.bincount(target, minlength=2)
    minority = int(counts.min())
    if minority < 2:
        raise LeakageError(f"Both classes need at least two subjects; counts={counts.tolist()}")
    if int(n_splits) > minority:
        raise LeakageError(
            f"{n_splits} folds exceed the minority class size {minority}; "
            "reduce cv.n_splits instead of silently changing the estimand"
        )

    records: list[FoldSplit] = []
    for repeat in range(max(1, int(n_repeats))):
        splitter = StratifiedKFold(
            n_splits=int(n_splits), shuffle=True, random_state=int(seed) + repeat * 1009
        )
        seen = np.zeros(len(target), dtype=np.int64)
        for fold, (train_index, validation_index) in enumerate(
            splitter.split(np.zeros((len(target), 1)), target)
        ):
            context = f"repeat={repeat}/fold={fold}"
            assert_disjoint_subjects(
                subjects[train_index], subjects[validation_index], context=context
            )
            positives = int(target[validation_index].sum())
            if positives < int(min_positive_per_validation_fold):
                raise LeakageError(
                    f"{context}: only {positives} positive subject(s) in the validation fold; "
                    f"cv.min_positive_per_validation_fold={min_positive_per_validation_fold}"
                )
            if len(set(target[train_index].tolist())) < 2:
                raise LeakageError(f"{context}: training fold is single-class")
            seen[validation_index] += 1
            records.append(
                FoldSplit(
                    split_id=f"r{repeat:02d}_f{fold:02d}",
                    repeat=repeat,
                    fold=fold,
                    train_indices=np.asarray(train_index, dtype=np.int64),
                    validation_indices=np.asarray(validation_index, dtype=np.int64),
                )
            )
        if np.any(seen != 1):
            raise LeakageError(f"repeat={repeat}: every subject must be validated exactly once")

    import hashlib

    material = json.dumps(
        [
            {
                "split_id": record.split_id,
                "validation": sorted(record.validation_indices.tolist()),
            }
            for record in records
        ],
        sort_keys=True,
    )
    plan_hash = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return SplitPlan(
        n_subjects=len(target),
        n_splits=int(n_splits),
        n_repeats=max(1, int(n_repeats)),
        seed=int(seed),
        records=tuple(records),
        plan_hash=plan_hash,
    )


def split_plan_payload(
    plan: SplitPlan, *, subject_ids: Sequence[str], y: Sequence[int], salt: str
) -> dict[str, Any]:
    """Serializable registry that stores hashed subject ids only."""

    subjects = np.asarray([str(value) for value in subject_ids], dtype=str)
    target = np.asarray(y, dtype=np.int64)
    return {
        "plan_hash": plan.plan_hash,
        "n_subjects": plan.n_subjects,
        "n_splits": plan.n_splits,
        "n_repeats": plan.n_repeats,
        "seed": plan.seed,
        "label_definition": {"negative": "CN", "positive": "MCI or Dem"},
        "records": [
            {
                "split_id": record.split_id,
                "repeat": record.repeat,
                "fold": record.fold,
                "train_subject_hashes": [
                    hash_subject_id(value, salt=salt) for value in subjects[record.train_indices]
                ],
                "validation_subject_hashes": [
                    hash_subject_id(value, salt=salt)
                    for value in subjects[record.validation_indices]
                ],
                "validation_class_counts": {
                    "negative": int(np.sum(target[record.validation_indices] == 0)),
                    "positive": int(np.sum(target[record.validation_indices] == 1)),
                },
            }
            for record in plan.records
        ],
    }


def save_split_plan(
    path: str | Path,
    plan: SplitPlan,
    *,
    subject_ids: Sequence[str],
    y: Sequence[int],
    salt: str,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            split_plan_payload(plan, subject_ids=subject_ids, y=y, salt=salt),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return destination
