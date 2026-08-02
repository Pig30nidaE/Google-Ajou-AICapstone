"""Subject-independent splits: StratifiedGroupKFold over subject ids.

These answer Estimand B (a subject the model has never seen).  The group is
always the subject, so every one of a subject's days lands in exactly one fold,
and sequences are built after the subjects are divided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ..data import schema


@dataclass
class SubjectSplit:
    """One outer (or inner) fold, expressed purely as two disjoint subject sets."""

    train_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]
    repeat: int = 0
    fold: int = 0
    name: str = "group_kfold"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        overlap = set(self.train_subjects) & set(self.test_subjects)
        if overlap:
            raise AssertionError(
                f"{len(overlap)} subjects are in both sides of split "
                f"{self.name} r{self.repeat}f{self.fold}: {sorted(overlap)[:3]}"
            )

    def describe(self, labels: pd.Series | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "repeat": self.repeat,
            "fold": self.fold,
            "n_train_subjects": len(self.train_subjects),
            "n_test_subjects": len(self.test_subjects),
            **self.meta,
        }
        if labels is not None:
            train = labels.reindex(list(self.train_subjects))
            test = labels.reindex(list(self.test_subjects))
            out["n_train_positive"] = int(train.sum())
            out["n_test_positive"] = int(test.sum())
            out["test_prevalence"] = float(test.mean()) if len(test) else 0.0
        return out


def stratified_group_splits(
    subjects: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: int = 42,
    name: str = "group_kfold",
    stratify_on: str = "label",
) -> list[SubjectSplit]:
    """Repeated StratifiedGroupKFold over subjects.

    ``stratify_on='diagnosis'`` stratifies on CN/MCI/Dem instead of the binary
    label, which keeps the 12 dementia subjects from clustering into one fold.
    The binary label is the default because that is the paper's target.
    """
    frame = subjects.sort_values(schema.SUBJECT_ID).reset_index(drop=True)
    ids = frame[schema.SUBJECT_ID].to_numpy()
    if stratify_on == "diagnosis":
        strata = frame[schema.DIAGNOSIS_COL].to_numpy()
    elif stratify_on == "label":
        strata = frame[schema.LABEL_COL].to_numpy()
    else:
        raise ValueError("stratify_on must be 'label' or 'diagnosis'")

    splits: list[SubjectSplit] = []
    for repeat in range(n_repeats):
        # Each subject is its own group, so StratifiedGroupKFold degenerates to a
        # stratified split over subjects -- which is exactly what is wanted, and
        # it keeps the same call shape as the sequence-level guard tests.
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed + repeat
        )
        for fold, (train_idx, test_idx) in enumerate(splitter.split(ids, strata, groups=ids)):
            splits.append(
                SubjectSplit(
                    train_subjects=tuple(ids[train_idx]),
                    test_subjects=tuple(ids[test_idx]),
                    repeat=repeat,
                    fold=fold,
                    name=name,
                    meta={"estimand": "B", "stratify_on": stratify_on, "seed": seed + repeat},
                )
            )
    return splits


def inner_splits(
    subjects: pd.DataFrame,
    outer: SubjectSplit,
    *,
    n_splits: int = 3,
    seed: int = 42,
    stratify_on: str = "label",
) -> list[SubjectSplit]:
    """Inner folds drawn from the outer *train* subjects only.

    The outer test subjects are not passed in, so nothing selected in here can
    have seen them.
    """
    train_only = subjects[subjects[schema.SUBJECT_ID].isin(set(outer.train_subjects))]
    folds = stratified_group_splits(
        train_only,
        n_splits=n_splits,
        n_repeats=1,
        seed=seed + 1000 * (outer.repeat + 1) + outer.fold,
        name=f"inner(r{outer.repeat}f{outer.fold})",
        stratify_on=stratify_on,
    )
    for fold in folds:
        fold.meta["outer_repeat"] = outer.repeat
        fold.meta["outer_fold"] = outer.fold
        leaked = set(fold.train_subjects + fold.test_subjects) & set(outer.test_subjects)
        if leaked:
            raise AssertionError(
                f"{len(leaked)} outer-test subjects reached an inner fold; "
                "the inner CV must only ever see outer-train subjects"
            )
    return folds


def iter_days(daily: pd.DataFrame, subjects: tuple[str, ...]) -> pd.DataFrame:
    """All daily rows belonging to *subjects*, sorted for sequence building."""
    wanted = set(subjects)
    return (
        daily[daily[schema.SUBJECT_ID].isin(wanted)]
        .sort_values([schema.SUBJECT_ID, schema.DATE_COL])
        .reset_index(drop=True)
    )


def check_split_viability(
    splits: list[SubjectSplit], labels: pd.Series
) -> dict[str, Any]:
    """Every fold needs both classes in test, or its ROC-AUC is undefined."""
    per_fold = []
    for split in splits:
        test = labels.reindex(list(split.test_subjects))
        per_fold.append(
            {
                "repeat": split.repeat,
                "fold": split.fold,
                "n_test_positive": int(test.sum()),
                "n_test_negative": int((1 - test).sum()),
            }
        )
    min_pos = min(f["n_test_positive"] for f in per_fold) if per_fold else 0
    min_neg = min(f["n_test_negative"] for f in per_fold) if per_fold else 0
    return {
        "per_fold": per_fold,
        "min_test_positive": min_pos,
        "min_test_negative": min_neg,
        "viable": bool(min_pos >= 1 and min_neg >= 1),
    }


def repeated_seeds(base_seed: int, n_repeats: int) -> list[int]:
    return [int(base_seed) + i for i in range(int(n_repeats))]


def group_sizes(splits: list[SubjectSplit]) -> np.ndarray:
    return np.asarray([len(s.test_subjects) for s in splits], dtype=int)
