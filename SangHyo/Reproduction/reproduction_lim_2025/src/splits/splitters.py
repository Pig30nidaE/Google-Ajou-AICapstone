"""Subject-level splits.

Every splitter yields *subject ids*, never row indices.  Representations are built
after the split so that a sequence can never be assembled from days that straddle
the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from ..data import schema


@dataclass
class SubjectSplit:
    """One train/test partition expressed as subject ids."""

    train_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]
    repeat: int = 0
    fold: int = 0
    name: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        overlap = set(self.train_subjects) & set(self.test_subjects)
        if overlap:
            raise ValueError(
                f"split {self.name!r} puts {len(overlap)} subjects on both sides: "
                f"{sorted(overlap)[:5]}"
            )

    def describe(self, labels: pd.Series | None = None) -> dict[str, Any]:
        record: dict[str, Any] = {
            "name": self.name,
            "repeat": self.repeat,
            "fold": self.fold,
            "n_train_subjects": len(self.train_subjects),
            "n_test_subjects": len(self.test_subjects),
            **self.meta,
        }
        if labels is not None:
            for side, ids in (("train", self.train_subjects), ("test", self.test_subjects)):
                values = labels.reindex(list(ids)).dropna().astype(int)
                record[f"n_{side}_positive"] = int(values.sum())
                record[f"n_{side}_negative"] = int((values == 0).sum())
        return record


@dataclass
class RowSplit:
    """A row-level partition -- only ever produced by the leaky diagnostic variant."""

    train_rows: np.ndarray
    test_rows: np.ndarray
    name: str = "assumption_variant_random_row_holdout"
    meta: dict[str, Any] = field(default_factory=dict)


def official_partition(data) -> SubjectSplit:
    """AI-Hub's own Training(141) / Validation(33) partition.

    ``reproduction_spec.md`` §2 reconstructs this as the paper's "80:20" split:
    141:33 = 81:19, and every reported metric decomposes over 33 subjects with
    7 positives.
    """
    subjects = data.subjects
    train = subjects.loc[subjects["split_origin"] == "train", schema.SUBJECT_ID]
    test = subjects.loc[subjects["split_origin"] == "validation", schema.SUBJECT_ID]
    return SubjectSplit(
        train_subjects=tuple(sorted(train.astype(str))),
        test_subjects=tuple(sorted(test.astype(str))),
        name="official_partition",
        meta={
            "ratio": f"{len(train)}:{len(test)}",
            "paper_description": "80:20",
            "reconstruction_note": (
                "Accuracy values in the paper are all k/33 and recall values k/7, "
                "matching this partition exactly."
            ),
        },
    )


def random_subject_holdout(data, *, test_size: float = 0.2, seed: int = 42) -> SubjectSplit:
    """Stratified random subject holdout -- the literal reading of '80:20'."""
    labels = data.labels_by_subject()
    ids = np.asarray(labels.index, dtype=object)
    train_ids, test_ids = train_test_split(
        ids, test_size=test_size, random_state=seed, stratify=labels.to_numpy()
    )
    return SubjectSplit(
        train_subjects=tuple(sorted(map(str, train_ids))),
        test_subjects=tuple(sorted(map(str, test_ids))),
        name="assumption_variant_random_subject_holdout",
        meta={"test_size": test_size, "seed": seed},
    )


def random_row_holdout(data, *, test_size: float = 0.2, seed: int = 42) -> RowSplit:
    """Random split over daily rows.  **Leaks subjects across the boundary.**

    Kept only to quantify how large that leak is; the caller must have set
    ``split.leakage_diagnostic_only: true`` for the config to validate.
    """
    daily = data.daily
    indices = np.arange(len(daily))
    train_rows, test_rows = train_test_split(
        indices, test_size=test_size, random_state=seed,
        stratify=daily[schema.LABEL_COL].to_numpy(),
    )
    train_subjects = set(daily.iloc[train_rows][schema.SUBJECT_ID])
    test_subjects = set(daily.iloc[test_rows][schema.SUBJECT_ID])
    shared = train_subjects & test_subjects
    return RowSplit(
        train_rows=np.sort(train_rows),
        test_rows=np.sort(test_rows),
        meta={
            "test_size": test_size,
            "seed": seed,
            "n_shared_subjects": len(shared),
            "shared_subject_fraction": round(
                len(shared) / max(len(train_subjects | test_subjects), 1), 4
            ),
            "leakage_expected": True,
            "interpret_as": "leakage_diagnostic_not_performance",
        },
    )


def stratified_group_kfold(
    data, *, n_splits: int = 5, repeats: int = 1, seed: int = 42
) -> Iterator[SubjectSplit]:
    """Repeated subject-level StratifiedGroupKFold over all subjects."""
    labels = data.labels_by_subject()
    ids = np.asarray(labels.index, dtype=object)
    y = labels.to_numpy()

    for repeat in range(repeats):
        # Each subject is its own group, so grouping is what keeps a person whole;
        # shuffling the subject order is what makes the repeats differ.
        rng = np.random.default_rng(seed + repeat)
        order = rng.permutation(len(ids))
        ids_r, y_r = ids[order], y[order]
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed + repeat
        )
        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(np.zeros(len(ids_r)), y_r, groups=ids_r)
        ):
            yield SubjectSplit(
                train_subjects=tuple(sorted(map(str, ids_r[train_idx]))),
                test_subjects=tuple(sorted(map(str, ids_r[test_idx]))),
                repeat=repeat,
                fold=fold,
                name="stratified_group_kfold",
                meta={"n_splits": n_splits, "seed": seed + repeat},
            )


def inner_stratified_group_kfold(
    subjects: Sequence[str],
    labels: pd.Series,
    *,
    n_splits: int = 3,
    seed: int = 42,
) -> list[SubjectSplit]:
    """Inner folds over the *outer training* subjects only.

    The caller must pass the outer-training subject ids; passing anything else is
    what ``test_outer_test_isolation.py`` checks against.
    """
    ids = np.asarray(sorted(map(str, subjects)), dtype=object)
    y = labels.reindex(list(ids)).to_numpy()
    if np.isnan(y.astype(float)).any():
        raise ValueError("inner CV received subjects without labels")
    y = y.astype(int)

    n_positive = int(y.sum())
    n_negative = int(len(y) - n_positive)
    usable = max(2, min(n_splits, n_positive, n_negative))
    splitter = StratifiedGroupKFold(n_splits=usable, shuffle=True, random_state=seed)

    folds: list[SubjectSplit] = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(ids)), y, groups=ids)
    ):
        folds.append(
            SubjectSplit(
                train_subjects=tuple(sorted(map(str, ids[train_idx]))),
                test_subjects=tuple(sorted(map(str, ids[test_idx]))),
                fold=fold,
                name="inner_stratified_group_kfold",
                meta={"n_splits": usable, "requested_n_splits": n_splits, "seed": seed},
            )
        )
    return folds


def build_outer_splits(data, config) -> list[SubjectSplit]:
    """Materialise the outer splits for whichever experiment is configured."""
    mode = config.split_mode
    seed = config.seed

    if mode == "official_partition":
        return [official_partition(data)]
    if mode == "assumption_variant_random_subject_holdout":
        return [
            random_subject_holdout(
                data, test_size=float(config.get("split.test_size", 0.2)), seed=seed
            )
        ]
    if mode in ("stratified_group_kfold", "nested_stratified_group_kfold"):
        return list(
            stratified_group_kfold(
                data,
                n_splits=int(config.get("split.outer_k", 5)),
                repeats=int(config.get("split.repeats", 1)),
                seed=seed,
            )
        )
    if mode == "assumption_variant_random_row_holdout":
        raise ValueError(
            "assumption_variant_random_row_holdout is a row-level split; call "
            "random_row_holdout() directly and treat its output as a diagnostic"
        )
    raise ValueError(f"unknown split mode: {mode!r}")
