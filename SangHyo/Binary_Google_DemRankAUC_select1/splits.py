"""Subject-level repeated stratified CV, with the leakage assertions inline.

Why stratified K-fold over subjects is already group-aware
----------------------------------------------------------
``features.py`` reduces every subject's 35-120 daily rows to **one row**, so a
row is a subject and a row-level split is a subject-level split.  The failure
mode documented in ``Binary_PaperLGBM_*`` -- randomly splitting *days*, which put
the same person on both sides and pushed AUC from 0.52 to 0.95 -- is structurally
impossible here.

That argument is not taken on trust: :func:`iter_splits` asserts subject-id
disjointness on every fold it yields, and :func:`assert_split_integrity` re-checks
the full plan.

Fold count
----------
There are 12 positives.  ``outer_k=5`` puts 2-3 Dem subjects in every test fold,
which is the most folds that still leaves each fold scoreable; ``safe_k`` caps
the request so a smaller cohort (the quality-filtered arm, or a smoke run) can
never silently produce a single-class fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class Fold:
    repeat: int
    fold: int
    train_index: np.ndarray
    test_index: np.ndarray

    @property
    def key(self) -> str:
        return f"r{self.repeat:02d}f{self.fold:02d}"


def safe_k(y: Sequence[int], requested: int, *, min_per_fold: int = 1) -> int:
    """Largest usable fold count: never more than ``minority // min_per_fold``."""

    y = np.asarray(y, dtype=np.int64)
    minority = int(np.bincount(y, minlength=2).min())
    allowed = max(2, minority // max(1, min_per_fold))
    return int(max(2, min(requested, allowed)))


def make_folds(y: Sequence[int], *, n_splits: int, n_repeats: int, seed: int,
               min_positives_per_fold: int = 2) -> list[Fold]:
    """Repeated stratified splits over subjects.

    Repeats -- not a single split -- are the unit of reporting.  With 12
    positives one 5-fold split moves by >0.05 AUC on the seed alone, so a single
    seed is not an estimate.
    """

    y = np.asarray(y, dtype=np.int64)
    k = safe_k(y, n_splits, min_per_fold=min_positives_per_fold)
    folds: list[Fold] = []
    for repeat in range(n_repeats):
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(seed) + repeat)
        for position, (train_index, test_index) in enumerate(splitter.split(np.zeros(len(y)), y)):
            folds.append(
                Fold(repeat=repeat, fold=position,
                     train_index=np.asarray(train_index, dtype=np.int64),
                     test_index=np.asarray(test_index, dtype=np.int64))
            )
    assert_split_integrity(folds, y, n_repeats=n_repeats)
    return folds


def iter_splits(folds: Sequence[Fold], subject_ids: Sequence[str]) -> Iterator[Fold]:
    """Yield folds after re-checking subject disjointness on each one."""

    subject_ids = np.asarray(subject_ids, dtype=str)
    for fold in folds:
        assert_no_subject_overlap(subject_ids[fold.train_index], subject_ids[fold.test_index])
        yield fold


def assert_no_subject_overlap(train_ids: Sequence[str], test_ids: Sequence[str]) -> None:
    overlap = sorted(set(map(str, train_ids)) & set(map(str, test_ids)))
    if overlap:
        raise AssertionError(
            f"Subject leakage: {len(overlap)} subject(s) in both train and test "
            f"(e.g. {overlap[:3]})"
        )


def assert_both_classes(y: Sequence[int], *, where: str) -> None:
    y = np.asarray(y, dtype=np.int64)
    present = np.unique(y)
    if len(present) < 2:
        raise AssertionError(f"{where}: single-class fold (only class {present.tolist()})")


def assert_split_integrity(folds: Sequence[Fold], y: Sequence[int], *, n_repeats: int) -> None:
    """Every contract a split plan has to satisfy, checked once up front."""

    y = np.asarray(y, dtype=np.int64)
    n = len(y)
    if not folds:
        raise AssertionError("Empty split plan")
    for fold in folds:
        if np.intersect1d(fold.train_index, fold.test_index).size:
            raise AssertionError(f"{fold.key}: train/test index overlap")
        assert_both_classes(y[fold.train_index], where=f"{fold.key} train")
        if int(y[fold.test_index].sum()) < 1:
            raise AssertionError(f"{fold.key}: test fold contains no Dem subject")
        if int((y[fold.test_index] == 0).sum()) < 1:
            raise AssertionError(f"{fold.key}: test fold contains no negative subject")
    for repeat in range(n_repeats):
        covered = np.concatenate([f.test_index for f in folds if f.repeat == repeat])
        if np.unique(covered).size != n or covered.size != n:
            raise AssertionError(
                f"repeat {repeat}: test folds must partition the cohort exactly once"
            )


def inner_folds(y_train: Sequence[int], *, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Inner splits used for tuning, blending and thresholds.

    Indices are relative to the outer-training block, so nothing here can address
    an outer-test row even by accident.
    """

    y_train = np.asarray(y_train, dtype=np.int64)
    k = safe_k(y_train, n_splits, min_per_fold=1)
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(seed))
    return [
        (np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64))
        for train, test in splitter.split(np.zeros(len(y_train)), y_train)
    ]


def split_summary(folds: Sequence[Fold], y: Sequence[int]) -> dict:
    y = np.asarray(y, dtype=np.int64)
    positives = [int(y[f.test_index].sum()) for f in folds]
    sizes = [int(f.test_index.size) for f in folds]
    return {
        "n_folds": len(folds),
        "n_repeats": len({f.repeat for f in folds}),
        "outer_k": len({f.fold for f in folds}),
        "test_positives_min": int(min(positives)),
        "test_positives_max": int(max(positives)),
        "test_size_min": int(min(sizes)),
        "test_size_max": int(max(sizes)),
    }


__all__ = [
    "Fold", "assert_both_classes", "assert_no_subject_overlap", "assert_split_integrity",
    "inner_folds", "iter_splits", "make_folds", "safe_k", "split_summary",
]
