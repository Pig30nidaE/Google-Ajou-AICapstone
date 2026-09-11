"""Fold construction and subject-disjointness assertions.

R: record-level StratifiedKFold (A10).  G/N: StratifiedGroupKFold with group = subject_id.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def record_folds(y: np.ndarray, n_splits: int, seed: int, shuffle: bool = True) -> list[tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)
    return [(tr, te) for tr, te in skf.split(np.zeros((len(y), 1)), y)]


def subject_folds(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [(tr, te) for tr, te in sgkf.split(np.zeros((len(y), 1)), y, groups)]
    for tr, te in folds:
        assert_subject_disjoint(groups, tr, te)
    return folds


def assert_subject_disjoint(groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
    tr_s = set(np.asarray(groups)[train_idx].tolist())
    te_s = set(np.asarray(groups)[test_idx].tolist())
    overlap = tr_s & te_s
    if overlap:
        raise AssertionError(f"Subject overlap between train and test: {len(overlap)} subjects")
    if set(train_idx.tolist()) & set(test_idx.tolist()):
        raise AssertionError("Record index overlap between train and test")
    return {"n_train_subjects": len(tr_s), "n_test_subjects": len(te_s), "subject_overlap": 0,
            "n_train_records": int(len(train_idx)), "n_test_records": int(len(test_idx))}


def record_overlap_subjects(groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> int:
    """Number of subjects present in both sides (diagnostic for record-level splits; expected > 0 in R)."""
    return len(set(np.asarray(groups)[train_idx].tolist()) & set(np.asarray(groups)[test_idx].tolist()))
