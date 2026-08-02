"""Simple undersampling of the majority class, training sequences only.

The paper says only "we performed simple undersampling to balance the data
distribution across classes in the training set" (§4.2).  It does not say whether
sampling was done over sequences or over subjects, nor with what seed, so both
strategies are implemented and the choice is assumption A-07.

Undersampling sequences at random is not harmless on this cohort: a subject
contributes between 2 and 118 five-day sequences, so a naive draw can delete a
whole subject from the majority class.  Every call therefore returns a diagnostic
report, and the engine writes it out.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..data import schema
from ..sequences.builder import SequenceSet

STRATEGIES = ("random_sequence", "subject_balanced", "none")


def undersample(
    train: SequenceSet,
    *,
    strategy: str = "random_sequence",
    target_ratio: float = 1.0,
    seed: int = 42,
) -> tuple[SequenceSet, dict[str, Any]]:
    """Downsample the majority class in *train*.

    Parameters
    ----------
    strategy:
        ``random_sequence`` draws majority sequences uniformly at random.
        ``subject_balanced`` draws the same *proportion* from each majority
        subject, so no subject disappears and the per-subject shape of the
        majority class is preserved.
        ``none`` returns the input unchanged (use with ``class_weight`` instead).
    target_ratio:
        Majority-to-minority sequence ratio to aim for.  ``1.0`` is balanced.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if train.split_name.startswith(("test", "outer_test", "literal_test", "validation")):
        raise AssertionError(
            f"undersampling was called on split {train.split_name!r}; it is only "
            "ever allowed on a training split"
        )

    before = _class_report(train)
    if strategy == "none" or not len(train):
        return train, {
            "strategy": strategy,
            "applied": False,
            "target_ratio": target_ratio,
            "before": before,
            "after": before,
            "n_removed": 0,
            "warnings": [],
        }

    rng = np.random.default_rng(seed)
    y = train.y
    counts = np.bincount(y, minlength=2)
    minority_class = int(np.argmin(counts))
    majority_class = 1 - minority_class
    n_minority = int(counts[minority_class])
    n_keep = int(round(n_minority * float(target_ratio)))

    majority_idx = np.flatnonzero(y == majority_class)
    minority_idx = np.flatnonzero(y == minority_class)

    if n_keep >= len(majority_idx):
        keep_majority = majority_idx
    elif strategy == "random_sequence":
        keep_majority = rng.choice(majority_idx, size=n_keep, replace=False)
    else:
        keep_majority = _subject_balanced_choice(train, majority_idx, n_keep, rng)

    keep = np.sort(np.concatenate([minority_idx, keep_majority]))
    mask = np.zeros(len(train), dtype=bool)
    mask[keep] = True
    sampled = train.select(mask)

    after = _class_report(sampled)
    warnings = _sampling_warnings(train, sampled, majority_class)
    return sampled, {
        "strategy": strategy,
        "applied": True,
        "target_ratio": target_ratio,
        "seed": seed,
        "minority_class": minority_class,
        "majority_class": majority_class,
        "before": before,
        "after": after,
        "n_removed": len(train) - len(sampled),
        "warnings": warnings,
    }


def _subject_balanced_choice(
    train: SequenceSet, majority_idx: np.ndarray, n_keep: int, rng: np.random.Generator
) -> np.ndarray:
    """Keep the same fraction of every majority subject's sequences.

    Largest-remainder allocation, with a floor of one sequence per subject so no
    majority subject is deleted outright.
    """
    subjects = train.provenance[schema.SUBJECT_ID].to_numpy()[majority_idx]
    unique = pd.unique(subjects)
    per_subject = {s: majority_idx[subjects == s] for s in unique}

    if n_keep <= len(unique):
        chosen = [rng.choice(per_subject[s], size=1, replace=False) for s in unique]
        picked = np.concatenate(chosen)
        return rng.choice(picked, size=max(1, n_keep), replace=False)

    total = len(majority_idx)
    exact = {s: len(idx) * n_keep / total for s, idx in per_subject.items()}
    allocation = {s: max(1, int(np.floor(v))) for s, v in exact.items()}

    # Hand out or claw back the rounding remainder, largest fractional part first.
    remaining = n_keep - sum(allocation.values())
    order = sorted(unique, key=lambda s: exact[s] - np.floor(exact[s]), reverse=True)
    i = 0
    while remaining > 0 and order:
        subject = order[i % len(order)]
        if allocation[subject] < len(per_subject[subject]):
            allocation[subject] += 1
            remaining -= 1
        i += 1
        if i > 100 * len(order):
            break
    while remaining < 0:
        subject = max(allocation, key=lambda s: allocation[s])
        if allocation[subject] <= 1:
            break
        allocation[subject] -= 1
        remaining += 1

    chosen = [
        rng.choice(per_subject[s], size=min(allocation[s], len(per_subject[s])), replace=False)
        for s in unique
    ]
    return np.concatenate(chosen)


def _class_report(sequences: SequenceSet) -> dict[str, Any]:
    if not len(sequences):
        return {"n_sequences": 0, "n_positive": 0, "n_negative": 0,
                "n_subjects": 0, "n_positive_subjects": 0, "n_negative_subjects": 0}
    frame = sequences.provenance
    by_subject = frame.groupby(schema.SUBJECT_ID)["y"].first()
    return {
        "n_sequences": len(sequences),
        "n_positive": int((sequences.y == 1).sum()),
        "n_negative": int((sequences.y == 0).sum()),
        "n_subjects": int(frame[schema.SUBJECT_ID].nunique()),
        "n_positive_subjects": int((by_subject == 1).sum()),
        "n_negative_subjects": int((by_subject == 0).sum()),
        "sequences_per_subject_max": int(frame.groupby(schema.SUBJECT_ID).size().max()),
    }


def _sampling_warnings(
    before: SequenceSet, after: SequenceSet, majority_class: int
) -> list[str]:
    """The specific failure modes the spec asks to be checked for."""
    warnings: list[str] = []
    before_counts = before.provenance.groupby(schema.SUBJECT_ID).size()
    after_counts = after.provenance.groupby(schema.SUBJECT_ID).size()
    after_counts = after_counts.reindex(before_counts.index).fillna(0).astype(int)

    dropped = before_counts.index[after_counts == 0]
    if len(dropped):
        warnings.append(
            f"{len(dropped)}명의 피험자가 undersampling으로 학습에서 완전히 제거되었다."
        )

    survivors = after_counts[after_counts > 0]
    if len(survivors) > 1:
        share = survivors / survivors.sum()
        if float(share.max()) > 0.10:
            warnings.append(
                f"한 피험자가 sampling 후 학습 시퀀스의 {float(share.max()):.1%}를 "
                "차지한다. 소수 피험자에 과적합될 위험이 있다."
            )

    majority_before = int((before.y == majority_class).sum())
    majority_after = int((after.y == majority_class).sum())
    if majority_before and majority_after / majority_before < 0.4:
        warnings.append(
            f"다수 클래스 시퀀스의 {1 - majority_after / majority_before:.0%}가 "
            "제거되었다. class_weight 대안과 비교하는 것이 좋다."
        )
    return warnings


def class_weights(y: np.ndarray) -> dict[int, float]:
    """Balanced class weights, the alternative to throwing sequences away."""
    counts = np.bincount(y.astype(int), minlength=2)
    total = counts.sum()
    return {
        int(k): float(total / (2 * counts[k])) if counts[k] else 1.0 for k in range(2)
    }
