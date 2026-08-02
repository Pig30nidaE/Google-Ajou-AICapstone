"""Fail-closed leakage assertions shared across the pipeline."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np


class LeakageError(RuntimeError):
    """Raised when an identity, target, fold, or preprocessing contract fails."""


FORBIDDEN_EXACT_FEATURES = frozenset(
    {
        "DIAG_NM",
        "DIAG_SEQ",
        "DOCTOR_NM",
        "MMSE_NUM",
        "MMSE_KIND",
        "EMAIL",
        "SAMPLE_EMAIL",
        "subject_id",
        "patient_id",
        "target",
        "label",
    }
)
FORBIDDEN_EXACT_FEATURES_CASEFOLD = frozenset(
    name.casefold() for name in FORBIDDEN_EXACT_FEATURES
)
FORBIDDEN_FEATURE_PATTERNS = (
    re.compile(r"(^|__)(diag(_nm|_seq)?|diagnosis)($|__)", re.IGNORECASE),
    re.compile(r"(^|__)dementia(_result|_label)?($|__)", re.IGNORECASE),
    re.compile(r"(^|__)target($|__)", re.IGNORECASE),
    re.compile(r"(^|__)label($|__)", re.IGNORECASE),
    re.compile(r"(^|__)(sample_)?email($|__)", re.IGNORECASE),
    re.compile(r"(^|__)subject(_id|_hash)?($|__)", re.IGNORECASE),
    re.compile(r"(^|__)patient(_id)?($|__)", re.IGNORECASE),
    re.compile(r"(^|__)doctor_nm($|__)", re.IGNORECASE),
    re.compile(r"(^|__)mmse_(num|kind)($|__)", re.IGNORECASE),
)


def hash_subject_id(subject_id: Any) -> str:
    """Stable non-reversible identifier used in persisted artifacts."""

    return hashlib.sha256(str(subject_id).encode("utf-8")).hexdigest()[:20]


def assert_no_forbidden_features(feature_names: Sequence[str]) -> None:
    bad: list[str] = []
    for value in feature_names:
        name = str(value)
        if name.casefold() in FORBIDDEN_EXACT_FEATURES_CASEFOLD or any(
            pattern.search(name) for pattern in FORBIDDEN_FEATURE_PATTERNS
        ):
            bad.append(name)
    if bad:
        raise LeakageError(f"Forbidden target/identity features: {sorted(set(bad))}")


def assert_unique_subjects(subject_ids: Sequence[Any]) -> None:
    normalized = [str(value).strip() for value in subject_ids]
    if any(not value or value.lower() in {"nan", "none", "null"} for value in normalized):
        raise LeakageError("Missing or invalid subject identifier")
    if len(set(normalized)) != len(normalized):
        raise LeakageError("Expected exactly one model row per subject")


def assert_group_label_consistency(
    groups: Sequence[Any], labels: Sequence[int]
) -> None:
    group_to_label: dict[str, int] = {}
    for group, raw_label in zip(groups, labels, strict=True):
        key = str(group)
        label = int(raw_label)
        if label not in {0, 1}:
            raise LeakageError(f"Non-binary target for {key!r}: {label}")
        previous = group_to_label.setdefault(key, label)
        if previous != label:
            raise LeakageError(f"Conflicting labels within subject {key!r}")


def assert_disjoint_groups(
    train_groups: Iterable[Any],
    validation_groups: Iterable[Any],
    *,
    context: str,
) -> None:
    overlap = set(map(str, train_groups)) & set(map(str, validation_groups))
    if overlap:
        examples = sorted(overlap)[:5]
        raise LeakageError(
            f"{context}: {len(overlap)} subject(s) cross the split; examples={examples}"
        )


def assert_binary_fold(
    train_y: Sequence[int],
    validation_y: Sequence[int],
    *,
    minimum_positive_validation: int,
    context: str,
) -> None:
    train = np.asarray(train_y, dtype=np.int64)
    validation = np.asarray(validation_y, dtype=np.int64)
    if set(np.unique(train)) != {0, 1}:
        raise LeakageError(f"{context}: training fold does not contain both classes")
    if set(np.unique(validation)) != {0, 1}:
        raise LeakageError(f"{context}: validation fold does not contain both classes")
    if int(validation.sum()) < int(minimum_positive_validation):
        raise LeakageError(
            f"{context}: validation fold has only {int(validation.sum())} positive(s)"
        )


def assert_fold_local_fit(
    fitted_subjects: Iterable[Any],
    scored_subjects: Iterable[Any],
    *,
    component: str,
) -> None:
    """Assert a fitted transformer/model never saw scored subjects."""

    assert_disjoint_groups(
        fitted_subjects,
        scored_subjects,
        context=f"fold-local {component}",
    )


def assert_prediction_coverage(
    counts: Sequence[int], *, expected_repeats: int, context: str
) -> None:
    values = np.asarray(counts, dtype=np.int64)
    if np.any(values != int(expected_repeats)):
        observed = sorted(set(values.tolist()))
        raise LeakageError(
            f"{context}: each subject needs {expected_repeats} OOF predictions; "
            f"observed counts={observed}"
        )


__all__ = [
    "LeakageError",
    "assert_binary_fold",
    "assert_disjoint_groups",
    "assert_fold_local_fit",
    "assert_group_label_consistency",
    "assert_no_forbidden_features",
    "assert_prediction_coverage",
    "assert_unique_subjects",
    "hash_subject_id",
]
