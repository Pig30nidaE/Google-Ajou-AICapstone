"""Fail-closed leakage checks, run before any model is fitted.

Every check in the spec's §12 list lives here.  The design rule is that a check
returns a *record*, and :meth:`AuditLog.raise_if_failed` turns records into an
exception -- so the engine can log everything it found and still refuse to
continue.

Two checks deserve a note on what they can and cannot prove:

* ``scaler_fitted_on_train_only`` compares fingerprints, so it detects a scaler
  fitted on the wrong ``SequenceSet``.  It cannot detect a scaler fitted on a
  hand-built array, which is why ``SequenceScaler.fit`` is the only fit path.
* ``sequence_length_not_chosen_on_test`` is a provenance check: it verifies the
  config declares where the length came from.  Whether a human peeked at a test
  score is outside any code's reach, which is precisely why it is written down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..data import schema
from ..preprocessing.scaler import SequenceScaler, fingerprint
from ..sequences.builder import SequenceSet


class LeakageError(AssertionError):
    """Raised when a leakage contract is violated."""


@dataclass
class AuditLog:
    context: str
    records: list[dict[str, Any]] = field(default_factory=list)

    def add(self, check: str, passed: bool, detail: Any = None, *, severity: str = "fatal") -> bool:
        self.records.append(
            {"check": check, "passed": bool(passed), "detail": detail, "severity": severity}
        )
        return bool(passed)

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [r for r in self.records if not r["passed"] and r["severity"] == "fatal"]

    @property
    def warnings(self) -> list[dict[str, Any]]:
        return [r for r in self.records if not r["passed"] and r["severity"] == "warning"]

    @property
    def passed(self) -> bool:
        return not self.failures

    def raise_if_failed(self) -> None:
        if self.failures:
            lines = [f"- {r['check']}: {r['detail']}" for r in self.failures]
            raise LeakageError(
                f"[{self.context}] {len(self.failures)} leakage check(s) failed:\n"
                + "\n".join(lines)
            )

    def summary(self) -> dict[str, Any]:
        return {
            "context": self.context,
            "checks": self.records,
            "n_checks": len(self.records),
            "n_failures": len(self.failures),
            "n_warnings": len(self.warnings),
            "all_passed": self.passed,
        }


# --- dataset level ------------------------------------------------------------

def audit_dataset(data: Any) -> AuditLog:
    """Checks that must hold before any split is drawn."""
    log = AuditLog("dataset")
    daily, subjects = data.daily, data.subjects

    log.add(
        "one_row_per_subject_day",
        not daily.duplicated([schema.SUBJECT_ID, schema.DATE_COL]).any(),
        f"{int(daily.duplicated([schema.SUBJECT_ID, schema.DATE_COL]).sum())} duplicate subject-days",
    )
    log.add(
        "raw_row_id_unique",
        daily[schema.RAW_ROW_ID].is_unique,
        "raw_row_id must uniquely identify a daily record",
    )
    log.add(
        "every_subject_labelled",
        set(daily[schema.SUBJECT_ID]) == set(subjects[schema.SUBJECT_ID]),
        "daily table and subject table disagree on the subject set",
    )
    log.add(
        "label_constant_within_subject",
        int(daily.groupby(schema.SUBJECT_ID)[schema.LABEL_COL].nunique().max()) == 1,
        "a subject changes label between days",
    )
    log.add(
        "no_forbidden_features",
        *_forbidden_check(data.feature_columns),
    )
    log.add(
        "feature_count_matches_paper",
        len(data.feature_columns) == 32,
        f"{len(data.feature_columns)} features, paper Table 4 lists 32",
    )
    log.add(
        "features_are_finite",
        bool(np.isfinite(daily[list(data.feature_columns)].to_numpy(dtype=float)).all()),
        "non-finite feature values would silently poison the scaler",
    )
    return log


def _forbidden_check(feature_columns: Iterable[str]) -> tuple[bool, Any]:
    offenders = [c for c in feature_columns if schema.is_forbidden_feature(c)]
    return (not offenders), (f"forbidden feature columns present: {offenders}" if offenders else None)


def assert_no_forbidden_features(feature_columns: Iterable[str]) -> None:
    passed, detail = _forbidden_check(feature_columns)
    if not passed:
        raise LeakageError(detail)


# --- split level --------------------------------------------------------------

def audit_sequence_split(
    train: SequenceSet,
    test: SequenceSet,
    *,
    context: str,
    estimand: str,
    validation: SequenceSet | None = None,
    scaler: SequenceScaler | None = None,
    scaler_fit_source: SequenceSet | None = None,
    sampling_report: dict[str, Any] | None = None,
    sequence_length_source: str | None = None,
    hyperparameter_source: str | None = None,
    early_stopping_source: str | None = None,
    expect_subject_overlap: bool = False,
    allow_boundary_crossing: bool = False,
) -> AuditLog:
    """Run every §12 check on one already-built train/test pair."""
    log = AuditLog(context)

    train_subjects = set(train.subjects.tolist())
    test_subjects = set(test.subjects.tolist())
    subject_overlap = train_subjects & test_subjects

    # 1. subject overlap -- required to be absent unless the estimand says otherwise
    if expect_subject_overlap:
        log.add(
            "subject_overlap_is_declared",
            estimand == "A",
            f"{len(subject_overlap)} subjects on both sides; only valid for estimand A",
        )
    else:
        log.add(
            "no_subject_overlap",
            not subject_overlap,
            f"{len(subject_overlap)} subjects in both train and test: "
            f"{sorted(subject_overlap)[:3]}",
        )

    # 2-3. raw row and (subject, date) overlap -- fatal in every estimand
    row_overlap = train.raw_row_ids() & test.raw_row_ids()
    log.add(
        "no_raw_row_overlap",
        not row_overlap,
        f"{len(row_overlap)} raw daily records appear in both splits",
    )
    date_overlap = train.subject_date_pairs() & test.subject_date_pairs()
    log.add(
        "no_shared_subject_dates",
        not date_overlap,
        f"{len(date_overlap)} (subject, date) pairs appear in both splits: "
        f"{sorted(map(str, date_overlap))[:3]}",
    )

    # 4. windows that quietly bridge a calendar gap
    for name, sequences in (("train", train), ("test", test)):
        n_gap = int((~sequences.provenance["is_calendar_consecutive"]).sum()) if len(sequences) else 0
        log.add(
            f"{name}_sequences_are_calendar_consecutive",
            n_gap == 0 or allow_boundary_crossing,
            f"{n_gap} {name} sequences span a calendar gap",
            severity="fatal" if not allow_boundary_crossing else "warning",
        )

    # 5. near-duplicate windows across the split
    duplicate = _duplicate_windows(train, test)
    log.add(
        "no_identical_windows_across_split",
        duplicate == 0,
        f"{duplicate} sequences with identical feature content exist in both splits",
    )

    # 6. the scaler saw training data and nothing else
    if scaler is not None:
        source = scaler_fit_source if scaler_fit_source is not None else train
        log.add(
            "scaler_fitted_on_train_only",
            scaler.fitted_on == fingerprint(source),
            f"scaler fingerprint {scaler.fitted_on} != training fingerprint "
            f"{fingerprint(source)}",
        )

    # 7. sampling touched training only
    if sampling_report is not None:
        log.add(
            "undersampling_applied_to_train_only",
            sampling_report.get("split_applied_to", "train") == "train",
            f"undersampling was applied to {sampling_report.get('split_applied_to')!r}",
        )

    # 8-10. selection provenance
    log.add(
        "sequence_length_not_chosen_on_test",
        sequence_length_source in {"paper_reported", "config_fixed", "inner_cv"},
        f"sequence length provenance is {sequence_length_source!r}; it must not be "
        "'test' or unset",
    )
    log.add(
        "hyperparameters_not_chosen_on_test",
        hyperparameter_source in {"paper_reported", "config_fixed", "inner_cv"},
        f"hyperparameter provenance is {hyperparameter_source!r}",
    )
    log.add(
        "early_stopping_not_monitored_on_test",
        early_stopping_source in {None, "none", "train_holdout", "inner_cv", "validation_period"},
        f"early stopping monitored {early_stopping_source!r}; the outer test set "
        "must never be the monitor",
    )

    # 11-12. features
    log.add("no_forbidden_features", *_forbidden_check(train.feature_columns))
    log.add(
        "subject_id_not_a_feature",
        schema.SUBJECT_ID not in train.feature_columns,
        "subject_id must not be an input variable",
    )

    if validation is not None and len(validation):
        validation_subjects = set(validation.subjects.tolist())
        log.add(
            "validation_disjoint_from_test",
            not (validation_subjects & test_subjects) if not expect_subject_overlap else True,
            "validation and test share subjects",
        )
        log.add(
            "validation_dates_disjoint_from_test",
            not (validation.subject_date_pairs() & test.subject_date_pairs()),
            "validation and test share (subject, date) pairs",
        )
    return log


def _duplicate_windows(train: SequenceSet, test: SequenceSet, *, max_check: int = 20000) -> int:
    """Count byte-identical feature windows appearing on both sides.

    Hashing the raw float bytes catches the case where two different (subject,
    date) windows still carry the same numbers -- which a date check alone misses.
    """
    if not len(train) or not len(test):
        return 0
    if len(train) > max_check or len(test) > max_check:
        rng = np.random.default_rng(0)
        train_idx = rng.choice(len(train), size=min(len(train), max_check), replace=False)
        test_idx = rng.choice(len(test), size=min(len(test), max_check), replace=False)
        left = train.X[train_idx]
        right = test.X[test_idx]
    else:
        left, right = train.X, test.X
    train_hashes = {hash(row.tobytes()) for row in left}
    return sum(1 for row in right if hash(row.tobytes()) in train_hashes)


# --- nested-specific ----------------------------------------------------------

def audit_outer_test_isolation(
    outer_test_subjects: Iterable[str],
    *,
    inner_splits: Iterable[Any],
    selection_scores_source: str,
    context: str = "nested",
) -> AuditLog:
    """No outer-test subject may appear anywhere in the inner selection loop."""
    log = AuditLog(context)
    outer = set(outer_test_subjects)
    leaked: set[str] = set()
    for split in inner_splits:
        leaked |= outer & set(split.train_subjects)
        leaked |= outer & set(split.test_subjects)
    log.add(
        "outer_test_absent_from_inner_cv",
        not leaked,
        f"{len(leaked)} outer-test subjects reached the inner CV: {sorted(leaked)[:3]}",
    )
    log.add(
        "selection_scores_come_from_inner_cv",
        selection_scores_source == "inner_cv",
        f"model selection used {selection_scores_source!r} scores; nested selection "
        "must use inner-CV scores only",
    )
    return log


def audit_temporal_split(split: Any, *, sequence_length: int) -> AuditLog:
    """Day-level checks for the within-subject temporal split."""
    log = AuditLog(f"temporal_split(L={sequence_length})")

    def pairs(frame: pd.DataFrame | None) -> set[tuple[str, pd.Timestamp]]:
        if frame is None or not len(frame):
            return set()
        return set(zip(frame[schema.SUBJECT_ID], frame[schema.DATE_COL]))

    train, test = pairs(split.train_days), pairs(split.test_days)
    log.add("train_test_days_disjoint", not (train & test),
            f"{len(train & test)} (subject, date) pairs on both sides")

    if split.validation_days is not None:
        validation = pairs(split.validation_days)
        log.add("validation_days_disjoint", not (validation & (train | test)),
                f"{len(validation & (train | test))} validation days overlap train/test")

    log.add(
        "test_is_strictly_later_than_train",
        *_ordering_check(split),
    )
    log.add(
        "embargo_covers_window",
        split.embargo_days == 0 or split.embargo_days >= sequence_length - 1,
        f"embargo of {split.embargo_days} days is shorter than the {sequence_length - 1} "
        "days a window reaches back",
        severity="warning",
    )
    return log


def _ordering_check(split: Any) -> tuple[bool, Any]:
    offenders = []
    train_max = split.train_days.groupby(schema.SUBJECT_ID)[schema.DATE_COL].max()
    test_min = split.test_days.groupby(schema.SUBJECT_ID)[schema.DATE_COL].min()
    for subject, latest in train_max.items():
        earliest = test_min.get(subject)
        if earliest is not None and latest >= earliest:
            offenders.append(subject)
    return (not offenders), f"{len(offenders)} subjects have a train day at or after their first test day"
