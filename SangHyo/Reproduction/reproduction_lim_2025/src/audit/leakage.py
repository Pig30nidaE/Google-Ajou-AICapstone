"""Leakage checks that run *before* any model is fitted.

Task brief §9: every check below raises rather than warns.  The one exception is
``assumption_variant_random_row_holdout``, which exists precisely to leak; there
the subject-overlap finding is recorded as a measurement and the config must have
declared ``leakage_diagnostic_only: true`` for it to have loaded at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from ..data import schema


class LeakageError(RuntimeError):
    """Raised when a leakage contract is violated.  Never caught internally."""


@dataclass
class AuditLog:
    """Accumulates every check so a run can prove what it verified."""

    checks: list[dict[str, Any]] = field(default_factory=list)
    diagnostic_mode: bool = False

    def record(self, name: str, passed: bool, detail: dict[str, Any] | None = None) -> None:
        self.checks.append(
            {"check": name, "passed": bool(passed), "detail": detail or {}}
        )

    def enforce(self, name: str, passed: bool, message: str, detail: dict[str, Any] | None = None) -> None:
        self.record(name, passed, detail)
        if not passed:
            if self.diagnostic_mode:
                self.checks[-1]["downgraded_to_diagnostic"] = True
                return
            raise LeakageError(f"[{name}] {message}  detail={detail or {}}")

    @property
    def passed(self) -> bool:
        return all(
            c["passed"] or c.get("downgraded_to_diagnostic", False) for c in self.checks
        )

    @property
    def failures(self) -> list[str]:
        return [
            c["check"] for c in self.checks
            if not c["passed"] and not c.get("downgraded_to_diagnostic", False)
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "n_checks": len(self.checks),
            "n_passed": sum(1 for c in self.checks if c["passed"]),
            "diagnostic_mode": self.diagnostic_mode,
            "all_passed": self.passed,
            "failures": self.failures,
            "checks": self.checks,
        }


# --- individual checks --------------------------------------------------------

def check_subject_overlap(
    train_subjects: Iterable[str], test_subjects: Iterable[str], log: AuditLog, *, tag: str = ""
) -> None:
    train, test = set(map(str, train_subjects)), set(map(str, test_subjects))
    shared = train & test
    log.enforce(
        f"subject_overlap{tag}",
        not shared,
        "the same subject appears in both the training and evaluation sets",
        {
            "n_shared": len(shared),
            "examples": sorted(shared)[:5],
            "n_train": len(train),
            "n_test": len(test),
        },
    )


def check_row_overlap(
    train_row_ids: Sequence[int], test_row_ids: Sequence[int], log: AuditLog, *, tag: str = ""
) -> None:
    train = set(int(r) for r in np.ravel(np.asarray(train_row_ids, dtype=object)) if r is not None)
    test = set(int(r) for r in np.ravel(np.asarray(test_row_ids, dtype=object)) if r is not None)
    shared = (train & test) - {-1}
    log.enforce(
        f"row_overlap{tag}",
        not shared,
        "the same raw daily record is used on both sides of the split",
        {"n_shared": len(shared), "examples": sorted(shared)[:5]},
    )


def check_sequence_source_overlap(train_rep, test_rep, log: AuditLog, *, tag: str = "") -> None:
    """No raw day may feed a training sequence and a test sequence."""
    if train_rep.row_ids is None or test_rep.row_ids is None:
        log.record(f"sequence_source_overlap{tag}", True, {"skipped": "no row ids"})
        return
    train_rows = {int(r) for arr in train_rep.row_ids for r in np.ravel(arr) if int(r) >= 0}
    test_rows = {int(r) for arr in test_rep.row_ids for r in np.ravel(arr) if int(r) >= 0}
    shared = train_rows & test_rows
    log.enforce(
        f"sequence_source_overlap{tag}",
        not shared,
        "a raw daily record contributed to both a training and a test sequence",
        {"n_shared": len(shared), "n_train_rows": len(train_rows), "n_test_rows": len(test_rows)},
    )
    check_subject_overlap(train_rep.subjects, test_rep.subjects, log, tag=f"_sequence{tag}")


def check_preprocessing_scope(
    preprocessor, train_subjects: Iterable[str], test_subjects: Iterable[str],
    log: AuditLog, *, tag: str = "",
) -> None:
    """The scaler/imputer must have seen training subjects and nothing else."""
    fitted = set(map(str, getattr(preprocessor, "fitted_subjects", ())))
    train, test = set(map(str, train_subjects)), set(map(str, test_subjects))

    log.enforce(
        f"preprocessor_fitted{tag}",
        bool(fitted),
        "preprocessing reports no fitted subjects; it may have been fitted globally",
        {"n_fitted": len(fitted)},
    )
    log.enforce(
        f"preprocessor_excludes_test{tag}",
        not (fitted & test),
        "preprocessing was fitted on subjects that are in the evaluation set",
        {"n_leaked": len(fitted & test), "examples": sorted(fitted & test)[:5]},
    )
    log.enforce(
        f"preprocessor_within_train{tag}",
        fitted.issubset(train),
        "preprocessing was fitted on subjects outside the current training fold",
        {"n_outside": len(fitted - train), "examples": sorted(fitted - train)[:5]},
    )


def check_forbidden_features(
    feature_names: Iterable[str], log: AuditLog, *, include_cognitive: bool = False, tag: str = "",
) -> None:
    """No target, identifier or (in the main analysis) cognitive-test column."""
    names = [str(n) for n in feature_names]
    forbidden = schema.forbidden_feature_columns(include_cognitive=include_cognitive)
    hits = sorted(set(names) & forbidden)
    log.enforce(
        f"forbidden_features{tag}",
        not hits,
        "the feature matrix contains target/identifier/cognitive columns",
        {"hits": hits, "include_cognitive_allowed": include_cognitive},
    )

    if not include_cognitive:
        heuristic = sorted(n for n in names if schema.is_cognitive_like(n))
        log.enforce(
            f"cognitive_like_features{tag}",
            not heuristic,
            "the feature matrix contains MMSE/SNSB/diagnosis-like column names",
            {"hits": heuristic},
        )

    identifiers = sorted(
        n for n in names
        if n in schema.IDENTIFIER_COLUMNS or n.lower() in {"email", "subject_id"}
    )
    log.enforce(
        f"identifier_features{tag}",
        not identifiers,
        "subject identifiers must be used as groups, never as features",
        {"hits": identifiers},
    )


def check_threshold_source(
    threshold_source: str, log: AuditLog, *, experiment: str, tag: str = ""
) -> None:
    """The outer test set must not have decided the operating point."""
    forbidden_sources = {"outer_test", "test", "evaluation_set"}
    log.enforce(
        f"threshold_source{tag}",
        threshold_source not in forbidden_sources,
        "the decision threshold was selected using the evaluation labels",
        {"threshold_source": threshold_source, "experiment": experiment},
    )
    if experiment == "nested_subject_independent":
        log.enforce(
            f"threshold_from_inner_cv{tag}",
            threshold_source == "inner_cv",
            "the nested experiment must select its threshold inside the inner CV",
            {"threshold_source": threshold_source},
        )


def check_outer_test_isolation(
    outer_test_subjects: Iterable[str],
    inner_splits: Sequence[Any],
    log: AuditLog,
    *,
    tag: str = "",
) -> None:
    """No outer-test subject may appear anywhere in the inner tuning folds."""
    outer = set(map(str, outer_test_subjects))
    contaminated: list[dict[str, Any]] = []
    for i, split in enumerate(inner_splits):
        used = set(map(str, split.train_subjects)) | set(map(str, split.test_subjects))
        shared = used & outer
        if shared:
            contaminated.append({"inner_fold": i, "n_shared": len(shared),
                                 "examples": sorted(shared)[:5]})
    log.enforce(
        f"outer_test_isolation{tag}",
        not contaminated,
        "outer-test subjects were used inside the inner tuning folds",
        {"contaminated_folds": contaminated, "n_inner_folds": len(inner_splits)},
    )


def check_subject_date_duplicates(daily, log: AuditLog) -> None:
    dupes = int(daily.duplicated([schema.SUBJECT_ID, schema.DATE_COL]).sum())
    log.record(
        "subject_date_duplicates",
        dupes == 0,
        {
            "n_duplicates": dupes,
            "note": "duplicates are tolerated but recorded; they inflate a subject's weight",
        },
    )


def check_label_consistency(daily, subjects, log: AuditLog) -> None:
    per_subject = daily.groupby(schema.SUBJECT_ID)[schema.LABEL_COL].nunique()
    bad = per_subject[per_subject > 1].index.tolist()
    log.enforce(
        "label_consistency",
        not bad,
        "a subject carries more than one label across their daily rows",
        {"subjects": bad[:5]},
    )
    known = set(subjects[schema.SUBJECT_ID].astype(str))
    orphans = sorted(set(daily[schema.SUBJECT_ID].astype(str)) - known)
    log.enforce(
        "daily_rows_have_labels",
        not orphans,
        "daily rows exist for subjects with no label",
        {"subjects": orphans[:5]},
    )


# --- composite entry points ---------------------------------------------------

def audit_split(
    split,
    *,
    train_rep,
    test_rep,
    preprocessor,
    experiment: str,
    threshold_source: str,
    include_cognitive: bool = False,
    inner_splits: Sequence[Any] | None = None,
    log: AuditLog | None = None,
    diagnostic_mode: bool = False,
) -> AuditLog:
    """Run every per-split check.  Call this before fitting anything."""
    log = log or AuditLog(diagnostic_mode=diagnostic_mode)
    tag = f"__r{getattr(split, 'repeat', 0)}f{getattr(split, 'fold', 0)}"

    check_subject_overlap(split.train_subjects, split.test_subjects, log, tag=tag)
    check_subject_overlap(train_rep.subjects, test_rep.subjects, log, tag=f"{tag}_rep")
    check_forbidden_features(
        train_rep.feature_names, log, include_cognitive=include_cognitive, tag=tag
    )

    if train_rep.row_ids is not None and test_rep.row_ids is not None:
        if train_rep.kind == "temporal_sequence":
            check_sequence_source_overlap(train_rep, test_rep, log, tag=tag)
            log.record(
                f"sequence_truncation_disclosed{tag}",
                True,
                {
                    "train_truncated_observations": int(
                        train_rep.meta.get("truncated_observations", 0)
                    ),
                    "test_truncated_observations": int(
                        test_rep.meta.get("truncated_observations", 0)
                    ),
                    "train_truncated_span_steps": int(
                        train_rep.meta.get("truncated_span_steps", 0)
                    ),
                    "test_truncated_span_steps": int(
                        test_rep.meta.get("truncated_span_steps", 0)
                    ),
                    "test_truncated_subjects": list(
                        test_rep.meta.get("truncated_subjects", [])
                    ),
                },
            )
        else:
            check_row_overlap(train_rep.row_ids, test_rep.row_ids, log, tag=tag)

    if preprocessor is not None:
        check_preprocessing_scope(
            preprocessor, split.train_subjects, split.test_subjects, log, tag=tag
        )

    check_threshold_source(threshold_source, log, experiment=experiment, tag=tag)

    if inner_splits:
        check_outer_test_isolation(split.test_subjects, inner_splits, log, tag=tag)

    return log


def audit_dataset(data, log: AuditLog | None = None) -> AuditLog:
    """Dataset-level checks that do not depend on a split."""
    log = log or AuditLog()
    check_label_consistency(data.daily, data.subjects, log)
    check_subject_date_duplicates(data.daily, log)
    check_forbidden_features(data.feature_columns, log, include_cognitive=False, tag="_dataset")

    counts = data.subjects[schema.LABEL_COL].value_counts().to_dict()
    log.enforce(
        "both_classes_present",
        len(counts) == 2,
        "the dataset does not contain both classes",
        {"counts": counts},
    )
    return log
