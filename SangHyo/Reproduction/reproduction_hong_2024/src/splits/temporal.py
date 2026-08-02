"""Within-subject temporal splits: the paper's "final week of each subject".

Every function here splits **raw days**, never sequences.  Sequences are built
afterwards, separately inside each side, which is what makes a boundary-crossing
window impossible rather than merely discouraged.

This split answers Estimand A (a known subject's later days).  It is not a
subject-independent split and must never be reported as one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..data import schema

FINAL_WEEK_MODES = ("calendar_days", "record_count")


@dataclass
class TemporalSplit:
    """A per-subject day-level split into train / (optional) validation / test."""

    train_days: pd.DataFrame
    test_days: pd.DataFrame
    validation_days: pd.DataFrame | None = None
    name: str = "paper_temporal"
    embargo_days: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        def block(frame: pd.DataFrame | None) -> dict[str, Any]:
            if frame is None or not len(frame):
                return {"n_rows": 0, "n_subjects": 0}
            return {
                "n_rows": int(len(frame)),
                "n_subjects": int(frame[schema.SUBJECT_ID].nunique()),
                "date_min": str(frame[schema.DATE_COL].min().date()),
                "date_max": str(frame[schema.DATE_COL].max().date()),
                "n_positive_subjects": int(
                    frame.groupby(schema.SUBJECT_ID)[schema.LABEL_COL].first().sum()
                ),
            }

        return {
            "name": self.name,
            "embargo_days": self.embargo_days,
            "train": block(self.train_days),
            "validation": block(self.validation_days),
            "test": block(self.test_days),
            **self.meta,
        }


def _cut_for_subject(
    group: pd.DataFrame, *, final_week_mode: str, final_week_length: int
) -> pd.Timestamp:
    """The first date that belongs to the subject's test period."""
    if final_week_mode == "calendar_days":
        return group[schema.DATE_COL].max() - pd.Timedelta(days=final_week_length - 1)
    if final_week_mode == "record_count":
        if len(group) <= final_week_length:
            return group[schema.DATE_COL].min()
        return group[schema.DATE_COL].iloc[-final_week_length]
    raise ValueError(f"final_week_mode must be one of {FINAL_WEEK_MODES}")


def first_test_dates(
    daily: pd.DataFrame,
    *,
    final_week_mode: str = "calendar_days",
    final_week_length: int = 7,
) -> dict[str, pd.Timestamp]:
    """Per-subject first test date, used by both the strict and literal builders."""
    out: dict[str, pd.Timestamp] = {}
    for subject, group in daily.groupby(schema.SUBJECT_ID, sort=True):
        group = group.sort_values(schema.DATE_COL)
        out[subject] = _cut_for_subject(
            group, final_week_mode=final_week_mode, final_week_length=final_week_length
        )
    return out


def final_week_split(
    daily: pd.DataFrame,
    *,
    final_week_mode: str = "calendar_days",
    final_week_length: int = 7,
    embargo_days: int = 0,
    validation_days: int = 0,
    name: str = "paper_temporal",
) -> TemporalSplit:
    """Hold out each subject's final week; optionally embargo and carve validation.

    Parameters
    ----------
    final_week_mode:
        ``calendar_days`` reads "final week" as the last 7 calendar dates;
        ``record_count`` reads it as the last 7 records.  The paper does not say
        which, and on this cohort they differ (assumption A-05), so both ship.
    embargo_days:
        Days dropped from the **end of train**, immediately before the cut.  Set
        this to ``sequence_length - 1`` and no train window can share a day with
        any test window even indirectly.
    validation_days:
        Days taken from the end of the remaining train period (after the embargo)
        for early stopping and model selection.  The paper mentions a validation
        loss and early stopping but never describes this period (A-06).
    """
    train_parts, test_parts, validation_parts = [], [], []
    embargoed_rows = 0

    for subject, group in daily.groupby(schema.SUBJECT_ID, sort=True):
        group = group.sort_values(schema.DATE_COL)
        cut = _cut_for_subject(
            group, final_week_mode=final_week_mode, final_week_length=final_week_length
        )
        test_parts.append(group[group[schema.DATE_COL] >= cut])

        before = group[group[schema.DATE_COL] < cut]
        if embargo_days > 0 and len(before):
            keep = before[schema.DATE_COL] < (cut - pd.Timedelta(days=embargo_days))
            embargoed_rows += int((~keep).sum())
            before = before[keep]

        if validation_days > 0 and len(before):
            validation_cut = before[schema.DATE_COL].max() - pd.Timedelta(days=validation_days - 1)
            validation_parts.append(before[before[schema.DATE_COL] >= validation_cut])
            before = before[before[schema.DATE_COL] < validation_cut]

        train_parts.append(before)

    def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
        frames = [p for p in parts if len(p)]
        if not frames:
            return daily.iloc[0:0].copy()
        return pd.concat(frames, ignore_index=True).sort_values(
            [schema.SUBJECT_ID, schema.DATE_COL]
        ).reset_index(drop=True)

    train = _concat(train_parts)
    test = _concat(test_parts)
    validation = _concat(validation_parts) if validation_days > 0 else None

    return TemporalSplit(
        train_days=train,
        test_days=test,
        validation_days=validation,
        name=name,
        embargo_days=embargo_days,
        meta={
            "final_week_mode": final_week_mode,
            "final_week_length": final_week_length,
            "validation_days": validation_days,
            "n_embargoed_rows": embargoed_rows,
            "estimand": "A",
            "subjects_appear_in_both_sides": True,
            "note": (
                "같은 피험자가 train과 test 양쪽에 존재한다. 이는 설계상 의도된 "
                "Estimand A이며, 신규 피험자 일반화 성능이 아니다."
            ),
        },
    )


def assert_no_shared_days(split: TemporalSplit) -> None:
    """Fail closed if any (subject, date) appears on both sides of the cut."""
    def pairs(frame: pd.DataFrame | None) -> set[tuple[str, pd.Timestamp]]:
        if frame is None or not len(frame):
            return set()
        return set(zip(frame[schema.SUBJECT_ID], frame[schema.DATE_COL]))

    train, test = pairs(split.train_days), pairs(split.test_days)
    shared = train & test
    if shared:
        raise AssertionError(
            f"{len(shared)} (subject, date) pairs are in both train and test, "
            f"e.g. {sorted(shared)[:3]}"
        )
    if split.validation_days is not None:
        validation = pairs(split.validation_days)
        for other_name, other in (("train", train), ("test", test)):
            overlap = validation & other
            if overlap:
                raise AssertionError(
                    f"{len(overlap)} (subject, date) pairs are in both validation "
                    f"and {other_name}"
                )


def subject_day_counts(split: TemporalSplit) -> pd.DataFrame:
    """Per-subject train/test day counts -- used by the dry run to spot thin subjects."""
    train = split.train_days.groupby(schema.SUBJECT_ID).size().rename("n_train_days")
    test = split.test_days.groupby(schema.SUBJECT_ID).size().rename("n_test_days")
    out = pd.concat([train, test], axis=1).fillna(0).astype(int)
    return out.reset_index()


def summarise_thin_subjects(split: TemporalSplit, sequence_length: int) -> dict[str, Any]:
    """How many subjects cannot contribute a test sequence at this length.

    Having ``L`` test days is not enough: they must be ``L`` *consecutive* days.
    On this cohort the gap-aware count is much larger than the day count, which
    is why both are reported -- the evaluable denominator is not 174.
    """
    counts = subject_day_counts(split)
    too_few_days = counts[counts["n_test_days"] < sequence_length]

    evaluable = 0
    for _, group in split.test_days.groupby(schema.SUBJECT_ID):
        dates = sorted(group[schema.DATE_COL])
        longest, current = 1, 1
        for previous, nxt in zip(dates, dates[1:]):
            current = current + 1 if (nxt - previous).days == 1 else 1
            longest = max(longest, current)
        if longest >= sequence_length:
            evaluable += 1

    n_subjects = int(counts[schema.SUBJECT_ID].nunique())
    return {
        "sequence_length": sequence_length,
        "n_subjects_total": n_subjects,
        "n_subjects_without_enough_test_days": int(len(too_few_days)),
        "n_subjects_with_evaluable_test_sequence": evaluable,
        "n_subjects_lost_to_gaps_or_length": n_subjects - evaluable,
        "min_test_days": int(counts["n_test_days"].min()),
        "median_test_days": float(counts["n_test_days"].median()),
        "min_train_days": int(counts["n_train_days"].min()),
        "note": (
            "연속 L일이 되지 않는 피험자는 시퀀스 단위 평가에서 자동으로 빠진다. "
            "피험자 단위 평가의 분모가 174명이 아님을 결과에 반드시 명시한다."
        ),
    }


def n_dates_in_common(left: pd.DataFrame, right: pd.DataFrame) -> int:
    if not len(left) or not len(right):
        return 0
    return len(
        set(zip(left[schema.SUBJECT_ID], left[schema.DATE_COL]))
        & set(zip(right[schema.SUBJECT_ID], right[schema.DATE_COL]))
    )


def date_bounds_by_subject(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(schema.SUBJECT_ID)[schema.DATE_COL]
    return pd.DataFrame(
        {"first": grouped.min(), "last": grouped.max(), "n_days": grouped.size()}
    ).reset_index()


def as_numpy_dates(frame: pd.DataFrame) -> np.ndarray:
    return frame[schema.DATE_COL].to_numpy()
