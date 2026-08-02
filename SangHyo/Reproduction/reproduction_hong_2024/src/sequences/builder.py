"""Turn the daily table into fixed-length sequences, with full provenance.

The single most important rule in this package lives here: **sequences are only
ever built inside one side of an already-decided split.**  ``build_sequences``
takes a frame that has already been restricted to a split, so a window can never
straddle a train/test boundary -- there is no boundary inside its input to cross.

The one exception is :func:`build_sequences_literal`, which reproduces the
reading of the paper where windows are cut across the whole record and only then
labelled train/test.  It exists to *measure* the leakage, and it refuses to hand
back a split unless the caller has asked for the diagnostic explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..data import schema


@dataclass
class SequenceSet:
    """Sequences plus every column needed to audit where they came from.

    ``X`` has shape ``(n_sequences, sequence_length, n_features)``.  ``provenance``
    has one row per sequence and carries, per the spec: subject_id, start_date,
    end_date, raw_row_ids, raw_dates, sequence_length, split_name, outer_fold and
    inner_fold.
    """

    X: np.ndarray
    y: np.ndarray
    provenance: pd.DataFrame
    feature_columns: tuple[str, ...]
    sequence_length: int
    split_name: str

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def subjects(self) -> np.ndarray:
        return self.provenance[schema.SUBJECT_ID].to_numpy()

    def raw_row_ids(self) -> set[int]:
        out: set[int] = set()
        for ids in self.provenance["raw_row_ids"]:
            out.update(int(i) for i in ids)
        return out

    def raw_dates(self) -> set[pd.Timestamp]:
        out: set[pd.Timestamp] = set()
        for dates in self.provenance["raw_dates"]:
            out.update(pd.Timestamp(d) for d in dates)
        return out

    def subject_date_pairs(self) -> set[tuple[str, pd.Timestamp]]:
        """(subject, date) pairs -- the key any honest overlap check uses.

        Two different subjects sharing a calendar date is normal and harmless;
        the same subject's date on both sides of a split is not.
        """
        out: set[tuple[str, pd.Timestamp]] = set()
        for subject, dates in zip(
            self.provenance[schema.SUBJECT_ID], self.provenance["raw_dates"]
        ):
            for date in dates:
                out.add((subject, pd.Timestamp(date)))
        return out

    def describe(self) -> dict[str, Any]:
        counts = self.provenance.groupby(schema.SUBJECT_ID).size()
        n_pos = int(self.y.sum())
        return {
            "split_name": self.split_name,
            "sequence_length": self.sequence_length,
            "n_sequences": len(self),
            "n_subjects": int(self.provenance[schema.SUBJECT_ID].nunique()),
            "n_positive_sequences": n_pos,
            "n_negative_sequences": len(self) - n_pos,
            "n_positive_subjects": int(
                self.provenance.groupby(schema.SUBJECT_ID)["y"].first().sum()
            ),
            "sequences_per_subject": {
                "min": int(counts.min()) if len(counts) else 0,
                "max": int(counts.max()) if len(counts) else 0,
                "median": float(counts.median()) if len(counts) else 0.0,
            },
            "X_shape": list(self.X.shape),
        }

    def select(self, mask: np.ndarray, *, split_name: str | None = None) -> "SequenceSet":
        mask = np.asarray(mask)
        return SequenceSet(
            X=self.X[mask],
            y=self.y[mask],
            provenance=self.provenance.loc[mask].reset_index(drop=True),
            feature_columns=self.feature_columns,
            sequence_length=self.sequence_length,
            split_name=split_name or self.split_name,
        )


def consecutive_runs(dates: Iterable[pd.Timestamp]) -> list[list[int]]:
    """Split positional indices into runs of consecutive calendar days.

    ``dates`` must already be sorted ascending and free of duplicates.
    """
    dates = list(dates)
    if not dates:
        return []
    runs: list[list[int]] = []
    current = [0]
    for i in range(1, len(dates)):
        if (pd.Timestamp(dates[i]) - pd.Timestamp(dates[i - 1])).days == 1:
            current.append(i)
        else:
            runs.append(current)
            current = [i]
    runs.append(current)
    return runs


def build_sequences(
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    sequence_length: int,
    stride: int = 1,
    split_name: str = "unnamed",
    require_consecutive: bool = True,
    outer_fold: int | None = None,
    inner_fold: int | None = None,
) -> SequenceSet:
    """Build sequences from *frame*, which must already be one side of a split.

    Parameters
    ----------
    require_consecutive:
        ``True`` (default) only emits windows whose days are genuinely
        consecutive calendar days.  ``False`` reproduces the naive "take the next
        N rows" behaviour and is only there so the audit can quantify what that
        costs -- on this cohort it silently glues across a calendar gap in 16.6%
        of 3-day windows and 28.6% of 5-day windows.
    """
    feature_columns = tuple(feature_columns)
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    blocks: list[np.ndarray] = []
    records: list[dict[str, Any]] = []

    for subject, group in frame.groupby(schema.SUBJECT_ID, sort=True):
        group = group.sort_values(schema.DATE_COL)
        dates = list(group[schema.DATE_COL])
        if group[schema.DATE_COL].duplicated().any():
            raise ValueError(
                f"subject {subject!r} has duplicate dates; the loader must reduce "
                "the table to one row per subject-day before sequences are built"
            )
        values = group[list(feature_columns)].to_numpy(dtype=np.float32)
        row_ids = group[schema.RAW_ROW_ID].to_numpy()
        label = int(group[schema.LABEL_COL].iloc[0])

        runs = consecutive_runs(dates) if require_consecutive else [list(range(len(dates)))]
        for run in runs:
            for offset in range(0, max(0, len(run) - sequence_length + 1), stride):
                positions = run[offset : offset + sequence_length]
                window_dates = [dates[p] for p in positions]
                blocks.append(values[positions])
                records.append(
                    {
                        schema.SUBJECT_ID: subject,
                        "y": label,
                        "start_date": window_dates[0],
                        "end_date": window_dates[-1],
                        "raw_row_ids": tuple(int(row_ids[p]) for p in positions),
                        "raw_dates": tuple(pd.Timestamp(d) for d in window_dates),
                        "sequence_length": sequence_length,
                        "is_calendar_consecutive": (
                            (window_dates[-1] - window_dates[0]).days == sequence_length - 1
                        ),
                        "split_name": split_name,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                    }
                )

    n_features = len(feature_columns)
    if blocks:
        X = np.stack(blocks).astype(np.float32)
        provenance = pd.DataFrame(records)
    else:
        X = np.empty((0, sequence_length, n_features), dtype=np.float32)
        provenance = pd.DataFrame(
            columns=[
                schema.SUBJECT_ID, "y", "start_date", "end_date", "raw_row_ids",
                "raw_dates", "sequence_length", "is_calendar_consecutive",
                "split_name", "outer_fold", "inner_fold",
            ]
        )

    provenance = provenance.reset_index(drop=True)
    provenance["sequence_id"] = [
        f"{split_name}|L{sequence_length}|{row[schema.SUBJECT_ID]}|"
        f"{pd.Timestamp(row['start_date']).date()}"
        for _, row in provenance.iterrows()
    ] if len(provenance) else []

    y = provenance["y"].to_numpy(dtype=np.int64) if len(provenance) else np.empty(0, dtype=np.int64)
    return SequenceSet(
        X=X,
        y=y,
        provenance=provenance,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        split_name=split_name,
    )


def build_sequences_literal(
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    sequence_length: int,
    stride: int = 1,
    test_start_by_subject: dict[str, pd.Timestamp],
    require_consecutive: bool = False,
    leakage_diagnostic_only: bool = False,
) -> tuple[SequenceSet, SequenceSet, dict[str, Any]]:
    """The *literal* reading of the paper: window first, label the windows after.

    Section 4.2 of the paper says the 12,183 records were "transformed into
    time-series data" and that test data were then "constructed by isolating the
    final week of data from each subject".  Read in that order, windows are cut
    before the split exists, so windows that span the cut have to be assigned to
    one side -- and every raw day inside them goes with them.

    Returns ``(train, test, report)``.  A window is test only if *all* of its days
    are in the subject's final week; a window with days on both sides is a
    boundary crosser and is assigned to train, which is what leaks.

    Raises unless *leakage_diagnostic_only* is set, so this can never be reached
    by accident from an experiment that reports a performance number.
    """
    if not leakage_diagnostic_only:
        raise ValueError(
            "build_sequences_literal reproduces a leaking construction. Set "
            "leakage_diagnostic_only=True to acknowledge that its numbers are a "
            "leakage measurement and not a performance claim."
        )

    everything = build_sequences(
        frame,
        feature_columns,
        sequence_length=sequence_length,
        stride=stride,
        split_name="literal_all",
        require_consecutive=require_consecutive,
    )
    if not len(everything):
        raise ValueError("no sequences were produced")

    is_test, is_cross = [], []
    for _, row in everything.provenance.iterrows():
        cut = test_start_by_subject[row[schema.SUBJECT_ID]]
        flags = [pd.Timestamp(d) >= cut for d in row["raw_dates"]]
        is_test.append(all(flags))
        is_cross.append(any(flags) and not all(flags))
    is_test = np.asarray(is_test)
    is_cross = np.asarray(is_cross)

    test = everything.select(is_test, split_name="literal_test")
    train = everything.select(~is_test, split_name="literal_train")

    shared = train.subject_date_pairs() & test.subject_date_pairs()
    report = {
        "mode": "paper_literal_variant",
        "sequence_length": sequence_length,
        "n_sequences_total": len(everything),
        "n_boundary_crossing_sequences": int(is_cross.sum()),
        "boundary_crossing_fraction": float(is_cross.mean()),
        "n_gap_spanning_sequences": int(
            (~everything.provenance["is_calendar_consecutive"]).sum()
        ),
        "gap_spanning_fraction": float(
            (~everything.provenance["is_calendar_consecutive"]).mean()
        ),
        "n_train_sequences": len(train),
        "n_test_sequences": len(test),
        "n_subject_dates_in_both_splits": len(shared),
        "n_subject_dates_in_test": len(test.subject_date_pairs()),
        "subject_date_leak_fraction": (
            len(shared) / len(test.subject_date_pairs()) if len(test) else 0.0
        ),
        "warning": (
            "이 구성은 train/test 경계를 가로지르는 윈도우를 허용한다. "
            "여기서 나오는 성능은 누수 진단값이며 일반화 성능이 아니다."
        ),
    }
    return train, test, report
