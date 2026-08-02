"""Load the AI-Hub sleep lifelog into one daily table with the paper's 32 features.

The unit of the returned table is **one subject-day**, which is what Hong et al.
(2024) call "a single subject's sleep data for one day".  Sequences are built on
top of this table later, never here -- see ``sequences/builder.py``.

Nothing in this module looks at a split, so it cannot leak by construction: the
same daily table feeds every experiment, and every split is drawn afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from . import schema

SLEEP_SOURCE = {
    "train": Path("1.Training") / "SourceData" / "2.Sleep" / "train_sleep.csv",
    "validation": Path("2.Validation") / "SourceData" / "2.Sleep" / "val_sleep.csv",
}
SLEEP_LABELS = {
    "train": Path("1.Training") / "LabelingData" / "2.Sleep" / "training_label.csv",
    "validation": Path("2.Validation") / "LabelingData" / "2.Sleep" / "val_label.csv",
}
#: Read only to check that the label copies agree (AGENTS.md §1).  Never featurised.
GAIT_LABELS = {
    "train": Path("1.Training") / "LabelingData" / "1.Gait" / "training_label.csv",
    "validation": Path("2.Validation") / "LabelingData" / "1.Gait" / "val_label.csv",
}

DUPLICATE_POLICIES = ("longest_duration", "latest_bedtime_end", "first")
DATE_SOURCES = ("bedtime_end", "bedtime_start")


class DataError(RuntimeError):
    """Raised when the data on disk contradicts what this package requires."""


@dataclass
class LifelogData:
    """One daily table, one subject table, and the feature column list.

    ``daily`` is sorted by (subject, date) and carries a stable ``raw_row_id``
    that every downstream audit uses to prove which raw records ended up in which
    split.
    """

    daily: pd.DataFrame
    subjects: pd.DataFrame
    feature_columns: tuple[str, ...]
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return len(self.daily)

    @property
    def n_subjects(self) -> int:
        return int(self.daily[schema.SUBJECT_ID].nunique())

    def labels_by_subject(self) -> pd.Series:
        return self.subjects.set_index(schema.SUBJECT_ID)[schema.LABEL_COL]

    def subject_ids(self) -> tuple[str, ...]:
        return tuple(self.subjects[schema.SUBJECT_ID])

    def rows_for(self, subjects: Iterable[str]) -> pd.DataFrame:
        wanted = set(subjects)
        return self.daily[self.daily[schema.SUBJECT_ID].isin(wanted)]

    def describe(self) -> dict[str, Any]:
        return {
            "n_subjects": self.n_subjects,
            "n_daily_rows": self.n_rows,
            "n_features": len(self.feature_columns),
            "date_min": str(self.daily[schema.DATE_COL].min().date()),
            "date_max": str(self.daily[schema.DATE_COL].max().date()),
            "subject_class_counts": self.subjects[schema.LABEL_COL]
            .value_counts()
            .sort_index()
            .to_dict(),
            "diagnosis_counts": self.subjects[schema.DIAGNOSIS_COL]
            .value_counts()
            .to_dict(),
        }


# --- intraday series ----------------------------------------------------------

def parse_intraday(cell: Any) -> np.ndarray:
    """Parse a ``"63/61/59/..."`` slash-separated series into a float array.

    Trailing empty fields (the files end the series with a ``/``) are dropped.
    """
    text = str(cell).strip()
    if not text or text in {"nan", "..."}:
        return np.empty(0, dtype=float)
    values = [part for part in text.split("/") if part.strip() != ""]
    try:
        return np.asarray(values, dtype=float)
    except ValueError:
        return np.asarray(
            [float(v) for v in values if v.lstrip("-").replace(".", "", 1).isdigit()],
            dtype=float,
        )


def _drop_sentinel(series: np.ndarray) -> np.ndarray:
    """Drop the device's 0 "no reading" sentinel."""
    return series[series != schema.INTRADAY_MISSING_SENTINEL]


def _intraday_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute every feature that has to come out of the 5-minute series."""
    hr_col = schema.INTRADAY_SOURCE["hr"]
    hyp_col = schema.INTRADAY_SOURCE["hypnogram"]
    rmssd_col = schema.INTRADAY_SOURCE["rmssd"]

    hr_max, hr_median, hypnogram_avg, rmssd_avg = [], [], [], []
    empty_series = 0
    for hr_cell, hyp_cell, rmssd_cell in zip(
        raw[hr_col].to_numpy(), raw[hyp_col].to_numpy(), raw[rmssd_col].to_numpy()
    ):
        hr = _drop_sentinel(parse_intraday(hr_cell))
        hyp = _drop_sentinel(parse_intraday(hyp_cell))
        rmssd = _drop_sentinel(parse_intraday(rmssd_cell))
        if hr.size == 0 or hyp.size == 0 or rmssd.size == 0:
            empty_series += 1
        hr_max.append(hr.max() if hr.size else np.nan)
        hr_median.append(float(np.median(hr)) if hr.size else np.nan)
        hypnogram_avg.append(float(hyp.mean()) if hyp.size else np.nan)
        rmssd_avg.append(float(rmssd.mean()) if rmssd.size else np.nan)

    frame = pd.DataFrame(
        {
            "sleep_hr_max": hr_max,
            "sleep_hr_median": hr_median,
            "sleep_hypnogram_average": hypnogram_avg,
            "rmssd_average": rmssd_avg,
        },
        index=raw.index,
    )
    frame.attrs["rows_with_empty_intraday_series"] = empty_series
    return frame


def _time_bin_onehot(timestamps: pd.Series, prefix: str) -> pd.DataFrame:
    """Table A1: which of the six four-hour bins the timestamp's hour falls in."""
    bins = (timestamps.dt.hour // schema.TIME_BIN_HOURS).astype(int)
    columns = {
        f"{prefix}{i + 1}": (bins == i).astype(np.int8) for i in range(schema.N_TIME_BINS)
    }
    return pd.DataFrame(columns, index=timestamps.index)


# --- loading ------------------------------------------------------------------

def resolve_data_root(candidates: Sequence[str | Path]) -> Path:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if (path / "1.Training" / "SourceData" / "2.Sleep" / "train_sleep.csv").is_file():
            return path.resolve()
    tried = ", ".join(str(c) for c in candidates if c)
    raise FileNotFoundError(f"could not find the AI-Hub Data/ directory. tried: {tried}")


def _load_labels(data_root: Path) -> pd.DataFrame:
    frames = []
    for origin, relative in SLEEP_LABELS.items():
        frame = pd.read_csv(data_root / relative)
        frame[schema.SPLIT_ORIGIN] = origin
        frames.append(frame)
    labels = pd.concat(frames, ignore_index=True)
    labels = labels.rename(columns={schema.RAW_LABEL_KEY: schema.SUBJECT_ID})

    # AGENTS.md: the Gait and Sleep label copies must agree.
    gait = pd.concat(
        [pd.read_csv(data_root / relative) for relative in GAIT_LABELS.values()],
        ignore_index=True,
    ).rename(columns={schema.RAW_LABEL_KEY: schema.SUBJECT_ID})
    merged = labels.merge(gait, on=schema.SUBJECT_ID, how="outer", suffixes=("_sleep", "_gait"))
    mismatched = merged[merged["DIAG_NM_sleep"] != merged["DIAG_NM_gait"]]
    if len(mismatched):
        raise DataError(
            f"sleep and gait label copies disagree for {len(mismatched)} subjects: "
            f"{sorted(mismatched[schema.SUBJECT_ID].head(5))}"
        )

    duplicated = labels[schema.SUBJECT_ID].duplicated().sum()
    if duplicated:
        raise DataError(f"{duplicated} subject ids appear in more than one label file")

    labels[schema.LABEL_COL] = labels[schema.DIAGNOSIS_COL].map(schema.diagnosis_to_label)
    return labels


def load_lifelog(
    data_root: str | Path,
    *,
    sleep_date_source: str = "bedtime_end",
    duplicate_policy: str = "longest_duration",
    rmssd_source: str = "intraday_mean",
) -> LifelogData:
    """Build the daily table with the paper's 32 features.

    Parameters
    ----------
    sleep_date_source:
        Which end of the sleep period names the day.  ``bedtime_end`` (waking day,
        the Oura convention) leaves 24 same-day collisions across 11 subjects;
        ``bedtime_start`` leaves 1044.  ``bedtime_end`` is the default for that
        reason and the choice is recorded in ``assumptions.md`` A-01.
    duplicate_policy:
        How to reduce the remaining same-day collisions to one row per day.
    rmssd_source:
        ``intraday_mean`` computes ``rmssd_average`` as the mean of the non-zero
        5-minute RMSSD series (Table A1: "Average heart rate variability").
        ``sleep_rmssd_column`` uses the pre-aggregated column instead; the two
        correlate at r = 0.997 but agree exactly on only 37.5% of rows, so the
        choice is an assumption variant (A-04), not a free relabelling.
    """
    if sleep_date_source not in DATE_SOURCES:
        raise ValueError(f"sleep_date_source must be one of {DATE_SOURCES}")
    if duplicate_policy not in DUPLICATE_POLICIES:
        raise ValueError(f"duplicate_policy must be one of {DUPLICATE_POLICIES}")

    data_root = Path(data_root)
    frames = []
    for origin, relative in SLEEP_SOURCE.items():
        frame = pd.read_csv(data_root / relative)
        frame[schema.SPLIT_ORIGIN] = origin
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    n_raw_rows = len(raw)

    raw = raw.rename(columns={schema.RAW_SUBJECT_KEY: schema.SUBJECT_ID})
    for column in ("sleep_bedtime_start", "sleep_bedtime_end"):
        raw[column] = pd.to_datetime(raw[column], format="ISO8601")

    date_column = f"sleep_{sleep_date_source}"
    raw[schema.DATE_COL] = raw[date_column].dt.tz_localize(None).dt.normalize()

    # --- assemble the 32 features -------------------------------------------
    features = pd.DataFrame(index=raw.index)
    missing = [
        source for source in schema.PASSTHROUGH_FEATURES.values() if source not in raw.columns
    ]
    if missing:
        raise DataError(f"sleep CSV is missing expected columns: {missing}")
    for paper_name, source in schema.PASSTHROUGH_FEATURES.items():
        features[paper_name] = pd.to_numeric(raw[source], errors="coerce")

    intraday = _intraday_frame(raw)
    for column in intraday.columns:
        features[column] = intraday[column]
    if rmssd_source == "sleep_rmssd_column":
        features["rmssd_average"] = pd.to_numeric(raw["sleep_rmssd"], errors="coerce")
    elif rmssd_source != "intraday_mean":
        raise ValueError("rmssd_source must be 'intraday_mean' or 'sleep_rmssd_column'")

    starts = _time_bin_onehot(raw["sleep_bedtime_start"], "start")
    ends = _time_bin_onehot(raw["sleep_bedtime_end"], "end")
    features = pd.concat([features, starts, ends], axis=1)

    unexpected = set(features.columns) ^ set(schema.PAPER_FEATURES)
    if unexpected:
        raise DataError(f"feature set does not match Table 4: symmetric difference {unexpected}")
    features = features[list(schema.PAPER_FEATURES)]

    # ``sleep_duration`` is already one of the 32 features, so it is not repeated
    # here -- the dedup below sorts on the feature column.
    daily = pd.concat(
        [
            raw[[schema.SUBJECT_ID, schema.DATE_COL, schema.SPLIT_ORIGIN]],
            raw[["sleep_bedtime_start", "sleep_bedtime_end"]],
            features,
        ],
        axis=1,
    )

    # --- one row per subject-day --------------------------------------------
    before = len(daily)
    if duplicate_policy == "longest_duration":
        daily = daily.sort_values(
            [schema.SUBJECT_ID, schema.DATE_COL, "sleep_duration"]
        ).drop_duplicates([schema.SUBJECT_ID, schema.DATE_COL], keep="last")
    elif duplicate_policy == "latest_bedtime_end":
        daily = daily.sort_values(
            [schema.SUBJECT_ID, schema.DATE_COL, "sleep_bedtime_end"]
        ).drop_duplicates([schema.SUBJECT_ID, schema.DATE_COL], keep="last")
    else:
        daily = daily.sort_values([schema.SUBJECT_ID, schema.DATE_COL]).drop_duplicates(
            [schema.SUBJECT_ID, schema.DATE_COL], keep="first"
        )
    n_dropped_duplicates = before - len(daily)

    daily = daily.sort_values([schema.SUBJECT_ID, schema.DATE_COL]).reset_index(drop=True)
    daily[schema.RAW_ROW_ID] = np.arange(len(daily), dtype=np.int64)

    labels = _load_labels(data_root)
    label_lookup = labels.set_index(schema.SUBJECT_ID)
    unlabelled = set(daily[schema.SUBJECT_ID]) - set(label_lookup.index)
    if unlabelled:
        raise DataError(f"{len(unlabelled)} subjects have sleep rows but no label")
    daily[schema.LABEL_COL] = daily[schema.SUBJECT_ID].map(label_lookup[schema.LABEL_COL])
    daily[schema.DIAGNOSIS_COL] = daily[schema.SUBJECT_ID].map(label_lookup[schema.DIAGNOSIS_COL])

    n_missing = int(daily[list(schema.PAPER_FEATURES)].isna().sum().sum())
    if n_missing:
        # The paper states the delivered data was already cleaned.  If that ever
        # stops being true here, the run must stop rather than silently impute:
        # an imputer fitted on all rows would be a preprocessing leak.
        raise DataError(
            f"{n_missing} missing feature values found. The paper reports the data "
            "arrived already cleaned; resolve this before training rather than "
            "imputing across the split boundary."
        )

    subjects = (
        labels[labels[schema.SUBJECT_ID].isin(set(daily[schema.SUBJECT_ID]))]
        .loc[:, [schema.SUBJECT_ID, schema.DIAGNOSIS_COL, schema.LABEL_COL, schema.SPLIT_ORIGIN]]
        .sort_values(schema.SUBJECT_ID)
        .reset_index(drop=True)
    )

    notes = {
        "data_root": str(data_root),
        "sleep_date_source": sleep_date_source,
        "duplicate_policy": duplicate_policy,
        "rmssd_source": rmssd_source,
        "n_raw_rows": n_raw_rows,
        "n_daily_rows": len(daily),
        "n_dropped_duplicate_day_rows": n_dropped_duplicates,
        "rows_with_empty_intraday_series": int(
            intraday.attrs.get("rows_with_empty_intraday_series", 0)
        ),
        "records_per_subject": {
            "min": int(daily.groupby(schema.SUBJECT_ID).size().min()),
            "max": int(daily.groupby(schema.SUBJECT_ID).size().max()),
            "median": float(daily.groupby(schema.SUBJECT_ID).size().median()),
        },
    }

    return LifelogData(
        daily=daily,
        subjects=subjects,
        feature_columns=tuple(schema.PAPER_FEATURES),
        notes=notes,
    )
