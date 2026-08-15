"""Per-subject day grids: the SensorFM-style model input built from our data.

Output object per subject (``DayBank`` holds the whole cohort):

    values  float32 (n_days, 1440, 8)   raw channel values (NOT normalized)
    mask    bool    (n_days, 1440, 8)   True = observed minute
    meta    int32   (n_days, 3)         [day_of_week, day_of_year, year]

Channel layout follows ``config.CHANNELS``:

    met          1-min MET from CONVERT(activity_met_1min); observed where the
                 value is finite and > 0 AND the surrounding 5-min activity
                 class is not non-wear (class 0)
    act_class    5-min class 1..5 upsampled to minutes; class 0 (non-wear) and
                 absent epochs are missing
    stage_*      hypnogram one-hot (deep/light/rem/awake); observed only inside
                 recorded sleep periods, placed by bedtime_start on the
                 04:00-anchored day grid (periods crossing 04:00 spill into the
                 neighboring day window, as they physically do)
    sleep_hr     5-min nocturnal HR; Oura encodes missing epochs as 0
    sleep_rmssd  5-min nocturnal RMSSD; 0 = missing

Leakage argument
----------------
Everything here is a function of ONE subject's own rows: no cross-subject
statistic, no label, no cohort reference.  Normalization (z-score + clip) is
deliberately NOT done here -- it is fold-local and lives in ``pretrain.py``,
fed by the per-subject moment sums this module precomputes.

Day admission (paper M.3.2): a day window with more than 80% missing cells or
fewer than ``MIN_OBSERVED_TOKENS`` observable 20-min tokens is dropped.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ACTIVITY_CLASS_EPOCHS_PER_DAY,
    ACTIVITY_DAY_START_HOUR,
    ACTIVITY_EPOCH_MINUTES,
    CHANNELS,
    INTRADAY_COLUMNS,
    LOCAL_TZ,
    MAX_DAY_MISSING_FRACTION,
    MINUTES_PER_DAY,
    MIN_OBSERVED_TOKENS,
    N_CHANNELS,
    PATCH_MINUTES,
    SLEEP_EPOCH_MINUTES,
    SOURCE_FILES,
    SPLIT_DIRS,
    TOKEN_OBSERVED_MIN_FRACTION,
)
from .data import read_csv

_CH = {name: i for i, name in enumerate(CHANNELS)}
_STAGE_CHANNEL = {1: _CH["stage_deep"], 2: _CH["stage_light"],
                  3: _CH["stage_rem"], 4: _CH["stage_awake"]}
_STAGE_CHANNELS = tuple(_STAGE_CHANNEL.values())


def parse_series(text: object, *, expected: int | None = None) -> np.ndarray:
    """Slash-separated intraday string -> float array (empty when absent)."""

    if text is None:
        return np.zeros(0, dtype=np.float64)
    raw = str(text)
    if raw in ("", "nan", "...", "None"):
        return np.zeros(0, dtype=np.float64)
    try:
        values = np.fromstring(raw, sep="/", dtype=np.float64)
    except Exception:  # np.fromstring text mode removed/changed in this numpy
        tokens = [t for t in raw.split("/") if t.strip() != ""]
        try:
            values = np.asarray([float(t) for t in tokens], dtype=np.float64)
        except ValueError:
            return np.zeros(0, dtype=np.float64)
    if expected is not None and values.size > expected:
        values = values[:expected]
    return values


def _local_ts(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_convert(LOCAL_TZ)


def _window_minute(ts: pd.Timestamp) -> tuple[int, int] | None:
    """(day_ordinal, minute_in_window) for one local timestamp.

    minute 0 == 04:00 local; the day ordinal is the calendar ordinal of the
    window's start date, so windows never overlap and cover all of time.
    """

    if ts is pd.NaT or ts is None:
        return None
    shifted = ts - pd.Timedelta(hours=ACTIVITY_DAY_START_HOUR)
    return shifted.toordinal(), shifted.hour * 60 + shifted.minute


@dataclass
class SubjectGrid:
    subject_id: str
    values: np.ndarray   # float32 (n_days, 1440, N_CHANNELS)
    mask: np.ndarray     # bool    (n_days, 1440, N_CHANNELS)
    meta: np.ndarray     # int32   (n_days, 3) [dow, doy, year]

    @property
    def n_days(self) -> int:
        return int(self.values.shape[0])


class _DayAccumulator:
    """Mutable per-subject store keyed by day ordinal, subject-local only."""

    def __init__(self) -> None:
        self.days: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def day(self, ordinal: int) -> tuple[np.ndarray, np.ndarray]:
        if ordinal not in self.days:
            values = np.zeros((MINUTES_PER_DAY, N_CHANNELS), dtype=np.float32)
            mask = np.zeros((MINUTES_PER_DAY, N_CHANNELS), dtype=bool)
            self.days[ordinal] = (values, mask)
        return self.days[ordinal]


def _place_activity_row(acc: _DayAccumulator, start_ts: pd.Timestamp,
                        met_text: object, class_text: object) -> None:
    """Place one activity row (a 04:00-anchored day) onto the grid.

    ``activity_day_start`` is contractually 04:00 local, but the placement uses
    absolute-minute arithmetic anyway, so a drifted row lands at its true clock
    position (possibly spilling into the neighboring window) instead of being
    silently snapped.
    """

    anchor = _window_minute(start_ts)
    if anchor is None:
        return
    ordinal, minute0 = anchor
    met = parse_series(met_text, expected=MINUTES_PER_DAY)
    classes = parse_series(class_text, expected=ACTIVITY_CLASS_EPOCHS_PER_DAY)

    # 5-min class upsampled to minutes; NaN where the epoch is absent.
    class_minutes = np.full(MINUTES_PER_DAY, np.nan, dtype=np.float64)
    if classes.size:
        upsampled = np.repeat(classes, ACTIVITY_EPOCH_MINUTES)[:MINUTES_PER_DAY]
        class_minutes[: upsampled.size] = upsampled

    met_minutes = np.full(MINUTES_PER_DAY, np.nan, dtype=np.float64)
    if met.size:
        met_minutes[: met.size] = met

    absolute = ordinal * MINUTES_PER_DAY + minute0 + np.arange(MINUTES_PER_DAY)
    day_ordinals = absolute // MINUTES_PER_DAY
    minute_index = absolute % MINUTES_PER_DAY
    for day_ordinal in np.unique(day_ordinals):
        values, mask = acc.day(int(day_ordinal))
        in_day = day_ordinals == day_ordinal
        rows = minute_index[in_day]
        cls = class_minutes[in_day]
        met_vals = met_minutes[in_day]

        wear = np.isfinite(cls) & (cls > 0)
        values[rows[wear], _CH["act_class"]] = cls[wear].astype(np.float32)
        mask[rows[wear], _CH["act_class"]] = True

        nonwear = np.isfinite(cls) & (cls == 0)
        met_ok = np.isfinite(met_vals) & (met_vals > 0) & ~nonwear
        values[rows[met_ok], _CH["met"]] = met_vals[met_ok].astype(np.float32)
        mask[rows[met_ok], _CH["met"]] = True


def _place_sleep_row(acc: _DayAccumulator, start_ts: pd.Timestamp,
                     hypnogram_text: object, hr_text: object,
                     rmssd_text: object) -> None:
    anchor = _window_minute(start_ts)
    if anchor is None:
        return
    ordinal, minute0 = anchor
    stages = parse_series(hypnogram_text)
    if stages.size == 0:
        return
    hr = parse_series(hr_text)
    rmssd = parse_series(rmssd_text)

    n_minutes = int(stages.size) * SLEEP_EPOCH_MINUTES
    absolute = ordinal * MINUTES_PER_DAY + minute0 + np.arange(n_minutes)
    epoch_of_minute = np.arange(n_minutes) // SLEEP_EPOCH_MINUTES
    day_ordinals = absolute // MINUTES_PER_DAY
    minute_index = absolute % MINUTES_PER_DAY

    for day_ordinal in np.unique(day_ordinals):
        values, mask = acc.day(int(day_ordinal))
        in_day = day_ordinals == day_ordinal
        rows = minute_index[in_day]
        epochs = epoch_of_minute[in_day]

        stage = stages[epochs]
        valid_stage = np.isin(stage, list(_STAGE_CHANNEL))
        stage_rows = rows[valid_stage]
        stage_vals = stage[valid_stage].astype(int)
        for code, channel in _STAGE_CHANNEL.items():
            values[stage_rows, channel] = (stage_vals == code).astype(np.float32)
        for channel in _STAGE_CHANNELS:
            mask[stage_rows, channel] = True

        for series, channel in ((hr, _CH["sleep_hr"]), (rmssd, _CH["sleep_rmssd"])):
            if series.size == 0:
                continue
            have = epochs < series.size
            vals = np.where(have, series[np.minimum(epochs, series.size - 1)], np.nan)
            ok = np.isfinite(vals) & (vals > 0)  # Oura: 0 encodes a missing epoch
            values[rows[ok], channel] = vals[ok].astype(np.float32)
            mask[rows[ok], channel] = True


def _admit_day(mask: np.ndarray) -> bool:
    """Paper M.3.2 rule + a floor on observable tokens for the embedding mean."""

    if 1.0 - float(mask.mean()) > MAX_DAY_MISSING_FRACTION:
        return False
    per_token = mask.reshape(MINUTES_PER_DAY // PATCH_MINUTES, PATCH_MINUTES,
                             N_CHANNELS).mean(axis=1)
    return int((per_token.T >= TOKEN_OBSERVED_MIN_FRACTION).sum()) >= MIN_OBSERVED_TOKENS


def _finalize_subject(subject_id: str, acc: _DayAccumulator) -> SubjectGrid | None:
    kept_values, kept_mask, kept_meta = [], [], []
    for ordinal in sorted(acc.days):
        values, mask = acc.days[ordinal]
        if not _admit_day(mask):
            continue
        date = pd.Timestamp.fromordinal(int(ordinal))
        kept_values.append(values)
        kept_mask.append(mask)
        kept_meta.append((int(date.dayofweek), int(date.dayofyear) - 1, int(date.year)))
    if not kept_values:
        return None
    return SubjectGrid(
        subject_id=subject_id,
        values=np.stack(kept_values).astype(np.float32),
        mask=np.stack(kept_mask),
        meta=np.asarray(kept_meta, dtype=np.int32),
    )


def build_subject_grids(activity: pd.DataFrame, sleep: pd.DataFrame,
                        *, max_days_per_subject: int | None = None
                        ) -> dict[str, SubjectGrid]:
    """Subject-local day grids for one split.  Deterministic, label-blind."""

    for frame, name, needed in (
        (activity, "activity", ("EMAIL", "activity_day_start",
                                INTRADAY_COLUMNS["met_1min"],
                                INTRADAY_COLUMNS["activity_class_5min"])),
        (sleep, "sleep", ("EMAIL", "sleep_bedtime_start",
                          INTRADAY_COLUMNS["hypnogram_5min"])),
    ):
        missing = [c for c in needed if c not in frame.columns]
        if missing:
            raise KeyError(f"{name} file lacks columns: {missing}")

    activity_sids = activity["EMAIL"].astype(str).str.strip().tolist()
    activity_starts = _local_ts(activity["activity_day_start"]).tolist()
    met_texts = activity[INTRADAY_COLUMNS["met_1min"]].tolist()
    class_texts = activity[INTRADAY_COLUMNS["activity_class_5min"]].tolist()

    sleep_sids = sleep["EMAIL"].astype(str).str.strip().tolist()
    sleep_starts = _local_ts(sleep["sleep_bedtime_start"]).tolist()
    hypnogram_texts = sleep[INTRADAY_COLUMNS["hypnogram_5min"]].tolist()
    hr_col = INTRADAY_COLUMNS["sleep_hr_5min"]
    rmssd_col = INTRADAY_COLUMNS["sleep_rmssd_5min"]
    hr_texts = sleep[hr_col].tolist() if hr_col in sleep.columns else [None] * len(sleep)
    rmssd_texts = (sleep[rmssd_col].tolist() if rmssd_col in sleep.columns
                   else [None] * len(sleep))

    accumulators: dict[str, _DayAccumulator] = {}
    for sid, start, met_text, class_text in zip(
            activity_sids, activity_starts, met_texts, class_texts):
        acc = accumulators.setdefault(sid, _DayAccumulator())
        _place_activity_row(acc, start, met_text, class_text)

    for sid, start, hyp_text, hr_text, rmssd_text in zip(
            sleep_sids, sleep_starts, hypnogram_texts, hr_texts, rmssd_texts):
        acc = accumulators.setdefault(sid, _DayAccumulator())
        _place_sleep_row(acc, start, hyp_text, hr_text, rmssd_text)

    grids: dict[str, SubjectGrid] = {}
    for sid in sorted(accumulators):
        grid = _finalize_subject(sid, accumulators[sid])
        if grid is None:
            continue
        if max_days_per_subject is not None and grid.n_days > max_days_per_subject:
            grid = SubjectGrid(
                subject_id=grid.subject_id,
                values=grid.values[:max_days_per_subject],
                mask=grid.mask[:max_days_per_subject],
                meta=grid.meta[:max_days_per_subject],
            )
        grids[sid] = grid
    return grids


def load_split_grids(data_root: str | Path, split: str,
                     *, max_days_per_subject: int | None = None
                     ) -> dict[str, SubjectGrid]:
    root = Path(data_root) / SPLIT_DIRS[split]
    activity = read_csv(root / SOURCE_FILES[split]["activity"])
    sleep = read_csv(root / SOURCE_FILES[split]["sleep"])
    return build_subject_grids(activity, sleep,
                               max_days_per_subject=max_days_per_subject)


# ---------------------------------------------------------------- day bank ---
@dataclass
class DayBank:
    """Cohort grids flattened into contiguous arrays for training/embedding."""

    subject_ids: list[str]                 # index -> subject id
    day_subject: np.ndarray                # int32 (n_total_days,)
    values: np.ndarray                     # float32 (n_total_days, 1440, C)
    mask: np.ndarray                       # bool    (n_total_days, 1440, C)
    meta: np.ndarray                       # int32   (n_total_days, 3)
    channel_count: np.ndarray              # float64 (n_subjects, C) moment sums
    channel_sum: np.ndarray                # float64 (n_subjects, C)
    channel_sumsq: np.ndarray              # float64 (n_subjects, C)

    def days_of_subjects(self, subject_indices: np.ndarray) -> np.ndarray:
        wanted = np.zeros(len(self.subject_ids), dtype=bool)
        wanted[np.asarray(subject_indices, dtype=int)] = True
        return np.flatnonzero(wanted[self.day_subject])

    def fold_channel_stats(self, subject_indices: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray]:
        """Fold-local per-channel mean/std from precomputed subject moments."""

        index = np.asarray(subject_indices, dtype=int)
        count = self.channel_count[index].sum(axis=0)
        total = self.channel_sum[index].sum(axis=0)
        sumsq = self.channel_sumsq[index].sum(axis=0)
        safe = np.maximum(count, 1.0)
        mean = total / safe
        var = np.maximum(sumsq / safe - mean**2, 0.0)
        std = np.maximum(np.sqrt(var), 1e-6)
        return mean.astype(np.float32), std.astype(np.float32)


def build_day_bank(grids: dict[str, SubjectGrid], subject_order: list[str]) -> DayBank:
    """Stack per-subject grids following ``subject_order`` (the label index).

    Subjects present in the labels but with zero admitted days are an error:
    every labeled subject must be scoreable.
    """

    missing = [sid for sid in subject_order if sid not in grids]
    if missing:
        from .data import subject_hash

        raise AssertionError(
            f"{len(missing)} labeled subjects have no admitted wearable days: "
            f"{[subject_hash(m) for m in missing[:5]]}"
        )

    values, mask, meta, day_subject = [], [], [], []
    channel_count = np.zeros((len(subject_order), N_CHANNELS), dtype=np.float64)
    channel_sum = np.zeros_like(channel_count)
    channel_sumsq = np.zeros_like(channel_count)
    for index, sid in enumerate(subject_order):
        grid = grids[sid]
        values.append(grid.values)
        mask.append(grid.mask)
        meta.append(grid.meta)
        day_subject.append(np.full(grid.n_days, index, dtype=np.int32))
        observed = np.where(grid.mask, grid.values.astype(np.float64), 0.0)
        channel_count[index] = grid.mask.sum(axis=(0, 1))
        channel_sum[index] = observed.sum(axis=(0, 1))
        channel_sumsq[index] = (observed**2).sum(axis=(0, 1))

    return DayBank(
        subject_ids=list(subject_order),
        day_subject=np.concatenate(day_subject),
        values=np.concatenate(values, axis=0),
        mask=np.concatenate(mask, axis=0),
        meta=np.concatenate(meta, axis=0),
        channel_count=channel_count,
        channel_sum=channel_sum,
        channel_sumsq=channel_sumsq,
    )


def grid_fingerprint(bank: DayBank) -> str:
    digest = hashlib.sha256()
    digest.update(",".join(bank.subject_ids).encode("utf-8"))
    digest.update(bank.day_subject.tobytes())
    digest.update(np.ascontiguousarray(bank.values).tobytes())
    digest.update(np.ascontiguousarray(bank.mask).tobytes())
    digest.update(np.ascontiguousarray(bank.meta).tobytes())
    return digest.hexdigest()[:16]


def bank_summary(bank: DayBank) -> dict:
    per_subject = np.bincount(bank.day_subject, minlength=len(bank.subject_ids))
    return {
        "n_subjects": len(bank.subject_ids),
        "n_days_total": int(bank.day_subject.size),
        "days_per_subject_min": int(per_subject.min()),
        "days_per_subject_median": float(np.median(per_subject)),
        "days_per_subject_max": int(per_subject.max()),
        "observed_cell_fraction": float(bank.mask.mean()),
        "per_channel_observed_fraction": {
            name: float(bank.mask[:, :, i].mean()) for i, name in enumerate(CHANNELS)
        },
        "memory_mb_values": round(bank.values.nbytes / 1e6, 1),
    }


__all__ = [
    "DayBank", "SubjectGrid", "bank_summary", "build_day_bank",
    "build_subject_grids", "grid_fingerprint", "load_split_grids", "parse_series",
]
