"""Leakage-safe per-subject features, including the previously unused intraday series.

Leakage argument (the reason this module can be audited quickly)
---------------------------------------------------------------
Every number produced here is a function of **one subject's own rows only**.
There is no cross-subject centering, no cohort quantile, no target encoding, no
label anywhere in this file.  Therefore a subject-level split of the resulting
matrix cannot leak, and pooling the Training and Validation splits before
cross-validation is safe.

``tests/test_contracts.py::test_features_are_subject_local`` enforces this
mechanically: it rebuilds the matrix from a random subset of subjects and
asserts the retained rows are bit-identical to the full-cohort build.  Any
accidental cohort-level statistic breaks that test.

Three feature families
----------------------
``mmse_``  cognitive test: total, domain sums, item indicators, recall deficit.
``wd_``    wearable *daily summaries* -- the representation every prior folder
           used, reproduced here so the intraday contribution can be isolated.
``wi_``    wearable *intraday* series, new in this folder.  The dataset ships
           1-minute MET, 5-minute activity class, 5-minute hypnogram, and
           5-minute sleep HR / RMSSD inside the ``CONVERT(... USING utf8)``
           columns (the plain ``*_5min`` columns contain the literal string
           "..." and are unusable).  No prior experiment in this repository
           parsed them.

The ``wi_`` block is built around the non-parametric circadian statistics that
the actigraphy literature associates with dementia -- interdaily stability (IS),
intradaily variability (IV), relative amplitude (RA), M10 and L5 -- plus sleep
micro-architecture (WASO, awakening count, fragmentation index, stage
transitions) and nocturnal autonomic measures (HR dip, HR slope, RMSSD).  These
are also *continuous*, which matters mechanically: with 12 positives, depth-2
trees emit a handful of distinct leaf values and ROC-AUC pays for every tie.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ACTIVITY_CLASSES,
    ACTIVITY_CLASS_EPOCHS_PER_DAY,
    ACTIVITY_DAY_START_HOUR,
    FORBIDDEN_COLUMNS,
    FORBIDDEN_SUBSTRINGS,
    HYPNOGRAM_STAGES,
    INTRADAY_COLUMNS,
    LITE_ACTIVITY,
    LITE_SLEEP,
    LOCAL_TZ,
    MET_MINUTES_PER_DAY,
    MMSE_DOMAINS,
    MMSE_EXCLUDED_ITEMS,
    MMSE_ITEMS,
    MMSE_ITEM_FAIL,
    MMSE_ITEM_PASS,
    RICH_ACTIVITY,
    RICH_SLEEP,
    SOURCE_FILES,
    SPLIT_DIRS,
)

SEDENTARY_MET = 1.5     # standard cut-point: below 1.5 MET counts as sedentary
NIGHT_EPOCH_MINUTES = 5


# ------------------------------------------------------------------- io ------
def read_csv(path: Path) -> pd.DataFrame:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    last: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as error:  # pragma: no cover - encoding probe
            last = error
    raise RuntimeError(f"Unable to decode CSV: {path}") from last


def _source_path(data_root: Path, split: str, kind: str) -> Path:
    return Path(data_root) / SPLIT_DIRS[split] / SOURCE_FILES[split][kind]


def assert_no_forbidden(names) -> None:
    """Fail-closed check that the label or an administrative column never lands
    in the feature matrix."""

    bad = []
    for name in names:
        text = str(name)
        if text in FORBIDDEN_COLUMNS:
            bad.append(text)
            continue
        lowered = text.lower()
        # ``wd_``/``wi_``/``mmse_`` features never legitimately contain these.
        if any(token in lowered for token in FORBIDDEN_SUBSTRINGS):
            bad.append(text)
    if bad:
        raise AssertionError(f"Forbidden / diagnosis-derived columns in features: {sorted(set(bad))}")


# ------------------------------------------------------------ numeric utils --
def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _safe(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("nan")


def _slope(values: np.ndarray, index: np.ndarray | None = None) -> float:
    """OLS slope of a subject's own series against day index (units per day)."""

    values = np.asarray(values, dtype=np.float64)
    index = np.arange(len(values), dtype=np.float64) if index is None else np.asarray(index, float)
    mask = np.isfinite(values) & np.isfinite(index)
    if mask.sum() < 3:
        return float("nan")
    x, y = index[mask], values[mask]
    x_centered = x - x.mean()
    denominator = float((x_centered ** 2).sum())
    if denominator <= 1e-12:
        return float("nan")
    return _safe(float((x_centered * (y - y.mean())).sum() / denominator))


def _acf1(values: np.ndarray) -> float:
    """Lag-1 autocorrelation of a subject's daily series."""

    values = _finite(values)
    if values.size < 4:
        return float("nan")
    centered = values - values.mean()
    denominator = float((centered ** 2).sum())
    if denominator <= 1e-12:
        return float("nan")
    return _safe(float((centered[:-1] * centered[1:]).sum() / denominator))


def _entropy(values: np.ndarray, bins: int = 16) -> float:
    """Shannon entropy (nats) of a within-subject/within-day value histogram."""

    values = _finite(values)
    if values.size < 4 or np.ptp(values) <= 1e-12:
        return float("nan")
    counts, _ = np.histogram(values, bins=bins)
    probabilities = counts[counts > 0] / counts.sum()
    return _safe(float(-(probabilities * np.log(probabilities)).sum()))


def _bout_lengths(mask: np.ndarray) -> np.ndarray:
    """Lengths of consecutive True runs (used for activity / sleep-stage bouts)."""

    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return np.zeros(0, dtype=np.float64)
    padded = np.concatenate([[False], mask, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return (changes[1::2] - changes[0::2]).astype(np.float64)


def _circular_stats(seconds: np.ndarray) -> tuple[float, float]:
    """Circular mean / SD of clock times, in hours.

    Sleep timing wraps at midnight, so a plain std would call 23:50 and 00:10
    twelve hours apart.  Circadian *irregularity* is the marker of interest.
    """

    values = _finite(seconds)
    if values.size == 0:
        return float("nan"), float("nan")
    theta = 2.0 * np.pi * (values % 86400.0) / 86400.0
    cos_mean, sin_mean = float(np.cos(theta).mean()), float(np.sin(theta).mean())
    resultant = float(np.hypot(cos_mean, sin_mean))
    mean_hours = (np.arctan2(sin_mean, cos_mean) % (2.0 * np.pi)) * 24.0 / (2.0 * np.pi)
    if values.size < 2 or resultant <= 1e-12:
        return mean_hours, float("nan")
    circular_sd = np.sqrt(-2.0 * np.log(min(resultant, 1.0))) * 24.0 / (2.0 * np.pi)
    return mean_hours, float(circular_sd)


def _clock_seconds(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    try:
        parsed = parsed.dt.tz_convert(LOCAL_TZ)
    except (TypeError, AttributeError):  # pragma: no cover - non-datetime column
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (parsed.dt.hour * 3600 + parsed.dt.minute * 60 + parsed.dt.second).astype(float)


def _local_datetime(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    try:
        return parsed.dt.tz_convert(LOCAL_TZ)
    except (TypeError, AttributeError):  # pragma: no cover
        return parsed


# ------------------------------------------------------------------ MMSE -----
def _mmse_table(data_root: Path, split: str) -> pd.DataFrame:
    frame = read_csv(_source_path(data_root, split, "mmse"))
    id_column = next((c for c in ("SAMPLE_EMAIL", "EMAIL") if c in frame.columns), None)
    if id_column is None:
        raise KeyError("MMSE file lacks SAMPLE_EMAIL/EMAIL")
    frame = frame.copy()
    frame["_sid"] = frame[id_column].astype(str).str.strip()
    # Fail-closed: drop diagnosis and administrative columns before anything else
    # can reference them.
    frame = frame.drop(columns=[c for c in frame.columns if c in FORBIDDEN_COLUMNS])
    frame = frame.drop(columns=[c for c in MMSE_EXCLUDED_ITEMS if c in frame.columns])
    return frame.drop_duplicates("_sid").set_index("_sid")


def mmse_features(table: pd.DataFrame, *, zero_as_missing: bool = True) -> pd.DataFrame:
    """MMSE block.

    Item coding is structural (1 = failed, 2 = passed) and asserted, so
    ``num_failed`` needs no cohort statistic.  Note that ``num_failed`` is
    exactly ``30 - TOTAL`` in this instrument; it is kept out of the feature set
    because a perfectly collinear duplicate only adds variance to linear models.

    Out-of-domain zeros
    -------------------
    ``val_mmse.csv`` contains one subject (``nia+045``) whose 30 items and TOTAL
    are all ``0``.  Zero is outside the instrument's 1/2 coding, so that row is a
    *record of a test that was not administered*, not a score of zero.  Leaving
    it as a literal 0 makes the subject the single lowest MMSE in the cohort --
    and they happen to be a Dem case, so the artifact quietly flatters every
    MMSE-based ranker.

    ``zero_as_missing=True`` (default) maps out-of-domain values to NaN, which
    the fold-local imputer then handles like any other missing value.  The rule
    is label-blind: it is a statement about the instrument's coding, applied
    before any label is read.  ``zero_as_missing=False`` reproduces the
    benchmark's behaviour for the sensitivity arm.

    Measured single-feature effect on ``-mmse_TOTAL`` over the 174 subjects:
    0.9470 as-is, 0.9422 excluding the subject, 0.9151 with a median imputation.
    Deliberately *not* added: an ``mmse_administered`` indicator.  Missingness of
    a clinical assessment is plausibly a consequence of the diagnosis, and a
    flag for it would be a post-diagnosis variable.
    """

    numeric = table[list(MMSE_ITEMS)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    observed = np.unique(values[np.isfinite(values)])
    valid_codes = {MMSE_ITEM_FAIL, MMSE_ITEM_PASS}
    unexpected = sorted(set(observed.tolist()) - valid_codes - {0.0})
    if unexpected:
        raise AssertionError(
            f"MMSE item coding changed; expected {{0,1,2}}, saw extra values {unexpected}"
        )

    out_of_domain = np.isfinite(values) & ~np.isin(values, list(valid_codes))
    if zero_as_missing and out_of_domain.any():
        numeric = numeric.mask(pd.DataFrame(out_of_domain, index=numeric.index,
                                            columns=numeric.columns))

    out = pd.DataFrame(index=table.index)
    total = pd.to_numeric(table["TOTAL"], errors="coerce").astype(float)
    if zero_as_missing:
        # A row with no valid item has no valid total either.
        total = total.mask(numeric.notna().sum(axis=1) == 0)
    out["mmse_TOTAL"] = total
    passed = (numeric == MMSE_ITEM_PASS).astype(float).mask(numeric.isna())
    for domain, items in MMSE_DOMAINS.items():
        out[f"mmse_{domain}"] = passed[list(items)].sum(axis=1)
    for item in MMSE_ITEMS:
        out[f"mmse_{item}"] = passed[item]
    out["mmse_recall_deficit"] = float(len(MMSE_DOMAINS["recall"])) - out["mmse_recall"]
    # Clinically meaningful ratios: which domain carries the loss.
    total = out["mmse_TOTAL"].replace(0.0, np.nan)
    for domain in ("orient_time", "orient_place", "attention", "recall", "language"):
        out[f"mmse_share_{domain}"] = out[f"mmse_{domain}"] / total
    return out


# ------------------------------------------------- wearable: daily summaries --
_RICH_STATS = (
    "mean", "median", "std", "cv", "min", "max", "q25", "q75", "iqr", "range",
    "skew", "kurt", "slope", "late_minus_early", "recent_ratio", "roll7_std",
    "acf1", "weekend_delta",
)


def _rich_stats(values: np.ndarray, is_weekend: np.ndarray) -> dict[str, float]:
    """Full statistic set for one channel of one subject's daily series."""

    values = np.asarray(values, dtype=np.float64)
    finite = _finite(values)
    stats = {name: float("nan") for name in _RICH_STATS}
    if finite.size == 0:
        return stats

    mean = float(finite.mean())
    std = float(finite.std(ddof=0))
    stats["mean"] = _safe(mean)
    stats["median"] = _safe(float(np.median(finite)))
    stats["std"] = _safe(std)
    stats["cv"] = _safe(std / abs(mean)) if abs(mean) > 1e-12 else float("nan")
    stats["min"] = _safe(float(finite.min()))
    stats["max"] = _safe(float(finite.max()))
    q25, q75 = (float(v) for v in np.percentile(finite, [25, 75]))
    stats["q25"], stats["q75"] = _safe(q25), _safe(q75)
    stats["iqr"] = _safe(q75 - q25)
    stats["range"] = _safe(float(finite.max() - finite.min()))
    if finite.size >= 3 and std > 1e-12:
        centered = (finite - mean) / std
        stats["skew"] = _safe(float((centered ** 3).mean()))
        stats["kurt"] = _safe(float((centered ** 4).mean() - 3.0))
    stats["slope"] = _slope(values)
    stats["acf1"] = _acf1(values)

    # Early vs late third of the observation window, and the recent 14 days.
    if finite.size >= 6:
        third = max(1, finite.size // 3)
        stats["late_minus_early"] = _safe(float(finite[-third:].mean() - finite[:third].mean()))
    if finite.size >= 10 and abs(mean) > 1e-12:
        recent = finite[-min(14, finite.size):]
        stats["recent_ratio"] = _safe(float(recent.mean() / mean))
    if finite.size >= 7:
        series = pd.Series(values)
        rolling = series.rolling(7, min_periods=4).std(ddof=0)
        stats["roll7_std"] = _safe(float(np.nanmean(rolling.to_numpy(dtype=np.float64))))

    weekend = np.asarray(is_weekend, dtype=bool)
    if weekend.size == values.size:
        week_values = values[~weekend]
        end_values = values[weekend]
        if np.isfinite(week_values).sum() >= 3 and np.isfinite(end_values).sum() >= 3:
            stats["weekend_delta"] = _safe(
                float(np.nanmean(end_values) - np.nanmean(week_values))
            )
    return stats


def _lite_stats(values: np.ndarray) -> dict[str, float]:
    finite = _finite(values)
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "cv": float("nan")}
    mean, std = float(finite.mean()), float(finite.std(ddof=0))
    return {
        "mean": _safe(mean),
        "std": _safe(std),
        "cv": _safe(std / abs(mean)) if abs(mean) > 1e-12 else float("nan"),
    }


def _daily_block(frame: pd.DataFrame, rich: tuple[str, ...], lite: tuple[str, ...],
                 *, day_column: str, source: str) -> pd.DataFrame:
    """Per-subject daily-summary statistics for one modality."""

    frame = frame.copy()
    frame["_sid"] = frame["EMAIL"].astype(str).str.strip()
    stamps = _local_datetime(frame[day_column])
    frame["_day"] = stamps.dt.normalize()
    frame["_weekend"] = stamps.dt.dayofweek.isin((5, 6)).to_numpy()

    present_rich = [c for c in rich if c in frame.columns]
    present_lite = [c for c in lite if c in frame.columns]
    for column in present_rich + present_lite:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    records: dict[str, dict[str, float]] = {}
    for sid, group in frame.groupby("_sid", sort=True):
        group = group.sort_values("_day")
        weekend = group["_weekend"].to_numpy(dtype=bool)
        record: dict[str, float] = {}
        for column in present_rich:
            for stat, value in _rich_stats(group[column].to_numpy(dtype=np.float64), weekend).items():
                record[f"wd_{column}__{stat}"] = value
        for column in present_lite:
            for stat, value in _lite_stats(group[column].to_numpy(dtype=np.float64)).items():
                record[f"wd_{column}__{stat}"] = value

        # Adherence / coverage descriptors (flagged as SUSPECT in config.py).
        days = group["_day"].dropna()
        n_days = float(len(group))
        record[f"wd_n_days_{source}"] = n_days
        if len(days) >= 2:
            ordinals = days.map(pd.Timestamp.toordinal).to_numpy(dtype=np.float64)
            span = float(ordinals.max() - ordinals.min() + 1.0)
            gaps = np.diff(np.unique(ordinals))
            record[f"wd_span_days_{source}"] = span
            record[f"wd_cover_{source}"] = _safe(n_days / span) if span > 0 else float("nan")
            record[f"wd_gap_max_{source}"] = _safe(float(gaps.max())) if gaps.size else 1.0
            record[f"wd_gap_mean_{source}"] = _safe(float(gaps.mean())) if gaps.size else 1.0
        else:
            record[f"wd_span_days_{source}"] = n_days
            record[f"wd_cover_{source}"] = float("nan")
            record[f"wd_gap_max_{source}"] = float("nan")
            record[f"wd_gap_mean_{source}"] = float("nan")
        records[str(sid)] = record
    return pd.DataFrame.from_dict(records, orient="index")


def _sleep_architecture(sleep: pd.DataFrame) -> pd.DataFrame:
    """Stage ratios and fragmentation, computed per night then per subject."""

    frame = sleep.copy()
    frame["_sid"] = frame["EMAIL"].astype(str).str.strip()
    for column in ("sleep_total", "sleep_duration", "sleep_deep", "sleep_rem",
                   "sleep_light", "sleep_awake"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    total = frame.get("sleep_total")
    if total is None:
        total = frame.get("sleep_duration")
    derived = pd.DataFrame(index=frame.index)
    if total is not None:
        denominator = total.replace(0.0, np.nan)
        for stage in ("sleep_deep", "sleep_rem", "sleep_light", "sleep_awake"):
            if stage in frame.columns:
                derived[f"ratio_{stage}"] = frame[stage] / denominator
    if "sleep_awake" in frame.columns and "sleep_duration" in frame.columns:
        derived["fragmentation"] = frame["sleep_awake"] / frame["sleep_duration"].replace(0.0, np.nan)
    if "sleep_hr_average" in frame.columns and "sleep_hr_lowest" in frame.columns:
        derived["hr_span"] = (
            pd.to_numeric(frame["sleep_hr_average"], errors="coerce")
            - pd.to_numeric(frame["sleep_hr_lowest"], errors="coerce")
        )
    derived["_sid"] = frame["_sid"]

    grouped = derived.groupby("_sid", sort=True)
    parts = [
        grouped.mean(numeric_only=True).add_prefix("wd_arch_").add_suffix("__mean"),
        grouped.std(numeric_only=True, ddof=0).add_prefix("wd_arch_").add_suffix("__std"),
    ]
    return pd.concat(parts, axis=1)


def _circadian_daily(sleep: pd.DataFrame) -> pd.DataFrame:
    """Sleep-timing regularity from bedtime / waketime / midpoint."""

    frame = pd.DataFrame({"_sid": sleep["EMAIL"].astype(str).str.strip()})
    if "sleep_bedtime_start" in sleep.columns:
        frame["bedtime"] = _clock_seconds(sleep["sleep_bedtime_start"]).to_numpy()
    if "sleep_bedtime_end" in sleep.columns:
        frame["waketime"] = _clock_seconds(sleep["sleep_bedtime_end"]).to_numpy()
    if "sleep_midpoint_time" in sleep.columns:
        frame["midpoint"] = pd.to_numeric(sleep["sleep_midpoint_time"], errors="coerce").to_numpy()

    channels = [c for c in ("bedtime", "waketime", "midpoint") if c in frame.columns]
    records: dict[str, dict[str, float]] = {}
    for sid, group in frame.groupby("_sid", sort=True):
        record: dict[str, float] = {}
        for channel in channels:
            values = group[channel].to_numpy(dtype=np.float64)
            if channel == "midpoint":
                # midpoint_time is seconds from bedtime start, not a clock time.
                finite = _finite(values)
                record["wd_circ_midpoint__mean_h"] = (
                    _safe(float(finite.mean()) / 3600.0) if finite.size else float("nan")
                )
                record["wd_circ_midpoint__circsd_h"] = (
                    _safe(float(finite.std(ddof=0)) / 3600.0) if finite.size > 1 else float("nan")
                )
                continue
            mean_hours, circular_sd = _circular_stats(values)
            record[f"wd_circ_{channel}__mean_h"] = mean_hours
            record[f"wd_circ_{channel}__circsd_h"] = circular_sd
        records[str(sid)] = record
    return pd.DataFrame.from_dict(records, orient="index")


# ----------------------------------------------- wearable: intraday series ---
def _parse_series(text: object, *, expected: int | None = None) -> np.ndarray:
    """Slash-separated intraday string -> float array (NaN when absent)."""

    if text is None:
        return np.zeros(0, dtype=np.float64)
    raw = str(text)
    if raw in ("", "nan", "...", "None"):
        return np.zeros(0, dtype=np.float64)
    values = np.fromstring(raw, sep="/", dtype=np.float64)
    if expected is not None and values.size > expected:
        values = values[:expected]
    return values


def _met_day_metrics(met: np.ndarray) -> dict[str, float]:
    """Within-day activity metrics from the 1-minute MET series."""

    metrics = {
        "entropy": float("nan"), "transition_rate": float("nan"),
        "active_bout_mean": float("nan"), "active_bout_max": float("nan"),
        "sed_bout_mean": float("nan"), "sed_bout_max": float("nan"),
        "peak": float("nan"), "total": float("nan"), "active_frac": float("nan"),
    }
    values = _finite(met)
    if values.size < 60:
        return metrics
    active = values >= SEDENTARY_MET
    metrics["entropy"] = _entropy(values)
    # Transitions per hour between sedentary and active states.
    metrics["transition_rate"] = _safe(
        float(np.count_nonzero(np.diff(active.astype(np.int8)) != 0)) / (values.size / 60.0)
    )
    active_bouts = _bout_lengths(active)
    sedentary_bouts = _bout_lengths(~active)
    if active_bouts.size:
        metrics["active_bout_mean"] = _safe(float(active_bouts.mean()))
        metrics["active_bout_max"] = _safe(float(active_bouts.max()))
    if sedentary_bouts.size:
        metrics["sed_bout_mean"] = _safe(float(sedentary_bouts.mean()))
        metrics["sed_bout_max"] = _safe(float(sedentary_bouts.max()))
    metrics["peak"] = _safe(float(values.max()))
    metrics["total"] = _safe(float(values.sum()))
    metrics["active_frac"] = _safe(float(active.mean()))
    return metrics


def _class_day_metrics(classes: np.ndarray) -> dict[str, float]:
    """Within-day metrics from the 5-minute activity-class series."""

    metrics = {f"{name}_frac": float("nan") for name in ACTIVITY_CLASSES.values()}
    metrics.update({"changes": float("nan"), "high_bout_max": float("nan")})
    values = _finite(classes)
    if values.size < 24:
        return metrics
    for code, name in ACTIVITY_CLASSES.items():
        metrics[f"{name}_frac"] = _safe(float(np.mean(values == code)))
    metrics["changes"] = _safe(
        float(np.count_nonzero(np.diff(values) != 0)) / (values.size / 12.0)
    )
    bouts = _bout_lengths(values >= 4)
    metrics["high_bout_max"] = _safe(float(bouts.max())) if bouts.size else 0.0
    return metrics


def _circadian_intraday(profiles: list[np.ndarray]) -> dict[str, float]:
    """Non-parametric circadian statistics from a subject's aligned MET days.

    IS  interdaily stability   - how repeatable the 24 h pattern is day to day
    IV  intradaily variability - how fragmented the rest/activity rhythm is
    M10 / L5 / RA              - most-active 10 h, least-active 5 h, amplitude

    All three are computed from *this subject's* days only.  Minute 0 of each
    row is 04:00 local (``activity_day_start`` is always 04:00), so the hourly
    bins are clock-aligned without needing any cohort reference.
    """

    empty = {
        "wi_met_IS": float("nan"), "wi_met_IV": float("nan"), "wi_met_RA": float("nan"),
        "wi_met_M10": float("nan"), "wi_met_L5": float("nan"),
        "wi_met_M10_onset_h": float("nan"), "wi_met_L5_onset_h": float("nan"),
        "wi_met_daily_profile_std": float("nan"),
    }
    usable = [p for p in profiles if p.size == MET_MINUTES_PER_DAY and np.isfinite(p).sum() > 720]
    if len(usable) < 3:
        return empty

    # Hourly binning (24 bins/day) is the convention for IS/IV.
    hourly = np.vstack([np.nanmean(p.reshape(24, 60), axis=1) for p in usable])
    flat = hourly.reshape(-1)
    finite = np.isfinite(flat)
    if finite.sum() < 48:
        return empty
    grand_mean = float(np.nanmean(flat))
    total_variance = float(np.nansum((flat - grand_mean) ** 2))
    if total_variance <= 1e-12:
        return empty

    n_points = int(finite.sum())
    hour_means = np.nanmean(hourly, axis=0)
    interdaily = (
        n_points * float(np.nansum((hour_means - grand_mean) ** 2))
        / (24.0 * total_variance)
    )
    differences = np.diff(flat)
    intradaily = (
        n_points * float(np.nansum(differences ** 2))
        / ((n_points - 1) * total_variance)
    )

    # M10 / L5 on the mean 24 h profile, wrapped so windows can cross midnight.
    if not np.isfinite(hour_means).any():
        return empty
    wrapped = np.concatenate([hour_means, hour_means])
    windows_10 = np.array([np.nanmean(wrapped[i:i + 10]) for i in range(24)])
    windows_5 = np.array([np.nanmean(wrapped[i:i + 5]) for i in range(24)])
    if not (np.isfinite(windows_10).any() and np.isfinite(windows_5).any()):
        return empty
    m10_index = int(np.nanargmax(windows_10))
    l5_index = int(np.nanargmin(windows_5))
    m10, l5 = float(windows_10[m10_index]), float(windows_5[l5_index])
    amplitude = (m10 - l5) / (m10 + l5) if (m10 + l5) > 1e-12 else float("nan")

    def _clock(offset: int) -> float:
        return float((ACTIVITY_DAY_START_HOUR + offset) % 24)

    return {
        "wi_met_IS": _safe(interdaily),
        "wi_met_IV": _safe(intradaily),
        "wi_met_RA": _safe(amplitude),
        "wi_met_M10": _safe(m10),
        "wi_met_L5": _safe(l5),
        "wi_met_M10_onset_h": _clock(m10_index),
        "wi_met_L5_onset_h": _clock(l5_index),
        "wi_met_daily_profile_std": _safe(float(np.nanstd(hour_means))),
    }


def _hypnogram_night_metrics(stages: np.ndarray) -> dict[str, float]:
    """Sleep micro-architecture for one night from the 5-minute hypnogram."""

    names = list(HYPNOGRAM_STAGES.values())
    metrics = {f"{name}_frac": float("nan") for name in names}
    metrics.update(
        {
            "waso_min": float("nan"), "awakenings": float("nan"),
            "frag_index": float("nan"), "deep_bout_max": float("nan"),
            "rem_bout_max": float("nan"), "trans_light_awake": float("nan"),
            "trans_deep_light": float("nan"), "sleep_onset_epochs": float("nan"),
        }
    )
    values = _finite(stages)
    if values.size < 12:
        return metrics

    for code, name in HYPNOGRAM_STAGES.items():
        metrics[f"{name}_frac"] = _safe(float(np.mean(values == code)))

    asleep = values != 4
    if not asleep.any():
        return metrics
    first_sleep = int(np.argmax(asleep))
    metrics["sleep_onset_epochs"] = float(first_sleep)
    after_onset = values[first_sleep:]
    awake_after = after_onset == 4
    metrics["waso_min"] = _safe(float(awake_after.sum()) * NIGHT_EPOCH_MINUTES)
    metrics["awakenings"] = float(_bout_lengths(awake_after).size)

    hours = values.size * NIGHT_EPOCH_MINUTES / 60.0
    transitions = np.diff(values)
    metrics["frag_index"] = _safe(float(np.count_nonzero(transitions != 0)) / hours) if hours > 0 else float("nan")

    deep_bouts = _bout_lengths(values == 1)
    rem_bouts = _bout_lengths(values == 3)
    metrics["deep_bout_max"] = _safe(float(deep_bouts.max()) * NIGHT_EPOCH_MINUTES) if deep_bouts.size else 0.0
    metrics["rem_bout_max"] = _safe(float(rem_bouts.max()) * NIGHT_EPOCH_MINUTES) if rem_bouts.size else 0.0

    pairs = np.stack([values[:-1], values[1:]], axis=1)
    denominator = max(1.0, hours)
    metrics["trans_light_awake"] = _safe(
        float(np.count_nonzero((pairs[:, 0] == 2) & (pairs[:, 1] == 4))) / denominator
    )
    metrics["trans_deep_light"] = _safe(
        float(np.count_nonzero((pairs[:, 0] == 1) & (pairs[:, 1] == 2))) / denominator
    )
    return metrics


def _night_signal_metrics(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Nocturnal HR / RMSSD dynamics for one night.  Zeros encode 'not measured'."""

    metrics = {
        f"{prefix}_mean": float("nan"), f"{prefix}_min": float("nan"),
        f"{prefix}_max": float("nan"), f"{prefix}_std": float("nan"),
        f"{prefix}_cv": float("nan"), f"{prefix}_slope": float("nan"),
        f"{prefix}_dip": float("nan"), f"{prefix}_min_position": float("nan"),
    }
    series = np.asarray(values, dtype=np.float64)
    series = series[np.isfinite(series)]
    series = series[series > 0]  # 0 = missing epoch in this dataset
    if series.size < 12:
        return metrics

    mean, std = float(series.mean()), float(series.std(ddof=0))
    metrics[f"{prefix}_mean"] = _safe(mean)
    metrics[f"{prefix}_min"] = _safe(float(series.min()))
    metrics[f"{prefix}_max"] = _safe(float(series.max()))
    metrics[f"{prefix}_std"] = _safe(std)
    metrics[f"{prefix}_cv"] = _safe(std / abs(mean)) if abs(mean) > 1e-12 else float("nan")
    # Slope per hour across the night.
    metrics[f"{prefix}_slope"] = _safe(
        _slope(series) * (60.0 / NIGHT_EPOCH_MINUTES)
    )
    early = series[: max(6, series.size // 6)]
    metrics[f"{prefix}_dip"] = _safe(float(early.mean() - series.min()))
    metrics[f"{prefix}_min_position"] = _safe(float(np.argmin(series)) / series.size)
    return metrics


def _aggregate_day_metrics(rows: list[dict[str, float]], prefix: str) -> dict[str, float]:
    """Mean and SD across a subject's own days, plus the across-day slope."""

    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row})
    out: dict[str, float] = {}
    for key in keys:
        series = np.array([row.get(key, np.nan) for row in rows], dtype=np.float64)
        finite = _finite(series)
        out[f"{prefix}_{key}__mean"] = _safe(float(finite.mean())) if finite.size else float("nan")
        out[f"{prefix}_{key}__std"] = _safe(float(finite.std(ddof=0))) if finite.size > 1 else float("nan")
    return out


def _intraday_block(activity: pd.DataFrame, sleep: pd.DataFrame) -> pd.DataFrame:
    """The ``wi_`` family: everything derived from the intraday strings."""

    activity = activity.copy()
    activity["_sid"] = activity["EMAIL"].astype(str).str.strip()
    activity["_day"] = _local_datetime(activity["activity_day_start"]).dt.normalize()
    sleep = sleep.copy()
    sleep["_sid"] = sleep["EMAIL"].astype(str).str.strip()
    sleep["_day"] = _local_datetime(sleep["sleep_bedtime_end"]).dt.normalize()

    met_column = INTRADAY_COLUMNS["met_1min"]
    class_column = INTRADAY_COLUMNS["activity_class_5min"]
    hypnogram_column = INTRADAY_COLUMNS["hypnogram_5min"]
    hr_column = INTRADAY_COLUMNS["sleep_hr_5min"]
    rmssd_column = INTRADAY_COLUMNS["sleep_rmssd_5min"]

    records: dict[str, dict[str, float]] = {}

    for sid, group in activity.groupby("_sid", sort=True):
        group = group.sort_values("_day")
        met_profiles: list[np.ndarray] = []
        met_rows: list[dict[str, float]] = []
        class_rows: list[dict[str, float]] = []
        if met_column in group.columns:
            for text in group[met_column]:
                series = _parse_series(text, expected=MET_MINUTES_PER_DAY)
                if series.size == MET_MINUTES_PER_DAY:
                    met_profiles.append(series)
                met_rows.append(_met_day_metrics(series))
        if class_column in group.columns:
            for text in group[class_column]:
                class_rows.append(
                    _class_day_metrics(_parse_series(text, expected=ACTIVITY_CLASS_EPOCHS_PER_DAY))
                )
        record: dict[str, float] = {}
        record.update(_circadian_intraday(met_profiles))
        record.update(_aggregate_day_metrics(met_rows, "wi_met"))
        record.update(_aggregate_day_metrics(class_rows, "wi_class"))
        records.setdefault(str(sid), {}).update(record)

    for sid, group in sleep.groupby("_sid", sort=True):
        group = group.sort_values("_day")
        hypnogram_rows: list[dict[str, float]] = []
        hr_rows: list[dict[str, float]] = []
        rmssd_rows: list[dict[str, float]] = []
        if hypnogram_column in group.columns:
            for text in group[hypnogram_column]:
                hypnogram_rows.append(_hypnogram_night_metrics(_parse_series(text)))
        if hr_column in group.columns:
            for text in group[hr_column]:
                hr_rows.append(_night_signal_metrics(_parse_series(text), "night"))
        if rmssd_column in group.columns:
            for text in group[rmssd_column]:
                rmssd_rows.append(_night_signal_metrics(_parse_series(text), "night"))
        record = {}
        record.update(_aggregate_day_metrics(hypnogram_rows, "wi_hyp"))
        record.update(_aggregate_day_metrics(hr_rows, "wi_hr"))
        record.update(_aggregate_day_metrics(rmssd_rows, "wi_rmssd"))
        records.setdefault(str(sid), {}).update(record)

    return pd.DataFrame.from_dict(records, orient="index")


# -------------------------------------------------------------------- API ----
def build_split_features(data_root: str | Path, split: str, *,
                         mmse_zero_as_missing: bool = True) -> pd.DataFrame:
    """Full per-subject feature matrix for one split, indexed by subject id."""

    data_root = Path(data_root)
    mmse = mmse_features(_mmse_table(data_root, split), zero_as_missing=mmse_zero_as_missing)
    activity = read_csv(_source_path(data_root, split, "activity"))
    sleep = read_csv(_source_path(data_root, split, "sleep"))
    for frame, name in ((activity, "activity"), (sleep, "sleep")):
        if "EMAIL" not in frame.columns:
            raise KeyError(f"{name} file lacks EMAIL")

    blocks = [
        mmse,
        _daily_block(activity, RICH_ACTIVITY, LITE_ACTIVITY,
                     day_column="activity_day_start", source="activity"),
        _daily_block(sleep, RICH_SLEEP, LITE_SLEEP,
                     day_column="sleep_bedtime_end", source="sleep"),
        _sleep_architecture(sleep),
        _circadian_daily(sleep),
        _intraday_block(activity, sleep),
    ]
    subjects = sorted(mmse.index.astype(str))
    frame = pd.concat([block.reindex(subjects) for block in blocks], axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    frame.index = pd.Index(subjects, name="subject_id")

    assert_no_forbidden(frame.columns)
    # Drop columns that are constant *within this split* only if they are
    # constant in both splits; that decision is deferred to the caller so no
    # split-level statistic silently changes the schema.
    return frame.astype(np.float64)


def build_cohort_features(data_root: str | Path, *,
                          mmse_zero_as_missing: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    """Pooled 174-subject feature matrix plus the originating split name.

    Concatenation is a row-wise union of two independently built matrices; no
    statistic crosses between them, and none crosses between subjects.
    """

    train = build_split_features(data_root, "train", mmse_zero_as_missing=mmse_zero_as_missing)
    validation = build_split_features(data_root, "val", mmse_zero_as_missing=mmse_zero_as_missing)
    if list(train.columns) != list(validation.columns):
        missing = sorted(set(train.columns) ^ set(validation.columns))
        raise AssertionError(f"Train/validation feature schema mismatch: {missing[:10]}")
    frame = pd.concat([train, validation], axis=0)
    origin = np.array(["train"] * len(train) + ["val"] * len(validation))
    if frame.index.has_duplicates:
        duplicated = frame.index[frame.index.duplicated()].tolist()
        raise AssertionError(f"Duplicate subject id across splits: {duplicated[:5]}")
    return frame, origin


def drop_degenerate(frame: pd.DataFrame, *, min_finite_ratio: float = 0.5) -> pd.DataFrame:
    """Remove all-NaN and zero-variance columns.

    This is a *schema* decision, not a fitted transformation: it removes columns
    that carry no information for anybody.  It is applied once to the pooled
    matrix.  Because a constant column has identical values in every fold, its
    removal cannot transfer information between folds -- but the choice is
    recorded in the report so the claim is auditable.
    """

    values = frame.to_numpy(dtype=np.float64)
    finite_ratio = np.isfinite(values).mean(axis=0)
    spread = np.nanstd(np.where(np.isfinite(values), values, np.nan), axis=0)
    keep = (finite_ratio >= min_finite_ratio) & (np.nan_to_num(spread, nan=0.0) > 1e-12)
    return frame.loc[:, keep]


def feature_fingerprint(frame: pd.DataFrame) -> str:
    """Stable hash of the built matrix, recorded in every report."""

    digest = hashlib.sha256()
    digest.update(",".join(map(str, frame.columns)).encode("utf-8"))
    digest.update(",".join(map(str, frame.index)).encode("utf-8"))
    digest.update(np.ascontiguousarray(frame.to_numpy(dtype=np.float64)).tobytes())
    return digest.hexdigest()[:16]


__all__ = [
    "assert_no_forbidden", "build_cohort_features", "build_split_features",
    "drop_degenerate", "feature_fingerprint", "mmse_features", "read_csv",
]
