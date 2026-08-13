"""Leakage-safe per-subject features: MMSE block + intraday circadian block.

Leakage argument
----------------
Every number produced here is a function of **one subject's own rows only**:
no cross-subject centering, no cohort quantile, no target encoding, no label.
A subject-level split of the resulting matrix therefore cannot leak through
feature construction.  ``tests/test_contracts.py::test_features_are_subject_local``
enforces this mechanically by rebuilding the matrix from a subject subset and
asserting bit-identical rows.

The intraday parsing (CONVERT columns) and the metric definitions follow the
audited implementation in ``Binary_Google_DemRankAUC_select1/features.py``;
this module keeps only the blocks used by this experiment's pre-registered
views (MMSE, MET circadian, hypnogram, nocturnal HR/RMSSD, sleep timing).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ACTIVITY_DAY_START_HOUR,
    FORBIDDEN_COLUMNS,
    FORBIDDEN_FEATURE_TOKENS,
    FORBIDDEN_SUBSTRINGS,
    HYPNOGRAM_STAGES,
    INTRADAY_COLUMNS,
    LOCAL_TZ,
    MET_MINUTES_PER_DAY,
    MMSE_DOMAINS,
    MMSE_EXCLUDED_ITEMS,
    MMSE_ITEMS,
    MMSE_ITEM_FAIL,
    MMSE_ITEM_PASS,
    NIGHT_EPOCH_MINUTES,
    SEDENTARY_MET,
    SOURCE_FILES,
    SPLIT_DIRS,
)
from .data import read_csv


# ------------------------------------------------------------------- guards --
def assert_no_forbidden(names) -> None:
    """Fail-closed: label/administrative/identifier/collection-proxy names."""

    bad = []
    for name in names:
        text = str(name)
        lowered = text.lower()
        if text in FORBIDDEN_COLUMNS:
            bad.append(text)
        elif any(token in lowered for token in FORBIDDEN_SUBSTRINGS):
            bad.append(text)
        elif any(token in lowered for token in FORBIDDEN_FEATURE_TOKENS):
            bad.append(text)
    if bad:
        raise AssertionError(f"Forbidden columns in feature matrix: {sorted(set(bad))}")


# ------------------------------------------------------------ numeric utils --
def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _safe(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("nan")


def _slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    index = np.arange(len(values), dtype=np.float64)
    mask = np.isfinite(values)
    if mask.sum() < 3:
        return float("nan")
    x, y = index[mask], values[mask]
    x_centered = x - x.mean()
    denominator = float((x_centered ** 2).sum())
    if denominator <= 1e-12:
        return float("nan")
    return _safe(float((x_centered * (y - y.mean())).sum() / denominator))


def _entropy(values: np.ndarray, bins: int = 16) -> float:
    values = _finite(values)
    if values.size < 4 or np.ptp(values) <= 1e-12:
        return float("nan")
    counts, _ = np.histogram(values, bins=bins)
    probabilities = counts[counts > 0] / counts.sum()
    return _safe(float(-(probabilities * np.log(probabilities)).sum()))


def _bout_lengths(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return np.zeros(0, dtype=np.float64)
    padded = np.concatenate([[False], mask, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return (changes[1::2] - changes[0::2]).astype(np.float64)


def _circular_stats(seconds: np.ndarray) -> tuple[float, float]:
    """Circular mean / SD of clock times, in hours (wraps at midnight)."""

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
    return float(mean_hours), float(circular_sd)


def _clock_seconds(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    try:
        parsed = parsed.dt.tz_convert(LOCAL_TZ)
    except (TypeError, AttributeError):  # pragma: no cover - non-datetime column
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (parsed.dt.hour * 3600 + parsed.dt.minute * 60 + parsed.dt.second).astype(float)


def _local_day(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    try:
        return parsed.dt.tz_convert(LOCAL_TZ).dt.normalize()
    except (TypeError, AttributeError):  # pragma: no cover
        return parsed


# ------------------------------------------------------------------ MMSE -----
def _mmse_table(data_root: Path, split: str) -> pd.DataFrame:
    frame = read_csv(Path(data_root) / SPLIT_DIRS[split] / SOURCE_FILES[split]["mmse"])
    id_column = next((c for c in ("SAMPLE_EMAIL", "EMAIL") if c in frame.columns), None)
    if id_column is None:
        raise KeyError("MMSE file lacks SAMPLE_EMAIL/EMAIL")
    frame = frame.copy()
    frame["_sid"] = frame[id_column].astype(str).str.strip()
    # Fail-closed: diagnosis / administrative columns are dropped before any
    # other code can reference them.
    frame = frame.drop(columns=[c for c in frame.columns if c in FORBIDDEN_COLUMNS])
    frame = frame.drop(columns=[c for c in MMSE_EXCLUDED_ITEMS if c in frame.columns])
    return frame.drop_duplicates("_sid").set_index("_sid")


def mmse_features(table: pd.DataFrame, *, zero_as_missing: bool = True) -> pd.DataFrame:
    """MMSE block: TOTAL, domain sums, item indicators, recall deficit.

    Items are coded 1 = failed / 2 = passed (asserted).  ``val_mmse.csv``
    contains one subject whose 30 items and TOTAL are all 0; zero is outside
    the instrument's coding, i.e. the test was not administered.  With
    ``zero_as_missing=True`` (default) out-of-domain values become NaN and the
    fold-local imputer treats them as missing -- a label-blind rule about the
    instrument, applied before any label is read.
    """

    numeric = table[list(MMSE_ITEMS)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    observed = np.unique(values[np.isfinite(values)])
    valid_codes = {MMSE_ITEM_FAIL, MMSE_ITEM_PASS}
    unexpected = sorted(set(observed.tolist()) - valid_codes - {0.0})
    if unexpected:
        raise AssertionError(
            f"MMSE item coding changed; expected {{0,1,2}}, saw {unexpected}"
        )

    out_of_domain = np.isfinite(values) & ~np.isin(values, list(valid_codes))
    if zero_as_missing and out_of_domain.any():
        numeric = numeric.mask(
            pd.DataFrame(out_of_domain, index=numeric.index, columns=numeric.columns)
        )

    out = pd.DataFrame(index=table.index)
    total = pd.to_numeric(table["TOTAL"], errors="coerce").astype(float)
    if zero_as_missing:
        total = total.mask(numeric.notna().sum(axis=1) == 0)
    out["mmse_TOTAL"] = total
    passed = (numeric == MMSE_ITEM_PASS).astype(float).mask(numeric.isna())
    for domain, items in MMSE_DOMAINS.items():
        out[f"mmse_{domain}"] = passed[list(items)].sum(axis=1)
    for item in MMSE_ITEMS:
        out[f"mmse_{item}"] = passed[item]
    out["mmse_recall_deficit"] = float(len(MMSE_DOMAINS["recall"])) - out["mmse_recall"]
    # Rows with no administered items carry no domain information either.
    no_items = numeric.notna().sum(axis=1) == 0
    if zero_as_missing and no_items.any():
        domain_cols = [f"mmse_{d}" for d in MMSE_DOMAINS] + ["mmse_recall_deficit"]
        out.loc[no_items, domain_cols] = np.nan
    return out


# ----------------------------------------------- wearable: intraday series ---
def _parse_series(text: object, *, expected: int | None = None) -> np.ndarray:
    """Slash-separated intraday string -> float array (empty when absent)."""

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
        "active_bout_mean": float("nan"), "sed_bout_mean": float("nan"),
        "active_frac": float("nan"),
    }
    values = _finite(met)
    if values.size < 60:
        return metrics
    active = values >= SEDENTARY_MET
    metrics["entropy"] = _entropy(values)
    metrics["transition_rate"] = _safe(
        float(np.count_nonzero(np.diff(active.astype(np.int8)) != 0)) / (values.size / 60.0)
    )
    active_bouts = _bout_lengths(active)
    sedentary_bouts = _bout_lengths(~active)
    if active_bouts.size:
        metrics["active_bout_mean"] = _safe(float(active_bouts.mean()))
    if sedentary_bouts.size:
        metrics["sed_bout_mean"] = _safe(float(sedentary_bouts.mean()))
    metrics["active_frac"] = _safe(float(active.mean()))
    return metrics


def _circadian_intraday(profiles: list[np.ndarray]) -> dict[str, float]:
    """IS / IV / RA / M10 / L5 from a subject's aligned 1-min MET days.

    Minute 0 of each day is 04:00 local (``activity_day_start`` is always
    04:00), so hourly bins are clock-aligned without any cohort reference.
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
            "trans_light_awake": float("nan"), "sleep_onset_epochs": float("nan"),
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
    metrics["frag_index"] = (
        _safe(float(np.count_nonzero(transitions != 0)) / hours) if hours > 0 else float("nan")
    )
    deep_bouts = _bout_lengths(values == 1)
    metrics["deep_bout_max"] = (
        _safe(float(deep_bouts.max()) * NIGHT_EPOCH_MINUTES) if deep_bouts.size else 0.0
    )
    pairs = np.stack([values[:-1], values[1:]], axis=1)
    denominator = max(1.0, hours)
    metrics["trans_light_awake"] = _safe(
        float(np.count_nonzero((pairs[:, 0] == 2) & (pairs[:, 1] == 4))) / denominator
    )
    return metrics


def _night_signal_metrics(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Nocturnal HR / RMSSD dynamics for one night; 0 encodes a missing epoch."""

    metrics = {
        f"{prefix}_mean": float("nan"), f"{prefix}_std": float("nan"),
        f"{prefix}_cv": float("nan"), f"{prefix}_slope": float("nan"),
        f"{prefix}_dip": float("nan"),
    }
    series = np.asarray(values, dtype=np.float64)
    series = series[np.isfinite(series)]
    series = series[series > 0]
    if series.size < 12:
        return metrics
    mean, std = float(series.mean()), float(series.std(ddof=0))
    metrics[f"{prefix}_mean"] = _safe(mean)
    metrics[f"{prefix}_std"] = _safe(std)
    metrics[f"{prefix}_cv"] = _safe(std / abs(mean)) if abs(mean) > 1e-12 else float("nan")
    metrics[f"{prefix}_slope"] = _safe(_slope(series) * (60.0 / NIGHT_EPOCH_MINUTES))
    early = series[: max(6, series.size // 6)]
    metrics[f"{prefix}_dip"] = _safe(float(early.mean() - series.min()))
    return metrics


def _aggregate_day_metrics(rows: list[dict[str, float]], prefix: str) -> dict[str, float]:
    """Mean and SD across a subject's own days."""

    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row})
    out: dict[str, float] = {}
    for key in keys:
        series = np.array([row.get(key, np.nan) for row in rows], dtype=np.float64)
        finite = _finite(series)
        out[f"{prefix}_{key}__mean"] = _safe(float(finite.mean())) if finite.size else float("nan")
        out[f"{prefix}_{key}__std"] = (
            _safe(float(finite.std(ddof=0))) if finite.size > 1 else float("nan")
        )
    return out


def _intraday_block(activity: pd.DataFrame, sleep: pd.DataFrame) -> pd.DataFrame:
    """The ``wi_`` family: everything derived from the CONVERT intraday strings."""

    activity = activity.copy()
    activity["_sid"] = activity["EMAIL"].astype(str).str.strip()
    activity["_day"] = _local_day(activity["activity_day_start"])
    sleep = sleep.copy()
    sleep["_sid"] = sleep["EMAIL"].astype(str).str.strip()
    sleep["_day"] = _local_day(sleep["sleep_bedtime_end"])

    met_column = INTRADAY_COLUMNS["met_1min"]
    hypnogram_column = INTRADAY_COLUMNS["hypnogram_5min"]
    hr_column = INTRADAY_COLUMNS["sleep_hr_5min"]
    rmssd_column = INTRADAY_COLUMNS["sleep_rmssd_5min"]
    for frame, column in ((activity, met_column), (sleep, hypnogram_column)):
        if column not in frame.columns:
            raise KeyError(f"Expected intraday column missing: {column}")

    records: dict[str, dict[str, float]] = {}

    for sid, group in activity.groupby("_sid", sort=True):
        group = group.sort_values("_day")
        met_profiles: list[np.ndarray] = []
        met_rows: list[dict[str, float]] = []
        for text in group[met_column]:
            series = _parse_series(text, expected=MET_MINUTES_PER_DAY)
            if series.size == MET_MINUTES_PER_DAY:
                met_profiles.append(series)
            met_rows.append(_met_day_metrics(series))
        record: dict[str, float] = {}
        record.update(_circadian_intraday(met_profiles))
        record.update(_aggregate_day_metrics(met_rows, "wi_met"))
        records.setdefault(str(sid), {}).update(record)

    for sid, group in sleep.groupby("_sid", sort=True):
        group = group.sort_values("_day")
        hypnogram_rows = [
            _hypnogram_night_metrics(_parse_series(text)) for text in group[hypnogram_column]
        ]
        record = {}
        record.update(_aggregate_day_metrics(hypnogram_rows, "wi_hyp"))
        if hr_column in group.columns:
            hr_rows = [
                _night_signal_metrics(_parse_series(text), "night") for text in group[hr_column]
            ]
            record.update(_aggregate_day_metrics(hr_rows, "wi_hr"))
        if rmssd_column in group.columns:
            rmssd_rows = [
                _night_signal_metrics(_parse_series(text), "night")
                for text in group[rmssd_column]
            ]
            record.update(_aggregate_day_metrics(rmssd_rows, "wi_rmssd"))
        records.setdefault(str(sid), {}).update(record)

    return pd.DataFrame.from_dict(records, orient="index")


def _sleep_timing_block(sleep: pd.DataFrame) -> pd.DataFrame:
    """Sleep-timing regularity (circular mean / SD of bedtime and waketime)."""

    frame = pd.DataFrame({"_sid": sleep["EMAIL"].astype(str).str.strip()})
    frame["bedtime"] = _clock_seconds(sleep["sleep_bedtime_start"]).to_numpy()
    frame["waketime"] = _clock_seconds(sleep["sleep_bedtime_end"]).to_numpy()

    records: dict[str, dict[str, float]] = {}
    for sid, group in frame.groupby("_sid", sort=True):
        record: dict[str, float] = {}
        for channel in ("bedtime", "waketime"):
            mean_hours, circular_sd = _circular_stats(group[channel].to_numpy(np.float64))
            record[f"wd_circ_{channel}__mean_h"] = mean_hours
            record[f"wd_circ_{channel}__circsd_h"] = circular_sd
        records[str(sid)] = record
    return pd.DataFrame.from_dict(records, orient="index")


# -------------------------------------------------------------------- API ----
def build_split_features(data_root: str | Path, split: str, *,
                         mmse_zero_as_missing: bool = True) -> pd.DataFrame:
    """Full per-subject feature matrix for one split, indexed by subject id."""

    data_root = Path(data_root)
    mmse = mmse_features(_mmse_table(data_root, split),
                         zero_as_missing=mmse_zero_as_missing)
    activity = read_csv(data_root / SPLIT_DIRS[split] / SOURCE_FILES[split]["activity"])
    sleep = read_csv(data_root / SPLIT_DIRS[split] / SOURCE_FILES[split]["sleep"])
    for frame, name in ((activity, "activity"), (sleep, "sleep")):
        if "EMAIL" not in frame.columns:
            raise KeyError(f"{name} file lacks EMAIL")

    blocks = [mmse, _intraday_block(activity, sleep), _sleep_timing_block(sleep)]
    subjects = sorted(mmse.index.astype(str))
    frame = pd.concat([block.reindex(subjects) for block in blocks], axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    frame.index = pd.Index(subjects, name="subject_id")

    assert_no_forbidden(frame.columns)
    return frame.astype(np.float64)


def select_view(features: pd.DataFrame, view_columns: tuple[str, ...]) -> pd.DataFrame:
    """Column subset for one pre-registered view; missing names are an error."""

    missing = [c for c in view_columns if c not in features.columns]
    if missing:
        raise KeyError(f"View columns absent from the built matrix: {missing}")
    return features.loc[:, list(view_columns)]


def drop_degenerate_columns(train: pd.DataFrame) -> list[str]:
    """Names of columns that are all-NaN or constant on the TRAIN cohort.

    Removing a constant column cannot transfer fold information (it is constant
    for every subject), but the decision is computed on the training cohort
    only and recorded in the report so the schema stays auditable.
    """

    values = train.to_numpy(dtype=np.float64)
    finite_ratio = np.isfinite(values).mean(axis=0)
    spread = np.nanstd(np.where(np.isfinite(values), values, np.nan), axis=0)
    degenerate = (finite_ratio <= 0.0) | (np.nan_to_num(spread, nan=0.0) <= 1e-12)
    return [name for name, flag in zip(train.columns, degenerate) if flag]


def feature_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(",".join(map(str, frame.columns)).encode("utf-8"))
    digest.update(",".join(map(str, frame.index)).encode("utf-8"))
    digest.update(np.ascontiguousarray(frame.to_numpy(dtype=np.float64)).tobytes())
    return digest.hexdigest()[:16]


__all__ = [
    "assert_no_forbidden", "build_split_features", "drop_degenerate_columns",
    "feature_fingerprint", "mmse_features", "select_view",
]
