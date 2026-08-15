"""The paper's supervised baseline features (SensorFM M.3.6 / Table ED.14).

For each subject-day and each of our 8 grid channels we compute 21 daily
summary statistics tracking Table ED.14's four families, then aggregate per
subject with mean and std across days (matching the M.3.4 person-level
aggregation used for embeddings):

    distributional  mean, std, median, iqr, skew, kurtosis, cv, rms, p05,
                    p95, proportion of exact zeros among valid points
    volatility      mean |minute-to-minute diff| ("Mean Abs Change"), RMSSD
    morphology      mean-centered zero-crossing rate, Hjorth complexity
    chronobiology   cosinor mesor / amplitude / acrophase (sin, cos), IV
                    (= var(first derivative)/var(signal), per the paper),
                    lag-1 autocorrelation of the interpolated signal

Documented deviations from Table ED.14 (README_KO.md D6): the paper's
"Missing Rate" feature is a collection-process proxy and is forbidden by this
repository's contract, so it is dropped; the acrophase is encoded as (sin,
cos) instead of a raw angle to avoid the circular discontinuity; the cosinor
mesor is additionally kept.

As in the paper, derivative-based metrics run on a linearly interpolated
series (back/forward filled at the edges); distributional metrics use the raw
observed minutes.  Everything is subject-local: no cross-subject statistic,
no label.  Fold-local imputation / scaling / PCA live in the probe pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CHANNELS, MINUTES_PER_DAY, N_CHANNELS
from .grids import DayBank

FE_STATS = (
    "mean", "std", "median", "iqr", "skew", "kurtosis", "cv", "rms",
    "p05", "p95", "prop_zeros",
    "mad1", "rmssd", "zcr", "hjorth_complexity",
    "cosinor_mesor", "cosinor_amplitude", "cosinor_acro_sin", "cosinor_acro_cos",
    "iv", "lag1_autocorr",
)
N_FE_STATS = len(FE_STATS)

_MINUTE_ANGLE = 2.0 * np.pi * np.arange(MINUTES_PER_DAY) / MINUTES_PER_DAY
_COS_T = np.cos(_MINUTE_ANGLE)
_SIN_T = np.sin(_MINUTE_ANGLE)


def _interpolate(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Linear interpolation over the day with back/forward edge filling."""

    if not observed.any():
        return np.full(values.shape, np.nan)
    x = np.flatnonzero(observed)
    return np.interp(np.arange(values.size), x, values[x])


def _day_channel_stats(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    out = np.full(N_FE_STATS, np.nan, dtype=np.float64)
    v = values[observed].astype(np.float64)
    if v.size < 30:  # under half an hour of data: day carries no signal
        return out

    mean = float(v.mean())
    std = float(v.std(ddof=0))
    out[0] = mean
    out[1] = std
    out[2] = float(np.median(v))
    q75, q25 = np.percentile(v, [75, 25])
    out[3] = float(q75 - q25)
    if std > 1e-12:
        centered = (v - mean) / std
        out[4] = float(np.mean(centered**3))
        out[5] = float(np.mean(centered**4) - 3.0)
        out[6] = std / abs(mean) if abs(mean) > 1e-12 else np.nan
    out[7] = float(np.sqrt(np.mean(v**2)))
    out[8] = float(np.percentile(v, 5))
    out[9] = float(np.percentile(v, 95))
    out[10] = float(np.mean(v == 0.0))  # "Proportion Zeros" among valid points

    filled = _interpolate(values.astype(np.float64), observed)
    diffs = np.diff(filled)
    out[11] = float(np.mean(np.abs(diffs)))
    out[12] = float(np.sqrt(np.mean(diffs**2)))

    centered_series = filled - filled.mean()
    signs = np.sign(centered_series)
    signs[signs == 0] = 1.0
    out[13] = float(np.count_nonzero(np.diff(signs) != 0)) / max(1, filled.size - 1)
    var0 = float(centered_series.var())
    var1 = float(diffs.var())
    if var0 > 1e-12 and var1 > 1e-12:
        mobility = np.sqrt(var1 / var0)
        var2 = float(np.diff(diffs).var())
        if var2 > 1e-12 and mobility > 1e-12:
            out[14] = float(np.sqrt(var2 / var1) / mobility)

    # 24-h cosinor on observed minutes: y = M + a*cos(theta) + b*sin(theta)
    design = np.column_stack(
        [np.ones(v.size), _COS_T[observed], _SIN_T[observed]]
    )
    try:
        coef, *_ = np.linalg.lstsq(design, v, rcond=None)
        mesor, a, b = (float(c) for c in coef)
        amplitude = float(np.hypot(a, b))
        out[15] = mesor
        out[16] = amplitude
        if amplitude > 1e-12:
            out[17] = b / amplitude   # sin(acrophase)
            out[18] = a / amplitude   # cos(acrophase)
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate design
        pass

    # Table ED.14: IV = var(first derivative) / var(signal); ACF_1 on the
    # interpolated, mean-centered signal.
    if var0 > 1e-12:
        out[19] = var1 / var0
        out[20] = float(
            np.dot(centered_series[:-1], centered_series[1:])
            / max(1e-12, np.dot(centered_series, centered_series))
        )
    return out


def fe_feature_names() -> list[str]:
    return [
        f"fe_{channel}__{stat}__{agg}"
        for channel in CHANNELS for stat in FE_STATS for agg in ("mean", "std")
    ]


def build_fe_features(bank: DayBank) -> pd.DataFrame:
    """Per-subject FE matrix (paper M.3.6 baseline), indexed by subject id."""

    n_subjects = len(bank.subject_ids)
    per_day = np.full((bank.day_subject.size, N_CHANNELS, N_FE_STATS), np.nan)
    for day in range(bank.day_subject.size):
        for channel in range(N_CHANNELS):
            per_day[day, channel] = _day_channel_stats(
                bank.values[day, :, channel], bank.mask[day, :, channel]
            )

    rows = np.zeros((n_subjects, N_CHANNELS * N_FE_STATS * 2), dtype=np.float64)
    for index in range(n_subjects):
        day_stats = per_day[bank.day_subject == index]  # (n_days, C, S)
        with np.errstate(invalid="ignore"):
            means = np.nanmean(day_stats, axis=0)
            stds = np.nanstd(day_stats, axis=0, ddof=0)
        stacked = np.stack([means, stds], axis=-1)      # (C, S, 2)
        rows[index] = stacked.reshape(-1)

    frame = pd.DataFrame(rows, index=pd.Index(bank.subject_ids, name="subject_id"),
                         columns=fe_feature_names())
    return frame.astype(np.float64)


__all__ = ["FE_STATS", "N_FE_STATS", "build_fe_features", "fe_feature_names"]
