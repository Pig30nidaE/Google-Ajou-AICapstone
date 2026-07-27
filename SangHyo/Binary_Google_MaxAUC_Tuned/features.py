"""Rich, leakage-safe per-subject feature construction for the max-AUC experiment.

Design notes
------------
The earlier ``Binary_MMSE_MaxAUC`` run topped out at subject-level OOF ROC-AUC
0.766 using **MMSE only** (39 features).  Its own EDA, however, showed the
MMSE+wearable feature set scoring slightly *higher* in a quick CV (0.752 vs
0.735), i.e. the wearables were not actually dead weight -- the earlier folder
just used a very thin wearable summary (5 std-only channels).

So this experiment builds a much richer per-subject wearable description and
lets **fold-internal feature selection + hyperparameter tuning** decide what to
keep, instead of hand-fixing the feature set up front:

* MMSE: TOTAL, 6 domain sums, 28 raw items, and the two engineered composites
  that matched TOTAL in univariate AUC (``num_failed``, ``recall_deficit``).
* Wearable: mean / std / CV for the clinically informative channels, mean-only
  for the rest, plus derived sleep-architecture ratios and circadian-regularity
  features (bedtime-midpoint variability is a well-known dementia marker).

Every aggregate is computed **within a single subject**, so a plain subject-level
split stays leakage-free by construction.  Diagnosis columns and MMSE
administrative metadata are hard-excluded (fail-closed).

Adherence caveat
----------------
``n_days`` / ``non_wear`` describe how much the subject actually wore the device.
They can carry real signal (apathy, disengagement) but they can equally be a
*protocol artifact* if cohorts were enrolled in different waves.  They are
therefore tagged in ``SUSPECT_FEATURES`` and the pipeline reports an ablation
with and without them (``MAXAUC_DROP_SUSPECT=1`` drops them entirely).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from SangHyo.Binary_Wearable_SequenceFusion_Google.data import load_binary_labels

# ---------------------------------------------------------------- MMSE --------
MMSE_DOMAINS = {
    "orient_time": ["Q01", "Q02", "Q03", "Q04", "Q05"],
    "orient_place": ["Q06", "Q07", "Q08", "Q09", "Q10"],
    "registration": ["Q11_1", "Q11_2", "Q11_3"],
    "attention": ["Q12_1", "Q12_2", "Q12_3", "Q12_4", "Q12_5"],
    "recall": ["Q13_1", "Q13_2", "Q13_3"],              # delayed recall: MCI hallmark
    "language": ["Q14_1", "Q14_2", "Q15", "Q16_1", "Q16_2", "Q16_3", "Q17", "Q18", "Q19"],
}
MMSE_ITEMS = [c for cols in MMSE_DOMAINS.values() for c in cols]
MMSE_FORBIDDEN = frozenset(
    {"DIAG_NM", "DIAG_SEQ", "DOCTOR_NM", "MMSE_NUM", "MMSE_KIND", "SAMPLE_EMAIL", "EMAIL"}
)

# ------------------------------------------------------------ wearable --------
# Channels that get the full mean/std/cv treatment (EDA-flagged or clinically
# meaningful for cognitive decline).
RICH_SLEEP = [
    "sleep_duration", "sleep_efficiency", "sleep_awake", "sleep_deep", "sleep_light",
    "sleep_rem", "sleep_restless", "sleep_onset_latency", "sleep_midpoint_time",
    "sleep_hr_average", "sleep_hr_lowest", "sleep_rmssd", "sleep_breath_average",
    "sleep_score", "sleep_score_deep", "sleep_temperature_deviation",
]
RICH_ACTIVITY = [
    "activity_score", "activity_steps", "activity_rest", "activity_inactive",
    "activity_low", "activity_medium", "activity_high", "activity_daily_movement",
    "activity_average_met", "activity_cal_active",
]
# Channels summarised by mean only (weaker / more redundant).
MEAN_ONLY_SLEEP = [
    "sleep_score_alignment", "sleep_score_disturbances", "sleep_score_efficiency",
    "sleep_score_latency", "sleep_score_rem", "sleep_score_total", "sleep_total",
]
MEAN_ONLY_ACTIVITY = [
    "activity_score_meet_daily_targets", "activity_score_move_every_hour",
    "activity_score_recovery_time", "activity_score_stay_active",
    "activity_score_training_frequency", "activity_score_training_volume",
    "activity_cal_total", "activity_inactivity_alerts", "activity_met_min_high",
    "activity_met_min_medium", "activity_met_min_low", "activity_met_min_inactive",
]

# Adherence / wear-time features -- possibly protocol artifacts, see module docstring.
SUSPECT_FEATURES = (
    "w_n_sleep_days", "w_n_activity_days", "w_activity_non_wear__mean",
)

_SPLIT = {
    "train": {"dir": "1.Training", "mmse": "train_mmse.csv", "act": "train_activity.csv",
              "slp": "train_sleep.csv", "n": 141, "counts": {0: 85, 1: 56}},
    "val": {"dir": "2.Validation", "mmse": "val_mmse.csv", "act": "val_activity.csv",
            "slp": "val_sleep.csv", "n": 33, "counts": {0: 26, 1: 7}},
}


@dataclass(frozen=True)
class SubjectData:
    subject_ids: np.ndarray
    y: np.ndarray | None
    X: np.ndarray
    feature_names: tuple[str, ...]
    item_max: dict          # per-item MMSE maximum, learned on training only

    @property
    def n_subjects(self) -> int:
        return len(self.subject_ids)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def select(self, names: Sequence[str]) -> "SubjectData":
        index = {n: i for i, n in enumerate(self.feature_names)}
        missing = [n for n in names if n not in index]
        if missing:
            raise KeyError(f"Requested features not built: {missing}")
        cols = [index[n] for n in names]
        return SubjectData(self.subject_ids, self.y, self.X[:, cols], tuple(names), self.item_max)

    def drop(self, names: Sequence[str]) -> "SubjectData":
        drop = set(names)
        keep = [n for n in self.feature_names if n not in drop]
        return self.select(keep)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Unable to decode CSV: {path}")


def _resolve_split_root(split_root: str | Path, split: str) -> Path:
    root = Path(split_root).expanduser().resolve()
    expected = _SPLIT[split]["dir"]
    if root.name == expected:
        return root
    for candidate in (root / expected, root / "Data" / expected):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not resolve {expected} under {root}")


def _assert_no_forbidden(names: Sequence[str]) -> None:
    bad = [n for n in names if n in MMSE_FORBIDDEN or "diag" in str(n).lower()]
    if bad:
        raise AssertionError(f"Diagnosis/metadata leaked into features: {bad}")


# ---------------------------------------------------------------- MMSE --------
def _mmse_frame(root: Path, split: str) -> pd.DataFrame:
    frame = _read_csv(root / "SourceData" / "3.CognitiveFunction" / _SPLIT[split]["mmse"])
    id_col = next((c for c in ("SAMPLE_EMAIL", "EMAIL") if c in frame.columns), None)
    if id_col is None:
        raise KeyError("MMSE file lacks SAMPLE_EMAIL/EMAIL")
    frame = frame.copy()
    frame["_sid"] = frame[id_col].astype(str).str.strip()
    frame = frame.drop(columns=[c for c in frame.columns if c in MMSE_FORBIDDEN])
    return frame.drop_duplicates("_sid").set_index("_sid")


def _mmse_features(table: pd.DataFrame, item_max: dict | None) -> tuple[pd.DataFrame, dict]:
    out = pd.DataFrame(index=table.index)
    out["mmse_TOTAL"] = pd.to_numeric(table["TOTAL"], errors="coerce").astype(float)
    for domain, cols in MMSE_DOMAINS.items():
        out[f"mmse_{domain}"] = table[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    for item in MMSE_ITEMS:
        out[f"mmse_{item}"] = pd.to_numeric(table[item], errors="coerce").astype(float)
    if item_max is None:
        item_max = {c: float(pd.to_numeric(table[c], errors="coerce").max()) for c in MMSE_ITEMS}
    below = pd.DataFrame(
        {c: (pd.to_numeric(table[c], errors="coerce") < item_max[c]).astype(float) for c in MMSE_ITEMS}
    )
    out["mmse_num_failed"] = below.sum(axis=1)
    out["mmse_recall_deficit"] = (
        float(sum(item_max[c] for c in MMSE_DOMAINS["recall"])) - out["mmse_recall"]
    )
    return out, item_max


# ------------------------------------------------------------ wearable --------
def _aggregate(frame: pd.DataFrame, rich: Sequence[str], mean_only: Sequence[str],
               prefix: str) -> pd.DataFrame:
    """Per-subject aggregates. Every statistic is computed within one subject."""

    frame = frame.copy()
    frame["_sid"] = frame["EMAIL"].astype(str).str.strip()
    present_rich = [c for c in rich if c in frame.columns]
    present_mean = [c for c in mean_only if c in frame.columns]
    for col in present_rich + present_mean:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    grouped = frame.groupby("_sid")
    parts: list[pd.DataFrame] = []

    if present_rich:
        mean = grouped[present_rich].mean()
        std = grouped[present_rich].std(ddof=0)
        cv = std / mean.abs().replace(0.0, np.nan)
        parts += [
            mean.add_prefix("w_").add_suffix("__mean"),
            std.add_prefix("w_").add_suffix("__std"),
            cv.add_prefix("w_").add_suffix("__cv"),
        ]
    if present_mean:
        parts.append(grouped[present_mean].mean().add_prefix("w_").add_suffix("__mean"))

    parts.append(grouped.size().rename(f"w_n_{prefix}_days").to_frame().astype(float))
    if "activity_non_wear" in frame.columns:
        non_wear = pd.to_numeric(frame["activity_non_wear"], errors="coerce")
        parts.append(non_wear.groupby(frame["_sid"]).mean().rename("w_activity_non_wear__mean").to_frame())
    return pd.concat(parts, axis=1)


LOCAL_TZ = "Asia/Seoul"


def _clock_seconds(values: pd.Series) -> pd.Series:
    """ISO timestamp -> seconds past local midnight (clock time, not duration)."""

    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    try:
        parsed = parsed.dt.tz_convert(LOCAL_TZ)
    except (TypeError, AttributeError):
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (parsed.dt.hour * 3600 + parsed.dt.minute * 60 + parsed.dt.second).astype(float)


def _circular_stats(seconds: np.ndarray) -> tuple[float, float]:
    """Circular mean (hours) and circular SD (hours) of a clock-time sample.

    Sleep timing wraps at midnight, so an ordinary std would call 23:50 and
    00:10 twelve hours apart.  The resultant-vector form handles the wrap, which
    matters here because circadian *irregularity* -- not average bedtime -- is
    the sleep marker associated with cognitive decline.
    """

    values = np.asarray(seconds, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    theta = 2.0 * np.pi * (values % 86400.0) / 86400.0
    cos_mean, sin_mean = float(np.cos(theta).mean()), float(np.sin(theta).mean())
    resultant = float(np.hypot(cos_mean, sin_mean))
    mean_hours = (np.arctan2(sin_mean, cos_mean) % (2.0 * np.pi)) * 24.0 / (2.0 * np.pi)
    if values.size < 2 or resultant <= 1e-12:
        return mean_hours, float("nan")
    circ_sd = np.sqrt(-2.0 * np.log(min(resultant, 1.0))) * 24.0 / (2.0 * np.pi)
    return mean_hours, float(circ_sd)


def _circadian(sleep: pd.DataFrame) -> pd.DataFrame:
    """Per-subject sleep-timing regularity from bedtime / waketime / midpoint."""

    frame = pd.DataFrame({"_sid": sleep["EMAIL"].astype(str).str.strip()})
    if "sleep_bedtime_start" in sleep.columns:
        frame["bedtime"] = _clock_seconds(sleep["sleep_bedtime_start"]).to_numpy()
    if "sleep_bedtime_end" in sleep.columns:
        frame["waketime"] = _clock_seconds(sleep["sleep_bedtime_end"]).to_numpy()
    if "sleep_midpoint_time" in sleep.columns:
        frame["midpoint"] = pd.to_numeric(sleep["sleep_midpoint_time"], errors="coerce").to_numpy()

    channels = [c for c in ("bedtime", "waketime", "midpoint") if c in frame.columns]
    rows: dict[str, dict[str, float]] = {}
    for sid, group in frame.groupby("_sid"):
        record: dict[str, float] = {}
        for channel in channels:
            mean_hours, circ_sd = _circular_stats(group[channel].to_numpy())
            record[f"w_circ_{channel}__mean_h"] = mean_hours
            record[f"w_circ_{channel}__circsd_h"] = circ_sd
        rows[str(sid)] = record
    return pd.DataFrame.from_dict(rows, orient="index")


def _sleep_architecture(sleep: pd.DataFrame) -> pd.DataFrame:
    """Derived sleep-architecture ratios, per subject."""

    frame = sleep.copy()
    frame["_sid"] = frame["EMAIL"].astype(str).str.strip()
    for col in ("sleep_total", "sleep_duration", "sleep_deep", "sleep_rem", "sleep_light",
                "sleep_awake"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    total = frame.get("sleep_total")
    if total is None:
        total = frame.get("sleep_duration")
    denominator = total.replace(0.0, np.nan) if total is not None else None

    derived = pd.DataFrame(index=frame.index)
    if denominator is not None:
        for stage in ("sleep_deep", "sleep_rem", "sleep_light", "sleep_awake"):
            if stage in frame.columns:
                derived[f"ratio_{stage}"] = frame[stage] / denominator
    if "sleep_awake" in frame.columns and "sleep_duration" in frame.columns:
        derived["fragmentation"] = frame["sleep_awake"] / frame["sleep_duration"].replace(0.0, np.nan)
    derived["_sid"] = frame["_sid"]

    grouped = derived.groupby("_sid")
    stats = [grouped.mean(numeric_only=True).add_prefix("w_arch_").add_suffix("__mean")]
    if "fragmentation" in derived.columns:
        stats.append(grouped["fragmentation"].std(ddof=0).rename("w_arch_fragmentation__std").to_frame())
    return pd.concat(stats, axis=1)


def _wearable_features(root: Path, split: str) -> pd.DataFrame:
    act = _read_csv(root / "SourceData" / "1.Gait" / _SPLIT[split]["act"])
    slp = _read_csv(root / "SourceData" / "2.Sleep" / _SPLIT[split]["slp"])
    for frame in (act, slp):
        if "EMAIL" not in frame.columns:
            raise KeyError("Wearable file lacks EMAIL")
    return pd.concat(
        [
            _aggregate(act, RICH_ACTIVITY, MEAN_ONLY_ACTIVITY, "activity"),
            _aggregate(slp, RICH_SLEEP, MEAN_ONLY_SLEEP, "sleep"),
            _sleep_architecture(slp),
            _circadian(slp),
        ],
        axis=1,
    )


# ------------------------------------------------------------------ API -------
def load_split(split_root: str | Path, *, require_labels: bool, split: str,
               item_max: dict | None = None, include_wearable: bool = True,
               drop_suspect: bool = False) -> SubjectData:
    """Build the full per-subject feature matrix for one split."""

    root = _resolve_split_root(split_root, split)
    mmse_table = _mmse_frame(root, split)

    if require_labels:
        labels = load_binary_labels(root)
        subjects = [str(s) for s in labels.index]
        y = labels.to_numpy(np.int64)
    else:
        subjects = sorted(str(s) for s in mmse_table.index)
        y = None

    expected = _SPLIT[split]["n"]
    if len(subjects) != expected:
        raise AssertionError(f"{split}: expected {expected} subjects, got {len(subjects)}")
    if y is not None:
        observed = {c: int((y == c).sum()) for c in (0, 1)}
        if observed != _SPLIT[split]["counts"]:
            raise AssertionError(f"{split} class contract failed: {observed}")

    missing = [s for s in subjects if s not in mmse_table.index]
    if missing:
        raise AssertionError(f"{len(missing)} subjects missing MMSE rows (e.g. {missing[:3]})")

    mmse_df, resolved_item_max = _mmse_features(mmse_table.reindex(subjects), item_max)
    frames = [mmse_df]
    if include_wearable:
        frames.append(_wearable_features(root, split).reindex(subjects))
    X_df = pd.concat(frames, axis=1)
    X_df = X_df.loc[:, ~X_df.columns.duplicated()]
    _assert_no_forbidden(list(X_df.columns))

    data = SubjectData(
        subject_ids=np.asarray(subjects, dtype=str),
        y=y,
        X=X_df.to_numpy(np.float64),
        feature_names=tuple(X_df.columns),
        item_max=resolved_item_max,
    )
    if drop_suspect:
        data = data.drop([n for n in SUSPECT_FEATURES if n in data.feature_names])
    return data


def load_validation_labels_checked(validation_root: str | Path, subject_ids: Sequence[str]) -> np.ndarray:
    root = _resolve_split_root(validation_root, "val")
    labels = load_binary_labels(root)
    wanted = [str(s) for s in subject_ids]
    if set(labels.index.astype(str)) != set(wanted):
        raise AssertionError("Validation label subjects differ from frozen predictions")
    y = labels.loc[wanted].to_numpy(np.int64)
    counts = {c: int((y == c).sum()) for c in (0, 1)}
    if len(y) != 33 or counts != {0: 26, 1: 7}:
        raise AssertionError(f"Validation label contract changed: n={len(y)}, counts={counts}")
    return y


def assert_disjoint_subjects(train_ids: Sequence[str], val_ids: Sequence[str]) -> None:
    overlap = sorted(set(map(str, train_ids)) & set(map(str, val_ids)))
    if overlap:
        raise AssertionError(f"Train/Validation subject leakage: {len(overlap)} subjects")


def hash_subject_id(subject_id: str) -> str:
    return hashlib.sha256(str(subject_id).encode("utf-8")).hexdigest()[:16]


__all__ = ["SUSPECT_FEATURES", "SubjectData", "assert_disjoint_subjects", "hash_subject_id",
           "load_split", "load_validation_labels_checked"]
