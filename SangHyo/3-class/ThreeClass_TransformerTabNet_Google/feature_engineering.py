"""Leakage-aware subject-level features for CN / MCI / DEM classification.

This module only transforms data.  It never fits a predictive model.  All
learned preprocessing (imputation, scaling, feature selection) is deliberately
left to ``train.py`` so that it can be fitted inside each cross-validation fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal
import re

import numpy as np
import pandas as pd


CLASS_NAMES = ("CN", "MCI", "DEM")
LABEL_TO_ID = {
    "cn": 0,
    "normal": 0,
    "mci": 1,
    "dem": 2,
    "dementia": 2,
}
FeatureMode = Literal["clinical_plus_lifelog", "wearable_only"]

# A match on any of these tokens is a hard failure, not a soft warning.
FORBIDDEN_FEATURE_TOKENS = (
    "diag_nm",
    "diag_seq",
    "doctor_nm",
    "sample_email",
    "email",
    "subject_id",
    "mmse_num",
    "mmse_kind",
    "sleep_period_id",
)


@dataclass(frozen=True)
class SplitFiles:
    activity: Path
    sleep: Path
    mmse: Path
    label: Path | None


@dataclass
class SubjectDataset:
    subject_ids: np.ndarray
    X: pd.DataFrame
    y: np.ndarray | None
    feature_mode: str
    audit: dict


def _one(root: Path, pattern: str, role: str) -> Path:
    matches = sorted(p for p in root.glob(pattern) if p.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {role} file under {root} with {pattern!r}; "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def discover_split_files(split_root: str | Path, require_label: bool = True) -> SplitFiles:
    """Resolve the official directory layout without depending on train/val names."""
    root = Path(split_root).expanduser().resolve()
    activity = _one(root, "SourceData/1.Gait/*activity.csv", "activity")
    sleep = _one(root, "SourceData/2.Sleep/*sleep.csv", "sleep")
    mmse = _one(root, "SourceData/3.CognitiveFunction/*mmse.csv", "MMSE")
    label_matches = sorted(root.glob("LabelingData/1.Gait/*label.csv"))
    if require_label and len(label_matches) != 1:
        raise FileNotFoundError(
            f"Expected one gait label file under {root}; found {label_matches}"
        )
    label = label_matches[0] if len(label_matches) == 1 else None
    return SplitFiles(activity=activity, sleep=sleep, mmse=mmse, label=label)


def normalize_label(value: object) -> int:
    key = str(value).strip().lower()
    if key not in LABEL_TO_ID:
        raise ValueError(f"Unknown diagnosis label: {value!r}")
    return LABEL_TO_ID[key]


def load_labels(label_path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(label_path, dtype=str)
    id_candidates = [c for c in raw.columns if c.upper() in {"SAMPLE_EMAIL", "EMAIL"}]
    if len(id_candidates) != 1 or "DIAG_NM" not in raw.columns:
        raise ValueError(f"Unexpected label schema: {list(raw.columns)}")
    labels = raw[[id_candidates[0], "DIAG_NM"]].rename(
        columns={id_candidates[0]: "subject_id", "DIAG_NM": "label_raw"}
    )
    labels["subject_id"] = labels["subject_id"].astype(str).str.strip()
    labels["target"] = labels["label_raw"].map(normalize_label).astype(np.int64)
    if labels["subject_id"].duplicated().any():
        conflicts = labels.groupby("subject_id")["target"].nunique()
        if (conflicts > 1).any():
            raise ValueError("Conflicting labels exist for at least one subject")
        labels = labels.drop_duplicates("subject_id", keep="last")
    return labels[["subject_id", "target"]].sort_values("subject_id").reset_index(drop=True)


def _numeric_tokens(value: object) -> np.ndarray:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.empty(0, dtype=np.float32)
    text = str(value).strip()
    if not text or text == "...":
        return np.empty(0, dtype=np.float32)
    tokens = [token for token in re.split(r"[/,|\s]+", text) if token and token != "..."]
    if not tokens:
        return np.empty(0, dtype=np.float32)
    values = pd.to_numeric(pd.Series(tokens), errors="coerce").to_numpy(dtype=np.float32)
    return values[np.isfinite(values)]


def _entropy_from_counts(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    _, counts = np.unique(values, return_counts=True)
    probs = counts.astype(float) / counts.sum()
    return float(-(probs * np.log(probs + 1e-12)).sum())


def _longest_run(mask: np.ndarray) -> float:
    if mask.size == 0:
        return np.nan
    padded = np.concatenate(([False], mask.astype(bool), [False])).astype(np.int8)
    changes = np.flatnonzero(np.diff(padded))
    return float((changes[1::2] - changes[::2]).max(initial=0))


def _numeric_sequence_summary(value: object, prefix: str) -> dict[str, float]:
    seq = _numeric_tokens(value)
    seq = seq[np.isfinite(seq)]
    seq_nonzero = seq[seq != 0]
    if seq_nonzero.size == 0:
        return {f"{prefix}_{name}": np.nan for name in (
            "mean", "std", "q10", "median", "q90", "iqr", "low_high_delta"
        )}
    half = max(1, seq_nonzero.size // 2)
    q10, q25, q50, q75, q90 = np.quantile(seq_nonzero, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        f"{prefix}_mean": float(np.mean(seq_nonzero)),
        f"{prefix}_std": float(np.std(seq_nonzero)),
        f"{prefix}_q10": float(q10),
        f"{prefix}_median": float(q50),
        f"{prefix}_q90": float(q90),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_low_high_delta": float(
            np.mean(seq_nonzero[half:]) - np.mean(seq_nonzero[:half])
        ) if seq_nonzero.size >= 4 else np.nan,
    }


def _state_sequence_summary(
    value: object,
    prefix: str,
    states: Iterable[int],
    active_states: Iterable[int],
) -> dict[str, float]:
    seq = _numeric_tokens(value)
    seq = seq[np.isfinite(seq)].astype(np.int16, copy=False)
    seq = seq[seq > 0]
    state_values = tuple(int(s) for s in states)
    if seq.size == 0:
        names = [f"state_{state}_ratio" for state in state_values]
        names += ["entropy", "transition_rate", "active_ratio", "longest_inactive_run"]
        return {f"{prefix}_{name}": np.nan for name in names}
    active_mask = np.isin(seq, list(active_states))
    out = {
        f"{prefix}_state_{state}_ratio": float(np.mean(seq == state))
        for state in state_values
    }
    out.update(
        {
            f"{prefix}_entropy": _entropy_from_counts(seq),
            f"{prefix}_transition_rate": float(np.mean(seq[1:] != seq[:-1]))
            if seq.size > 1 else 0.0,
            f"{prefix}_active_ratio": float(np.mean(active_mask)),
            f"{prefix}_longest_inactive_run": _longest_run(~active_mask),
        }
    )
    return out


def _converted_column(frame: pd.DataFrame, raw_name: str) -> str:
    candidates = [
        c for c in frame.columns
        if c.startswith("CONVERT(") and raw_name.lower() in c.lower()
    ]
    if len(candidates) != 1:
        raise ValueError(f"Could not uniquely resolve converted sequence for {raw_name}: {candidates}")
    return candidates[0]


def _append_dict_features(frame: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    expanded = pd.DataFrame(values.tolist(), index=frame.index)
    return pd.concat([frame, expanded], axis=1)


def prepare_activity_daily(path: str | Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, low_memory=False)
    if "EMAIL" not in raw or "activity_day_end" not in raw:
        raise ValueError("Activity schema is missing EMAIL or activity_day_end")
    raw = raw.rename(columns={"EMAIL": "subject_id"})
    raw["subject_id"] = raw["subject_id"].astype(str).str.strip()
    raw["event_time"] = pd.to_datetime(raw["activity_day_end"], errors="coerce", utc=True)
    if raw["event_time"].isna().any():
        raise ValueError("Activity contains an invalid activity_day_end")

    blocked = {
        "subject_id", "activity_day_start", "activity_day_end", "event_time",
        "activity_class_5min", "activity_met_1min",
    }
    blocked.update(c for c in raw.columns if c.startswith("CONVERT("))
    scalar_cols = [c for c in raw.columns if c not in blocked]
    daily = raw[["subject_id", "event_time"]].copy()
    for col in scalar_cols:
        values = pd.to_numeric(raw[col], errors="coerce")
        if values.notna().any():
            daily[f"scalar_{col}"] = values.astype(np.float32)

    class_col = _converted_column(raw, "activity_class_5min")
    met_col = _converted_column(raw, "activity_met_1min")
    daily = _append_dict_features(
        daily,
        raw[class_col].map(
            lambda x: _state_sequence_summary(x, "seq_activity_state", (1, 2, 3, 4), (3, 4))
        ),
    )
    daily = _append_dict_features(
        daily,
        raw[met_col].map(lambda x: _numeric_sequence_summary(x, "seq_activity_met")),
    )
    audit = {
        "rows": int(len(raw)),
        "subjects": int(raw["subject_id"].nunique()),
        "duplicate_subject_timestamp_rows": int(
            raw.duplicated(["subject_id", "event_time"]).sum()
        ),
        "feature_columns_per_day": int(daily.shape[1] - 2),
    }
    return daily.sort_values(["subject_id", "event_time"]), audit


def _clock_features(timestamp: pd.Series, prefix: str) -> pd.DataFrame:
    local = timestamp.dt.tz_convert("Asia/Seoul")
    hour = local.dt.hour + local.dt.minute / 60.0 + local.dt.second / 3600.0
    angle = 2.0 * np.pi * hour / 24.0
    return pd.DataFrame(
        {f"{prefix}_sin": np.sin(angle), f"{prefix}_cos": np.cos(angle)},
        index=timestamp.index,
    )


def prepare_sleep_daily(
    path: str | Path,
    prediction_index: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, low_memory=False).rename(columns={"EMAIL": "subject_id"})
    required = {"subject_id", "sleep_bedtime_start", "sleep_bedtime_end", "sleep_duration"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Sleep schema missing: {sorted(required - set(raw.columns))}")
    raw["subject_id"] = raw["subject_id"].astype(str).str.strip()
    raw["bedtime_start"] = pd.to_datetime(raw["sleep_bedtime_start"], errors="coerce", utc=True)
    raw["event_time"] = pd.to_datetime(raw["sleep_bedtime_end"], errors="coerce", utc=True)
    before = len(raw)
    raw = raw.merge(prediction_index.rename("prediction_index"), left_on="subject_id", right_index=True)
    future_mask = raw["event_time"] > raw["prediction_index"]
    invalid_time = raw["event_time"].isna() | raw["bedtime_start"].isna()
    future_rows = int(future_mask.sum())
    raw = raw.loc[~future_mask & ~invalid_time].copy()
    raw["wake_date"] = raw["event_time"].dt.tz_convert("Asia/Seoul").dt.date
    raw["_duration"] = pd.to_numeric(raw["sleep_duration"], errors="coerce").fillna(-1)
    duplicate_rows = int(raw.duplicated(["subject_id", "wake_date"], keep=False).sum())
    # Main sleep selection uses physiology/time only. The device flag and period id
    # are intentionally not part of either selection or the model input.
    raw = raw.sort_values(
        ["subject_id", "wake_date", "_duration", "bedtime_start", "event_time"],
        ascending=[True, True, False, True, True],
        kind="mergesort",
    ).drop_duplicates(["subject_id", "wake_date"], keep="first")

    blocked = {
        "subject_id", "sleep_bedtime_start", "sleep_bedtime_end", "bedtime_start",
        "event_time", "prediction_index", "wake_date", "_duration", "sleep_period_id",
        "sleep_is_longest", "sleep_hr_5min", "sleep_hypnogram_5min", "sleep_rmssd_5min",
    }
    blocked.update(c for c in raw.columns if c.startswith("CONVERT("))
    daily = raw[["subject_id", "event_time"]].copy()
    for col in [c for c in raw.columns if c not in blocked]:
        values = pd.to_numeric(raw[col], errors="coerce")
        if values.notna().any():
            daily[f"scalar_{col}"] = values.astype(np.float32)
    daily = pd.concat(
        [
            daily,
            _clock_features(raw["bedtime_start"], "clock_bedtime_start"),
            _clock_features(raw["event_time"], "clock_bedtime_end"),
        ],
        axis=1,
    )

    hr_col = _converted_column(raw, "sleep_hr_5min")
    rmssd_col = _converted_column(raw, "sleep_rmssd_5min")
    hypno_col = _converted_column(raw, "sleep_hypnogram_5min")
    daily = _append_dict_features(
        daily, raw[hr_col].map(lambda x: _numeric_sequence_summary(x, "seq_sleep_hr"))
    )
    daily = _append_dict_features(
        daily, raw[rmssd_col].map(lambda x: _numeric_sequence_summary(x, "seq_sleep_rmssd"))
    )
    daily = _append_dict_features(
        daily,
        raw[hypno_col].map(
            lambda x: _state_sequence_summary(x, "seq_sleep_stage", (1, 2, 3, 4), (1, 2, 3))
        ),
    )
    audit = {
        "raw_rows": int(before),
        "rows_after_time_guard_and_main_sleep": int(len(raw)),
        "subjects": int(raw["subject_id"].nunique()),
        "future_rows_excluded": future_rows,
        "duplicate_subject_wake_date_rows_before_dedup": duplicate_rows,
        "feature_columns_per_day": int(daily.shape[1] - 2),
    }
    return daily.sort_values(["subject_id", "event_time"]), audit


def _safe_slope(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if valid.sum() < 4:
        return np.nan
    y = values[valid].astype(float)
    x = np.linspace(0.0, 1.0, y.size)
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    return float(np.dot(x_centered, y - y.mean()) / denominator) if denominator else 0.0


def _recent_early_delta(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 4:
        return np.nan
    half = values.size // 2
    return float(np.median(values[-half:]) - np.median(values[:half]))


def aggregate_subject_features(daily: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Aggregate daily biology without exposing collection counts or calendar gaps."""
    features = [c for c in daily.columns if c not in {"subject_id", "event_time"}]
    rows: list[dict[str, float | str]] = []
    for subject_id, group in daily.groupby("subject_id", sort=True):
        group = group.sort_values("event_time")
        row: dict[str, float | str] = {"subject_id": subject_id}
        for col in features:
            values = pd.to_numeric(group[col], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            stem = f"{prefix}__{col}"
            if finite.size == 0:
                stats = [np.nan] * 8
            else:
                q10, q25, q50, q75, q90 = np.quantile(finite, [0.10, 0.25, 0.50, 0.75, 0.90])
                stats = [
                    float(q50), float(np.mean(finite)), float(np.std(finite)),
                    float(q10), float(q90), float(q75 - q25),
                    _safe_slope(values), _recent_early_delta(values),
                ]
            for suffix, value in zip(
                ("median", "mean", "std", "q10", "q90", "iqr", "rank_slope", "recent_early"),
                stats,
            ):
                row[f"{stem}__{suffix}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_mmse_subject(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Keep test answers while blocking fields that directly reveal the diagnosis."""
    raw = pd.read_csv(path, low_memory=False).rename(columns={"SAMPLE_EMAIL": "subject_id"})
    if "subject_id" not in raw:
        raise ValueError("MMSE schema has no SAMPLE_EMAIL")
    raw["subject_id"] = raw["subject_id"].astype(str).str.strip()
    if "MMSE_NUM" in raw:
        raw["_order"] = pd.to_numeric(raw["MMSE_NUM"], errors="coerce").fillna(-1)
    else:
        raw["_order"] = np.arange(len(raw))
    latest = raw.sort_values(["subject_id", "_order"]).groupby("subject_id", as_index=False).tail(1)
    answer_cols = [c for c in latest.columns if re.match(r"^Q\d", str(c), flags=re.IGNORECASE)]
    if "TOTAL" in latest:
        answer_cols.append("TOTAL")
    if not answer_cols:
        raise ValueError("No MMSE answer columns were found")
    out = latest[["subject_id"]].copy()
    for col in answer_cols:
        out[f"mmse__{col.lower()}"] = pd.to_numeric(latest[col], errors="coerce").astype(np.float32)

    # Compact domain summaries supplement, rather than replace, individual answers.
    q_cols = [c for c in out.columns if c.startswith("mmse__q") and "total" not in c]
    orientation = [c for c in q_cols if re.match(r"mmse__q(0[1-9]|10)$", c)]
    recall = [c for c in q_cols if re.match(r"mmse__q1[1-9]", c)]
    if orientation:
        out["mmse__orientation_sum"] = out[orientation].sum(axis=1, min_count=1)
    if recall:
        out["mmse__memory_language_sum"] = out[recall].sum(axis=1, min_count=1)
    audit = {
        "rows": int(len(raw)),
        "subjects": int(raw["subject_id"].nunique()),
        "records_per_subject_max": int(raw.groupby("subject_id").size().max()),
        "answer_feature_count": int(out.shape[1] - 1),
        "forbidden_source_columns_present_but_excluded": sorted(
            c for c in ("DIAG_NM", "DIAG_SEQ", "DOCTOR_NM", "MMSE_NUM", "MMSE_KIND")
            if c in raw.columns
        ),
    }
    return out.sort_values("subject_id"), audit


def assert_no_forbidden_features(columns: Iterable[str]) -> None:
    bad = [
        col for col in columns
        if any(token in str(col).lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if bad:
        raise AssertionError(f"Forbidden leakage/identifier features detected: {bad[:20]}")


def build_subject_dataset(
    split_root: str | Path,
    *,
    feature_mode: FeatureMode = "clinical_plus_lifelog",
    require_labels: bool = True,
) -> SubjectDataset:
    files = discover_split_files(split_root, require_label=require_labels)
    labels = load_labels(files.label) if require_labels and files.label is not None else None

    activity_daily, activity_audit = prepare_activity_daily(files.activity)
    prediction_index = activity_daily.groupby("subject_id")["event_time"].max()
    sleep_daily, sleep_audit = prepare_sleep_daily(files.sleep, prediction_index)
    activity_subject = aggregate_subject_features(activity_daily, "activity")
    sleep_subject = aggregate_subject_features(sleep_daily, "sleep")
    mmse_subject, mmse_audit = prepare_mmse_subject(files.mmse)

    if labels is not None:
        base = labels[["subject_id"]].copy()
    else:
        subject_union = sorted(
            set(activity_subject["subject_id"])
            | set(sleep_subject["subject_id"])
            | set(mmse_subject["subject_id"])
        )
        base = pd.DataFrame({"subject_id": subject_union})

    merged = base.merge(activity_subject, on="subject_id", how="left", validate="one_to_one")
    merged = merged.merge(sleep_subject, on="subject_id", how="left", validate="one_to_one")
    if feature_mode == "clinical_plus_lifelog":
        merged = merged.merge(mmse_subject, on="subject_id", how="left", validate="one_to_one")
    elif feature_mode != "wearable_only":
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    feature_cols = [c for c in merged.columns if c != "subject_id"]
    assert_no_forbidden_features(feature_cols)
    X = merged[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.astype(np.float32)
    y = None
    if labels is not None:
        target_map = labels.set_index("subject_id")["target"]
        y = merged["subject_id"].map(target_map).to_numpy(dtype=np.int64)
        if np.isnan(y.astype(float)).any():
            raise AssertionError("At least one labeled subject lost its target during joining")

    activity_subjects = set(activity_subject["subject_id"])
    sleep_subjects = set(sleep_subject["subject_id"])
    mmse_subjects = set(mmse_subject["subject_id"])
    expected = set(base["subject_id"])
    audit = {
        "split_root": str(Path(split_root)),
        "feature_mode": feature_mode,
        "subjects": int(len(base)),
        "features": int(X.shape[1]),
        "overall_missing_fraction": float(X.isna().to_numpy().mean()),
        "subjects_missing_activity": int(len(expected - activity_subjects)),
        "subjects_missing_sleep": int(len(expected - sleep_subjects)),
        "subjects_missing_mmse": int(len(expected - mmse_subjects)),
        "activity": activity_audit,
        "sleep": sleep_audit,
        "mmse": mmse_audit,
        "direct_diagnosis_columns_used": False,
        "identifier_columns_used_as_features": False,
        "calendar_date_columns_used_as_features": False,
        "coverage_count_or_gap_features_used": False,
    }
    return SubjectDataset(
        subject_ids=merged["subject_id"].astype(str).to_numpy(),
        X=X,
        y=y,
        feature_mode=feature_mode,
        audit=audit,
    )

