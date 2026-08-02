"""Leakage-audited core for the preregistered three-class PerformanceLab.

The module has deliberately narrow responsibilities:

* it reads only an explicitly supplied ``Data/1.Training`` directory;
* it never reads MMSE data;
* it builds label-free activity/sleep representations using local (+09:00)
  chronology;
* it keeps coverage controls and the legacy calendar representation physically
  separate from the primary observed-event representations; and
* it performs every learned transform inside the relevant subject fold.

Raw subject identifiers are allowed only in transient in-memory joins.  Public
artifacts use HMAC-SHA256 subject hashes; the secret key is never returned,
logged, or serialized.  scikit-learn, PyTorch, and joblib are imported lazily so
the deterministic feature code needs only NumPy and pandas.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import platform
import random
import re
import tempfile
import time
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd


DESIGN_VERSION = "performance_lab_preregistered_v1"
FEATURE_VERSION_SUMMARY = "event_summary_v1"
FEATURE_VERSION_SEQUENCE = "event_sequence28_v1"
COVERAGE_VERSION = "coverage_only_v1"
LEGACY_VERSION = "mask_tcn_35d_legacy_v1"

CLASS_NAMES = ("CN", "MCI", "DEM")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {index: name for name, index in CLASS_TO_ID.items()}
EXPECTED_SUBJECTS = 141
EXPECTED_CLASS_COUNTS = (85, 47, 9)

EVENT_LIMIT = 28
EVENT_SEQUENCE_STEPS = 28
LEGACY_CALENDAR_DAYS = 35
OUTER_SEEDS = (137, 1009, 2027, 4099, 8191)
MODEL_SEEDS = (17011, 27011)
N_SPLITS = 3

PRIMARY_CANDIDATES = (
    "event_elastic_v1",
    "event_extra_trees_v1",
    "event_tcn28_v1",
    "event_elastic_tcn_equal_v1",
)
PRIMARY_COMPLEXITY = {name: index for index, name in enumerate(PRIMARY_CANDIDATES)}
CONTROL_CANDIDATES = (LEGACY_VERSION, COVERAGE_VERSION, "class_prior_v1")

LOCKED_MODEL_CONTRACT = {
    "outer_seeds": list(OUTER_SEEDS),
    "inner_seed_offsets": [50021, 90001],
    "n_splits": N_SPLITS,
    "model_seeds": list(MODEL_SEEDS),
    "event_limit": EVENT_LIMIT,
    "event_sequence_steps": EVENT_SEQUENCE_STEPS,
    "elastic": {
        "C": 0.1,
        "l1_ratio": 0.5,
        "class_weight": "balanced",
        "imputation": "median",
        "scaler": "robust_median_iqr",
    },
    "extra_trees": {
        "n_estimators": 1000,
        "max_depth": 5,
        "min_samples_leaf": 4,
        "max_features": 0.35,
        "class_weight": "balanced_subsample",
    },
    "tcn": {
        "hidden": 24,
        "kernel_size": 3,
        "dilations": [1, 2, 4, 8],
        "convolutions_per_block": 2,
        "normalization": "GroupNorm",
        "activation": "GELU",
        "dropout": 0.35,
        "learning_rate": 8e-4,
        "weight_decay": 2e-3,
        "gradient_clip": 1.0,
        "label_smoothing": 0.05,
        "epochs": 120,
        "mixed_precision": "disabled_locked_float32",
    },
}

ACTIVITY_CLASS_BLOB = "CONVERT(activity_class_5min USING utf8)"
ACTIVITY_MET_BLOB = "CONVERT(activity_met_1min USING utf8)"
SLEEP_HR_BLOB = "CONVERT(sleep_hr_5min USING utf8)"
SLEEP_STAGE_BLOB = "CONVERT(sleep_hypnogram_5min USING utf8)"
SLEEP_RMSSD_BLOB = "CONVERT(sleep_rmssd_5min USING utf8)"

ACTIVITY_REQUIRED_COLUMNS = {
    "EMAIL",
    "activity_day_start",
    "activity_day_end",
    ACTIVITY_CLASS_BLOB,
    ACTIVITY_MET_BLOB,
}
SLEEP_REQUIRED_COLUMNS = {
    "EMAIL",
    "sleep_bedtime_start",
    "sleep_bedtime_end",
    "sleep_duration",
    SLEEP_HR_BLOB,
    SLEEP_STAGE_BLOB,
    SLEEP_RMSSD_BLOB,
}

# Fixed, label-blind scalar source contract.  Non-wear is intentionally absent:
# acquisition/wear information belongs to coverage_only_v1, not the primary view.
ACTIVITY_PRIMARY_SCALARS = (
    "activity_average_met",
    "activity_cal_active",
    "activity_cal_total",
    "activity_daily_movement",
    "activity_high",
    "activity_inactive",
    "activity_inactivity_alerts",
    "activity_low",
    "activity_medium",
    "activity_met_min_high",
    "activity_met_min_inactive",
    "activity_met_min_low",
    "activity_met_min_medium",
    "activity_rest",
    "activity_score",
    "activity_score_meet_daily_targets",
    "activity_score_move_every_hour",
    "activity_score_recovery_time",
    "activity_score_stay_active",
    "activity_score_training_frequency",
    "activity_score_training_volume",
    "activity_steps",
    "activity_total",
)

SLEEP_PRIMARY_SCALARS = (
    "sleep_awake",
    "sleep_breath_average",
    "sleep_deep",
    "sleep_duration",
    "sleep_efficiency",
    "sleep_hr_average",
    "sleep_hr_lowest",
    "sleep_light",
    "sleep_midpoint_at_delta",
    "sleep_onset_latency",
    "sleep_rem",
    "sleep_restless",
    "sleep_rmssd",
    "sleep_score",
    "sleep_score_alignment",
    "sleep_score_deep",
    "sleep_score_disturbances",
    "sleep_score_efficiency",
    "sleep_score_latency",
    "sleep_score_rem",
    "sleep_score_total",
    "sleep_temperature_delta",
    "sleep_total",
)

# Exact historical 35-day comparator value view from ThreeClass_NextStage.
LEGACY_ACTIVITY_COLUMNS = (
    "act__activity_average_met",
    "act__activity_cal_active",
    "act__activity_daily_movement",
    "act__activity_high",
    "act__activity_inactive",
    "act__activity_low",
    "act__activity_medium",
    "act__activity_non_wear",
    "act__activity_rest",
    "act__activity_steps",
    "act__activity_total",
    "actseq__class_entropy",
    "actseq__class_transition_rate",
    "actseq__met__std",
    "actseq__met_sedentary_ratio",
    "actseq__met_moderate_ratio",
    "actseq__met_vigorous_ratio",
    "actseq__relative_amplitude",
)
LEGACY_SLEEP_COLUMNS = (
    "sleep__sleep_awake",
    "sleep__sleep_breath_average",
    "sleep__sleep_deep",
    "sleep__sleep_duration",
    "sleep__sleep_efficiency",
    "sleep__sleep_hr_average",
    "sleep__sleep_hr_lowest",
    "sleep__sleep_light",
    "sleep__sleep_midpoint_at_delta",
    "sleep__sleep_onset_latency",
    "sleep__sleep_rem",
    "sleep__sleep_restless",
    "sleep__sleep_rmssd",
    "sleep__sleep_temperature_delta",
    "sleep__sleep_total",
    "sleep__sleep_awake_ratio_duration",
    "sleep__sleep_deep_ratio_duration",
    "sleep__sleep_light_ratio_duration",
    "sleep__sleep_rem_ratio_duration",
    "sleep__sleep_total_ratio_duration",
    "sleep__daily_sleep_count",
    "sleep__hr_drop",
    "sleep__hr_drop_ratio",
    "sleepseq__stage_entropy",
    "sleepseq__stage_transition_rate",
    "sleepseq__awake_bouts",
    "sleepseq__hr__std",
    "sleepseq__rmssd__std",
    "sleep__lowest_hr_position",
    "sleep__lowest_hr_clock_sin",
    "sleep__lowest_hr_clock_cos",
)

SUMMARY_STATS = (
    "median",
    "trimmed_mean_10",
    "p10",
    "p90",
    "iqr",
    "mad",
    "theil_sen_rank_slope",
    "late_half_minus_early_half",
)

FORBIDDEN_PRIMARY_TOKENS = (
    "email",
    "diag",
    "mmse",
    "doctor",
    "period_id",
    "sample_order",
    "observed_count",
    "observed_day",
    "observed_night",
    "valid_count",
    "valid_n",
    "valid_ratio",
    "raw_length",
    "sequence_length",
    "missing_ratio",
    "coverage",
    "calendar_gap",
    "span_day",
    "duplicate",
    "nonwear",
    "non_wear",
    "class0",
    "mask",
    "delta_since",
    "year",
    "month",
    "weekday",
    "absolute_date",
)

_EMAIL_LIKE_RE = re.compile(r"[^\s@]+@[^\s@]+")
_KST_SUFFIX_RE = re.compile(r"(?:\+09:00|\+0900)\s*$")


@dataclass(frozen=True)
class TrainingInputPaths:
    """Exact Training-only files.  There is intentionally no MMSE field."""

    training_root: Path
    activity: Path
    sleep: Path
    labels: tuple[Path, Path, Path]

    def files(self) -> tuple[Path, ...]:
        return (self.activity, self.sleep, *self.labels)


@dataclass
class TrainingDataset:
    """Transient Training cohort; raw IDs remain only in memory."""

    activity: pd.DataFrame
    sleep: pd.DataFrame
    labels: pd.Series
    subject_ids: list[str]
    audit: dict[str, Any]
    input_paths: TrainingInputPaths


@dataclass
class FeatureBundle:
    """Aligned, deterministic label-free feature representations."""

    subject_ids: list[str]
    event_summary: pd.DataFrame
    activity_sequence: np.ndarray
    sleep_sequence: np.ndarray
    activity_sequence_features: list[str]
    sleep_sequence_features: list[str]
    coverage: pd.DataFrame
    legacy_values: np.ndarray
    legacy_features: list[str]
    diagnostics: dict[str, Any]

    @property
    def views(self) -> dict[str, pd.DataFrame]:
        return {FEATURE_VERSION_SUMMARY: self.event_summary}

    def summary_manifest(self) -> dict[str, Any]:
        return {
            "version": FEATURE_VERSION_SUMMARY,
            "event_limit": EVENT_LIMIT,
            "statistics": list(SUMMARY_STATS),
            "feature_count": int(self.event_summary.shape[1]),
            "feature_names": list(self.event_summary.columns),
            "forbidden_explicit_signals": [
                "observed count", "calendar gap", "mask", "absolute date", "non-wear"
            ],
        }

    def sequence_manifest(self) -> dict[str, Any]:
        return {
            "version": FEATURE_VERSION_SEQUENCE,
            "steps": EVENT_SEQUENCE_STEPS,
            "interpolation": "linear_on_normalized_observed_event_rank",
            "activity_feature_names": list(self.activity_sequence_features),
            "sleep_feature_names": list(self.sleep_sequence_features),
            "activity_shape": list(self.activity_sequence.shape),
            "sleep_shape": list(self.sleep_sequence.shape),
            "model_input_excludes": ["mask", "count", "calendar gap", "absolute date"],
        }

    def coverage_audit(self) -> dict[str, Any]:
        numeric = self.coverage.apply(pd.to_numeric, errors="coerce")
        return {
            "version": COVERAGE_VERSION,
            "feature_names": list(numeric.columns),
            "feature_count": int(numeric.shape[1]),
            "descriptive_summary": {
                column: {
                    "median": _finite_or_none(numeric[column].median()),
                    "min": _finite_or_none(numeric[column].min()),
                    "max": _finite_or_none(numeric[column].max()),
                }
                for column in numeric.columns
            },
            "warning": "Negative control only; never eligible for primary selection.",
        }

    def public_summary(self) -> dict[str, Any]:
        return {
            **self.diagnostics,
            "summary_feature_count": int(self.event_summary.shape[1]),
            "activity_sequence_shape": list(self.activity_sequence.shape),
            "sleep_sequence_shape": list(self.sleep_sequence.shape),
            "legacy_shape": list(self.legacy_values.shape),
            "coverage_feature_count": int(self.coverage.shape[1]),
        }


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def normalize_label(value: Any) -> str:
    """Normalize allowed raw labels to ``CN``, ``MCI``, or ``DEM``."""

    if isinstance(value, (int, np.integer)) and int(value) in ID_TO_CLASS:
        return ID_TO_CLASS[int(value)]
    mapping = {"cn": "CN", "mci": "MCI", "dem": "DEM", "dementia": "DEM"}
    key = str(value).strip().lower()
    if key not in mapping:
        raise ValueError("Unknown diagnosis label encountered.")
    return mapping[key]


def label_to_id(value: Any) -> int:
    return CLASS_TO_ID[normalize_label(value)]


def training_input_paths(training_root: str | Path) -> TrainingInputPaths:
    """Resolve only an explicit ``1.Training`` split directory.

    Passing ``Data/`` or a repository root is rejected.  This prevents a
    discovery notebook from accidentally traversing or fingerprinting the
    sibling official benchmark.
    """

    root = Path(training_root).expanduser().resolve(strict=True)
    if root.name != "1.Training" or not root.is_dir():
        raise ValueError("training_root must point exactly to the 1.Training directory.")
    if any(part.lower() in {"2.validation", "validation"} for part in root.parts):
        raise ValueError("A validation path cannot be used by the Training loader.")
    paths = TrainingInputPaths(
        training_root=root,
        activity=root / "SourceData" / "1.Gait" / "train_activity.csv",
        sleep=root / "SourceData" / "2.Sleep" / "train_sleep.csv",
        labels=(
            root / "LabelingData" / "1.Gait" / "training_label.csv",
            root / "LabelingData" / "2.Sleep" / "training_label.csv",
            root / "LabelingData" / "3.CognitiveFunction" / "training_label.csv",
        ),
    )
    missing = [path.name for path in paths.files() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required Training files are missing: {sorted(missing)}")
    # Deliberately do not resolve, stat, hash, or read train_mmse.csv.
    return paths


def load_consistent_labels(paths: Sequence[str | Path]) -> pd.Series:
    """Load three label copies and require identical one-label-per-subject maps."""

    copies: list[pd.Series] = []
    for supplied in paths:
        path = Path(supplied)
        frame = pd.read_csv(path, usecols=["SAMPLE_EMAIL", "DIAG_NM"])
        if frame["SAMPLE_EMAIL"].isna().any() or frame["DIAG_NM"].isna().any():
            raise AssertionError(f"Missing subject or diagnosis in {path.name}.")
        if frame["SAMPLE_EMAIL"].astype(str).duplicated().any():
            raise AssertionError(f"Duplicate subject in {path.name}.")
        normalized = frame["DIAG_NM"].map(label_to_id).astype(np.int64)
        copy = pd.Series(
            normalized.to_numpy(),
            index=frame["SAMPLE_EMAIL"].astype(str),
            name="label",
        ).sort_index()
        copies.append(copy)
    if len(copies) != 3 or any(not copies[0].equals(other) for other in copies[1:]):
        raise AssertionError("The three modality label mappings are not identical.")
    return copies[0]


def load_training_dataset(training_root: str | Path) -> TrainingDataset:
    """Read the immutable Training activity/sleep sources and labels only.

    The expected 141-subject / 85-47-9 contract is asserted.  MMSE is wholly
    excluded: neither its source nor its label-like fields are opened.
    """

    paths = training_input_paths(training_root)
    activity = pd.read_csv(paths.activity)
    sleep = pd.read_csv(paths.sleep)
    labels = load_consistent_labels(paths.labels)
    _require_columns(activity, ACTIVITY_REQUIRED_COLUMNS, "activity")
    _require_columns(sleep, SLEEP_REQUIRED_COLUMNS, "sleep")
    for frame, name in ((activity, "activity"), (sleep, "sleep")):
        if frame["EMAIL"].isna().any():
            raise AssertionError(f"Missing subject identifier in {name} source.")
        frame["EMAIL"] = frame["EMAIL"].astype(str)
    activity_subjects = set(activity["EMAIL"])
    sleep_subjects = set(sleep["EMAIL"])
    label_subjects = set(labels.index.astype(str))
    if activity_subjects != sleep_subjects or activity_subjects != label_subjects:
        raise AssertionError("Training modality subject sets are inconsistent.")
    subject_ids = sorted(label_subjects)
    labels = labels.reindex(subject_ids)
    counts = tuple(np.bincount(labels.to_numpy(dtype=int), minlength=3).tolist())
    if len(subject_ids) != EXPECTED_SUBJECTS or counts != EXPECTED_CLASS_COUNTS:
        raise AssertionError(
            "Training cohort does not match preregistered 141 / 85-47-9 contract."
        )
    audit = {
        "cohort": "Data/1.Training only",
        "subjects": len(subject_ids),
        "class_counts": {CLASS_NAMES[i]: counts[i] for i in range(3)},
        "activity_rows": int(len(activity)),
        "sleep_rows": int(len(sleep)),
        "label_copies_consistent": True,
        "modalities_have_identical_subject_sets": True,
        "mmse_opened": False,
    }
    return TrainingDataset(activity, sleep, labels, subject_ids, audit, paths)


def _require_columns(frame: pd.DataFrame, required: Iterable[str], source_name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise AssertionError(f"{source_name} source is missing required columns: {missing}")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_input_manifest(paths: TrainingInputPaths | str | Path) -> dict[str, Any]:
    """Return a Training-only, PII-free file fingerprint manifest."""

    resolved = paths if isinstance(paths, TrainingInputPaths) else training_input_paths(paths)
    role_paths = {
        "activity_source": resolved.activity,
        "sleep_source": resolved.sleep,
        "activity_labels": resolved.labels[0],
        "sleep_labels": resolved.labels[1],
        "cognitive_modality_labels": resolved.labels[2],
    }
    return {
        "scope": "Data/1.Training only; MMSE excluded",
        "files": {
            role: {
                "name": path.name,
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for role, path in role_paths.items()
        },
    }


def keyed_subject_hash(value: Any, secret_key: str | bytes, length: int = 24) -> str:
    """HMAC-hash one transient subject ID without exposing or persisting the key."""

    key = secret_key.encode("utf-8") if isinstance(secret_key, str) else bytes(secret_key)
    if len(key) < 16:
        raise ValueError("subject_hash_key must contain at least 16 bytes of entropy.")
    digest = hmac.new(key, str(value).encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[:length]


def hash_subjects(subject_ids: Sequence[Any], secret_key: str | bytes) -> list[str]:
    hashes = [keyed_subject_hash(value, secret_key) for value in subject_ids]
    if len(set(hashes)) != len(hashes):
        raise AssertionError("Keyed subject hash collision.")
    return hashes


def json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _finite_or_none(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _assert_public_payload(value: Any, path: str = "root") -> None:
    """Fail if a JSON/CSV-like public payload appears to contain raw PII."""

    if isinstance(value, pd.DataFrame):
        bad_columns = [
            column for column in value.columns
            if str(column).lower() in {"email", "sample_email", "subject_id", "raw_subject_id"}
        ]
        if bad_columns:
            raise AssertionError("Raw identifier column in public artifact.")
        if value.index.name and str(value.index.name).lower() in {
            "email", "sample_email", "subject_id", "raw_subject_id"
        }:
            raise AssertionError("Raw identifier index in public artifact.")
        if pd.Index(value.index).astype(str).str.contains(_EMAIL_LIKE_RE, regex=True).any():
            raise AssertionError("Email-like index value in public artifact.")
        for column in value.select_dtypes(include=["object", "string"]).columns:
            if value[column].astype(str).str.contains(_EMAIL_LIKE_RE, regex=True).any():
                raise AssertionError("Email-like value in public artifact.")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in {"email", "sample_email", "subject_id", "raw_subject_id", "hash_key"}:
                raise AssertionError("Raw identifier/key field in public artifact.")
            _assert_public_payload(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple, set)):
        for index, item in enumerate(value):
            _assert_public_payload(item, f"{path}[{index}]")
        return
    if isinstance(value, str) and _EMAIL_LIKE_RE.search(value):
        raise AssertionError("Email-like value in public artifact.")


def write_json(path: str | Path, payload: Any) -> None:
    _assert_public_payload(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, destination)


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    _assert_public_payload(frame)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def write_parquet(path: str | Path, frame: pd.DataFrame) -> None:
    _assert_public_payload(frame)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)


def atomic_joblib_dump(payload: Any, path: str | Path, compress: int = 3) -> None:
    """Atomically serialize a known-safe checkpoint/model payload (lazy joblib)."""

    import joblib

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    joblib.dump(payload, temporary, compress=compress)
    os.replace(temporary, destination)


def atomic_joblib_load(path: str | Path) -> Any:
    import joblib

    return joblib.load(Path(path))


def stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        json_ready(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_array(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8").copy(order="C")
    array[~np.isfinite(array)] = np.nan
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
                raise RuntimeError(
                    "Strict CUDA determinism requires CUBLAS_WORKSPACE_CONFIG "
                    "to be set before CUDA initialization."
                )
            torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


def parse_slash_sequence(value: Any) -> np.ndarray:
    """Parse the real ``CONVERT(... USING utf8)`` slash-delimited sequence."""

    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return np.empty(0, dtype=np.float32)
    text = str(value).strip().rstrip("/")
    if not text or text == "...":
        return np.empty(0, dtype=np.float32)
    parsed: list[float] = []
    for token in text.split("/"):
        try:
            number = float(token)
        except (TypeError, ValueError):
            number = np.nan
        parsed.append(number if np.isfinite(number) else np.nan)
    return np.asarray(parsed, dtype=np.float32)


def _finite(values: Any, zero_is_missing: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=float).copy()
    array[~np.isfinite(array)] = np.nan
    if zero_is_missing:
        array[array == 0] = np.nan
    return array


def _parse_kst_timestamps(values: pd.Series, name: str) -> pd.Series:
    text = values.astype("string")
    nonmissing = text.notna()
    if nonmissing.any() and not text.loc[nonmissing].str.contains(_KST_SUFFIX_RE, regex=True).all():
        raise AssertionError(f"{name} must carry an explicit +09:00 offset.")
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if parsed.isna().any():
        raise AssertionError(f"Invalid timestamp in {name}.")
    return parsed.dt.tz_convert("Asia/Seoul")


def _local_dates(values: pd.Series, name: str) -> pd.Series:
    return _parse_kst_timestamps(values, name).dt.tz_localize(None).dt.normalize()


def _clock_phase(value: Any) -> tuple[float, float]:
    try:
        timestamp = pd.to_datetime(value, errors="raise", utc=True).tz_convert("Asia/Seoul")
    except Exception:
        return np.nan, np.nan
    hour = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
    angle = 2.0 * np.pi * hour / 24.0
    return float(np.sin(angle)), float(np.cos(angle))


def _entropy(values: np.ndarray) -> float:
    finite = _finite(values)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan
    _, counts = np.unique(finite, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def _longest_run_ratio(mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not len(mask):
        return np.nan
    best = current = 0
    for flag in mask:
        current = current + 1 if flag else 0
        best = max(best, current)
    return float(best / len(mask))


def _trimmed_mean_10(values: np.ndarray) -> float:
    finite = np.sort(_finite(values)[np.isfinite(_finite(values))])
    if not len(finite):
        return np.nan
    trim = int(math.floor(0.10 * len(finite)))
    kept = finite[trim:len(finite) - trim] if trim else finite
    return float(np.mean(kept)) if len(kept) else np.nan


def _robust_vector_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Within-event biological summaries; no length or valid-ratio outputs."""

    finite = _finite(values)
    finite = finite[np.isfinite(finite)]
    names = ("median", "trimmed_mean_10", "p10", "p90", "iqr", "mad")
    if not len(finite):
        return {f"{prefix}__{name}": np.nan for name in names}
    p10, q25, median, q75, p90 = np.quantile(finite, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        f"{prefix}__median": float(median),
        f"{prefix}__trimmed_mean_10": _trimmed_mean_10(finite),
        f"{prefix}__p10": float(p10),
        f"{prefix}__p90": float(p90),
        f"{prefix}__iqr": float(q75 - q25),
        f"{prefix}__mad": float(np.median(np.abs(finite - median))),
    }


def _theil_sen_slope(values: np.ndarray, ranks: np.ndarray) -> float:
    values = _finite(values)
    ranks = _finite(ranks)
    keep = np.isfinite(values) & np.isfinite(ranks)
    values, ranks = values[keep], ranks[keep]
    if len(values) < 3 or np.unique(ranks).size < 2:
        return np.nan
    slopes: list[float] = []
    for left in range(len(values) - 1):
        denominator = ranks[left + 1:] - ranks[left]
        valid = denominator != 0
        slopes.extend(((values[left + 1:][valid] - values[left]) / denominator[valid]).tolist())
    return float(np.median(slopes)) if slopes else np.nan


def _event_summary_stats(values: np.ndarray, ranks: np.ndarray) -> dict[str, float]:
    """The exactly eight preregistered event_summary_v1 statistics.

    Half-change uses ``numpy.array_split`` on event positions (the earlier half
    receives the middle event when length is odd), then takes finite medians.
    """

    values = _finite(values)
    finite = values[np.isfinite(values)]
    result = {name: np.nan for name in SUMMARY_STATS}
    if len(finite):
        p10, q25, median, q75, p90 = np.quantile(finite, [0.10, 0.25, 0.50, 0.75, 0.90])
        result.update(
            median=float(median),
            trimmed_mean_10=_trimmed_mean_10(finite),
            p10=float(p10),
            p90=float(p90),
            iqr=float(q75 - q25),
            mad=float(np.median(np.abs(finite - median))),
        )
    result["theil_sen_rank_slope"] = _theil_sen_slope(values, ranks)
    if len(values) >= 2:
        early_indices, late_indices = np.array_split(np.arange(len(values)), 2)
        early = values[early_indices]
        late = values[late_indices]
        if np.isfinite(early).any() and np.isfinite(late).any():
            result["late_half_minus_early_half"] = float(
                np.nanmedian(late) - np.nanmedian(early)
            )
    return result


def _activity_sequence_features(row: pd.Series) -> dict[str, float]:
    output: dict[str, float] = {}
    classes = _finite(parse_slash_sequence(row[ACTIVITY_CLASS_BLOB]))
    # Code 0 is non-wear/acquisition information and is excluded, not treated as
    # a behavioural state or exposed through a validity ratio.
    wear_states = classes[np.isfinite(classes) & np.isin(classes, [1, 2, 3, 4, 5])]
    output["activity__class__wear_state_entropy"] = _entropy(wear_states)
    output["activity__class__wear_state_transition_rate"] = (
        float(np.mean(wear_states[1:] != wear_states[:-1])) if len(wear_states) >= 2 else np.nan
    )
    for code, name in ((1, "rest"), (2, "inactive"), (3, "low"), (4, "medium"), (5, "high")):
        output[f"activity__class__{name}_ratio_within_wear"] = (
            float(np.mean(wear_states == code)) if len(wear_states) else np.nan
        )
        output[f"activity__class__{name}_longest_run_ratio_within_wear"] = (
            _longest_run_ratio(wear_states == code) if len(wear_states) else np.nan
        )

    met = _finite(parse_slash_sequence(row[ACTIVITY_MET_BLOB]))
    # The activity-state stream is 5-minute while MET is 1-minute.  Expand the
    # known wear-state mask and remove code-0/non-wear MET samples before every
    # biological statistic.  Unknown/uncovered minutes are also missing.
    expanded_wear = np.repeat(np.isin(classes, [1, 2, 3, 4, 5]), 5)
    wear_met = np.full_like(met, np.nan, dtype=float)
    aligned = min(len(met), len(expanded_wear))
    if aligned:
        keep = expanded_wear[:aligned] & np.isfinite(met[:aligned])
        aligned_indices = np.flatnonzero(keep)
        wear_met[aligned_indices] = met[aligned_indices]
    met = wear_met
    output.update(_robust_vector_stats(met, "activity__met"))
    valid_met = met[np.isfinite(met)]
    output["activity__met__sedentary_ratio"] = (
        float(np.mean(valid_met <= 1.5)) if len(valid_met) else np.nan
    )
    output["activity__met__moderate_ratio"] = (
        float(np.mean((valid_met > 1.5) & (valid_met <= 3.0))) if len(valid_met) else np.nan
    )
    output["activity__met__vigorous_ratio"] = (
        float(np.mean(valid_met > 3.0)) if len(valid_met) else np.nan
    )
    if len(met) == 1440:
        hourly_matrix = met.reshape(24, 60)
        hourly_counts = np.isfinite(hourly_matrix).sum(axis=1)
        hourly = np.divide(
            np.nansum(hourly_matrix, axis=1),
            hourly_counts,
            out=np.full(24, np.nan, dtype=float),
            where=hourly_counts > 0,
        )
        finite_hours = np.flatnonzero(np.isfinite(hourly))
        if finite_hours.size >= 12:
            # Circularly interpolate hourly physiology without exporting the
            # underlying wear mask/count.  Coverage remains audited separately.
            x = np.r_[finite_hours - 24, finite_hours, finite_hours + 24]
            y = np.tile(hourly[finite_hours], 3)
            hourly = np.interp(np.arange(24), x, y)
        else:
            hourly = np.full(24, np.nan)
    else:
        hourly = np.full(24, np.nan)
    if np.isfinite(hourly).all():
        doubled = np.r_[hourly, hourly]
        m10 = np.asarray([doubled[index:index + 10].mean() for index in range(24)])
        l5 = np.asarray([doubled[index:index + 5].mean() for index in range(24)])
        m10_value, l5_value = float(m10.max()), float(l5.min())
        output["activity__circadian__m10"] = m10_value
        output["activity__circadian__l5"] = l5_value
        output["activity__circadian__relative_amplitude"] = (
            (m10_value - l5_value) / (m10_value + l5_value + 1e-8)
        )
        peak_hour = int(np.argmax(hourly))
        try:
            activity_start = pd.to_datetime(
                row["activity_day_start"], errors="raise", utc=True
            ).tz_convert("Asia/Seoul")
            peak_clock_hour = (
                activity_start.hour
                + activity_start.minute / 60.0
                + peak_hour
            ) % 24.0
        except Exception:
            peak_clock_hour = np.nan
        angle = 2.0 * np.pi * peak_clock_hour / 24.0
        output["activity__circadian__peak_sin"] = float(np.sin(angle))
        output["activity__circadian__peak_cos"] = float(np.cos(angle))
        spectrum = np.fft.rfft(hourly - hourly.mean())
        output["activity__circadian__first_harmonic_ratio"] = float(
            abs(spectrum[1]) / (np.abs(spectrum[1:]).sum() + 1e-8)
        )
    else:
        for name in ("m10", "l5", "relative_amplitude", "peak_sin", "peak_cos", "first_harmonic_ratio"):
            output[f"activity__circadian__{name}"] = np.nan
    return output


def _sleep_sequence_features(row: pd.Series) -> dict[str, float]:
    output: dict[str, float] = {}
    stages = _finite(parse_slash_sequence(row[SLEEP_STAGE_BLOB]))
    stages = stages[np.isfinite(stages) & np.isin(stages, [1, 2, 3, 4])]
    output["sleep__stage__entropy"] = _entropy(stages)
    output["sleep__stage__transition_rate"] = (
        float(np.mean(stages[1:] != stages[:-1])) if len(stages) >= 2 else np.nan
    )
    for code, name in ((1, "deep"), (2, "light"), (3, "rem"), (4, "awake")):
        output[f"sleep__stage__{name}_ratio"] = (
            float(np.mean(stages == code)) if len(stages) else np.nan
        )
        output[f"sleep__stage__{name}_longest_run_ratio"] = (
            _longest_run_ratio(stages == code) if len(stages) else np.nan
        )

    # Sensor zero is semantic missingness for both HR and RMSSD.
    hr = _finite(parse_slash_sequence(row[SLEEP_HR_BLOB]), zero_is_missing=True)
    rmssd = _finite(parse_slash_sequence(row[SLEEP_RMSSD_BLOB]), zero_is_missing=True)
    output.update(_robust_vector_stats(hr, "sleep__hr"))
    output.update(_robust_vector_stats(rmssd, "sleep__rmssd_seq"))
    if len(hr) and len(rmssd):
        count = min(len(hr), len(rmssd))
        keep = np.isfinite(hr[:count]) & np.isfinite(rmssd[:count])
        output["sleep__hr_rmssd__correlation"] = (
            float(np.corrcoef(hr[:count][keep], rmssd[:count][keep])[0, 1])
            if keep.sum() >= 4
            and np.std(hr[:count][keep]) > 0
            and np.std(rmssd[:count][keep]) > 0
            else np.nan
        )
    else:
        output["sleep__hr_rmssd__correlation"] = np.nan

    start_sin, start_cos = _clock_phase(row.get("sleep_bedtime_start"))
    wake_sin, wake_cos = _clock_phase(row.get("sleep_bedtime_end"))
    output.update(
        {
            "sleep__clock__bedtime_sin": start_sin,
            "sleep__clock__bedtime_cos": start_cos,
            "sleep__clock__wake_sin": wake_sin,
            "sleep__clock__wake_cos": wake_cos,
        }
    )
    duration = _finite_or_none(row.get("sleep_duration"))
    for column in ("sleep_awake", "sleep_deep", "sleep_light", "sleep_rem", "sleep_total"):
        value = _finite_or_none(row.get(column))
        output[f"sleep__duration__{column.removeprefix('sleep_')}_ratio"] = (
            value / duration if value is not None and duration is not None and duration > 0 else np.nan
        )
    average = _finite_or_none(row.get("sleep_hr_average"))
    lowest = _finite_or_none(row.get("sleep_hr_lowest"))
    output["sleep__hr__drop"] = (
        average - lowest if average is not None and lowest is not None else np.nan
    )
    output["sleep__hr__drop_ratio"] = (
        (average - lowest) / average
        if average is not None and lowest is not None and average > 0
        else np.nan
    )
    valid_positions = np.flatnonzero(np.isfinite(hr))
    if valid_positions.size:
        lowest_position = int(valid_positions[np.nanargmin(hr[valid_positions])])
        output["sleep__hr__lowest_relative_position"] = lowest_position / max(1, len(hr) - 1)
        try:
            start = pd.to_datetime(row["sleep_bedtime_start"], utc=True).tz_convert("Asia/Seoul")
            lowest_time = start + pd.Timedelta(minutes=5 * lowest_position)
            angle = 2.0 * np.pi * (
                lowest_time.hour + lowest_time.minute / 60.0
            ) / 24.0
            output["sleep__hr__lowest_clock_sin"] = float(np.sin(angle))
            output["sleep__hr__lowest_clock_cos"] = float(np.cos(angle))
        except Exception:
            output["sleep__hr__lowest_clock_sin"] = np.nan
            output["sleep__hr__lowest_clock_cos"] = np.nan
    else:
        output["sleep__hr__lowest_relative_position"] = np.nan
        output["sleep__hr__lowest_clock_sin"] = np.nan
        output["sleep__hr__lowest_clock_cos"] = np.nan
    return output


def make_activity_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """Build one biological activity event per local activity date.

    Row count, raw sequence length, field-validity ratio, non-wear, and dates are
    not returned as model features.  If duplicate subject-date rows ever occur,
    biological values are combined by a deterministic median rather than row
    order.
    """

    _require_columns(
        raw,
        ACTIVITY_REQUIRED_COLUMNS | set(ACTIVITY_PRIMARY_SCALARS),
        "activity",
    )
    work = raw.copy()
    work["_subject"] = work["EMAIL"].astype(str)
    start_timestamps = _parse_kst_timestamps(
        work["activity_day_start"], "activity_day_start"
    )
    work["_event_date"] = start_timestamps.dt.tz_localize(None).dt.normalize()
    work["_event_timestamp"] = _parse_kst_timestamps(
        work["activity_day_end"], "activity_day_end"
    )
    if (work["_event_timestamp"] <= start_timestamps).any():
        raise AssertionError("activity_day_end must follow activity_day_start.")
    end_dates = work["_event_timestamp"].dt.tz_localize(None).dt.normalize()
    if (end_dates < work["_event_date"]).any():
        raise AssertionError("Activity end precedes its local activity date.")
    scalar = work.loc[:, ACTIVITY_PRIMARY_SCALARS].apply(pd.to_numeric, errors="coerce")
    scalar.columns = [f"activity__raw__{column}" for column in scalar.columns]
    derived = pd.DataFrame(
        [_activity_sequence_features(row) for _, row in work.iterrows()],
        index=work.index,
    )
    output = pd.concat(
        [work[["_subject", "_event_date", "_event_timestamp"]], scalar, derived], axis=1
    )
    timestamps = output.groupby(["_subject", "_event_date"], sort=True)[
        "_event_timestamp"
    ].max()
    biological = output.drop(columns="_event_timestamp").groupby(
        ["_subject", "_event_date"], sort=True
    ).median(numeric_only=True)
    output = (
        biological.join(timestamps)
        .reset_index()
        .sort_values(["_subject", "_event_timestamp"], kind="mergesort")
        .reset_index(drop=True)
    )
    return output


def _sleep_tie_digest(row: pd.Series, columns: Sequence[str]) -> str:
    # sleep_period_id and row position are excluded from the digest.
    text = "\x1f".join("<NA>" if pd.isna(row[column]) else str(row[column]) for column in columns)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_sleep_daily(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one longest main-sleep event per local wake date.

    Returns the biological daily table and per-subject duplicate-episode count
    for the separate coverage control.  ``sleep_period_id`` is never consulted.
    """

    _require_columns(raw, SLEEP_REQUIRED_COLUMNS | set(SLEEP_PRIMARY_SCALARS), "sleep")
    work = raw.copy()
    work["_subject"] = work["EMAIL"].astype(str)
    work["_wake_timestamp"] = _parse_kst_timestamps(
        work["sleep_bedtime_end"], "sleep_bedtime_end"
    )
    work["_start_timestamp"] = _parse_kst_timestamps(
        work["sleep_bedtime_start"], "sleep_bedtime_start"
    )
    if (work["_wake_timestamp"] <= work["_start_timestamp"]).any():
        raise AssertionError("Sleep wake timestamp must follow bedtime start.")
    work["_event_date"] = work["_wake_timestamp"].dt.tz_localize(None).dt.normalize()
    work["_duration"] = pd.to_numeric(work["sleep_duration"], errors="coerce")
    if work["_duration"].isna().any() or (work["_duration"] <= 0).any():
        raise AssertionError("Sleep duration must be finite and positive.")
    duplicate_table = work[["_subject", "_event_date", "_wake_timestamp"]].copy()
    # Only an explicit, preregistered biological allowlist can break an exact
    # duration/start/end tie. Unknown future columns, diagnosis, device IDs, and
    # sleep_period_id can never affect main-sleep selection.
    tie_columns = [
        column
        for column in (
            *SLEEP_PRIMARY_SCALARS,
            SLEEP_HR_BLOB,
            SLEEP_STAGE_BLOB,
            SLEEP_RMSSD_BLOB,
        )
        if column in work.columns
    ]
    work["_tie_digest"] = [
        _sleep_tie_digest(row, tie_columns) for _, row in work.iterrows()
    ]
    work = (
        work.sort_values(
            [
                "_subject",
                "_event_date",
                "_duration",
                "_start_timestamp",
                "_wake_timestamp",
                "_tie_digest",
            ],
            ascending=[True, True, False, True, True, True],
            kind="mergesort",
        )
        .drop_duplicates(["_subject", "_event_date"], keep="first")
        .reset_index(drop=True)
    )
    scalar = work.loc[:, SLEEP_PRIMARY_SCALARS].apply(pd.to_numeric, errors="coerce")
    scalar.columns = [f"sleep__raw__{column}" for column in scalar.columns]
    derived = pd.DataFrame(
        [_sleep_sequence_features(row) for _, row in work.iterrows()],
        index=work.index,
    )
    output = pd.concat(
        [
            work[["_subject", "_event_date"]],
            work["_wake_timestamp"].rename("_event_timestamp"),
            scalar,
            derived,
        ],
        axis=1,
    ).sort_values(["_subject", "_event_date"], kind="mergesort")
    return output.reset_index(drop=True), duplicate_table


def _make_legacy_activity_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the historical 18-value activity view for the comparator only."""

    required_scalars = {
        "activity_average_met",
        "activity_cal_active",
        "activity_daily_movement",
        "activity_high",
        "activity_inactive",
        "activity_low",
        "activity_medium",
        "activity_non_wear",
        "activity_rest",
        "activity_steps",
        "activity_total",
    }
    _require_columns(raw, ACTIVITY_REQUIRED_COLUMNS | required_scalars, "legacy activity")
    rows: list[dict[str, Any]] = []
    starts = _parse_kst_timestamps(raw["activity_day_start"], "activity_day_start")
    ends = _parse_kst_timestamps(raw["activity_day_end"], "activity_day_end")
    for position, (_, row) in enumerate(raw.iterrows()):
        features: dict[str, Any] = {
            "_subject": str(row["EMAIL"]),
            "_event_date": starts.iloc[position].tz_localize(None).normalize(),
            "_event_timestamp": ends.iloc[position],
        }
        for source in required_scalars:
            features[f"act__{source}"] = _finite_or_none(row.get(source))
        classes = _finite(parse_slash_sequence(row[ACTIVITY_CLASS_BLOB]))
        classes = classes[np.isfinite(classes)]
        features["actseq__class_entropy"] = _entropy(classes)
        features["actseq__class_transition_rate"] = (
            float(np.mean(classes[1:] != classes[:-1]))
            if len(classes) >= 2
            else (0.0 if len(classes) == 1 else np.nan)
        )
        met = _finite(parse_slash_sequence(row[ACTIVITY_MET_BLOB]))
        valid_met = met[np.isfinite(met)]
        features["actseq__met__std"] = (
            float(np.std(valid_met)) if len(valid_met) else np.nan
        )
        features["actseq__met_sedentary_ratio"] = (
            float(np.mean(valid_met <= 1.5)) if len(valid_met) else np.nan
        )
        features["actseq__met_moderate_ratio"] = (
            float(np.mean((valid_met > 1.5) & (valid_met <= 3.0)))
            if len(valid_met)
            else np.nan
        )
        features["actseq__met_vigorous_ratio"] = (
            float(np.mean(valid_met > 3.0)) if len(valid_met) else np.nan
        )
        if len(met) == 1440 and np.isfinite(met).all():
            hourly = met.reshape(24, 60).mean(axis=1)
            doubled = np.r_[hourly, hourly]
            m10 = max(doubled[index:index + 10].mean() for index in range(24))
            l5 = min(doubled[index:index + 5].mean() for index in range(24))
            features["actseq__relative_amplitude"] = float(
                (m10 - l5) / (m10 + l5 + 1e-6)
            )
        else:
            features["actseq__relative_amplitude"] = np.nan
        rows.append(features)
    frame = pd.DataFrame(rows)
    timestamps = frame.groupby(["_subject", "_event_date"])["_event_timestamp"].max()
    biological = frame.drop(columns="_event_timestamp").groupby(
        ["_subject", "_event_date"]
    ).median(numeric_only=True)
    output = biological.join(timestamps).reset_index()
    missing = set(LEGACY_ACTIVITY_COLUMNS) - set(output.columns)
    if missing:
        raise AssertionError("Historical activity comparator columns are incomplete.")
    return output[["_subject", "_event_date", "_event_timestamp", *LEGACY_ACTIVITY_COLUMNS]]


def _make_legacy_sleep_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the historical 31-value longest-sleep comparator view."""

    work = raw.copy()
    work["_subject"] = work["EMAIL"].astype(str)
    work["_wake_timestamp"] = _parse_kst_timestamps(
        work["sleep_bedtime_end"], "sleep_bedtime_end"
    )
    work["_start_timestamp"] = _parse_kst_timestamps(
        work["sleep_bedtime_start"], "sleep_bedtime_start"
    )
    work["_event_date"] = work["_wake_timestamp"].dt.tz_localize(None).dt.normalize()
    work["_duration"] = pd.to_numeric(work["sleep_duration"], errors="coerce")
    work["_daily_sleep_count"] = work.groupby(
        ["_subject", "_event_date"]
    )["_subject"].transform("size").astype(float)
    tie_columns = [
        column
        for column in (
            *SLEEP_PRIMARY_SCALARS,
            SLEEP_HR_BLOB,
            SLEEP_STAGE_BLOB,
            SLEEP_RMSSD_BLOB,
        )
        if column in work.columns
    ]
    work["_tie_digest"] = [
        _sleep_tie_digest(row, tie_columns) for _, row in work.iterrows()
    ]
    work = (
        work.sort_values(
            [
                "_subject",
                "_event_date",
                "_duration",
                "_start_timestamp",
                "_wake_timestamp",
                "_tie_digest",
            ],
            ascending=[True, True, False, True, True, True],
            kind="mergesort",
        )
        .drop_duplicates(["_subject", "_event_date"], keep="first")
        .reset_index(drop=True)
    )
    raw_scalars = (
        "sleep_awake",
        "sleep_breath_average",
        "sleep_deep",
        "sleep_duration",
        "sleep_efficiency",
        "sleep_hr_average",
        "sleep_hr_lowest",
        "sleep_light",
        "sleep_midpoint_at_delta",
        "sleep_onset_latency",
        "sleep_rem",
        "sleep_restless",
        "sleep_rmssd",
        "sleep_temperature_delta",
        "sleep_total",
    )
    rows: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        features: dict[str, Any] = {
            "_subject": str(row["_subject"]),
            "_event_date": row["_event_date"],
            "_event_timestamp": row["_wake_timestamp"],
        }
        for source in raw_scalars:
            features[f"sleep__{source}"] = _finite_or_none(row.get(source))
        duration = _finite_or_none(row.get("sleep_duration"))
        for source in ("sleep_awake", "sleep_deep", "sleep_light", "sleep_rem", "sleep_total"):
            value = _finite_or_none(row.get(source))
            features[f"sleep__{source}_ratio_duration"] = (
                value / duration
                if value is not None and duration is not None and duration > 0
                else np.nan
            )
        features["sleep__daily_sleep_count"] = float(row["_daily_sleep_count"])
        average = _finite_or_none(row.get("sleep_hr_average"))
        lowest = _finite_or_none(row.get("sleep_hr_lowest"))
        features["sleep__hr_drop"] = (
            average - lowest if average is not None and lowest is not None else np.nan
        )
        features["sleep__hr_drop_ratio"] = (
            (average - lowest) / average
            if average is not None and lowest is not None and average > 0
            else np.nan
        )
        stages = _finite(parse_slash_sequence(row[SLEEP_STAGE_BLOB]))
        stages = stages[np.isfinite(stages)]
        features["sleepseq__stage_entropy"] = _entropy(stages)
        features["sleepseq__stage_transition_rate"] = (
            float(np.mean(stages[1:] != stages[:-1]))
            if len(stages) >= 2
            else (0.0 if len(stages) == 1 else np.nan)
        )
        awake = stages == 4
        features["sleepseq__awake_bouts"] = (
            float(np.sum(awake & np.r_[True, ~awake[:-1]])) if len(stages) else np.nan
        )
        hr = _finite(parse_slash_sequence(row[SLEEP_HR_BLOB]), zero_is_missing=True)
        rmssd = _finite(parse_slash_sequence(row[SLEEP_RMSSD_BLOB]), zero_is_missing=True)
        features["sleepseq__hr__std"] = (
            float(np.nanstd(hr)) if np.isfinite(hr).any() else np.nan
        )
        features["sleepseq__rmssd__std"] = (
            float(np.nanstd(rmssd)) if np.isfinite(rmssd).any() else np.nan
        )
        valid_positions = np.flatnonzero(np.isfinite(hr))
        if valid_positions.size:
            position = int(valid_positions[np.nanargmin(hr[valid_positions])])
            features["sleep__lowest_hr_position"] = position / max(1, len(hr) - 1)
            timestamp = row["_start_timestamp"] + pd.Timedelta(minutes=5 * position)
            angle = 2.0 * np.pi * (
                timestamp.hour + timestamp.minute / 60.0
            ) / 24.0
            features["sleep__lowest_hr_clock_sin"] = float(np.sin(angle))
            features["sleep__lowest_hr_clock_cos"] = float(np.cos(angle))
        else:
            features["sleep__lowest_hr_position"] = np.nan
            features["sleep__lowest_hr_clock_sin"] = np.nan
            features["sleep__lowest_hr_clock_cos"] = np.nan
        rows.append(features)
    output = pd.DataFrame(rows)
    missing = set(LEGACY_SLEEP_COLUMNS) - set(output.columns)
    if missing:
        raise AssertionError("Historical sleep comparator columns are incomplete.")
    return output[["_subject", "_event_date", "_event_timestamp", *LEGACY_SLEEP_COLUMNS]]


def _event_ranks(length: int) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=float)
    if length == 1:
        return np.asarray([0.5], dtype=float)
    return np.linspace(0.0, 1.0, length, dtype=float)


def _aggregate_event_summary(
    daily: pd.DataFrame,
    subjects: Sequence[str],
    feature_columns: Sequence[str],
    modality: str,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for subject in subjects:
        events = (
            daily.loc[daily["_subject"] == subject]
            .sort_values("_event_timestamp", kind="mergesort")
            .tail(EVENT_LIMIT)
        )
        ranks = _event_ranks(len(events))
        features: dict[str, float] = {}
        for column in feature_columns:
            values = pd.to_numeric(events[column], errors="coerce").to_numpy(dtype=float)
            for statistic, value in _event_summary_stats(values, ranks).items():
                features[f"{modality}__event28__{column}__{statistic}"] = value
        rows.append(features)
    return pd.DataFrame(rows, index=pd.Index(subjects, name="_subject"))


def _interpolate_event_sequence(
    daily: pd.DataFrame,
    subjects: Sequence[str],
    feature_columns: Sequence[str],
) -> np.ndarray:
    """Interpolate each daily feature onto 28 normalized event-rank positions.

    A single finite observation is repeated.  Zero finite observations remain
    NaN for fold-local imputation.  No mask or event count is returned.
    """

    target = np.linspace(0.0, 1.0, EVENT_SEQUENCE_STEPS, dtype=float)
    tensors: list[np.ndarray] = []
    for subject in subjects:
        events = (
            daily.loc[daily["_subject"] == subject]
            .sort_values("_event_timestamp", kind="mergesort")
            .tail(EVENT_LIMIT)
        )
        ranks = _event_ranks(len(events))
        tensor = np.full((EVENT_SEQUENCE_STEPS, len(feature_columns)), np.nan, dtype=np.float32)
        for feature_index, column in enumerate(feature_columns):
            values = pd.to_numeric(events[column], errors="coerce").to_numpy(dtype=float)
            keep = np.isfinite(values) & np.isfinite(ranks)
            if keep.sum() == 1:
                tensor[:, feature_index] = float(values[keep][0])
            elif keep.sum() >= 2:
                tensor[:, feature_index] = np.interp(
                    target, ranks[keep], values[keep]
                ).astype(np.float32)
        tensors.append(tensor)
    return np.stack(tensors, axis=0)


def _calendar_gap_statistics(dates: Sequence[pd.Timestamp]) -> tuple[float, float, float]:
    ordered = np.asarray(sorted(pd.Timestamp(value).value for value in set(dates)), dtype=np.int64)
    if not len(ordered):
        return 0.0, 0.0, 0.0
    span = float((ordered[-1] - ordered[0]) / (24 * 3600 * 1e9) + 1.0)
    if len(ordered) < 2:
        return span, 0.0, 0.0
    gaps = np.diff(ordered) / (24 * 3600 * 1e9)
    return span, float(np.max(gaps)), float(np.mean(gaps))


def _activity_missing_ratio(frame: pd.DataFrame) -> float:
    missing = total = 0
    scalar_columns = [column for column in ACTIVITY_PRIMARY_SCALARS if column in frame]
    scalar = frame[scalar_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    missing += int((~np.isfinite(scalar)).sum())
    total += int(scalar.size)
    for _, row in frame.iterrows():
        classes = _finite(parse_slash_sequence(row[ACTIVITY_CLASS_BLOB]))
        met = _finite(parse_slash_sequence(row[ACTIVITY_MET_BLOB]))
        # Class 0 is non-wear and is treated as acquisition missingness only here.
        missing += int((~np.isfinite(classes) | (classes == 0)).sum())
        total += int(len(classes))
        missing += int((~np.isfinite(met)).sum())
        total += int(len(met))
    return float(missing / total) if total else np.nan


def _sleep_missing_ratio(frame: pd.DataFrame) -> float:
    missing = total = 0
    scalar_columns = [column for column in SLEEP_PRIMARY_SCALARS if column in frame]
    scalar = frame[scalar_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    missing += int((~np.isfinite(scalar)).sum())
    total += int(scalar.size)
    for _, row in frame.iterrows():
        hr = _finite(parse_slash_sequence(row[SLEEP_HR_BLOB]), zero_is_missing=True)
        rmssd = _finite(parse_slash_sequence(row[SLEEP_RMSSD_BLOB]), zero_is_missing=True)
        stages = _finite(parse_slash_sequence(row[SLEEP_STAGE_BLOB]))
        missing += int((~np.isfinite(hr)).sum() + (~np.isfinite(rmssd)).sum())
        total += int(len(hr) + len(rmssd))
        missing += int((~np.isfinite(stages) | ~np.isin(stages, [1, 2, 3, 4])).sum())
        total += int(len(stages))
    return float(missing / total) if total else np.nan


def _build_coverage_table(
    activity_raw: pd.DataFrame,
    sleep_raw: pd.DataFrame,
    activity_daily: pd.DataFrame,
    sleep_daily: pd.DataFrame,
    duplicate_table: pd.DataFrame,
    anchor_timestamps: pd.Series,
    subjects: Sequence[str],
) -> pd.DataFrame:
    activity_source = activity_raw.copy()
    sleep_source = sleep_raw.copy()
    activity_source["_subject"] = activity_source["EMAIL"].astype(str)
    sleep_source["_subject"] = sleep_source["EMAIL"].astype(str)
    activity_source["_event_date"] = _local_dates(
        activity_source["activity_day_start"], "activity_day_start"
    )
    activity_source["_event_timestamp"] = _parse_kst_timestamps(
        activity_source["activity_day_end"], "activity_day_end"
    )
    sleep_source["_event_date"] = _local_dates(
        sleep_source["sleep_bedtime_end"], "sleep_bedtime_end"
    )
    sleep_source["_event_timestamp"] = _parse_kst_timestamps(
        sleep_source["sleep_bedtime_end"], "sleep_bedtime_end"
    )
    rows: list[dict[str, float]] = []
    for subject in subjects:
        anchor = anchor_timestamps.loc[subject]
        activity_events = activity_daily.loc[
            (activity_daily["_subject"] == subject)
            & (activity_daily["_event_timestamp"] <= anchor)
        ]
        sleep_events = sleep_daily.loc[
            (sleep_daily["_subject"] == subject)
            & (sleep_daily["_event_timestamp"] <= anchor)
        ]
        activity_dates = list(activity_events["_event_date"])
        sleep_dates = list(sleep_events["_event_date"])
        activity_span, activity_max_gap, activity_mean_gap = _calendar_gap_statistics(
            activity_dates
        )
        sleep_span, sleep_max_gap, sleep_mean_gap = _calendar_gap_statistics(sleep_dates)
        activity_set, sleep_set = set(activity_dates), set(sleep_dates)
        raw_activity_part = activity_source.loc[
            (activity_source["_subject"] == subject)
            & (activity_source["_event_timestamp"] <= anchor)
        ]
        raw_sleep_part = sleep_source.loc[
            (sleep_source["_subject"] == subject)
            & (sleep_source["_event_timestamp"] <= anchor)
        ]
        duplicate_part = duplicate_table.loc[
            (duplicate_table["_subject"] == subject)
            & (duplicate_table["_wake_timestamp"] <= anchor)
        ]
        duplicate_count = float(
            duplicate_part.groupby("_event_date").size().sub(1).clip(lower=0).sum()
        )
        rows.append(
            {
                "activity_observed_day_count": float(len(activity_set)),
                "sleep_observed_night_count": float(len(sleep_set)),
                "paired_day_count": float(len(activity_set & sleep_set)),
                "activity_observation_span_days": activity_span,
                "sleep_observation_span_days": sleep_span,
                "activity_max_calendar_gap_days": activity_max_gap,
                "activity_mean_calendar_gap_days": activity_mean_gap,
                "sleep_max_calendar_gap_days": sleep_max_gap,
                "sleep_mean_calendar_gap_days": sleep_mean_gap,
                "activity_raw_field_missing_ratio": _activity_missing_ratio(raw_activity_part),
                "sleep_raw_field_missing_ratio": _sleep_missing_ratio(raw_sleep_part),
                "main_sleep_duplicate_episode_count": duplicate_count,
            }
        )
    return pd.DataFrame(rows, index=pd.Index(subjects, name="_subject"))


def _build_legacy_calendar_tensor(
    activity_daily: pd.DataFrame,
    sleep_daily: pd.DataFrame,
    anchor_dates: pd.Series,
    subjects: Sequence[str],
    activity_columns: Sequence[str],
    sleep_columns: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    features = list(activity_columns) + list(sleep_columns)
    joined = activity_daily[["_subject", "_event_date", *activity_columns]].merge(
        sleep_daily[["_subject", "_event_date", *sleep_columns]],
        on=["_subject", "_event_date"],
        how="outer",
        validate="one_to_one",
    )
    tensors: list[np.ndarray] = []
    for subject in subjects:
        anchor = anchor_dates.loc[subject]
        dates = pd.date_range(
            anchor - pd.Timedelta(days=LEGACY_CALENDAR_DAYS - 1),
            anchor,
            freq="D",
        )
        part = (
            joined.loc[(joined["_subject"] == subject) & (joined["_event_date"] <= anchor)]
            .set_index("_event_date")
            .reindex(dates)
            .reindex(columns=features)
            .apply(pd.to_numeric, errors="coerce")
        )
        tensors.append(part.to_numpy(dtype=np.float32))
    return np.stack(tensors, axis=0), features


def assert_primary_feature_contract(feature_names: Sequence[str]) -> None:
    """Assert that explicit shortcut/leakage fields cannot enter a primary view."""

    lowered = [str(name).lower() for name in feature_names]
    offenders = sorted(
        name
        for name, low in zip(feature_names, lowered)
        if any(token in low for token in FORBIDDEN_PRIMARY_TOKENS)
    )
    if offenders:
        raise AssertionError(f"Forbidden primary feature names detected: {offenders[:10]}")
    if len(lowered) != len(set(lowered)):
        raise AssertionError("Duplicate primary feature names.")


def build_feature_bundle(
    activity_raw: pd.DataFrame,
    sleep_raw: pd.DataFrame,
    subject_ids: Sequence[str] | None = None,
) -> FeatureBundle:
    """Build all deterministic source-only representations without labels.

    Each subject's final valid local activity date is the prediction anchor.
    Sleep wake-events after that anchor are removed before recent-event
    selection, preventing future information from entering any representation.
    """

    activity_daily = make_activity_daily(activity_raw)
    raw_anchor_timestamps = (
        pd.DataFrame(
            {
                "_subject": activity_raw["EMAIL"].astype(str),
                "_end": _parse_kst_timestamps(
                    activity_raw["activity_day_end"], "activity_day_end"
                ),
            }
        )
        .groupby("_subject", sort=True)["_end"]
        .max()
    )
    guarded_sleep = sleep_raw.copy()
    guarded_sleep["_guard_wake"] = _parse_kst_timestamps(
        guarded_sleep["sleep_bedtime_end"], "sleep_bedtime_end"
    )
    guarded_sleep["_guard_anchor"] = guarded_sleep["EMAIL"].astype(str).map(
        raw_anchor_timestamps
    )
    if guarded_sleep["_guard_anchor"].isna().any():
        raise AssertionError("Sleep subject has no activity prediction timestamp.")
    sleep_post_index_rows = int(
        (guarded_sleep["_guard_wake"] > guarded_sleep["_guard_anchor"]).sum()
    )
    guarded_sleep = guarded_sleep.loc[
        guarded_sleep["_guard_wake"] <= guarded_sleep["_guard_anchor"]
    ].drop(columns=["_guard_wake", "_guard_anchor"])
    # Longest main-sleep selection occurs only after the exact timestamp guard.
    sleep_daily, duplicate_table = make_sleep_daily(guarded_sleep)
    legacy_activity_daily = _make_legacy_activity_daily(activity_raw)
    legacy_sleep_daily = _make_legacy_sleep_daily(guarded_sleep)
    activity_subjects = set(activity_daily["_subject"])
    sleep_subjects = set(sleep_daily["_subject"])
    if activity_subjects != sleep_subjects:
        raise AssertionError("Activity and sleep subject sets differ.")
    subjects = list(subject_ids) if subject_ids is not None else sorted(activity_subjects)
    if len(subjects) != len(set(subjects)) or set(subjects) != activity_subjects:
        raise AssertionError("Supplied subject order does not match deterministic source subjects.")
    last_activity = (
        activity_daily.sort_values("_event_timestamp", kind="mergesort")
        .groupby("_subject", sort=True)
        .tail(1)
        .set_index("_subject")
        .reindex(subjects)
    )
    anchor_timestamps = last_activity["_event_timestamp"]
    # The comparator preserves the historical calendar convention: its grid
    # ends on the start-local-date of the last activity event, while the exact
    # activity_day_end timestamp remains the future-information guard.
    anchor_dates = last_activity["_event_date"]
    if anchor_timestamps.isna().any() or anchor_dates.isna().any():
        raise AssertionError("Every subject needs a valid activity prediction anchor.")
    if sleep_daily.empty or set(sleep_daily["_subject"]) != set(subjects):
        raise AssertionError("Every subject needs at least one non-future sleep event.")
    if any(
        (part["_event_timestamp"] > anchor_timestamps.loc[subject]).any()
        for subject, part in sleep_daily.groupby("_subject")
    ):
        raise AssertionError("Future sleep event survived the activity-anchor chronology guard.")

    activity_columns = sorted(
        column for column in activity_daily.columns
        if column not in {"_subject", "_event_date", "_event_timestamp"}
    )
    sleep_columns = sorted(
        column for column in sleep_daily.columns
        if column not in {"_subject", "_event_date", "_event_timestamp"}
    )
    assert_primary_feature_contract(activity_columns)
    assert_primary_feature_contract(sleep_columns)
    activity_summary = _aggregate_event_summary(
        activity_daily, subjects, activity_columns, "activity"
    )
    sleep_summary = _aggregate_event_summary(sleep_daily, subjects, sleep_columns, "sleep")
    event_summary = activity_summary.join(sleep_summary, how="outer").reindex(subjects)
    assert_primary_feature_contract(event_summary.columns)
    activity_sequence = _interpolate_event_sequence(
        activity_daily, subjects, activity_columns
    )
    sleep_sequence = _interpolate_event_sequence(sleep_daily, subjects, sleep_columns)
    coverage = _build_coverage_table(
        activity_raw,
        sleep_raw,
        activity_daily,
        sleep_daily,
        duplicate_table,
        anchor_timestamps,
        subjects,
    )
    legacy_values, legacy_features = _build_legacy_calendar_tensor(
        legacy_activity_daily,
        legacy_sleep_daily,
        anchor_dates,
        subjects,
        LEGACY_ACTIVITY_COLUMNS,
        LEGACY_SLEEP_COLUMNS,
    )
    bundle = FeatureBundle(
        subject_ids=[str(value) for value in subjects],
        event_summary=event_summary.apply(pd.to_numeric, errors="coerce"),
        activity_sequence=activity_sequence.astype(np.float32),
        sleep_sequence=sleep_sequence.astype(np.float32),
        activity_sequence_features=activity_columns,
        sleep_sequence_features=sleep_columns,
        coverage=coverage.apply(pd.to_numeric, errors="coerce"),
        legacy_values=legacy_values.astype(np.float32),
        legacy_features=legacy_features,
        diagnostics={
            "design_version": DESIGN_VERSION,
            "subjects": int(len(subjects)),
            "prediction_index": "last valid activity_day_end timestamp (+09:00)",
            "activity_raw_rows": int(len(activity_raw)),
            "sleep_raw_rows": int(len(sleep_raw)),
            "activity_daily_events": int(len(activity_daily)),
            "sleep_main_events_before_anchor": int(len(sleep_daily)),
            "sleep_post_index_raw_episodes_excluded": sleep_post_index_rows,
            "activity_post_index_raw_episodes_excluded": 0,
            "primary_explicit_coverage_signals": 0,
            "legacy_value_features": len(legacy_features),
            "legacy_contract": (
                "chronology-corrected historical 49-feature comparator; calendar grid "
                "ends on last activity event start-local-date"
            ),
        },
    )
    validate_feature_bundle(bundle)
    return bundle


def validate_feature_bundle(bundle: FeatureBundle, expected_subjects: int | None = None) -> None:
    """Fail fast on alignment, dimensions, chronology-contract, or leakage errors."""

    count = len(bundle.subject_ids)
    if expected_subjects is not None and count != expected_subjects:
        raise AssertionError("Unexpected feature-bundle subject count.")
    if count != len(set(bundle.subject_ids)):
        raise AssertionError("Duplicate subject in feature bundle.")
    if list(bundle.event_summary.index.astype(str)) != bundle.subject_ids:
        raise AssertionError("event_summary subject order mismatch.")
    if list(bundle.coverage.index.astype(str)) != bundle.subject_ids:
        raise AssertionError("coverage subject order mismatch.")
    if bundle.activity_sequence.shape != (
        count,
        EVENT_SEQUENCE_STEPS,
        len(bundle.activity_sequence_features),
    ):
        raise AssertionError("Activity event-sequence shape mismatch.")
    if bundle.sleep_sequence.shape != (
        count,
        EVENT_SEQUENCE_STEPS,
        len(bundle.sleep_sequence_features),
    ):
        raise AssertionError("Sleep event-sequence shape mismatch.")
    if bundle.legacy_values.shape != (
        count,
        LEGACY_CALENDAR_DAYS,
        len(bundle.legacy_features),
    ):
        raise AssertionError("Legacy calendar tensor shape mismatch.")
    expected_legacy = list((*LEGACY_ACTIVITY_COLUMNS, *LEGACY_SLEEP_COLUMNS))
    if bundle.legacy_features != expected_legacy or len(bundle.legacy_features) != 49:
        raise AssertionError("Legacy comparator must retain the fixed ordered 49-value view.")
    if bundle.legacy_values.shape[2] * 3 != 147:
        raise AssertionError("Legacy transformed channel contract must be 147.")
    assert_primary_feature_contract(bundle.event_summary.columns)
    assert_primary_feature_contract(bundle.activity_sequence_features)
    assert_primary_feature_contract(bundle.sleep_sequence_features)
    if not bundle.event_summary.columns.is_unique or not bundle.coverage.columns.is_unique:
        raise AssertionError("Feature names must be unique.")


def deidentify_feature_bundle(
    bundle: FeatureBundle, secret_key: str | bytes
) -> dict[str, Any]:
    """Create a safe derived-data view; no anchors or raw IDs are retained."""

    hashes = hash_subjects(bundle.subject_ids, secret_key)
    return {
        "subject_hashes": hashes,
        "event_summary": bundle.event_summary.set_axis(hashes, axis=0).rename_axis(
            "subject_hash"
        ),
        "activity_sequence": bundle.activity_sequence,
        "sleep_sequence": bundle.sleep_sequence,
        "coverage": bundle.coverage.set_axis(hashes, axis=0).rename_axis("subject_hash"),
        "legacy_values": bundle.legacy_values,
        "summary_manifest": bundle.summary_manifest(),
        "sequence_manifest": bundle.sequence_manifest(),
        "public_summary": bundle.public_summary(),
    }


@dataclass
class FoldTabularPreprocessor:
    """Fold-local all-missing/constant removal, imputation, and robust scaling."""

    robust_scale: bool
    clip: bool = False
    input_features: list[str] = field(default_factory=list)
    output_features: list[str] = field(default_factory=list)
    medians: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    centers: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "FoldTabularPreprocessor":
        clean = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        self.input_features = list(clean.columns)
        usable: list[str] = []
        for column in clean.columns:
            values = clean[column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if len(finite) and np.unique(finite).size > 1:
                usable.append(column)
        if not usable:
            raise AssertionError("No nonconstant finite tabular feature in the training fold.")
        self.output_features = usable
        values = clean[usable].to_numpy(dtype=float)
        self.medians = np.nanmedian(values, axis=0)
        self.medians = np.nan_to_num(self.medians, nan=0.0)
        filled = np.where(np.isfinite(values), values, self.medians[None, :])
        if self.clip:
            self.lower = np.quantile(filled, 0.005, axis=0)
            self.upper = np.quantile(filled, 0.995, axis=0)
            filled = np.clip(filled, self.lower, self.upper)
        else:
            self.lower = np.full(filled.shape[1], -np.inf)
            self.upper = np.full(filled.shape[1], np.inf)
        if self.robust_scale:
            self.centers = np.median(filled, axis=0)
            q25 = np.quantile(filled, 0.25, axis=0)
            q75 = np.quantile(filled, 0.75, axis=0)
            self.scales = q75 - q25
            self.scales[~np.isfinite(self.scales) | (self.scales < 1e-8)] = 1.0
        else:
            self.centers = np.zeros(filled.shape[1], dtype=float)
            self.scales = np.ones(filled.shape[1], dtype=float)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians is None or self.lower is None or self.upper is None:
            raise RuntimeError("FoldTabularPreprocessor is not fitted.")
        if list(frame.columns) != self.input_features:
            raise AssertionError("Tabular inference feature names/order changed.")
        values = (
            frame.reindex(columns=self.output_features)
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .to_numpy(dtype=float)
        )
        values = np.where(np.isfinite(values), values, self.medians[None, :])
        values = np.clip(values, self.lower, self.upper)
        values = (values - self.centers) / self.scales
        if not np.isfinite(values).all():
            raise AssertionError("Non-finite value after fold-local tabular preprocessing.")
        return values.astype(np.float32)

    def manifest(self) -> dict[str, Any]:
        return {
            "input_features": list(self.input_features),
            "output_features": list(self.output_features),
            "removed_feature_count": len(self.input_features) - len(self.output_features),
            "median_imputation": True,
            "clip_percentiles": [0.5, 99.5] if self.clip else None,
            "scaler": "RobustScaler-equivalent median/IQR" if self.robust_scale else None,
        }


@dataclass
class FoldEventSequencePreprocessor:
    """Independent modality fold-local median/IQR sequence transform.

    The transform never emits a missingness mask, event count, or calendar
    delta.  Feature removal and every statistic are learned on fold-train only.
    """

    input_features: list[str] = field(default_factory=list)
    output_features: list[str] = field(default_factory=list)
    usable_indices: np.ndarray | None = None
    medians: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    iqrs: np.ndarray | None = None

    def fit(
        self, values: np.ndarray, feature_names: Sequence[str]
    ) -> "FoldEventSequencePreprocessor":
        values = np.asarray(values, dtype=float)
        if values.ndim != 3 or values.shape[2] != len(feature_names):
            raise AssertionError("Event sequence and feature-name dimensions disagree.")
        self.input_features = list(feature_names)
        usable: list[int] = []
        for index in range(values.shape[2]):
            finite = values[:, :, index][np.isfinite(values[:, :, index])]
            if len(finite) and np.unique(finite).size > 1:
                usable.append(index)
        if not usable:
            raise AssertionError("No nonconstant finite sequence feature in training fold.")
        self.usable_indices = np.asarray(usable, dtype=int)
        self.output_features = [self.input_features[index] for index in usable]
        selected = values[:, :, self.usable_indices]
        self.medians = np.nanmedian(selected, axis=(0, 1))
        self.medians = np.nan_to_num(self.medians, nan=0.0)
        filled = np.where(np.isfinite(selected), selected, self.medians[None, None, :])
        self.lower = np.quantile(filled, 0.005, axis=(0, 1))
        self.upper = np.quantile(filled, 0.995, axis=(0, 1))
        filled = np.clip(filled, self.lower, self.upper)
        q25 = np.quantile(filled, 0.25, axis=(0, 1))
        q75 = np.quantile(filled, 0.75, axis=(0, 1))
        self.iqrs = q75 - q25
        self.iqrs[~np.isfinite(self.iqrs) | (self.iqrs < 1e-8)] = 1.0
        return self

    def transform(self, values: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
        if self.usable_indices is None or self.medians is None or self.iqrs is None:
            raise RuntimeError("FoldEventSequencePreprocessor is not fitted.")
        if list(feature_names) != self.input_features:
            raise AssertionError("Event sequence feature order changed at inference.")
        selected = np.asarray(values, dtype=float)[:, :, self.usable_indices]
        selected = np.where(
            np.isfinite(selected), selected, self.medians[None, None, :]
        )
        selected = np.clip(selected, self.lower, self.upper)
        selected = (selected - self.medians[None, None, :]) / self.iqrs[None, None, :]
        if not np.isfinite(selected).all():
            raise AssertionError("Non-finite value after event-sequence preprocessing.")
        return selected.astype(np.float32)

    def manifest(self) -> dict[str, Any]:
        return {
            "input_features": list(self.input_features),
            "output_features": list(self.output_features),
            "removed_feature_count": len(self.input_features) - len(self.output_features),
            "median_imputation": True,
            "scale": "feature-wise fold-train IQR",
            "clip_percentiles": [0.5, 99.5],
            "explicit_mask_or_count_channels": 0,
        }


@dataclass
class FoldLegacySequencePreprocessor:
    """Legacy value + observed mask + normalized missing-run delta transform."""

    input_features: list[str] = field(default_factory=list)
    output_features: list[str] = field(default_factory=list)
    usable_indices: np.ndarray | None = None
    medians: np.ndarray | None = None
    iqrs: np.ndarray | None = None

    def fit(
        self, values: np.ndarray, feature_names: Sequence[str]
    ) -> "FoldLegacySequencePreprocessor":
        values = np.asarray(values, dtype=float)
        if values.ndim != 3 or values.shape[2] != len(feature_names):
            raise AssertionError("Legacy values and feature names disagree.")
        self.input_features = list(feature_names)
        # Historical SequenceFoldTransformer retained all fixed channels,
        # mapping all-missing medians to 0 and degenerate IQRs to 1.
        self.usable_indices = np.arange(values.shape[2], dtype=int)
        self.output_features = list(self.input_features)
        selected = values[:, :, self.usable_indices]
        self.medians = np.nanmedian(selected, axis=(0, 1))
        self.medians = np.nan_to_num(self.medians, nan=0.0)
        q25 = np.nanquantile(selected, 0.25, axis=(0, 1))
        q75 = np.nanquantile(selected, 0.75, axis=(0, 1))
        self.iqrs = np.nan_to_num(q75 - q25, nan=1.0)
        self.iqrs[self.iqrs < 1e-6] = 1.0
        return self

    def transform(
        self, values: np.ndarray, feature_names: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.usable_indices is None or self.medians is None or self.iqrs is None:
            raise RuntimeError("FoldLegacySequencePreprocessor is not fitted.")
        if list(feature_names) != self.input_features:
            raise AssertionError("Legacy feature order changed at inference.")
        selected = np.asarray(values, dtype=float)[:, :, self.usable_indices]
        observed = np.isfinite(selected)
        normalized = (
            np.where(observed, selected, self.medians[None, None, :])
            - self.medians[None, None, :]
        ) / self.iqrs[None, None, :]
        delta = np.zeros_like(normalized, dtype=float)
        for time_index in range(1, selected.shape[1]):
            delta[:, time_index, :] = np.where(
                observed[:, time_index, :],
                0.0,
                np.minimum(delta[:, time_index - 1, :] + 1.0, selected.shape[1]),
            )
        delta /= max(1, selected.shape[1])
        channels = np.concatenate([normalized, observed.astype(float), delta], axis=2)
        day_mask = observed.any(axis=2).astype(np.float32)
        if not np.isfinite(channels).all():
            raise AssertionError("Non-finite legacy channel after fold transform.")
        return channels.astype(np.float32), day_mask

    def manifest(self) -> dict[str, Any]:
        return {
            "input_features": list(self.input_features),
            "output_features": list(self.output_features),
            "channels": ["normalized_value", "observed_mask", "normalized_delta"],
            "calendar_days": LEGACY_CALENDAR_DAYS,
            "selection_eligible": False,
        }


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise AssertionError("Expected an N x 3 probability matrix.")
    if not np.isfinite(probabilities).all():
        raise AssertionError("Model returned a non-finite probability.")
    probabilities = np.clip(probabilities, 1e-7, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


@dataclass
class FittedTabularCandidate:
    name: str
    preprocessor: FoldTabularPreprocessor
    model: Any

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        values = self.preprocessor.transform(frame)
        probabilities = self.model.predict_proba(values)
        classes = np.asarray(self.model.classes_, dtype=int)
        if not np.array_equal(classes, np.arange(3)):
            raise AssertionError("Tabular model class order is not [0, 1, 2].")
        return _normalize_probabilities(probabilities)


def _fit_elastic(
    frame: pd.DataFrame, labels: np.ndarray, seed: int
) -> FittedTabularCandidate:
    from sklearn.linear_model import LogisticRegression

    preprocessor = FoldTabularPreprocessor(robust_scale=True, clip=True).fit(frame)
    values = preprocessor.transform(frame)
    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=0.1,
        l1_ratio=0.5,
        class_weight="balanced",
        max_iter=10000,
        tol=1e-4,
        random_state=int(seed),
        n_jobs=1,
    )
    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(values, labels)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("Elastic-net logistic regression did not converge.")
    if np.asarray(model.n_iter_).max() >= model.max_iter:
        raise RuntimeError("Elastic-net logistic regression reached max_iter.")
    return FittedTabularCandidate("event_elastic_v1", preprocessor, model)


def _fit_extra_trees(
    frame: pd.DataFrame, labels: np.ndarray, seed: int, fast_mode: bool
) -> FittedTabularCandidate:
    from sklearn.ensemble import ExtraTreesClassifier

    preprocessor = FoldTabularPreprocessor(robust_scale=False, clip=False).fit(frame)
    values = preprocessor.transform(frame)
    model = ExtraTreesClassifier(
        n_estimators=64 if fast_mode else 1000,
        max_depth=5,
        min_samples_leaf=4,
        max_features=0.35,
        class_weight="balanced_subsample",
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(values, labels)
    return FittedTabularCandidate("event_extra_trees_v1", preprocessor, model)


@dataclass
class DualModalityTCNBundle:
    """Two independent residual encoders with late pooled-embedding fusion."""

    seed: int
    device: str
    activity_preprocessor: FoldEventSequencePreprocessor
    sleep_preprocessor: FoldEventSequencePreprocessor
    activity_channels: int
    sleep_channels: int
    state_dict: dict[str, Any]
    epochs: int

    @staticmethod
    def _network(activity_channels: int, sleep_channels: int):
        import torch
        from torch import nn

        class ResidualBlock(nn.Module):
            def __init__(self, hidden: int, dilation: int):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation),
                    nn.GroupNorm(6, hidden),
                    nn.GELU(),
                    nn.Dropout(0.35),
                    nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation),
                    nn.GroupNorm(6, hidden),
                    nn.GELU(),
                    nn.Dropout(0.35),
                )

            def forward(self, values):
                return values + self.layers(values)

        class Encoder(nn.Module):
            def __init__(self, input_channels: int):
                super().__init__()
                self.input = nn.Conv1d(input_channels, 24, 1)
                self.blocks = nn.Sequential(
                    *[ResidualBlock(24, dilation) for dilation in (1, 2, 4, 8)]
                )

            def forward(self, values):
                hidden = self.blocks(self.input(values.transpose(1, 2)))
                return hidden.mean(dim=2)

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.activity = Encoder(activity_channels)
                self.sleep = Encoder(sleep_channels)
                self.head = nn.Sequential(nn.Dropout(0.35), nn.Linear(48, 3))

            def forward(self, activity, sleep):
                return self.head(torch.cat([self.activity(activity), self.sleep(sleep)], dim=1))

        return Network()

    @classmethod
    def fit(
        cls,
        activity: np.ndarray,
        sleep: np.ndarray,
        activity_features: Sequence[str],
        sleep_features: Sequence[str],
        labels: np.ndarray,
        seed: int,
        device: str,
        fast_mode: bool = False,
    ) -> "DualModalityTCNBundle":
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        set_all_seeds(seed)
        activity_preprocessor = FoldEventSequencePreprocessor().fit(
            activity, activity_features
        )
        sleep_preprocessor = FoldEventSequencePreprocessor().fit(sleep, sleep_features)
        activity_values = activity_preprocessor.transform(activity, activity_features)
        sleep_values = sleep_preprocessor.transform(sleep, sleep_features)
        labels = np.asarray(labels, dtype=int)
        target_device = torch.device(device if torch.cuda.is_available() else "cpu")
        dataset = TensorDataset(
            torch.tensor(activity_values, dtype=torch.float32),
            torch.tensor(sleep_values, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )
        loader = DataLoader(
            dataset,
            batch_size=min(32, len(dataset)),
            shuffle=True,
            generator=torch.Generator().manual_seed(int(seed)),
            num_workers=0,
        )
        network = cls._network(activity_values.shape[2], sleep_values.shape[2]).to(target_device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=8e-4, weight_decay=2e-3)
        counts = np.bincount(labels, minlength=3).astype(float)
        weights = np.sqrt(len(labels) / (3.0 * np.maximum(counts, 1.0)))
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=target_device),
            label_smoothing=0.05,
        )
        epochs = 3 if fast_mode else 120
        for _ in range(epochs):
            network.train()
            for batch_activity, batch_sleep, batch_labels in loader:
                batch_activity = batch_activity.to(target_device)
                batch_sleep = batch_sleep.to(target_device)
                batch_labels = batch_labels.to(target_device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(network(batch_activity, batch_sleep), batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()
        state = {
            name: value.detach().cpu().clone() for name, value in network.state_dict().items()
        }
        return cls(
            int(seed),
            str(device),
            activity_preprocessor,
            sleep_preprocessor,
            int(activity_values.shape[2]),
            int(sleep_values.shape[2]),
            state,
            epochs,
        )

    def predict_proba(
        self,
        activity: np.ndarray,
        sleep: np.ndarray,
        activity_features: Sequence[str],
        sleep_features: Sequence[str],
    ) -> np.ndarray:
        import torch

        activity_values = self.activity_preprocessor.transform(activity, activity_features)
        sleep_values = self.sleep_preprocessor.transform(sleep, sleep_features)
        target_device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        network = self._network(self.activity_channels, self.sleep_channels).to(target_device)
        network.load_state_dict(self.state_dict)
        network.eval()
        probabilities: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(activity_values), 128):
                logits = network(
                    torch.tensor(activity_values[start:start + 128], device=target_device),
                    torch.tensor(sleep_values[start:start + 128], device=target_device),
                )
                probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        return _normalize_probabilities(np.concatenate(probabilities, axis=0))

    def manifest(self) -> dict[str, Any]:
        return {
            "architecture": "dual-modality residual TCN, mean-pool late fusion",
            "hidden": 24,
            "kernel_size": 3,
            "dilations": [1, 2, 4, 8],
            "convolutions_per_block": 2,
            "normalization": "GroupNorm(6)",
            "activation": "GELU",
            "dropout": 0.35,
            "epochs": self.epochs,
            "optimizer": "AdamW(lr=8e-4, weight_decay=2e-3)",
            "gradient_clip": 1.0,
            "label_smoothing": 0.05,
            "class_weight": "sqrt-balanced fold-train only",
            "mixed_precision": "disabled_locked_float32",
            "activity_preprocessor": self.activity_preprocessor.manifest(),
            "sleep_preprocessor": self.sleep_preprocessor.manifest(),
        }


@dataclass
class LegacyTCNBundle:
    seed: int
    device: str
    preprocessor: FoldLegacySequencePreprocessor
    input_channels: int
    state_dict: dict[str, Any]
    epochs: int
    epoch_selector_seed: int
    epoch_selector_train_counts: list[int]
    epoch_selector_stop_counts: list[int]

    @staticmethod
    def _network(input_channels: int):
        import torch
        from torch import nn

        class ResidualBlock(nn.Module):
            def __init__(self, dilation: int):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Conv1d(24, 24, 3, padding=dilation, dilation=dilation),
                    nn.GroupNorm(6, 24),
                    nn.GELU(),
                    nn.Dropout(0.35),
                    nn.Conv1d(24, 24, 3, padding=dilation, dilation=dilation),
                    nn.GroupNorm(6, 24),
                    nn.GELU(),
                    nn.Dropout(0.35),
                )

            def forward(self, values):
                return values + self.layers(values)

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.input = nn.Conv1d(input_channels, 24, 1)
                self.blocks = nn.Sequential(
                    *[ResidualBlock(dilation) for dilation in (1, 2, 4, 8)]
                )
                self.head = nn.Sequential(
                    nn.Linear(48, 32), nn.GELU(), nn.Dropout(0.35), nn.Linear(32, 3)
                )

            def forward(self, values, day_mask):
                hidden = self.blocks(self.input(values.transpose(1, 2))).transpose(1, 2)
                denominator = day_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                mean_pool = (hidden * day_mask.unsqueeze(-1)).sum(dim=1) / denominator
                positions = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
                last_index = (positions * day_mask.long()).max(dim=1).values
                last_pool = hidden[
                    torch.arange(hidden.shape[0], device=hidden.device), last_index
                ]
                return self.head(torch.cat([mean_pool, last_pool], dim=1))

        return Network()

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        feature_names: Sequence[str],
        labels: np.ndarray,
        seed: int,
        device: str,
        fast_mode: bool = False,
    ) -> "LegacyTCNBundle":
        import torch
        from sklearn.metrics import f1_score
        from sklearn.model_selection import StratifiedShuffleSplit
        from torch.utils.data import DataLoader, TensorDataset

        set_all_seeds(seed)
        preprocessor = FoldLegacySequencePreprocessor().fit(values, feature_names)
        channels, day_mask = preprocessor.transform(values, feature_names)
        labels = np.asarray(labels, dtype=int)
        target_device = torch.device(device if torch.cuda.is_available() else "cpu")
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=int(seed))
        epoch_train, epoch_stop = next(split.split(np.zeros(len(labels)), labels))
        epoch_train = np.asarray(epoch_train, dtype=int)
        epoch_stop = np.asarray(epoch_stop, dtype=int)
        train_counts = np.bincount(labels[epoch_train], minlength=3).astype(int)
        stop_counts = np.bincount(labels[epoch_stop], minlength=3).astype(int)
        if np.any(train_counts == 0) or np.any(stop_counts == 0):
            raise AssertionError("Legacy epoch-selector split is missing a class.")
        dataset = TensorDataset(
            torch.tensor(channels[epoch_train], dtype=torch.float32),
            torch.tensor(day_mask[epoch_train], dtype=torch.float32),
            torch.tensor(labels[epoch_train], dtype=torch.long),
        )
        loader = DataLoader(
            dataset,
            batch_size=min(32, len(dataset)),
            shuffle=True,
            generator=torch.Generator().manual_seed(int(seed)),
            num_workers=0,
        )
        network = cls._network(channels.shape[2]).to(target_device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=8e-4, weight_decay=2e-3)
        weights = np.sqrt(
            len(epoch_train) / (3.0 * np.maximum(train_counts.astype(float), 1.0))
        )
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=target_device),
            label_smoothing=0.05,
        )
        stop_values = torch.tensor(channels[epoch_stop], dtype=torch.float32, device=target_device)
        stop_mask = torch.tensor(day_mask[epoch_stop], dtype=torch.float32, device=target_device)
        best_score = -np.inf
        selected_epoch = 1
        stale = 0
        max_epochs = 60 if fast_mode else 300
        patience = 12 if fast_mode else 30
        for epoch_index in range(max_epochs):
            network.train()
            for batch_values, batch_mask, batch_labels in loader:
                batch_values = batch_values.to(target_device)
                batch_mask = batch_mask.to(target_device)
                batch_labels = batch_labels.to(target_device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(network(batch_values, batch_mask), batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()
            network.eval()
            with torch.no_grad():
                stop_prediction = network(stop_values, stop_mask).argmax(dim=1).cpu().numpy()
            score = f1_score(
                labels[epoch_stop], stop_prediction, average="macro", zero_division=0
            )
            if score > best_score + 1e-6:
                best_score = float(score)
                selected_epoch = epoch_index + 1
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break

        # Historical contract: reset with a fixed offset and refit every
        # fold-train subject for exactly the selected number of epochs.
        refit_seed = int(seed) + 100003
        set_all_seeds(refit_seed)
        full_dataset = TensorDataset(
            torch.tensor(channels, dtype=torch.float32),
            torch.tensor(day_mask, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )
        full_loader = DataLoader(
            full_dataset,
            batch_size=min(32, len(full_dataset)),
            shuffle=True,
            generator=torch.Generator().manual_seed(refit_seed),
            num_workers=0,
        )
        final_network = cls._network(channels.shape[2]).to(target_device)
        final_optimizer = torch.optim.AdamW(
            final_network.parameters(), lr=8e-4, weight_decay=2e-3
        )
        full_counts = np.bincount(labels, minlength=3).astype(float)
        full_weights = np.sqrt(len(labels) / (3.0 * np.maximum(full_counts, 1.0)))
        final_criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(full_weights, dtype=torch.float32, device=target_device),
            label_smoothing=0.05,
        )
        for _ in range(selected_epoch):
            final_network.train()
            for batch_values, batch_mask, batch_labels in full_loader:
                batch_values = batch_values.to(target_device)
                batch_mask = batch_mask.to(target_device)
                batch_labels = batch_labels.to(target_device)
                final_optimizer.zero_grad(set_to_none=True)
                loss = final_criterion(final_network(batch_values, batch_mask), batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(final_network.parameters(), 1.0)
                final_optimizer.step()
        state = {
            name: value.detach().cpu().clone()
            for name, value in final_network.state_dict().items()
        }
        return cls(
            int(seed),
            str(device),
            preprocessor,
            int(channels.shape[2]),
            state,
            int(selected_epoch),
            int(seed),
            train_counts.tolist(),
            stop_counts.tolist(),
        )

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "selected_epoch": int(self.epochs),
            "epoch_selector_seed": int(self.epoch_selector_seed),
            "epoch_selector_train_counts": list(self.epoch_selector_train_counts),
            "epoch_selector_stop_counts": list(self.epoch_selector_stop_counts),
            "refit_seed": int(self.seed + 100003),
            "input_channels": int(self.input_channels),
        }

    def predict_proba(
        self, values: np.ndarray, feature_names: Sequence[str]
    ) -> np.ndarray:
        import torch

        channels, day_mask = self.preprocessor.transform(values, feature_names)
        target_device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        network = self._network(self.input_channels).to(target_device)
        network.load_state_dict(self.state_dict)
        network.eval()
        probabilities: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(channels), 128):
                logits = network(
                    torch.tensor(channels[start:start + 128], device=target_device),
                    torch.tensor(day_mask[start:start + 128], device=target_device),
                )
                probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        return _normalize_probabilities(np.concatenate(probabilities, axis=0))


def _aligned_labels(bundle: FeatureBundle, labels: Any) -> np.ndarray:
    """Align labels to bundle order without ever serializing the raw index."""

    if isinstance(labels, pd.Series):
        indexed = labels.copy()
        indexed.index = indexed.index.astype(str)
        if set(indexed.index) != set(bundle.subject_ids) or indexed.index.duplicated().any():
            raise AssertionError("Label subjects do not match feature-bundle subjects.")
        values = indexed.reindex(bundle.subject_ids).to_numpy()
    elif isinstance(labels, Mapping):
        if set(map(str, labels.keys())) != set(bundle.subject_ids):
            raise AssertionError("Label mapping does not match feature-bundle subjects.")
        values = np.asarray([labels[subject] for subject in bundle.subject_ids], dtype=object)
    else:
        values = np.asarray(labels)
        if values.shape != (len(bundle.subject_ids),):
            raise AssertionError("Label array length does not match feature bundle.")
    normalized = np.asarray([label_to_id(value) for value in values], dtype=np.int64)
    counts = tuple(np.bincount(normalized, minlength=3).tolist())
    if len(normalized) != EXPECTED_SUBJECTS or counts != EXPECTED_CLASS_COUNTS:
        raise AssertionError("Nested CV requires the preregistered 141 / 85-47-9 cohort.")
    return normalized


def evaluate_probabilities(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    """Compute the fixed subject-level three-class metric set."""

    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    labels = np.asarray(labels, dtype=int)
    probabilities = _normalize_probabilities(probabilities)
    if probabilities.shape[0] != len(labels):
        raise AssertionError("Metric labels and probabilities have different lengths.")
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=np.arange(3),
        zero_division=0,
    )
    per_class: dict[str, Any] = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        binary = (labels == class_id).astype(int)
        auroc = (
            float(roc_auc_score(binary, probabilities[:, class_id]))
            if np.unique(binary).size == 2
            else None
        )
        auprc = (
            float(average_precision_score(binary, probabilities[:, class_id]))
            if binary.sum() > 0
            else None
        )
        per_class[class_name] = {
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
            "support": int(support[class_id]),
            "ovr_auroc": auroc,
            "ovr_auprc": auprc,
            "correct": int(np.sum((labels == class_id) & (predictions == class_id))),
        }
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "log_loss": float(log_loss(labels, probabilities, labels=np.arange(3))),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=np.arange(3)
        ).astype(int).tolist(),
        "per_class": per_class,
        "n_subjects": int(len(labels)),
        "class_counts": np.bincount(labels, minlength=3).astype(int).tolist(),
        "decision_rule": "raw_probability_argmax",
    }


def summarize_metric_runs(metric_runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not metric_runs:
        raise ValueError("At least one metric run is required.")
    result: dict[str, Any] = {"n_runs": len(metric_runs)}
    for metric in ("macro_f1", "balanced_accuracy", "accuracy", "log_loss"):
        values = np.asarray([float(run[metric]) for run in metric_runs], dtype=float)
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result[f"{metric}_values"] = values.tolist()
    result["per_class_f1_mean"] = {
        class_name: float(
            np.mean([run["per_class"][class_name]["f1"] for run in metric_runs])
        )
        for class_name in CLASS_NAMES
    }
    result["per_class_recall_mean"] = {
        class_name: float(
            np.mean([run["per_class"][class_name]["recall"] for run in metric_runs])
        )
        for class_name in CLASS_NAMES
    }
    return result


def _resolve_device(device: str | None) -> str:
    if device:
        return str(device)
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _fit_predict_base_seed(
    candidate: str,
    bundle: FeatureBundle,
    labels: np.ndarray,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    model_seed: int,
    fast_mode: bool,
    device: str,
) -> tuple[Any, np.ndarray]:
    train_indices = np.asarray(train_indices, dtype=int)
    predict_indices = np.asarray(predict_indices, dtype=int)
    train_labels = labels[train_indices]
    if set(np.unique(train_labels)) != {0, 1, 2}:
        raise AssertionError("Every model-training fold must contain all three classes.")
    if candidate == "event_elastic_v1":
        model = _fit_elastic(bundle.event_summary.iloc[train_indices], train_labels, model_seed)
        probabilities = model.predict_proba(bundle.event_summary.iloc[predict_indices])
    elif candidate == "event_extra_trees_v1":
        model = _fit_extra_trees(
            bundle.event_summary.iloc[train_indices], train_labels, model_seed, fast_mode
        )
        probabilities = model.predict_proba(bundle.event_summary.iloc[predict_indices])
    elif candidate == "event_tcn28_v1":
        model = DualModalityTCNBundle.fit(
            bundle.activity_sequence[train_indices],
            bundle.sleep_sequence[train_indices],
            bundle.activity_sequence_features,
            bundle.sleep_sequence_features,
            train_labels,
            model_seed,
            device,
            fast_mode,
        )
        probabilities = model.predict_proba(
            bundle.activity_sequence[predict_indices],
            bundle.sleep_sequence[predict_indices],
            bundle.activity_sequence_features,
            bundle.sleep_sequence_features,
        )
    else:
        raise KeyError(f"Not a base primary candidate: {candidate}")
    return model, _normalize_probabilities(probabilities)


def _fit_predict_coverage(
    bundle: FeatureBundle,
    labels: np.ndarray,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
) -> tuple[FittedTabularCandidate, np.ndarray]:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    train = bundle.coverage.iloc[train_indices]
    predict = bundle.coverage.iloc[predict_indices]
    preprocessor = FoldTabularPreprocessor(robust_scale=True, clip=False).fit(train)
    train_values = preprocessor.transform(train)
    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=MODEL_SEEDS[0],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train_values, labels[train_indices])
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("Coverage-only logistic regression did not converge.")
    fitted = FittedTabularCandidate(COVERAGE_VERSION, preprocessor, model)
    return fitted, fitted.predict_proba(predict)


def _fit_predict_legacy_seed(
    bundle: FeatureBundle,
    labels: np.ndarray,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    model_seed: int,
    fast_mode: bool,
    device: str,
) -> tuple[LegacyTCNBundle, np.ndarray]:
    model = LegacyTCNBundle.fit(
        bundle.legacy_values[train_indices],
        bundle.legacy_features,
        labels[train_indices],
        model_seed,
        device,
        fast_mode,
    )
    probabilities = model.predict_proba(
        bundle.legacy_values[predict_indices], bundle.legacy_features
    )
    return model, probabilities


def _checkpointed_prediction(
    checkpoint_path: Path,
    run_hash: str,
    candidate: str,
    model_seed: int,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    fit_predict: Any,
    resume: bool,
) -> np.ndarray:
    """Resume one candidate/model-seed prediction with strict identity checks."""

    train_indices = np.asarray(train_indices, dtype=np.int64)
    predict_indices = np.asarray(predict_indices, dtype=np.int64)
    identity = {
        "run_hash": run_hash,
        "candidate": candidate,
        "model_seed": int(model_seed),
        "train_index_sha256": hashlib.sha256(train_indices.tobytes()).hexdigest(),
        "predict_index_sha256": hashlib.sha256(predict_indices.tobytes()).hexdigest(),
    }
    if resume and checkpoint_path.is_file():
        payload = atomic_joblib_load(checkpoint_path)
        if payload.get("identity") != identity:
            raise AssertionError("Checkpoint identity differs from the frozen run contract.")
        probabilities = np.asarray(payload["probabilities"], dtype=np.float64)
        if _hash_array(probabilities) != payload.get("probability_sha256"):
            raise AssertionError("Checkpoint probability hash mismatch.")
        if (
            probabilities.ndim != 2
            or probabilities.shape[1] != 3
            or not np.isfinite(probabilities).all()
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10, rtol=0.0)
        ):
            raise AssertionError("Checkpoint probability contract is invalid.")
        if len(probabilities) != len(predict_indices):
            raise AssertionError("Checkpoint probability row count mismatch.")
        return probabilities
    model, probabilities = fit_predict()
    probabilities = _normalize_probabilities(probabilities).astype(np.float64)
    model_metadata = (
        model.checkpoint_metadata() if hasattr(model, "checkpoint_metadata") else None
    )
    atomic_joblib_dump(
        {
            "identity": identity,
            "probabilities": probabilities,
            "probability_sha256": _hash_array(probabilities),
            "model_metadata": model_metadata,
        },
        checkpoint_path,
    )
    return probabilities


def _ensemble_base_predictions(
    candidate: str,
    bundle: FeatureBundle,
    labels: np.ndarray,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    checkpoint_directory: Path,
    run_hash: str,
    fast_mode: bool,
    resume: bool,
    device: str,
) -> np.ndarray:
    seeds = MODEL_SEEDS[:1] if fast_mode else MODEL_SEEDS
    predictions: list[np.ndarray] = []
    for model_seed in seeds:
        path = checkpoint_directory / candidate / f"model_seed_{model_seed}.joblib"
        predictions.append(
            _checkpointed_prediction(
                path,
                run_hash,
                candidate,
                model_seed,
                train_indices,
                predict_indices,
                lambda seed=model_seed: _fit_predict_base_seed(
                    candidate,
                    bundle,
                    labels,
                    train_indices,
                    predict_indices,
                    seed,
                    fast_mode,
                    device,
                ),
                resume,
            )
        )
    return _normalize_probabilities(np.mean(predictions, axis=0))


def _ensemble_legacy_predictions(
    bundle: FeatureBundle,
    labels: np.ndarray,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    checkpoint_directory: Path,
    run_hash: str,
    fast_mode: bool,
    resume: bool,
    device: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    seeds = MODEL_SEEDS[:1] if fast_mode else MODEL_SEEDS
    predictions: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for model_seed in seeds:
        path = checkpoint_directory / LEGACY_VERSION / f"model_seed_{model_seed}.joblib"
        predictions.append(
            _checkpointed_prediction(
                path,
                run_hash,
                LEGACY_VERSION,
                model_seed,
                train_indices,
                predict_indices,
                lambda seed=model_seed: _fit_predict_legacy_seed(
                    bundle,
                    labels,
                    train_indices,
                    predict_indices,
                    seed,
                    fast_mode,
                    device,
                ),
                resume,
            )
        )
        checkpoint_payload = atomic_joblib_load(path)
        if not isinstance(checkpoint_payload.get("model_metadata"), Mapping):
            raise AssertionError("Legacy checkpoint is missing epoch-selection metadata.")
        metadata.append(dict(checkpoint_payload["model_metadata"]))
    return _normalize_probabilities(np.mean(predictions, axis=0)), metadata


def _coverage_predictions(
    bundle: FeatureBundle,
    labels: np.ndarray,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    checkpoint_directory: Path,
    run_hash: str,
    resume: bool,
) -> np.ndarray:
    path = checkpoint_directory / COVERAGE_VERSION / f"model_seed_{MODEL_SEEDS[0]}.joblib"
    return _checkpointed_prediction(
        path,
        run_hash,
        COVERAGE_VERSION,
        MODEL_SEEDS[0],
        train_indices,
        predict_indices,
        lambda: _fit_predict_coverage(bundle, labels, train_indices, predict_indices),
        resume,
    )


def _prior_probabilities(labels: np.ndarray, train_indices: np.ndarray, count: int) -> np.ndarray:
    prior = np.bincount(labels[train_indices], minlength=3).astype(float)
    prior /= prior.sum()
    return np.tile(prior, (count, 1))


def _select_inner_candidate(
    repeated_metrics: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[str, dict[str, Any]]:
    """Apply the exact preregistered inner-selection tie sequence."""

    table: list[dict[str, Any]] = []
    for name in PRIMARY_CANDIDATES:
        runs = repeated_metrics[name]
        table.append(
            {
                "candidate": name,
                "macro_f1_mean": float(np.mean([run["macro_f1"] for run in runs])),
                "balanced_accuracy_mean": float(
                    np.mean([run["balanced_accuracy"] for run in runs])
                ),
                "log_loss_mean": float(np.mean([run["log_loss"] for run in runs])),
                "complexity": PRIMARY_COMPLEXITY[name],
                "table_order": PRIMARY_COMPLEXITY[name],
            }
        )
    best_f1 = max(row["macro_f1_mean"] for row in table)
    eligible = [row for row in table if best_f1 - row["macro_f1_mean"] <= 0.01 + 1e-12]
    eligible.sort(
        key=lambda row: (
            row["complexity"],
            -row["balanced_accuracy_mean"],
            row["log_loss_mean"],
            row["table_order"],
        )
    )
    selected = eligible[0]["candidate"]
    return selected, {
        "selected_candidate": selected,
        "best_macro_f1": best_f1,
        "within_0p01_candidates": [row["candidate"] for row in eligible],
        "candidate_table": table,
        "selection_order": [
            "mean_macro_f1",
            "within_0.01_choose_simpler",
            "balanced_accuracy",
            "log_loss",
            "preregistered_table_order",
        ],
    }


def _assert_fold_counts(
    labels: np.ndarray, indices: np.ndarray, expected_dem: int, level: str
) -> list[int]:
    counts = np.bincount(labels[np.asarray(indices, dtype=int)], minlength=3).astype(int)
    if np.any(counts == 0) or int(counts[2]) != int(expected_dem):
        raise AssertionError(f"{level} fold violates preregistered class-count contract.")
    return counts.tolist()


def _feature_bundle_hash(bundle: FeatureBundle, labels: np.ndarray) -> str:
    return stable_json_hash(
        {
            "summary_names": list(bundle.event_summary.columns),
            "summary_values": _hash_array(bundle.event_summary.to_numpy(dtype=float)),
            "activity_names": bundle.activity_sequence_features,
            "activity_values": _hash_array(bundle.activity_sequence),
            "sleep_names": bundle.sleep_sequence_features,
            "sleep_values": _hash_array(bundle.sleep_sequence),
            "coverage_names": list(bundle.coverage.columns),
            "coverage_values": _hash_array(bundle.coverage.to_numpy(dtype=float)),
            "legacy_names": bundle.legacy_features,
            "legacy_values": _hash_array(bundle.legacy_values),
            "labels": hashlib.sha256(np.asarray(labels, dtype=np.int8).tobytes()).hexdigest(),
        }
    )


def _validate_locked_config(config: Mapping[str, Any]) -> None:
    """Assert the complete shipped identity used by preprocessing, models, and GO gates."""

    expected: dict[str, Any] = {
        "config_version": DESIGN_VERSION,
        "target": {
            "type": "simultaneous_cognitive_status",
            "class_names": list(CLASS_NAMES),
            "class_mapping": CLASS_TO_ID,
            "primary_metric": "subject_macro_f1",
            "prediction_index": "last_activity_day_end_timestamp_plus09",
        },
        "data": {
            "training_subjects_expected": EXPECTED_SUBJECTS,
            "class_counts_expected": {"CN": 85, "MCI": 47, "DEM": 9},
            "include_mmse": False,
            "include_absolute_dates": False,
            "event_steps": EVENT_SEQUENCE_STEPS,
            "calendar_legacy_days": LEGACY_CALENDAR_DAYS,
        },
        "cv": {
            "global_seed": 137,
            "outer_folds": N_SPLITS,
            "outer_seeds": list(OUTER_SEEDS),
            "inner_folds": N_SPLITS,
            "inner_seed_offsets": [50021, 90001],
            "model_seeds": list(MODEL_SEEDS),
        },
        "candidates": {
            "selectable": list(PRIMARY_CANDIDATES),
            "comparators": list(CONTROL_CANDIDATES),
        },
        "models": {
            "event_elastic_v1": {
                "C": 0.1,
                "l1_ratio": 0.5,
                "class_weight": "balanced",
                "max_iter": 10000,
                "solver": "saga",
                "imputation": "fold_train_median",
                "clip_quantiles": [0.005, 0.995],
                "scaling": "fold_train_median_iqr",
            },
            "event_extra_trees_v1": {
                "n_estimators": 1000,
                "max_depth": 5,
                "min_samples_leaf": 4,
                "max_features": 0.35,
                "class_weight": "balanced_subsample",
                "imputation": "fold_train_median",
                "scaling": "none",
            },
            "event_tcn28_v1": {
                "hidden": 24,
                "kernel_size": 3,
                "dilations": [1, 2, 4, 8],
                "convolutions_per_block": 2,
                "normalization": "GroupNorm",
                "activation": "GELU",
                "dropout": 0.35,
                "optimizer": "AdamW",
                "learning_rate": 0.0008,
                "weight_decay": 0.002,
                "gradient_clip": 1.0,
                "class_weight": "sqrt_balanced",
                "label_smoothing": 0.05,
                "epochs": 120,
                "early_stopping": False,
                "imputation": "fold_train_feature_median",
                "clip_quantiles": [0.005, 0.995],
                "scaling": "fold_train_feature_iqr",
                "mixed_precision": False,
            },
            "event_elastic_tcn_equal_v1": {
                "event_elastic_v1": 0.5,
                "event_tcn28_v1": 0.5,
            },
            COVERAGE_VERSION: {
                "penalty": "l2",
                "solver": "lbfgs",
                "C": 1.0,
                "class_weight": "balanced",
                "max_iter": 5000,
                "imputation": "fold_train_median",
                "scaling": "fold_train_median_iqr",
                "selection_eligible": False,
            },
            LEGACY_VERSION: {
                "role": "chronology_corrected_historical_49_feature_comparator",
                "calendar_days": 35,
                "daily_value_features": 49,
                "channels": ["normalized_value", "observed_mask", "normalized_delta"],
                "input_channels": 147,
                "calendar_anchor": "last_activity_event_start_local_date",
                "future_guard": (
                    "sleep_bedtime_end_lte_exact_activity_day_end_index_before_main_sleep_selection"
                ),
                "main_sleep_tie": [
                    "duration_desc",
                    "bedtime_start_asc",
                    "bedtime_end_asc",
                    "biological_digest_asc",
                ],
                "hidden": 24,
                "kernel_size": 3,
                "dilations": [1, 2, 4, 8],
                "convolutions_per_block": 2,
                "normalization": "GroupNorm(6)",
                "activation": "GELU",
                "dropout": 0.35,
                "optimizer": "AdamW",
                "learning_rate": 0.0008,
                "weight_decay": 0.002,
                "gradient_clip": 1.0,
                "class_weight": "sqrt_balanced",
                "label_smoothing": 0.05,
                "epoch_selection": {
                    "split": "fold_train_stratified_shuffle",
                    "test_size": 0.25,
                    "metric": "macro_f1",
                    "max_epochs": 300,
                    "patience": 30,
                    "refit": "all_fold_train_at_selected_epoch",
                    "refit_seed_offset": 100003,
                },
                "selection_eligible": False,
            },
        },
        "selection": {
            "primary": "mean_inner_subject_macro_f1",
            "final_candidate_frequency_unit": "15_outer_folds",
            "simplicity_tolerance": 0.01,
            "complexity_order": list(PRIMARY_CANDIDATES),
            "calibration": "none",
            "class_thresholds": "none",
            "adaptive_ensemble_weights": False,
        },
        "go_gate": {
            "nested_macro_f1_min": 0.378,
            "final_candidate_macro_f1_min": 0.378,
            "primary_minus_coverage_macro_f1_min": 0.03,
            "primary_minus_coverage_positive_repeats_min": 4,
            "repeat_sd_max": 0.1,
            "selection_nested_gap_abs_max": 0.03,
            "incremental_vs_elastic_gates_apply_when_final_is_not_elastic": True,
            "paired_repeat_wins_min": 4,
            "paired_outer_fold_wins_min": 10,
            "mci_f1_delta_vs_elastic_min": -0.02,
            "dem_f1_delta_vs_elastic_min": -0.02,
            "zero_recall_repeats_max_per_mci_or_dem": 1,
            "log_loss_excess_over_class_prior_max": 0.05,
            "leave_one_subject_delta_positive_fraction_min": 0.9,
            "leave_two_subject_delta_positive_fraction_min": 0.9,
            "leave_two_subject_max_pairs": 10000,
            "config_identity_must_match_checkpoints": True,
        },
    }
    actual = json_ready(config)
    if actual != expected:
        differing_sections = sorted(
            section
            for section in set(actual) | set(expected)
            if actual.get(section) != expected.get(section)
        )
        raise AssertionError(
            f"Locked config differs from implementation in sections: {differing_sections}"
        )


def run_nested_cv(
    bundle: FeatureBundle,
    labels: Any,
    output_dir: str | Path,
    subject_hash_key: str | bytes,
    subject_ids: Sequence[str] | None = None,
    fast_mode: bool = False,
    resume: bool = True,
    device: str | None = None,
    locked_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the fixed repeated nested three-fold Training-only experiment.

    Normal mode is the preregistered 5 outer seeds x 3 folds, with two fixed
    inner seeds x 3 folds and model-seed probability averaging. ``fast_mode``
    is a clearly marked, non-reportable smoke test using one outer/inner/model
    seed, 64 trees, and three TCN epochs; it can never receive a GO decision.
    """

    from sklearn.model_selection import StratifiedKFold

    started = time.time()
    if locked_config is None and not fast_mode:
        raise ValueError("A full reportable run requires the complete locked_config.")
    validate_feature_bundle(bundle, EXPECTED_SUBJECTS)
    if subject_ids is not None and list(map(str, subject_ids)) != bundle.subject_ids:
        raise AssertionError("Explicit subject_ids do not match feature-bundle order.")
    label_values = _aligned_labels(bundle, labels)
    subject_hashes = hash_subjects(bundle.subject_ids, subject_hash_key)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target_device = _resolve_device(device)
    outer_seeds = OUTER_SEEDS[:1] if fast_mode else OUTER_SEEDS
    inner_offsets = (50021,) if fast_mode else (50021, 90001)
    model_seeds = MODEL_SEEDS[:1] if fast_mode else MODEL_SEEDS
    run_config = {
        "design_version": DESIGN_VERSION,
        "representation_versions": [
            FEATURE_VERSION_SUMMARY,
            FEATURE_VERSION_SEQUENCE,
            COVERAGE_VERSION,
            LEGACY_VERSION,
        ],
        "outer_seeds": list(outer_seeds),
        "inner_seed_offsets": list(inner_offsets),
        "n_splits": N_SPLITS,
        "model_seeds": list(model_seeds),
        "primary_candidates": list(PRIMARY_CANDIDATES),
        "controls": list(CONTROL_CANDIDATES),
        "model_contract": LOCKED_MODEL_CONTRACT,
        "device": target_device,
        "mixed_precision": "disabled_locked_float32",
        "fast_mode": bool(fast_mode),
        "reportable_preregistered_run": not fast_mode,
        "smoke_overrides": (
            {
                "outer_seeds": list(outer_seeds),
                "inner_seed_offsets": list(inner_offsets),
                "model_seeds": list(model_seeds),
                "extra_trees_n_estimators": 64,
                "event_tcn_epochs": 3,
                "legacy_epoch_selection_max_epochs": 60,
                "legacy_epoch_selection_patience": 12,
            }
            if fast_mode
            else None
        ),
    }
    if locked_config is not None:
        _validate_locked_config(locked_config)
    input_hash = _feature_bundle_hash(bundle, label_values)
    code_hash = sha256_file(Path(__file__))
    config_hash = stable_json_hash(run_config)
    locked_config_hash = stable_json_hash(locked_config) if locked_config is not None else None
    run_hash = stable_json_hash(
        {
            "input_hash": input_hash,
            "code_hash": code_hash,
            "config_hash": config_hash,
            "locked_config_hash": locked_config_hash,
        }
    )
    write_json(
        output / "nested_cv_config.json",
        {
            **run_config,
            "config_hash": config_hash,
            "locked_config_hash": locked_config_hash,
        },
    )
    write_json(output / "feature_manifest_event_summary.json", bundle.summary_manifest())
    write_json(output / "feature_manifest_event_sequence.json", bundle.sequence_manifest())
    write_json(output / "coverage_audit.json", bundle.coverage_audit())

    base_candidates = PRIMARY_CANDIDATES[:3]
    probability_by_seed: dict[int, dict[str, np.ndarray]] = {}
    selected_by_seed: dict[int, np.ndarray] = {}
    coverage_by_seed: dict[int, np.ndarray] = {}
    legacy_by_seed: dict[int, np.ndarray] = {}
    prior_by_seed: dict[int, np.ndarray] = {}
    selected_name_by_seed: dict[int, np.ndarray] = {}
    fold_records: list[dict[str, Any]] = []
    inner_metric_rows: list[dict[str, Any]] = []
    outer_metric_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    inner_split_audit_rows: list[dict[str, Any]] = []
    legacy_epoch_records: list[dict[str, Any]] = []

    for outer_seed in outer_seeds:
        print(
            f"[PerformanceLab] outer seed {outer_seed} started; elapsed={time.time() - started:.1f}s",
            flush=True,
        )
        candidate_oof = {
            name: np.zeros((len(label_values), 3), dtype=float)
            for name in PRIMARY_CANDIDATES
        }
        selected_oof = np.zeros((len(label_values), 3), dtype=float)
        coverage_oof = np.zeros((len(label_values), 3), dtype=float)
        legacy_oof = np.zeros((len(label_values), 3), dtype=float)
        prior_oof = np.zeros((len(label_values), 3), dtype=float)
        selected_names = np.full(len(label_values), "", dtype=object)
        coverage_counts = np.zeros(len(label_values), dtype=int)
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=outer_seed)
        for outer_fold, (outer_train, outer_valid) in enumerate(
            splitter.split(np.zeros(len(label_values)), label_values), start=1
        ):
            outer_train = np.asarray(outer_train, dtype=int)
            outer_valid = np.asarray(outer_valid, dtype=int)
            if np.intersect1d(outer_train, outer_valid).size:
                raise AssertionError("Outer subject overlap detected.")
            train_counts = np.bincount(label_values[outer_train], minlength=3).astype(int).tolist()
            valid_counts = _assert_fold_counts(
                label_values, outer_valid, expected_dem=3, level="outer-valid"
            )
            for index in outer_valid:
                assignment_rows.append(
                    {
                        "subject_hash": subject_hashes[index],
                        "split_level": "outer_valid",
                        "outer_seed": int(outer_seed),
                        "outer_fold": int(outer_fold),
                        "inner_seed": None,
                        "inner_fold": None,
                    }
                )

            repeated_inner_metrics: dict[str, list[dict[str, Any]]] = {
                name: [] for name in PRIMARY_CANDIDATES
            }
            for offset in inner_offsets:
                inner_seed = int(outer_seed + offset)
                inner_probabilities = {
                    name: np.zeros((len(outer_train), 3), dtype=float)
                    for name in PRIMARY_CANDIDATES
                }
                inner_counts = np.zeros(len(outer_train), dtype=int)
                inner_splitter = StratifiedKFold(
                    n_splits=N_SPLITS, shuffle=True, random_state=inner_seed
                )
                for inner_fold, (inner_train_local, inner_valid_local) in enumerate(
                    inner_splitter.split(
                        np.zeros(len(outer_train)), label_values[outer_train]
                    ),
                    start=1,
                ):
                    inner_train = outer_train[np.asarray(inner_train_local, dtype=int)]
                    inner_valid = outer_train[np.asarray(inner_valid_local, dtype=int)]
                    if np.intersect1d(inner_train, inner_valid).size:
                        raise AssertionError("Inner subject overlap detected.")
                    inner_valid_counts = _assert_fold_counts(
                        label_values, inner_valid, expected_dem=2, level="inner-valid"
                    )
                    inner_train_counts = np.bincount(
                        label_values[inner_train], minlength=3
                    ).astype(int).tolist()
                    if any(count == 0 for count in inner_train_counts):
                        raise AssertionError("Inner-train fold is missing a class.")
                    inner_split_audit_rows.append(
                        {
                            "outer_seed": int(outer_seed),
                            "outer_fold": int(outer_fold),
                            "inner_seed": inner_seed,
                            "inner_fold": int(inner_fold),
                            "train_CN": inner_train_counts[0],
                            "train_MCI": inner_train_counts[1],
                            "train_DEM": inner_train_counts[2],
                            "valid_CN": inner_valid_counts[0],
                            "valid_MCI": inner_valid_counts[1],
                            "valid_DEM": inner_valid_counts[2],
                        }
                    )
                    for global_index in inner_valid:
                        assignment_rows.append(
                            {
                                "subject_hash": subject_hashes[global_index],
                                "split_level": "inner_valid",
                                "outer_seed": int(outer_seed),
                                "outer_fold": int(outer_fold),
                                "inner_seed": inner_seed,
                                "inner_fold": int(inner_fold),
                            }
                        )
                    checkpoint = (
                        output
                        / "checkpoints"
                        / f"outer_seed_{outer_seed}"
                        / f"outer_fold_{outer_fold}"
                        / f"inner_seed_{inner_seed}"
                        / f"inner_fold_{inner_fold}"
                    )
                    base_predictions = {
                        name: _ensemble_base_predictions(
                            name,
                            bundle,
                            label_values,
                            inner_train,
                            inner_valid,
                            checkpoint,
                            run_hash,
                            fast_mode,
                            resume,
                            target_device,
                        )
                        for name in base_candidates
                    }
                    base_predictions["event_elastic_tcn_equal_v1"] = _normalize_probabilities(
                        0.5 * base_predictions["event_elastic_v1"]
                        + 0.5 * base_predictions["event_tcn28_v1"]
                    )
                    for name in PRIMARY_CANDIDATES:
                        inner_probabilities[name][inner_valid_local] = base_predictions[name]
                    inner_counts[inner_valid_local] += 1
                    print(
                        "[PerformanceLab] "
                        f"outer={outer_seed}/{outer_fold} inner={inner_seed}/{inner_fold} "
                        f"complete; valid_counts={inner_valid_counts}; "
                        f"elapsed={time.time() - started:.1f}s",
                        flush=True,
                    )
                if not np.all(inner_counts == 1):
                    raise AssertionError("Inner OOF did not cover each outer-train subject once.")
                for name in PRIMARY_CANDIDATES:
                    metrics = evaluate_probabilities(
                        label_values[outer_train], inner_probabilities[name]
                    )
                    repeated_inner_metrics[name].append(metrics)
                    inner_metric_rows.append(
                        {
                            "outer_seed": int(outer_seed),
                            "outer_fold": int(outer_fold),
                            "inner_seed": inner_seed,
                            "candidate": name,
                            "macro_f1": metrics["macro_f1"],
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "accuracy": metrics["accuracy"],
                            "log_loss": metrics["log_loss"],
                            "CN_f1": metrics["per_class"]["CN"]["f1"],
                            "MCI_f1": metrics["per_class"]["MCI"]["f1"],
                            "DEM_f1": metrics["per_class"]["DEM"]["f1"],
                        }
                    )
            selected_name, inner_selection = _select_inner_candidate(repeated_inner_metrics)

            outer_checkpoint = (
                output
                / "checkpoints"
                / f"outer_seed_{outer_seed}"
                / f"outer_fold_{outer_fold}"
                / "outer_refit"
            )
            outer_base = {
                name: _ensemble_base_predictions(
                    name,
                    bundle,
                    label_values,
                    outer_train,
                    outer_valid,
                    outer_checkpoint,
                    run_hash,
                    fast_mode,
                    resume,
                    target_device,
                )
                for name in base_candidates
            }
            outer_base["event_elastic_tcn_equal_v1"] = _normalize_probabilities(
                0.5 * outer_base["event_elastic_v1"]
                + 0.5 * outer_base["event_tcn28_v1"]
            )
            coverage_probabilities = _coverage_predictions(
                bundle,
                label_values,
                outer_train,
                outer_valid,
                outer_checkpoint,
                run_hash,
                resume,
            )
            legacy_probabilities, legacy_metadata = _ensemble_legacy_predictions(
                bundle,
                label_values,
                outer_train,
                outer_valid,
                outer_checkpoint,
                run_hash,
                fast_mode,
                resume,
                target_device,
            )
            for item in legacy_metadata:
                legacy_epoch_records.append(
                    {
                        "outer_seed": int(outer_seed),
                        "outer_fold": int(outer_fold),
                        **item,
                    }
                )
            prior_probabilities = _prior_probabilities(
                label_values, outer_train, len(outer_valid)
            )
            selected_probabilities = outer_base[selected_name]
            for name in PRIMARY_CANDIDATES:
                candidate_oof[name][outer_valid] = outer_base[name]
            selected_oof[outer_valid] = selected_probabilities
            coverage_oof[outer_valid] = coverage_probabilities
            legacy_oof[outer_valid] = legacy_probabilities
            prior_oof[outer_valid] = prior_probabilities
            selected_names[outer_valid] = selected_name
            coverage_counts[outer_valid] += 1

            candidate_metrics = {
                name: evaluate_probabilities(label_values[outer_valid], outer_base[name])
                for name in PRIMARY_CANDIDATES
            }
            selected_metrics = candidate_metrics[selected_name]
            control_metrics = {
                COVERAGE_VERSION: evaluate_probabilities(
                    label_values[outer_valid], coverage_probabilities
                ),
                LEGACY_VERSION: evaluate_probabilities(
                    label_values[outer_valid], legacy_probabilities
                ),
                "class_prior_v1": evaluate_probabilities(
                    label_values[outer_valid], prior_probabilities
                ),
            }
            selected_inner_mean = next(
                row["macro_f1_mean"]
                for row in inner_selection["candidate_table"]
                if row["candidate"] == selected_name
            )
            fold_records.append(
                {
                    "outer_seed": int(outer_seed),
                    "outer_fold": int(outer_fold),
                    "train_class_counts": train_counts,
                    "valid_class_counts": valid_counts,
                    "selected_candidate": selected_name,
                    "inner_selection": inner_selection,
                    "inner_selected_macro_f1_mean": selected_inner_mean,
                    "outer_selected_macro_f1": selected_metrics["macro_f1"],
                    "inner_minus_outer_gap": float(
                        selected_inner_mean - selected_metrics["macro_f1"]
                    ),
                    "selected_metrics": selected_metrics,
                    "candidate_metrics": candidate_metrics,
                    "control_metrics": control_metrics,
                }
            )
            for name, metrics in {
                **candidate_metrics,
                "NESTED_SELECTED_PIPELINE": selected_metrics,
                **control_metrics,
            }.items():
                outer_metric_rows.append(
                    {
                        "outer_seed": int(outer_seed),
                        "outer_fold": int(outer_fold),
                        "candidate": name,
                        "selected_candidate": selected_name,
                        "train_CN": train_counts[0],
                        "train_MCI": train_counts[1],
                        "train_DEM": train_counts[2],
                        "valid_CN": valid_counts[0],
                        "valid_MCI": valid_counts[1],
                        "valid_DEM": valid_counts[2],
                        "macro_f1": metrics["macro_f1"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "accuracy": metrics["accuracy"],
                        "log_loss": metrics["log_loss"],
                        "CN_f1": metrics["per_class"]["CN"]["f1"],
                        "MCI_f1": metrics["per_class"]["MCI"]["f1"],
                        "DEM_f1": metrics["per_class"]["DEM"]["f1"],
                    }
                )
            print(
                "[PerformanceLab] "
                f"outer={outer_seed}/{outer_fold} complete; selected={selected_name}; "
                f"valid_counts={valid_counts}; elapsed={time.time() - started:.1f}s",
                flush=True,
            )
        if not np.all(coverage_counts == 1):
            raise AssertionError("Outer repeat did not cover each subject exactly once.")
        probability_by_seed[int(outer_seed)] = candidate_oof
        selected_by_seed[int(outer_seed)] = selected_oof
        coverage_by_seed[int(outer_seed)] = coverage_oof
        legacy_by_seed[int(outer_seed)] = legacy_oof
        prior_by_seed[int(outer_seed)] = prior_oof
        selected_name_by_seed[int(outer_seed)] = selected_names
        print(
            f"[PerformanceLab] outer seed {outer_seed} complete; elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    repeat_metrics: dict[str, list[dict[str, Any]]] = {
        name: [] for name in (*PRIMARY_CANDIDATES, "NESTED_SELECTED_PIPELINE", *CONTROL_CANDIDATES)
    }
    repeat_metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for outer_seed in outer_seeds:
        repeat_probabilities: dict[str, np.ndarray] = {
            **probability_by_seed[int(outer_seed)],
            "NESTED_SELECTED_PIPELINE": selected_by_seed[int(outer_seed)],
            COVERAGE_VERSION: coverage_by_seed[int(outer_seed)],
            LEGACY_VERSION: legacy_by_seed[int(outer_seed)],
            "class_prior_v1": prior_by_seed[int(outer_seed)],
        }
        for name, probabilities in repeat_probabilities.items():
            metrics = evaluate_probabilities(label_values, probabilities)
            repeat_metrics[name].append({"outer_seed": int(outer_seed), **metrics})
            repeat_metric_rows.append(
                {
                    "outer_seed": int(outer_seed),
                    "candidate": name,
                    "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "accuracy": metrics["accuracy"],
                    "log_loss": metrics["log_loss"],
                    "CN_f1": metrics["per_class"]["CN"]["f1"],
                    "MCI_f1": metrics["per_class"]["MCI"]["f1"],
                    "DEM_f1": metrics["per_class"]["DEM"]["f1"],
                }
            )
        for index, subject_hash in enumerate(subject_hashes):
            row: dict[str, Any] = {
                "subject_hash": subject_hash,
                "outer_seed": int(outer_seed),
                "true_class_id": int(label_values[index]),
                "nested_selected_candidate": str(selected_name_by_seed[int(outer_seed)][index]),
            }
            for name, probabilities in repeat_probabilities.items():
                safe_name = name.lower()
                for class_id, class_name in enumerate(CLASS_NAMES):
                    row[f"{safe_name}__p_{class_name}"] = float(probabilities[index, class_id])
            prediction_rows.append(row)

    summaries = {
        name: summarize_metric_runs(metrics) for name, metrics in repeat_metrics.items()
    }
    selection_counts = Counter(record["selected_candidate"] for record in fold_records)
    report = {
        "design_version": DESIGN_VERSION,
        "run_hash": run_hash,
        "config_hash": config_hash,
        "run_config": run_config,
        "locked_config_hash": locked_config_hash,
        "input_hash": input_hash,
        "code_hash": code_hash,
        "fast_mode": bool(fast_mode),
        "primary_interpretation": (
            "Mean and sample SD across complete subject-level outer OOF repeats."
        ),
        "subject_count": int(len(label_values)),
        "class_counts": np.bincount(label_values, minlength=3).astype(int).tolist(),
        "repeat_summaries": summaries,
        "summary": summaries["NESTED_SELECTED_PIPELINE"],
        "repeat_metrics": repeat_metrics,
        "candidate_selection_counts": {
            name: int(selection_counts.get(name, 0)) for name in PRIMARY_CANDIDATES
        },
        "folds": fold_records,
        "inner_fold_split_audit": inner_split_audit_rows,
        "legacy_epoch_selection": legacy_epoch_records,
        "inner_outer_gap_mean": float(
            np.mean([record["inner_minus_outer_gap"] for record in fold_records])
        ),
        "elapsed_seconds": float(time.time() - started),
        "privacy": "Only keyed subject hashes were written; key not persisted.",
    }
    write_json(output / "nested_cv_report.json", report)
    write_csv(output / "fold_assignments_hashed.csv", pd.DataFrame(assignment_rows))
    write_csv(output / "inner_candidate_metrics.csv", pd.DataFrame(inner_metric_rows))
    write_csv(output / "inner_fold_split_audit.csv", pd.DataFrame(inner_split_audit_rows))
    write_csv(output / "outer_fold_metrics.csv", pd.DataFrame(outer_metric_rows))
    write_csv(output / "outer_repeat_metrics.csv", pd.DataFrame(repeat_metric_rows))
    write_parquet(
        output / "candidate_outer_predictions_hashed.parquet",
        pd.DataFrame(prediction_rows),
    )
    write_json(
        output / "coverage_negative_control_metrics.json",
        {
            "version": COVERAGE_VERSION,
            "summary": summaries[COVERAGE_VERSION],
            "repeat_metrics": repeat_metrics[COVERAGE_VERSION],
            "selection_eligible": False,
        },
    )
    write_json(
        output / "legacy_mask_tcn_metrics.json",
        {
            "version": LEGACY_VERSION,
            "summary": summaries[LEGACY_VERSION],
            "repeat_metrics": repeat_metrics[LEGACY_VERSION],
            "feature_manifest": {
                "calendar_days": LEGACY_CALENDAR_DAYS,
                "value_feature_count": len(bundle.legacy_features),
                "value_feature_names": bundle.legacy_features,
                "transformed_channel_count": 3 * len(bundle.legacy_features),
                "channels": ["normalized_value", "observed_mask", "normalized_delta"],
                "calendar_anchor": "last activity event start-local-date",
                "chronology_guard": (
                    "sleep_bedtime_end <= exact last activity_day_end before main-sleep selection"
                ),
            },
            "epoch_selection_records": legacy_epoch_records,
            "selection_eligible": False,
            "warning": (
                "Chronology-corrected 49-feature replication of a historical post-hoc "
                "hypothesis; not an exact numerical reproduction of the prior 0.404."
            ),
        },
    )
    return {
        **report,
        "output_dir": str(output),
        "subject_hashes": subject_hashes,
        "labels": label_values,
        "probabilities": {
            "candidates": probability_by_seed,
            "nested_selected": selected_by_seed,
            "coverage": coverage_by_seed,
            "legacy": legacy_by_seed,
            "prior": prior_by_seed,
            "selected_names": selected_name_by_seed,
        },
    }


def _choose_final_candidate(nested_result: Mapping[str, Any]) -> dict[str, Any]:
    counts = {
        name: int(nested_result["candidate_selection_counts"].get(name, 0))
        for name in PRIMARY_CANDIDATES
    }
    maximum = max(counts.values())
    frequency_tied = [name for name in PRIMARY_CANDIDATES if counts[name] == maximum]
    summaries = nested_result["repeat_summaries"]
    if len(frequency_tied) == 1:
        selected = frequency_tied[0]
        reason = "highest_outer_fold_selection_frequency"
    else:
        best_score = max(summaries[name]["macro_f1_mean"] for name in frequency_tied)
        performance_eligible = [
            name
            for name in frequency_tied
            if best_score - summaries[name]["macro_f1_mean"] <= 0.01 + 1e-12
        ]
        minimum_complexity = min(PRIMARY_COMPLEXITY[name] for name in performance_eligible)
        complexity_tied = [
            name
            for name in performance_eligible
            if PRIMARY_COMPLEXITY[name] == minimum_complexity
        ]
        selected = min(
            complexity_tied,
            key=lambda name: (
                summaries[name]["log_loss_mean"], PRIMARY_COMPLEXITY[name]
            ),
        )
        reason = "frequency_tie_then_oof_performance_with_0p01_simplicity_then_log_loss"
    return {
        "selected_candidate": selected,
        "selection_frequency": counts,
        "frequency_tied_candidates": frequency_tied,
        "reason": reason,
        "candidate_outer_repeat_summaries": {
            name: summaries[name] for name in PRIMARY_CANDIDATES
        },
        "legacy_and_coverage_excluded": True,
    }


def _macro_f1_from_confusion(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=float)
    scores = []
    for class_id in range(3):
        true_positive = matrix[class_id, class_id]
        false_positive = matrix[:, class_id].sum() - true_positive
        false_negative = matrix[class_id, :].sum() - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator <= 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def _robust_incremental_delta_audit(
    labels: np.ndarray,
    final_probabilities: Mapping[int, np.ndarray],
    elastic_probabilities: Mapping[int, np.ndarray],
    max_pairs: int,
    subject_hashes: Sequence[str],
) -> dict[str, Any]:
    """Leave-one/two subject paired Macro-F1 delta sensitivity audit."""

    leave_one: list[float] = []
    cached: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for outer_seed in sorted(final_probabilities):
        final_prediction = final_probabilities[outer_seed].argmax(axis=1)
        elastic_prediction = elastic_probabilities[outer_seed].argmax(axis=1)
        final_matrix = np.zeros((3, 3), dtype=int)
        elastic_matrix = np.zeros((3, 3), dtype=int)
        for truth, final_value, elastic_value in zip(
            labels, final_prediction, elastic_prediction
        ):
            final_matrix[truth, final_value] += 1
            elastic_matrix[truth, elastic_value] += 1
        cached[outer_seed] = (
            final_prediction,
            elastic_prediction,
            final_matrix,
            elastic_matrix,
        )
    for index in range(len(labels)):
        repeat_deltas: list[float] = []
        for outer_seed in sorted(cached):
            final_prediction, elastic_prediction, final_matrix, elastic_matrix = cached[outer_seed]
            final_reduced = final_matrix.copy()
            elastic_reduced = elastic_matrix.copy()
            final_reduced[labels[index], final_prediction[index]] -= 1
            elastic_reduced[labels[index], elastic_prediction[index]] -= 1
            repeat_deltas.append(
                _macro_f1_from_confusion(final_reduced)
                - _macro_f1_from_confusion(elastic_reduced)
            )
        leave_one.append(float(np.mean(repeat_deltas)))
    pair_entries = [
        (left, right)
        for left in range(len(labels) - 1)
        for right in range(left + 1, len(labels))
    ]
    if len(pair_entries) > max_pairs:
        pair_entries.sort(
            key=lambda pair: tuple(sorted((subject_hashes[pair[0]], subject_hashes[pair[1]])))
        )
        pair_entries = pair_entries[:max_pairs]
    leave_two: list[float] = []
    for left, right in pair_entries:
        repeat_deltas = []
        for outer_seed in sorted(cached):
            final_prediction, elastic_prediction, final_matrix, elastic_matrix = cached[outer_seed]
            final_reduced = final_matrix.copy()
            elastic_reduced = elastic_matrix.copy()
            for index in (left, right):
                final_reduced[labels[index], final_prediction[index]] -= 1
                elastic_reduced[labels[index], elastic_prediction[index]] -= 1
            repeat_deltas.append(
                _macro_f1_from_confusion(final_reduced)
                - _macro_f1_from_confusion(elastic_reduced)
            )
        leave_two.append(float(np.mean(repeat_deltas)))
    return {
        "leave_one_evaluations": len(leave_one),
        "leave_one_positive_fraction": float(np.mean(np.asarray(leave_one) > 0)),
        "leave_one_min_delta": float(np.min(leave_one)),
        "leave_two_evaluations": len(leave_two),
        "leave_two_positive_fraction": float(np.mean(np.asarray(leave_two) > 0)),
        "leave_two_min_delta": float(np.min(leave_two)),
        "aggregation": "mean repeat-specific final-minus-elastic delta per removed subject set",
        "pair_order_if_capped": "lexicographic keyed-subject-hash order",
    }


def select_and_assess(
    nested_result: Mapping[str, Any],
    locked_config: Mapping[str, Any],
    fast_mode: bool = False,
) -> dict[str, Any]:
    """Choose the deployment candidate and apply every preregistered STOP/GO gate."""

    _validate_locked_config(locked_config)
    if locked_config["go_gate"].get(
        "incremental_vs_elastic_gates_apply_when_final_is_not_elastic"
    ) is not True:
        raise AssertionError("Locked config must explicitly enable conditional incremental gates.")
    selection = _choose_final_candidate(nested_result)
    selected = selection["selected_candidate"]
    gates = locked_config["go_gate"]
    summaries = nested_result["repeat_summaries"]
    final_summary = summaries[selected]
    nested_summary = summaries["NESTED_SELECTED_PIPELINE"]
    coverage_summary = summaries[COVERAGE_VERSION]
    prior_summary = summaries["class_prior_v1"]
    elastic_summary = summaries["event_elastic_v1"]
    probabilities = nested_result["probabilities"]
    outer_seeds = sorted(probabilities["candidates"])
    final_probabilities = {
        seed: probabilities["candidates"][seed][selected] for seed in outer_seeds
    }
    elastic_probabilities = {
        seed: probabilities["candidates"][seed]["event_elastic_v1"] for seed in outer_seeds
    }
    coverage_repeat_delta = [
        nested_result["repeat_metrics"][selected][index]["macro_f1"]
        - nested_result["repeat_metrics"][COVERAGE_VERSION][index]["macro_f1"]
        for index in range(len(outer_seeds))
    ]
    paired_repeat_delta = [
        nested_result["repeat_metrics"][selected][index]["macro_f1"]
        - nested_result["repeat_metrics"]["event_elastic_v1"][index]["macro_f1"]
        for index in range(len(outer_seeds))
    ]
    paired_fold_delta = [
        record["candidate_metrics"][selected]["macro_f1"]
        - record["candidate_metrics"]["event_elastic_v1"]["macro_f1"]
        for record in nested_result["folds"]
    ]
    zero_recall_counts = {
        class_name: int(
            sum(
                run["per_class"][class_name]["recall"] == 0
                for run in nested_result["repeat_metrics"][selected]
            )
        )
        for class_name in ("MCI", "DEM")
    }
    selection_nested_gap = float(
        final_summary["macro_f1_mean"] - nested_summary["macro_f1_mean"]
    )
    log_loss_excess = float(
        final_summary["log_loss_mean"] - prior_summary["log_loss_mean"]
    )
    incremental = selected != "event_elastic_v1"
    if incremental:
        robustness = _robust_incremental_delta_audit(
            np.asarray(nested_result["labels"], dtype=int),
            final_probabilities,
            elastic_probabilities,
            int(gates["leave_two_subject_max_pairs"]),
            nested_result["subject_hashes"],
        )
    else:
        robustness = {
            "status": "not_applicable_pass",
            "reason": "Final candidate is the preregistered elastic baseline.",
            "leave_one_positive_fraction": None,
            "leave_two_positive_fraction": None,
        }

    gate_results: dict[str, dict[str, Any]] = {}

    def record_gate(name: str, passed: bool, value: Any, requirement: str) -> None:
        gate_results[name] = {
            "passed": bool(passed),
            "value": json_ready(value),
            "requirement": requirement,
        }

    record_gate(
        "reportable_full_run",
        not fast_mode and not nested_result.get("fast_mode", False) and len(outer_seeds) == 5,
        {"fast_mode": bool(fast_mode), "outer_repeats": len(outer_seeds)},
        "full 5-repeat preregistered run",
    )
    record_gate(
        "nested_macro_f1",
        nested_summary["macro_f1_mean"] >= gates["nested_macro_f1_min"],
        nested_summary["macro_f1_mean"],
        f">= {gates['nested_macro_f1_min']}",
    )
    record_gate(
        "final_candidate_macro_f1",
        final_summary["macro_f1_mean"] >= gates["final_candidate_macro_f1_min"],
        final_summary["macro_f1_mean"],
        f">= {gates['final_candidate_macro_f1_min']}",
    )
    coverage_gap = final_summary["macro_f1_mean"] - coverage_summary["macro_f1_mean"]
    record_gate(
        "primary_minus_coverage",
        coverage_gap >= gates["primary_minus_coverage_macro_f1_min"],
        coverage_gap,
        f">= {gates['primary_minus_coverage_macro_f1_min']}",
    )
    record_gate(
        "primary_beats_coverage_repeats",
        int(np.sum(np.asarray(coverage_repeat_delta) > 0))
        >= gates["primary_minus_coverage_positive_repeats_min"],
        {
            "positive_repeats": int(np.sum(np.asarray(coverage_repeat_delta) > 0)),
            "deltas": coverage_repeat_delta,
        },
        f">= {gates['primary_minus_coverage_positive_repeats_min']} positive repeats",
    )
    record_gate(
        "repeat_sd",
        max(nested_summary["macro_f1_sd"], final_summary["macro_f1_sd"])
        < gates["repeat_sd_max"],
        {
            "nested_selected_sd": nested_summary["macro_f1_sd"],
            "final_candidate_sd": final_summary["macro_f1_sd"],
        },
        f"both < {gates['repeat_sd_max']}",
    )
    identity_matches = nested_result.get("locked_config_hash") == stable_json_hash(locked_config)
    record_gate(
        "config_checkpoint_identity",
        identity_matches and gates.get("config_identity_must_match_checkpoints") is True,
        {
            "identity_matches": identity_matches,
            "identity_required": gates.get("config_identity_must_match_checkpoints"),
        },
        "locked config hash must match checkpointed nested run",
    )
    record_gate(
        "selection_nested_gap",
        abs(selection_nested_gap) <= gates["selection_nested_gap_abs_max"],
        selection_nested_gap,
        f"absolute value <= {gates['selection_nested_gap_abs_max']}",
    )
    record_gate(
        "zero_recall_mci_dem",
        all(
            count <= gates["zero_recall_repeats_max_per_mci_or_dem"]
            for count in zero_recall_counts.values()
        ),
        zero_recall_counts,
        f"each <= {gates['zero_recall_repeats_max_per_mci_or_dem']} repeats",
    )
    record_gate(
        "log_loss_vs_class_prior",
        log_loss_excess <= gates["log_loss_excess_over_class_prior_max"],
        log_loss_excess,
        f"<= {gates['log_loss_excess_over_class_prior_max']}",
    )
    if incremental:
        record_gate(
            "paired_repeat_wins_vs_elastic",
            int(np.sum(np.asarray(paired_repeat_delta) > 0))
            >= gates["paired_repeat_wins_min"],
            {
                "positive_repeats": int(np.sum(np.asarray(paired_repeat_delta) > 0)),
                "deltas": paired_repeat_delta,
            },
            f">= {gates['paired_repeat_wins_min']} positive repeats",
        )
        record_gate(
            "paired_outer_fold_wins_vs_elastic",
            int(np.sum(np.asarray(paired_fold_delta) > 0))
            >= gates["paired_outer_fold_wins_min"],
            {
                "positive_folds": int(np.sum(np.asarray(paired_fold_delta) > 0)),
                "deltas": paired_fold_delta,
            },
            f">= {gates['paired_outer_fold_wins_min']} positive folds",
        )
        for class_name, config_name in (
            ("MCI", "mci_f1_delta_vs_elastic_min"),
            ("DEM", "dem_f1_delta_vs_elastic_min"),
        ):
            delta = (
                final_summary["per_class_f1_mean"][class_name]
                - elastic_summary["per_class_f1_mean"][class_name]
            )
            record_gate(
                f"{class_name.lower()}_f1_vs_elastic",
                delta >= gates[config_name],
                delta,
                f">= {gates[config_name]}",
            )
        record_gate(
            "leave_one_subject_robustness",
            robustness["leave_one_positive_fraction"]
            >= gates["leave_one_subject_delta_positive_fraction_min"],
            robustness["leave_one_positive_fraction"],
            f">= {gates['leave_one_subject_delta_positive_fraction_min']}",
        )
        record_gate(
            "leave_two_subject_robustness",
            robustness["leave_two_positive_fraction"]
            >= gates["leave_two_subject_delta_positive_fraction_min"],
            robustness["leave_two_positive_fraction"],
            f">= {gates['leave_two_subject_delta_positive_fraction_min']}",
        )
    else:
        for name in (
            "paired_repeat_wins_vs_elastic",
            "paired_outer_fold_wins_vs_elastic",
            "mci_f1_vs_elastic",
            "dem_f1_vs_elastic",
            "leave_one_subject_robustness",
            "leave_two_subject_robustness",
        ):
            gate_results[name] = {
                "passed": True,
                "status": "not_applicable_pass",
                "value": None,
                "requirement": "incremental-complexity gate; final candidate is elastic",
            }

    failed = [name for name, result in gate_results.items() if not result["passed"]]
    stop_go = {
        "decision": "GO" if not failed else "NO-GO",
        "selected_candidate": selected,
        "failed_gates": failed,
        "gate_results": gate_results,
        "coverage_repeat_deltas": coverage_repeat_delta,
        "paired_repeat_deltas_vs_elastic": paired_repeat_delta,
        "paired_fold_deltas_vs_elastic": paired_fold_delta,
        "robustness_audit": robustness,
        "selection_nested_gap": selection_nested_gap,
        "warning": (
            "GO means a reproducible Training-cohort candidate, not confirmed external generalization."
        ),
    }
    selection.update(
        {
            "final_candidate_repeat_summary": final_summary,
            "nested_selected_pipeline_summary": nested_summary,
            "selection_nested_gap": selection_nested_gap,
        }
    )
    return {"selection": selection, "stop_go": stop_go}


@dataclass
class FrozenPrimaryEnsemble:
    """PII-free selected candidate with two full-Training stochastic refits."""

    selected_candidate: str
    component_models: dict[str, list[Any]]
    model_seeds: list[int]
    summary_features: list[str]
    activity_features: list[str]
    sleep_features: list[str]

    def predict_proba(self, bundle: FeatureBundle) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        validate_feature_bundle(bundle)
        if list(bundle.event_summary.columns) != self.summary_features:
            raise AssertionError("Frozen summary feature contract changed.")
        if bundle.activity_sequence_features != self.activity_features:
            raise AssertionError("Frozen activity sequence feature contract changed.")
        if bundle.sleep_sequence_features != self.sleep_features:
            raise AssertionError("Frozen sleep sequence feature contract changed.")
        component_probabilities: dict[str, np.ndarray] = {}
        for name, models in self.component_models.items():
            refit_predictions: list[np.ndarray] = []
            for model in models:
                if name in {"event_elastic_v1", "event_extra_trees_v1"}:
                    refit_predictions.append(model.predict_proba(bundle.event_summary))
                elif name == "event_tcn28_v1":
                    refit_predictions.append(
                        model.predict_proba(
                            bundle.activity_sequence,
                            bundle.sleep_sequence,
                            bundle.activity_sequence_features,
                            bundle.sleep_sequence_features,
                        )
                    )
                else:
                    raise KeyError(f"Unknown frozen component: {name}")
            component_probabilities[name] = _normalize_probabilities(
                np.mean(refit_predictions, axis=0)
            )
        if self.selected_candidate == "event_elastic_tcn_equal_v1":
            selected = _normalize_probabilities(
                0.5 * component_probabilities["event_elastic_v1"]
                + 0.5 * component_probabilities["event_tcn28_v1"]
            )
        else:
            selected = component_probabilities[self.selected_candidate]
        return selected, component_probabilities


def _fit_full_primary_ensemble(
    bundle: FeatureBundle,
    labels: np.ndarray,
    selected_candidate: str,
    device: str,
) -> FrozenPrimaryEnsemble:
    all_indices = np.arange(len(labels), dtype=int)
    components = (
        ("event_elastic_v1", "event_tcn28_v1")
        if selected_candidate == "event_elastic_tcn_equal_v1"
        else (selected_candidate,)
    )
    fitted: dict[str, list[Any]] = {name: [] for name in components}
    for name in components:
        for model_seed in MODEL_SEEDS:
            model, _ = _fit_predict_base_seed(
                name,
                bundle,
                labels,
                all_indices,
                all_indices[:1],
                model_seed,
                False,
                device,
            )
            fitted[name].append(model)
    return FrozenPrimaryEnsemble(
        selected_candidate=selected_candidate,
        component_models=fitted,
        model_seeds=list(MODEL_SEEDS),
        summary_features=list(bundle.event_summary.columns),
        activity_features=list(bundle.activity_sequence_features),
        sleep_features=list(bundle.sleep_sequence_features),
    )


def _extract_preprocessors(ensemble: FrozenPrimaryEnsemble) -> dict[str, Any]:
    extracted: dict[str, list[Any]] = {}
    for name, models in ensemble.component_models.items():
        extracted[name] = []
        for model in models:
            if isinstance(model, FittedTabularCandidate):
                extracted[name].append(model.preprocessor)
            elif isinstance(model, DualModalityTCNBundle):
                extracted[name].append(
                    {
                        "activity": model.activity_preprocessor,
                        "sleep": model.sleep_preprocessor,
                    }
                )
            else:
                raise TypeError("Unexpected frozen model type while extracting preprocessors.")
    return extracted


def _preprocessing_manifest(ensemble: FrozenPrimaryEnsemble) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for name, models in ensemble.component_models.items():
        manifest[name] = []
        for model in models:
            if isinstance(model, FittedTabularCandidate):
                manifest[name].append(model.preprocessor.manifest())
            else:
                manifest[name].append(model.manifest())
    return manifest


def fit_frozen_training_bundle(
    bundle: FeatureBundle,
    labels: Any,
    selected_candidate: str,
    output_dir: str | Path,
    locked_config: Mapping[str, Any],
    nested_result: Mapping[str, Any],
    decision: Mapping[str, Any],
    device: str | None = None,
) -> dict[str, Any]:
    """Refit a cryptographically bound, in-memory GO result on all Training data."""

    _validate_locked_config(locked_config)
    if selected_candidate not in PRIMARY_CANDIDATES:
        raise ValueError("Only a preregistered primary candidate can be frozen.")
    validate_feature_bundle(bundle, EXPECTED_SUBJECTS)
    label_values = _aligned_labels(bundle, labels)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    nested_evidence = nested_result
    selection_evidence = decision["selection"]
    stop_go_evidence = decision["stop_go"]
    if nested_evidence.get("fast_mode"):
        raise AssertionError("A smoke run can never create a frozen deployable model.")
    if nested_evidence.get("subject_count") != EXPECTED_SUBJECTS:
        raise AssertionError("Nested evidence has the wrong Training subject count.")
    if nested_evidence.get("class_counts") != list(EXPECTED_CLASS_COUNTS):
        raise AssertionError("Nested evidence has the wrong Training class counts.")
    if len(nested_evidence.get("folds", ())) != len(OUTER_SEEDS) * N_SPLITS:
        raise AssertionError("Frozen refit requires all 15 completed outer folds.")
    current_locked_hash = stable_json_hash(locked_config)
    current_input_hash = _feature_bundle_hash(bundle, label_values)
    current_code_hash = sha256_file(Path(__file__))
    if nested_evidence.get("locked_config_hash") != current_locked_hash:
        raise AssertionError("Nested evidence and frozen locked config hashes differ.")
    if nested_evidence.get("input_hash") != current_input_hash:
        raise AssertionError("Nested evidence and frozen Training input hashes differ.")
    if nested_evidence.get("code_hash") != current_code_hash:
        raise AssertionError("Nested evidence was produced by different core code.")
    nested_run_config = nested_evidence.get("run_config")
    nested_config_hash = nested_evidence.get("config_hash")
    if not isinstance(nested_run_config, Mapping):
        raise AssertionError("Nested evidence is missing its exact run configuration.")
    if stable_json_hash(nested_run_config) != nested_config_hash:
        raise AssertionError("Nested run configuration hash mismatch.")
    if stop_go_evidence.get("decision") != "GO":
        raise AssertionError("Frozen refit is forbidden unless the preregistered decision is GO.")
    if (
        selection_evidence.get("selected_candidate") != selected_candidate
        or stop_go_evidence.get("selected_candidate") != selected_candidate
    ):
        raise AssertionError("Frozen candidate differs from the completed selection evidence.")
    nested_run_hash = nested_evidence.get("run_hash")
    if not isinstance(nested_run_hash, str) or len(nested_run_hash) != 64:
        raise AssertionError("Nested run hash is missing or malformed.")
    expected_nested_run_hash = stable_json_hash(
        {
            "input_hash": current_input_hash,
            "code_hash": current_code_hash,
            "config_hash": nested_config_hash,
            "locked_config_hash": current_locked_hash,
        }
    )
    if nested_run_hash != expected_nested_run_hash:
        raise AssertionError("Nested run hash is not bound to this input/code/config identity.")
    recomputed_decision = select_and_assess(
        nested_result, locked_config=locked_config, fast_mode=False
    )
    if stable_json_hash(recomputed_decision) != stable_json_hash(decision):
        raise AssertionError("Supplied STOP/GO evidence differs from deterministic recomputation.")
    target_device = _resolve_device(device)
    ensemble = _fit_full_primary_ensemble(
        bundle, label_values, selected_candidate, target_device
    )
    model_path = output / "final_model_bundle.joblib"
    preprocessor_path = output / "selected_preprocessor.joblib"
    atomic_joblib_dump(ensemble, model_path)
    atomic_joblib_dump(_extract_preprocessors(ensemble), preprocessor_path)
    model_hash = sha256_file(model_path)
    preprocessor_hash = sha256_file(preprocessor_path)
    code_hash = current_code_hash
    locked_config_hash = current_locked_hash
    input_hash = current_input_hash
    frozen_config: dict[str, Any] = {
        "design_version": DESIGN_VERSION,
        "selected_candidate": selected_candidate,
        "class_mapping": CLASS_TO_ID,
        "prediction_index": "last valid activity_day_end timestamp (+09:00)",
        "event_lookback": "last up to 28 observed events per modality",
        "model_seeds": list(MODEL_SEEDS),
        "outer_split_seeds": list(OUTER_SEEDS),
        "probability_rule": (
            "0.5 elastic + 0.5 event TCN"
            if selected_candidate == "event_elastic_tcn_equal_v1"
            else "mean of two stochastic refits; raw argmax"
        ),
        "code_sha256": code_hash,
        "locked_config_sha256": locked_config_hash,
        "nested_run_sha256": nested_run_hash,
        "selection_evidence_sha256": stable_json_hash(selection_evidence),
        "stop_go_evidence_sha256": stable_json_hash(stop_go_evidence),
        "training_input_feature_sha256": input_hash,
        "model_artifact": model_path.name,
        "model_artifact_sha256": model_hash,
        "preprocessor_artifact": preprocessor_path.name,
        "preprocessor_artifact_sha256": preprocessor_hash,
        "summary_feature_manifest": bundle.summary_manifest(),
        "sequence_feature_manifest": bundle.sequence_manifest(),
        "preprocessing_manifest": _preprocessing_manifest(ensemble),
        "training_subject_count": int(len(label_values)),
        "training_class_counts": {
            CLASS_NAMES[index]: int(count)
            for index, count in enumerate(np.bincount(label_values, minlength=3))
        },
        "validation_seen": False,
        "benchmark_role": "historically reused official benchmark; not model selection",
    }
    frozen_config["frozen_config_sha256"] = stable_json_hash(frozen_config)
    training_report = {
        "selected_candidate": selected_candidate,
        "full_training_refit_complete": True,
        "model_seeds": list(MODEL_SEEDS),
        "model_sha256": model_hash,
        "preprocessor_sha256": preprocessor_hash,
        "selection_oof_warning": (
            "Candidate-specific outer OOF was used to finalize the rule and is optimistic; "
            "repeated nested CV remains the primary estimate."
        ),
    }
    write_json(output / "FINAL_TRAINING_REFIT.json", training_report)
    return {
        "ensemble": ensemble,
        "frozen_config": frozen_config,
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
        "training_refit_report": training_report,
    }


def _verify_frozen_mapping(frozen: Mapping[str, Any]) -> FrozenPrimaryEnsemble:
    """Verify code, configuration, and in-memory ensemble contracts together."""

    if "ensemble" not in frozen or "frozen_config" not in frozen:
        raise TypeError("Frozen inference requires both ensemble and frozen_config evidence.")
    ensemble = frozen["ensemble"]
    config = frozen["frozen_config"]
    if not isinstance(ensemble, FrozenPrimaryEnsemble) or not isinstance(config, Mapping):
        raise TypeError("Malformed verified frozen bundle.")
    if config.get("code_sha256") != sha256_file(Path(__file__)):
        raise AssertionError("Current core code differs from the frozen code snapshot.")
    expected = config.get("frozen_config_sha256")
    payload = dict(config)
    payload.pop("frozen_config_sha256", None)
    if expected != stable_json_hash(payload):
        raise AssertionError("Frozen configuration identity mismatch.")
    if config.get("design_version") != DESIGN_VERSION:
        raise AssertionError("Frozen design version mismatch.")
    if config.get("class_mapping") != CLASS_TO_ID:
        raise AssertionError("Frozen class mapping mismatch.")
    if config.get("selected_candidate") != ensemble.selected_candidate:
        raise AssertionError("Frozen candidate and serialized ensemble differ.")
    if config.get("model_seeds") != ensemble.model_seeds or ensemble.model_seeds != list(
        MODEL_SEEDS
    ):
        raise AssertionError("Frozen model-seed contract mismatch.")
    summary_manifest = config.get("summary_feature_manifest", {})
    sequence_manifest = config.get("sequence_feature_manifest", {})
    if summary_manifest.get("feature_names") != ensemble.summary_features:
        raise AssertionError("Frozen summary-feature manifest mismatch.")
    if sequence_manifest.get("activity_feature_names") != ensemble.activity_features:
        raise AssertionError("Frozen activity-feature manifest mismatch.")
    if sequence_manifest.get("sleep_feature_names") != ensemble.sleep_features:
        raise AssertionError("Frozen sleep-feature manifest mismatch.")
    expected_components = (
        {"event_elastic_v1", "event_tcn28_v1"}
        if ensemble.selected_candidate == "event_elastic_tcn_equal_v1"
        else {ensemble.selected_candidate}
    )
    if set(ensemble.component_models) != expected_components:
        raise AssertionError("Frozen ensemble component contract mismatch.")
    if any(len(models) != len(MODEL_SEEDS) for models in ensemble.component_models.values()):
        raise AssertionError("Frozen ensemble must contain both preregistered refits.")
    return ensemble


def predict_frozen_training_bundle(
    frozen: Mapping[str, Any],
    bundle: FeatureBundle,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Label-free inference after mandatory frozen-identity verification."""

    if not isinstance(frozen, Mapping):
        raise TypeError("Raw ensembles are not accepted; supply verified frozen evidence.")
    return _verify_frozen_mapping(frozen).predict_proba(bundle)


def load_frozen_training_bundle(
    model_path: str | Path, frozen_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Load a model only after mandatory artifact/config/code verification."""

    path = Path(model_path)
    if not isinstance(frozen_config, Mapping):
        raise TypeError("frozen_config is mandatory when loading a serialized model.")
    if path.name != frozen_config.get("model_artifact"):
        raise AssertionError("Frozen model file name differs from its configuration.")
    if sha256_file(path) != frozen_config.get("model_artifact_sha256"):
        raise AssertionError("Frozen model artifact hash mismatch.")
    preprocessor_name = frozen_config.get("preprocessor_artifact")
    preprocessor_hash = frozen_config.get("preprocessor_artifact_sha256")
    if not preprocessor_name or not preprocessor_hash:
        raise AssertionError("Frozen preprocessor identity is missing.")
    preprocessor_path = path.parent / str(preprocessor_name)
    if not preprocessor_path.is_file() or sha256_file(preprocessor_path) != preprocessor_hash:
        raise AssertionError("Frozen preprocessor artifact hash mismatch.")
    ensemble = atomic_joblib_load(path)
    verified = {"ensemble": ensemble, "frozen_config": dict(frozen_config)}
    _verify_frozen_mapping(verified)
    return verified


def audit_output_privacy(
    output_dir: str | Path,
    raw_subject_ids: Sequence[str],
    forbidden_secret: str | bytes,
) -> dict[str, Any]:
    """Scan every artifact byte stream for raw IDs or the HMAC secret.

    The report exposes only counts and safe relative file names, never the
    matched identifier or secret.
    """

    root = Path(output_dir)
    raw_needles = [str(value).encode("utf-8") for value in raw_subject_ids]
    secret = (
        forbidden_secret.encode("utf-8")
        if isinstance(forbidden_secret, str)
        else bytes(forbidden_secret)
    )
    needles = [needle for needle in (*raw_needles, secret) if needle]
    offending_count = 0
    stale_temporary_count = 0
    files_scanned = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files_scanned += 1
        data = path.read_bytes()
        if any(needle in data for needle in needles):
            offending_count += 1
        if path.name.endswith(".tmp"):
            stale_temporary_count += 1
    return {
        "passed": offending_count == 0 and stale_temporary_count == 0,
        "files_scanned": files_scanned,
        "offending_file_count": offending_count,
        "stale_temporary_file_count": stale_temporary_count,
        "offending_file_names_reported": False,
        "raw_identifier_values_reported": False,
        "secret_value_reported": False,
    }


def hash_public_artifacts(output_dir: str | Path) -> dict[str, str]:
    """Hash completed public artifacts, excluding checkpoints and circular marker."""

    root = Path(output_dir)
    hashes: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if (
            "checkpoints" in relative.parts
            or path.name.endswith(".tmp")
            or path.name == "TRAINING_COMPLETE.json"
        ):
            continue
        hashes[str(relative)] = sha256_file(path)
    return hashes


def build_final_training_report(
    nested_result: Mapping[str, Any],
    decision: Mapping[str, Any],
    environment: Mapping[str, Any],
    run_id: str,
    frozen: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the concise, privacy-safe final Training report payload."""

    selected = decision["selection"]["selected_candidate"]
    nested_repeats = nested_result["repeat_metrics"]["NESTED_SELECTED_PIPELINE"]
    confusion_sum = np.sum(
        [np.asarray(run["confusion_matrix"], dtype=int) for run in nested_repeats], axis=0
    )
    error_patterns = sorted(
        (
            {
                "true_class": CLASS_NAMES[truth],
                "predicted_class": CLASS_NAMES[predicted],
                "count_across_repeats": int(confusion_sum[truth, predicted]),
            }
            for truth in range(3)
            for predicted in range(3)
            if truth != predicted
        ),
        key=lambda item: -item["count_across_repeats"],
    )
    return {
        "run_id": str(run_id),
        "design_version": DESIGN_VERSION,
        "target": {
            "task": "simultaneous CN/MCI/DEM cognitive-status classification",
            "class_mapping": CLASS_TO_ID,
            "prediction_index": "last valid activity_day_end timestamp (+09:00)",
            "horizon": "contemporaneous state; not future conversion prognosis",
            "cohort": "Data/1.Training, 141 subjects (85/47/9)",
        },
        "leakage_audit": {
            "subject_grouping": "one subject row/sequence; fold-disjoint",
            "mmse": "entirely excluded and never opened",
            "labels": "three copies normalized and required identical",
            "primary_excludes": [
                "raw IDs",
                "diagnosis/MMSE",
                "absolute dates",
                "observed counts",
                "calendar gaps",
                "masks/deltas",
                "non-wear ratios",
            ],
            "sleep_chronology": "bedtime_end <= subject activity_day_end index before main-sleep selection",
            "preprocessing": "all learned statistics fit inside each train fold",
        },
        "primary_nested_cv": nested_result["summary"],
        "primary_repeat_details": nested_repeats,
        "confusion_matrix_sum_across_five_repeats": confusion_sum.tolist(),
        "major_error_patterns": error_patterns,
        "selected_candidate": selected,
        "selected_candidate_outer_diagnostic": nested_result["repeat_summaries"][selected],
        "coverage_negative_control": nested_result["repeat_summaries"][COVERAGE_VERSION],
        "legacy_comparator": nested_result["repeat_summaries"][LEGACY_VERSION],
        "selection": decision["selection"],
        "stop_go": decision["stop_go"],
        "environment": json_ready(environment),
        "frozen": None if frozen is None else frozen["frozen_config"],
        "artifact_index": {
            "nested_report": "nested_cv_report.json",
            "inner_metrics": "inner_candidate_metrics.csv",
            "inner_split_counts": "inner_fold_split_audit.csv",
            "outer_fold_metrics": "outer_fold_metrics.csv",
            "outer_repeat_metrics": "outer_repeat_metrics.csv",
            "hashed_predictions": "candidate_outer_predictions_hashed.parquet",
            "coverage_control": "coverage_negative_control_metrics.json",
            "legacy_comparator": "legacy_mask_tcn_metrics.json",
            "model": "final_model_bundle.joblib" if frozen is not None else None,
            "preprocessor": "selected_preprocessor.joblib" if frozen is not None else None,
        },
        "limitations": [
            "Only nine Training subjects have DEM; no claim of statistical certainty is warranted.",
            "The official benchmark has been historically reused and is not an untouched holdout.",
            "An independent external cohort with a different acquisition protocol is required.",
            "Feature effects are associations, not clinical causal explanations.",
            (
                "Class-labelled aggregate EDA on all 141 Training subjects informed the "
                "preregistered feature-family design; nested CV is conditional internal "
                "evidence, not a fully human-unseen experiment."
            ),
            (
                "Exact numeric preprocessing arrays live in the hashed model/preprocessor "
                "joblib artifacts; JSON records their method, feature order, and hashes."
            ),
        ],
    }
