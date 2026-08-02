"""Fail-closed source access for the Google YDF ROC-AUC experiment.

The MMSE feature source is opened with a source-level allow-list.  Diagnosis
and administrative fields therefore never enter the feature-building process.
Labels are opened separately from the two Activity/Sleep label copies and are
cross-checked before use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

PERSON_KEY = "SAMPLE_EMAIL"

MMSE_DOMAINS: Mapping[str, tuple[str, ...]] = {
    "orient_time": ("Q01", "Q02", "Q03", "Q04", "Q05"),
    "orient_place": ("Q06", "Q07", "Q08", "Q09", "Q10"),
    "registration": ("Q11_1", "Q11_2", "Q11_3"),
    "attention": ("Q12_1", "Q12_2", "Q12_3", "Q12_4", "Q12_5"),
    "recall": ("Q13_1", "Q13_2", "Q13_3"),
    "language": (
        "Q14_1",
        "Q14_2",
        "Q15",
        "Q16_1",
        "Q16_2",
        "Q16_3",
        "Q17",
        "Q18",
        "Q19",
    ),
}
MMSE_ITEMS = tuple(item for items in MMSE_DOMAINS.values() for item in items)
MMSE_ALLOWED_SOURCE_COLUMNS = (PERSON_KEY, "TOTAL", *MMSE_ITEMS)
MMSE_FORBIDDEN_SOURCE_COLUMNS = frozenset(
    {
        "DIAG_NM",
        "DIAG_SEQ",
        "DOCTOR_NM",
        "MMSE_NUM",
        "MMSE_KIND",
        "EMAIL",
    }
)

# The wearable allow-lists reproduce the audited 112-feature bank used by the
# earlier 151-feature MaxAUC experiment.  They contain measurements and the
# subject join key, but no diagnosis.
ACTIVITY_RICH_COLUMNS = (
    "activity_score",
    "activity_steps",
    "activity_rest",
    "activity_inactive",
    "activity_low",
    "activity_medium",
    "activity_high",
    "activity_daily_movement",
    "activity_average_met",
    "activity_cal_active",
)
ACTIVITY_MEAN_COLUMNS = (
    "activity_score_meet_daily_targets",
    "activity_score_move_every_hour",
    "activity_score_recovery_time",
    "activity_score_stay_active",
    "activity_score_training_frequency",
    "activity_score_training_volume",
    "activity_cal_total",
    "activity_inactivity_alerts",
    "activity_met_min_high",
    "activity_met_min_medium",
    "activity_met_min_low",
    "activity_met_min_inactive",
)
ACTIVITY_ALLOWED_SOURCE_COLUMNS = (
    "EMAIL",
    *ACTIVITY_RICH_COLUMNS,
    *ACTIVITY_MEAN_COLUMNS,
)

SLEEP_RICH_COLUMNS = (
    "sleep_duration",
    "sleep_efficiency",
    "sleep_awake",
    "sleep_deep",
    "sleep_light",
    "sleep_rem",
    "sleep_restless",
    "sleep_onset_latency",
    "sleep_midpoint_time",
    "sleep_hr_average",
    "sleep_hr_lowest",
    "sleep_rmssd",
    "sleep_breath_average",
    "sleep_score",
    "sleep_score_deep",
    "sleep_temperature_deviation",
)
SLEEP_MEAN_COLUMNS = (
    "sleep_score_alignment",
    "sleep_score_disturbances",
    "sleep_score_efficiency",
    "sleep_score_latency",
    "sleep_score_rem",
    "sleep_score_total",
    "sleep_total",
)
SLEEP_ALLOWED_SOURCE_COLUMNS = (
    "EMAIL",
    *SLEEP_RICH_COLUMNS,
    *SLEEP_MEAN_COLUMNS,
    "sleep_bedtime_start",
    "sleep_bedtime_end",
)

_LAYOUT = {
    "train": {
        "directory": "1.Training",
        "mmse": "train_mmse.csv",
        "activity": "train_activity.csv",
        "sleep": "train_sleep.csv",
        "label": "training_label.csv",
        "n_subjects": 141,
        "diagnoses": {"CN": 85, "MCI": 47, "Dem": 9},
    },
    "val": {
        "directory": "2.Validation",
        "mmse": "val_mmse.csv",
        "activity": "val_activity.csv",
        "sleep": "val_sleep.csv",
        "label": "val_label.csv",
        "n_subjects": 33,
        "diagnoses": {"CN": 26, "MCI": 4, "Dem": 3},
    },
}
_DIAGNOSIS_MAP = {
    "CN": "CN",
    "NORMAL": "CN",
    "MCI": "MCI",
    "DEM": "Dem",
    "DEMENTIA": "Dem",
    "AD": "Dem",
}


class LeakageContractError(RuntimeError):
    """Raised when a source, subject, or split violates a leakage contract."""


@dataclass
class AccessAudit:
    """Serializable record of every CSV opened by this experiment."""

    events: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        path: Path,
        *,
        purpose: str,
        selected_columns: Iterable[str],
    ) -> None:
        self.events.append(
            {
                "path": str(path.resolve()),
                "purpose": str(purpose),
                "selected_columns": list(map(str, selected_columns)),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {"n_reads": len(self.events), "events": list(self.events)}


@dataclass(frozen=True)
class SplitSources:
    """Raw, explicitly allowed sources for one split."""

    split: str
    mmse: pd.DataFrame
    activity: pd.DataFrame
    sleep: pd.DataFrame
    diagnoses: pd.Series | None


def normalise_split(split: str) -> str:
    value = str(split).strip().lower()
    aliases = {
        "training": "train",
        "1.training": "train",
        "validation": "val",
        "valid": "val",
        "2.validation": "val",
    }
    value = aliases.get(value, value)
    if value not in _LAYOUT:
        raise ValueError(f"split must be train or val; got {split!r}")
    return value


def resolve_data_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    for candidate in (root, root / "Data"):
        if (candidate / "1.Training").is_dir() and (
            candidate / "2.Validation"
        ).is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Data root with 1.Training and 2.Validation not found below {root}"
    )


def split_root(data_root: str | Path, split: str) -> Path:
    key = normalise_split(split)
    return (resolve_data_root(data_root) / _LAYOUT[key]["directory"]).resolve()


def _read_csv_allowlist(
    path: Path,
    columns: Iterable[str],
    *,
    audit: AccessAudit,
    purpose: str,
) -> pd.DataFrame:
    """Read exactly ``columns`` using pandas ``usecols``."""

    selected = tuple(map(str, columns))
    if not path.is_file():
        raise FileNotFoundError(path)
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            frame = pd.read_csv(
                path,
                encoding=encoding,
                usecols=list(selected),
                low_memory=False,
            )
            audit.record(path, purpose=purpose, selected_columns=selected)
            return frame.loc[:, list(selected)]
        except UnicodeDecodeError as error:
            last_error = error
    raise RuntimeError(f"Unable to decode CSV: {path}") from last_error


def _normalise_ids(values: pd.Series, source: str) -> pd.Series:
    if values.isna().any():
        raise LeakageContractError(f"{source}: missing subject identifier")
    result = values.astype(str).str.strip()
    invalid = result.str.lower().isin({"", "nan", "none", "null", "<na>"})
    if invalid.any():
        raise LeakageContractError(f"{source}: invalid subject identifier")
    return result


def load_mmse_allowed(
    data_root: str | Path,
    split: str,
    *,
    audit: AccessAudit,
) -> pd.DataFrame:
    """Open only the MMSE score allow-list; diagnosis is never read here."""

    key = normalise_split(split)
    path = (
        split_root(data_root, key)
        / "SourceData"
        / "3.CognitiveFunction"
        / _LAYOUT[key]["mmse"]
    )
    frame = _read_csv_allowlist(
        path,
        MMSE_ALLOWED_SOURCE_COLUMNS,
        audit=audit,
        purpose=f"{key} MMSE feature source; diagnosis/admin excluded by usecols",
    )
    if set(frame.columns) & MMSE_FORBIDDEN_SOURCE_COLUMNS:
        raise LeakageContractError("Forbidden MMSE columns entered memory")
    frame = frame.copy()
    frame[PERSON_KEY] = _normalise_ids(frame[PERSON_KEY], f"{key} MMSE")
    if frame[PERSON_KEY].duplicated().any():
        raise LeakageContractError(f"{key} MMSE has duplicate subjects")
    frame = frame.set_index(PERSON_KEY)
    expected = int(_LAYOUT[key]["n_subjects"])
    if len(frame) != expected:
        raise LeakageContractError(
            f"{key} MMSE subject count changed: {len(frame)} != {expected}"
        )
    return frame.sort_index()


def load_wearable_sources(
    data_root: str | Path,
    split: str,
    *,
    audit: AccessAudit,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Open the fixed Activity and Sleep feature-source allow-lists."""

    key = normalise_split(split)
    root = split_root(data_root, key) / "SourceData"
    activity = _read_csv_allowlist(
        root / "1.Gait" / _LAYOUT[key]["activity"],
        ACTIVITY_ALLOWED_SOURCE_COLUMNS,
        audit=audit,
        purpose=f"{key} Activity feature source",
    )
    sleep = _read_csv_allowlist(
        root / "2.Sleep" / _LAYOUT[key]["sleep"],
        SLEEP_ALLOWED_SOURCE_COLUMNS,
        audit=audit,
        purpose=f"{key} Sleep feature source",
    )
    activity = activity.copy()
    sleep = sleep.copy()
    activity["EMAIL"] = _normalise_ids(activity["EMAIL"], f"{key} Activity")
    sleep["EMAIL"] = _normalise_ids(sleep["EMAIL"], f"{key} Sleep")
    return activity, sleep


def _normalise_label_copy(frame: pd.DataFrame, source: str) -> pd.Series:
    work = frame.copy()
    work[PERSON_KEY] = _normalise_ids(work[PERSON_KEY], source)
    raw = work["DIAG_NM"].astype(str).str.strip().str.upper()
    diagnosis = raw.map(_DIAGNOSIS_MAP)
    if diagnosis.isna().any():
        raise LeakageContractError(
            f"{source}: unknown diagnoses {sorted(raw[diagnosis.isna()].unique())}"
        )
    output = pd.Series(
        diagnosis.to_numpy(dtype=str),
        index=work[PERSON_KEY].to_numpy(dtype=str),
        name="diagnosis",
    )
    if output.index.has_duplicates:
        counts = output.groupby(level=0).nunique()
        if (counts > 1).any():
            raise LeakageContractError(f"{source}: conflicting duplicate labels")
        output = output[~output.index.duplicated(keep="first")]
    return output.sort_index()


def load_diagnoses(
    data_root: str | Path,
    split: str,
    *,
    audit: AccessAudit,
) -> pd.Series:
    """Load and cross-check the independent Gait and Sleep label copies."""

    key = normalise_split(split)
    root = split_root(data_root, key) / "LabelingData"
    copies: dict[str, pd.Series] = {}
    for modality, directory in (("Gait", "1.Gait"), ("Sleep", "2.Sleep")):
        path = root / directory / _LAYOUT[key]["label"]
        frame = _read_csv_allowlist(
            path,
            (PERSON_KEY, "DIAG_NM"),
            audit=audit,
            purpose=f"{key} {modality} diagnosis copy",
        )
        copies[modality] = _normalise_label_copy(frame, f"{key} {modality}")
    if not copies["Gait"].equals(copies["Sleep"]):
        raise LeakageContractError(f"{key} Gait/Sleep diagnosis copies disagree")
    observed = copies["Gait"].value_counts().to_dict()
    expected = _LAYOUT[key]["diagnoses"]
    if observed != expected:
        raise LeakageContractError(
            f"{key} diagnosis contract changed: {observed} != {expected}"
        )
    return copies["Gait"].copy()


def load_split_sources(
    data_root: str | Path,
    split: str,
    *,
    include_labels: bool,
    audit: AccessAudit,
) -> SplitSources:
    key = normalise_split(split)
    mmse = load_mmse_allowed(data_root, key, audit=audit)
    activity, sleep = load_wearable_sources(data_root, key, audit=audit)
    diagnoses = (
        load_diagnoses(data_root, key, audit=audit) if include_labels else None
    )
    return SplitSources(
        split=key,
        mmse=mmse,
        activity=activity,
        sleep=sleep,
        diagnoses=diagnoses,
    )


def binary_target(diagnoses: pd.Series) -> pd.Series:
    values = diagnoses.astype(str)
    unknown = sorted(set(values) - {"CN", "MCI", "Dem"})
    if unknown:
        raise LeakageContractError(f"Unexpected diagnoses: {unknown}")
    return values.isin({"MCI", "Dem"}).astype("int64")


def assert_disjoint_subjects(
    train_subject_ids: Iterable[str],
    heldout_subject_ids: Iterable[str],
    *,
    role: str,
) -> None:
    overlap = sorted(
        set(map(str, train_subject_ids)) & set(map(str, heldout_subject_ids))
    )
    if overlap:
        raise LeakageContractError(
            f"{role} direct subject leakage: {len(overlap)} overlapping subjects"
        )


__all__ = [
    "ACTIVITY_ALLOWED_SOURCE_COLUMNS",
    "ACTIVITY_MEAN_COLUMNS",
    "ACTIVITY_RICH_COLUMNS",
    "AccessAudit",
    "LeakageContractError",
    "MMSE_ALLOWED_SOURCE_COLUMNS",
    "MMSE_DOMAINS",
    "MMSE_FORBIDDEN_SOURCE_COLUMNS",
    "MMSE_ITEMS",
    "PERSON_KEY",
    "SLEEP_ALLOWED_SOURCE_COLUMNS",
    "SLEEP_MEAN_COLUMNS",
    "SLEEP_RICH_COLUMNS",
    "SplitSources",
    "assert_disjoint_subjects",
    "binary_target",
    "load_diagnoses",
    "load_mmse_allowed",
    "load_split_sources",
    "load_wearable_sources",
    "normalise_split",
    "resolve_data_root",
]
