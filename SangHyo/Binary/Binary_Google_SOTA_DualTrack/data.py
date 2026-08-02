"""Data loading with fail-closed leakage guards.

Two feature tracks are supported:

``wearable``
    Activity + Sleep only.  ``Data/*/[Source|Labeling]Data/3.CognitiveFunction``
    is never opened; any attempt raises.

``mmse_fusion``
    Adds the MMSE item scores (``Q01``..``Q19``, ``TOTAL``).  The diagnosis
    columns that live in the *same* file -- ``DIAG_NM``, ``DIAG_SEQ`` -- and the
    administrative columns ``MMSE_NUM`` / ``MMSE_KIND`` are dropped before the
    frame is handed back, and a post-check re-verifies they are gone.

Ground-truth labels always come from the ``1.Gait`` / ``2.Sleep`` LabelingData
copies, never from ``train_mmse.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PERSON_KEY = "person_id"
LABEL_COLUMN = "y"
DIAG_COLUMN = "DIAG_NM"
POSITIVE_DIAGNOSES = ("MCI", "Dem")
NEGATIVE_DIAGNOSES = ("CN",)

COGNITIVE_DIRECTORY = "3.CognitiveFunction"
#: Columns of ``*_mmse.csv`` that must never reach the feature matrix.
MMSE_BANNED_COLUMNS = ("DIAG_NM", "DIAG_SEQ", "MMSE_NUM", "MMSE_KIND")
#: Feature-name tokens that indicate a label proxy or an observation-count proxy.
#: Matched on whole ``_``-delimited word boundaries, so the genuine Oura metric
#: ``activity_score_meet_daily_targets`` is not mistaken for an ML "target".
BANNED_FEATURE_TOKENS = (
    "diag", "diag_nm", "diag_seq", "label", "target", "y",
    "n_days", "num_days", "day_count", "days_observed",
    "n_obs", "n_rows", "count", "record_count", "observation_count",
)

TRACKS = ("wearable", "mmse_fusion")


class LeakageGuardError(RuntimeError):
    """Raised when a fail-closed data contract is violated."""


# --------------------------------------------------------------------- paths --
def resolve_roots(explicit_data_root: str | Path | None = None) -> tuple[Path, Path]:
    """Return ``(project_root, data_root)`` for both Colab and local runs."""

    project_root = Path(__file__).resolve().parents[2]
    if explicit_data_root is not None:
        data_root = Path(explicit_data_root).expanduser().resolve()
        if not data_root.exists():
            raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root}")
        return project_root, data_root

    candidates = [
        Path("/content/drive/Shareddrives/GoogleAI_contest/Data"),
        project_root / "Data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return project_root, candidate.resolve()
    raise FileNotFoundError(
        "Could not locate Data/. Tried: " + ", ".join(str(c) for c in candidates)
    )


_SPLIT_LAYOUT = {
    "train": dict(
        directory="1.Training",
        activity="SourceData/1.Gait/train_activity.csv",
        sleep="SourceData/2.Sleep/train_sleep.csv",
        mmse="SourceData/3.CognitiveFunction/train_mmse.csv",
        label_gait="LabelingData/1.Gait/training_label.csv",
        label_sleep="LabelingData/2.Sleep/training_label.csv",
    ),
    "validation": dict(
        directory="2.Validation",
        activity="SourceData/1.Gait/val_activity.csv",
        sleep="SourceData/2.Sleep/val_sleep.csv",
        mmse="SourceData/3.CognitiveFunction/val_mmse.csv",
        label_gait="LabelingData/1.Gait/val_label.csv",
        label_sleep="LabelingData/2.Sleep/val_label.csv",
    ),
}


def _split_path(data_root: Path, split: str, key: str) -> Path:
    if split not in _SPLIT_LAYOUT:
        raise ValueError(f"Unknown split {split!r}; expected one of {tuple(_SPLIT_LAYOUT)}")
    layout = _SPLIT_LAYOUT[split]
    return data_root / layout["directory"] / layout[key]


def _guarded_read_csv(path: Path, *, track: str) -> pd.DataFrame:
    """Read a CSV, refusing cognitive-function paths outside the fusion track."""

    if COGNITIVE_DIRECTORY in str(path) and track != "mmse_fusion":
        raise LeakageGuardError(
            f"track={track!r} must not open {COGNITIVE_DIRECTORY}: {path}"
        )
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return pd.read_csv(path)


# -------------------------------------------------------------------- labels --
def load_labels(data_root: Path, split: str) -> pd.DataFrame:
    """Load labels from the Gait and Sleep copies and verify they agree."""

    gait = _guarded_read_csv(_split_path(data_root, split, "label_gait"), track="wearable")
    sleep = _guarded_read_csv(_split_path(data_root, split, "label_sleep"), track="wearable")

    def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[["SAMPLE_EMAIL", DIAG_COLUMN]].copy()
        out["SAMPLE_EMAIL"] = out["SAMPLE_EMAIL"].astype(str).str.strip()
        out[DIAG_COLUMN] = out[DIAG_COLUMN].astype(str).str.strip()
        return out.sort_values("SAMPLE_EMAIL").reset_index(drop=True)

    gait_n, sleep_n = _normalise(gait), _normalise(sleep)
    if not gait_n.equals(sleep_n):
        raise LeakageGuardError(
            f"{split}: 1.Gait and 2.Sleep label copies disagree; refusing to guess."
        )

    unknown = set(gait_n[DIAG_COLUMN]) - set(POSITIVE_DIAGNOSES) - set(NEGATIVE_DIAGNOSES)
    if unknown:
        raise LeakageGuardError(f"{split}: unexpected DIAG_NM values {sorted(unknown)}")
    if gait_n["SAMPLE_EMAIL"].duplicated().any():
        raise LeakageGuardError(f"{split}: duplicated SAMPLE_EMAIL in labels")

    labels = gait_n.rename(columns={"SAMPLE_EMAIL": PERSON_KEY})
    labels[LABEL_COLUMN] = labels[DIAG_COLUMN].isin(POSITIVE_DIAGNOSES).astype(int)
    return labels


# ------------------------------------------------------------------ wearable --
def load_wearable(data_root: Path, split: str, *, track: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the day-level Activity and Sleep tables for one split."""

    activity = _guarded_read_csv(_split_path(data_root, split, "activity"), track=track)
    sleep = _guarded_read_csv(_split_path(data_root, split, "sleep"), track=track)
    for name, frame in (("activity", activity), ("sleep", sleep)):
        if "EMAIL" not in frame.columns:
            raise LeakageGuardError(f"{split}/{name}: missing EMAIL column")
        frame["EMAIL"] = frame["EMAIL"].astype(str).str.strip()
        frame.rename(columns={"EMAIL": PERSON_KEY}, inplace=True)
    return activity, sleep


# ---------------------------------------------------------------------- MMSE --
def load_mmse(data_root: Path, split: str, *, track: str) -> pd.DataFrame:
    """Load MMSE item scores with the diagnosis/administrative columns removed."""

    if track != "mmse_fusion":
        raise LeakageGuardError(
            f"load_mmse() called on track={track!r}; only 'mmse_fusion' may read "
            f"{COGNITIVE_DIRECTORY}."
        )

    raw = _guarded_read_csv(_split_path(data_root, split, "mmse"), track=track)
    if "SAMPLE_EMAIL" not in raw.columns:
        raise LeakageGuardError(f"{split}/mmse: missing SAMPLE_EMAIL column")

    keep = [c for c in raw.columns if c not in MMSE_BANNED_COLUMNS]
    frame = raw[keep].copy()
    frame["SAMPLE_EMAIL"] = frame["SAMPLE_EMAIL"].astype(str).str.strip()
    frame.rename(columns={"SAMPLE_EMAIL": PERSON_KEY}, inplace=True)

    # A person may appear more than once (repeat assessments); keep the first.
    frame = frame.drop_duplicates(subset=[PERSON_KEY], keep="first").reset_index(drop=True)

    leaked = [c for c in frame.columns if c in MMSE_BANNED_COLUMNS]
    if leaked:
        raise LeakageGuardError(f"{split}/mmse: banned columns survived the drop: {leaked}")

    numeric = frame.drop(columns=[PERSON_KEY]).apply(pd.to_numeric, errors="coerce")
    numeric.columns = [f"mmse_{c.lower()}" for c in numeric.columns]
    numeric.insert(0, PERSON_KEY, frame[PERSON_KEY].to_numpy())
    return numeric


# -------------------------------------------------------------------- guards --
def assert_feature_names_clean(columns) -> None:
    """Reject feature names that encode the label or the number of observations.

    Tokens are matched on whole ``_``-delimited word boundaries.  Substring
    matching would reject legitimate channels -- Oura ships
    ``activity_score_meet_daily_targets``, whose "targets" has nothing to do
    with the ML target.
    """

    offenders: list[str] = []
    for name in columns:
        lowered = str(name).lower()
        if lowered == PERSON_KEY:
            continue
        padded = f"_{lowered.strip('_')}_"
        for token in BANNED_FEATURE_TOKENS:
            if f"_{token}_" in padded:
                offenders.append(f"{name} (matched {token!r})")
                break
    if offenders:
        raise LeakageGuardError(
            "Forbidden feature names reached the matrix: " + "; ".join(sorted(offenders))
        )


def assert_no_mmse_features(columns) -> None:
    """Fail-closed check used by the wearable-only track."""

    offenders = [c for c in columns if "mmse" in str(c).lower() or "cognitive" in str(c).lower()]
    if offenders:
        raise LeakageGuardError(
            "track='wearable' but MMSE-derived columns are present: " + ", ".join(map(str, offenders))
        )


def assert_person_disjoint(train_ids, validation_ids) -> None:
    overlap = sorted(set(map(str, train_ids)) & set(map(str, validation_ids)))
    if overlap:
        raise LeakageGuardError(
            f"{len(overlap)} person id(s) appear in BOTH Training and Validation "
            f"(e.g. {overlap[:3]}); aborting."
        )


def align_features_and_labels(
    features: pd.DataFrame, labels: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Inner-join a person-level feature frame with its labels.

    Returns ``(X, y, person_ids)`` with rows in a deterministic person order.
    """

    merged = labels.merge(features, on=PERSON_KEY, how="inner", validate="one_to_one")
    missing = set(labels[PERSON_KEY]) - set(merged[PERSON_KEY])
    if missing:
        raise LeakageGuardError(
            f"{len(missing)} labelled person(s) have no wearable features "
            f"(e.g. {sorted(missing)[:3]})."
        )
    merged = merged.sort_values(PERSON_KEY).reset_index(drop=True)

    person_ids = merged[PERSON_KEY].to_numpy()
    y = merged[LABEL_COLUMN].to_numpy(dtype=np.int64)
    X = merged.drop(columns=[PERSON_KEY, LABEL_COLUMN, DIAG_COLUMN])
    assert_feature_names_clean(X.columns)
    return X, y, person_ids
