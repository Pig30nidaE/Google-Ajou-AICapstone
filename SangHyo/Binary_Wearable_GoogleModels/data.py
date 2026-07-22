"""MMSE-free wearable data contract for CN vs MCI/DEM classification.

This module intentionally does not implement a second feature pipeline.  It
dynamically loads the repository's audited Activity + Sleep observed-event
builder and then performs only the label recoding needed by the binary task.
The resulting table has exactly one row per subject and contains no identifier,
diagnosis, collection-coverage, calendar, or cognitive-score feature.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd


# Public binary-label contract.  Human-facing reports may render MCI_DEM as
# "MCI + DEM", but model code can rely on these exact two stable names.
CLASS_NAMES = ("CN", "MCI_DEM")
CLASS_DISPLAY_NAMES = {"CN": "CN", "MCI_DEM": "MCI + DEM"}
CLASS_TO_ID = {name: class_id for class_id, name in enumerate(CLASS_NAMES)}
TARGET_DESCRIPTION = "CN=0, MCI or DEM=1"
FEATURE_MODE = "wearable_activity_sleep_observed_events_only"

# These explicit literals are both a fail-closed check and an auditable promise
# that no score/source/protocol proxy can silently enter the model table.
FORBIDDEN_FEATURE_TOKENS = (
    "email",
    "subject_id",
    "diag",
    "label",
    "mmse",
    "cognitive",
    "doctor",
    "period_id",
    "sample_order",
    "observed_count",
    "observed_day",
    "observed_night",
    "valid_count",
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
    "mask",
    "delta_since",
    "absolute_date",
)

# Counts are ordered exactly as CLASS_NAMES.  Contract checks are explicit and
# opt-in so a label-free Validation table can still be built before evaluation.
OFFICIAL_SPLIT_CONTRACTS = {
    "training": {"subjects": 141, "class_counts": [85, 56]},
    "validation": {"subjects": 33, "class_counts": [26, 7]},
}

_BUILDER_MODULE_NAME = "binary_wearable_audited_feature_builder"
_BUILDER: ModuleType | None = None
EXPECTED_BUILDER_SHA256 = "8c99b1a22191ed7bc6de40715af61c8c474132e5751c4e314d4a908394084433"
EXPECTED_CORE_SHA256 = "00d601d24330ffd7bdb358a27932160435201015f9038d593b516cceeaf46984"


@dataclass
class BinaryDataset:
    """One-row-per-subject wearable table used by training and prediction."""

    subject_ids: np.ndarray
    X: pd.DataFrame
    y: np.ndarray | None
    audit: dict


def _audited_builder_path() -> Path:
    """Resolve one fixed repository file, never a raw cognitive source path."""

    package_root = Path(__file__).resolve().parents[1]
    builder_dir = (package_root / "ThreeClass_GoogleYDF_CNBoost").resolve()
    builder_path = (builder_dir / "feature_engineering.py").resolve()
    if builder_path.parent != builder_dir or not builder_path.is_file():
        raise FileNotFoundError(
            "Required audited wearable feature builder is missing: "
            f"{builder_path}"
        )
    return builder_path


def _audited_core_path() -> Path:
    core_path = (
        Path(__file__).resolve().parents[1]
        / "ThreeClass_PerformanceLab"
        / "performance_lab_core.py"
    ).resolve()
    if not core_path.is_file():
        raise FileNotFoundError(f"Required audited wearable core is missing: {core_path}")
    return core_path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_audited_code_hashes() -> None:
    observed = {
        _audited_builder_path(): EXPECTED_BUILDER_SHA256,
        _audited_core_path(): EXPECTED_CORE_SHA256,
    }
    mismatches = [
        (path, expected, _file_sha256(path))
        for path, expected in observed.items()
        if _file_sha256(path) != expected
    ]
    if mismatches:
        rendered = "; ".join(
            f"{path}: expected {expected}, observed {actual}"
            for path, expected, actual in mismatches
        )
        raise RuntimeError(
            "Audited wearable feature code changed. Review it and update the pinned "
            f"hash deliberately before training: {rendered}"
        )


def _load_audited_builder() -> ModuleType:
    """Load and cache the fixed, repository-local feature builder safely."""

    global _BUILDER
    if _BUILDER is not None:
        return _BUILDER

    _assert_audited_code_hashes()
    builder_path = _audited_builder_path()
    existing = sys.modules.get(_BUILDER_MODULE_NAME)
    if existing is not None:
        existing_path = Path(getattr(existing, "__file__", "")).resolve()
        if existing_path != builder_path:
            raise ImportError(
                f"Module-name collision for {_BUILDER_MODULE_NAME}: {existing_path}"
            )
        _BUILDER = existing
        return existing

    spec = importlib.util.spec_from_file_location(_BUILDER_MODULE_NAME, builder_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load audited feature builder: {builder_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    required = (
        "build_subject_dataset",
        "discover_split_files",
        "load_consistent_labels",
        "feature_family",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        sys.modules.pop(spec.name, None)
        raise ImportError(f"Audited builder is missing required API(s): {missing}")
    _BUILDER = module
    return module


def _builder_sha256() -> str:
    return _file_sha256(_audited_builder_path())


def assert_wearable_feature_contract(columns: Iterable[str]) -> None:
    """Reject leakage, identifiers, and collection-protocol proxy features."""

    names = [str(column) for column in columns]
    offenders = sorted(
        name
        for name in names
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    )
    non_wearable = sorted(
        name
        for name in names
        if not (name.startswith("activity__") or name.startswith("sleep__"))
    )
    if offenders:
        raise AssertionError(f"Forbidden model feature(s): {offenders[:20]}")
    if non_wearable:
        raise AssertionError(f"Non-wearable model feature(s): {non_wearable[:20]}")


def discover_wearable_split_files(
    split_root: str | Path,
    *,
    require_labels: bool,
):
    """Resolve only Activity/Sleep sources and their two label copies.

    Patterns are intentionally explicit.  In particular, this function never
    globs ``LabelingData/*`` and therefore cannot even resolve a path below the
    CognitiveFunction/MMSE directory.
    """

    builder = _load_audited_builder()
    root = Path(split_root).expanduser().resolve()

    def one(pattern: str, role: str) -> Path:
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {role} below {root} with {pattern!r}; "
                f"found {len(matches)}: {matches}"
            )
        return matches[0]

    activity = one("SourceData/1.Gait/*activity.csv", "Activity source")
    sleep = one("SourceData/2.Sleep/*sleep.csv", "Sleep source")
    labels: tuple[Path, ...] = ()
    if require_labels:
        labels = (
            one("LabelingData/1.Gait/*label.csv", "Gait diagnosis label"),
            one("LabelingData/2.Sleep/*label.csv", "Sleep diagnosis label"),
        )
    return builder.SplitFiles(root=root, activity=activity, sleep=sleep, labels=labels)


def load_consistent_label_copies(paths: Sequence[Path]) -> pd.Series:
    """Load only wearable-label copies and fail closed if either differs."""

    checked = tuple(Path(path).expanduser().resolve() for path in paths)
    forbidden = [
        path
        for path in checked
        if any(token in str(path).lower() for token in ("cognitivefunction", "mmse"))
    ]
    if forbidden:
        raise AssertionError(f"Cognitive/MMSE label path is forbidden: {forbidden}")
    allowed_parents = {"1.Gait", "2.Sleep"}
    if not checked or any(path.parent.name not in allowed_parents for path in checked):
        raise AssertionError(f"Only Gait/Sleep label copies are allowed: {checked}")
    return _load_audited_builder().load_consistent_labels(checked)


def feature_family(feature_name: str) -> str:
    """Return the audited Activity/Sleep behavior family for EDA only."""

    return str(_load_audited_builder().feature_family(feature_name))


def _assert_one_row_per_subject(subject_ids: np.ndarray, X: pd.DataFrame) -> None:
    if subject_ids.ndim != 1:
        raise AssertionError("subject_ids must be one-dimensional")
    if len(subject_ids) != len(X):
        raise AssertionError("Feature rows and subject IDs are not aligned")
    normalized = pd.Series(subject_ids, dtype="string").str.strip()
    if normalized.isna().any() or normalized.eq("").any():
        raise AssertionError("Empty subject identifier detected")
    # The audited builder performs groupby-based subject aggregation.  This
    # independent wrapper check fails closed if any subject still has >1 row.
    subject_row_counts = normalized.groupby(normalized, sort=False).size()
    if not subject_row_counts.eq(1).all():
        duplicates = subject_row_counts[subject_row_counts > 1].index.tolist()
        raise AssertionError(f"Subject-level aggregation failed: {duplicates[:10]}")
    if len(subject_row_counts) != len(X):
        raise AssertionError("One-row-per-subject contract failed")


def _binary_target(three_class_target: np.ndarray | None) -> np.ndarray | None:
    if three_class_target is None:
        return None
    target = np.asarray(three_class_target, dtype=np.int64)
    unexpected = sorted(set(np.unique(target).tolist()) - {0, 1, 2})
    if unexpected:
        raise AssertionError(f"Unexpected source diagnosis ID(s): {unexpected}")
    return (target != 0).astype(np.int64, copy=False)


def build_binary_dataset(
    split_root: str | Path,
    *,
    require_labels: bool,
) -> BinaryDataset:
    """Build audited 7/14/28-event wearable features and binary labels.

    ``require_labels=False`` is the prediction-safe path.  It never resolves a
    label file.  If labels are requested, only the Gait and Sleep diagnosis
    copies are checked; the CognitiveFunction directory is never resolved.
    """

    builder = _load_audited_builder()
    files = discover_wearable_split_files(split_root, require_labels=require_labels)
    activity = pd.read_csv(files.activity, low_memory=False)
    sleep = pd.read_csv(files.sleep, low_memory=False)
    for frame, role in ((activity, "Activity"), (sleep, "Sleep")):
        if "EMAIL" not in frame.columns:
            raise ValueError(f"{role} source does not contain EMAIL")
        frame["EMAIL"] = frame["EMAIL"].astype(str).str.strip()
        if frame["EMAIL"].eq("").any():
            raise ValueError(f"{role} source contains an empty subject identifier")
    activity_subjects = sorted(activity["EMAIL"].unique().tolist())
    sleep_subjects = sorted(sleep["EMAIL"].unique().tolist())
    if activity_subjects != sleep_subjects:
        raise AssertionError("Activity and Sleep subject sets differ")

    labels = load_consistent_label_copies(files.labels) if require_labels else None
    if labels is not None:
        ordered_subjects = labels.index.astype(str).tolist()
        if set(ordered_subjects) != set(activity_subjects):
            raise AssertionError("Wearable source and label subject sets differ")
    else:
        ordered_subjects = activity_subjects

    X, source_diagnostics = builder.build_multiscale_event_summary(
        activity,
        sleep,
        ordered_subjects,
    )
    X.index = pd.Index(ordered_subjects, name="subject_id")
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    builder.assert_no_forbidden_features(X.columns)
    builder.PERFORMANCE_CORE.assert_primary_feature_contract(X.columns)
    X = X.reset_index(drop=True)
    subject_ids = np.asarray(ordered_subjects, dtype=object)
    _assert_one_row_per_subject(subject_ids, X)
    if X.columns.duplicated().any():
        raise AssertionError("Engineered feature names are not unique")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise AssertionError("Every engineered model feature must be numeric")
    assert_wearable_feature_contract(X.columns)

    three_class_y: np.ndarray | None = None
    original_diagnosis_counts = None
    if labels is not None:
        aligned = labels.reindex(ordered_subjects)
        if aligned.isna().any():
            raise AssertionError("At least one subject lost its diagnosis during alignment")
        three_class_y = aligned.map(builder.CLASS_TO_ID).to_numpy(dtype=np.int64)
        original_diagnosis_counts = {
            name: int((aligned == name).sum()) for name in ("CN", "MCI", "DEM")
        }
    y = _binary_target(three_class_y)
    if require_labels and y is None:
        raise AssertionError("Labels were requested but are absent")
    if not require_labels and y is not None:
        raise AssertionError("Label-free construction unexpectedly returned labels")

    class_counts = None
    if y is not None:
        class_counts = {
            class_name: int(np.sum(y == class_id))
            for class_id, class_name in enumerate(CLASS_NAMES)
        }
        if sum(class_counts.values()) != len(subject_ids):
            raise AssertionError("Binary labels do not cover every subject")

    source_contract = {
        "version": "observed_event_multiscale_7_14_28_v1",
        "observed_event_windows": [7, 14, 28],
        "statistics": list(builder.PERFORMANCE_CORE.SUMMARY_STATS),
        "feature_count": int(X.shape[1]),
        "feature_names": list(X.columns),
        "forbidden_explicit_signals": [
            "observed count",
            "calendar gap",
            "mask",
            "absolute date",
            "non-wear",
        ],
    }
    audit = {
        "split_root": str(Path(split_root).expanduser().resolve()),
        "subjects": int(len(subject_ids)),
        "class_names": list(CLASS_NAMES),
        "class_display_names": dict(CLASS_DISPLAY_NAMES),
        "class_counts": class_counts,
        "target_contract": TARGET_DESCRIPTION,
        "labels_requested": bool(require_labels),
        "original_diagnosis_counts": original_diagnosis_counts,
        "feature_mode": FEATURE_MODE,
        "source_modalities": ["activity", "sleep"],
        "feature_count": int(X.shape[1]),
        "feature_contract": source_contract,
        "observed_event_windows": list(
            source_contract.get("observed_event_windows", [7, 14, 28])
        ),
        "one_row_per_subject": True,
        "subject_id_duplicate_rows": 0,
        "subject_aggregation": "audited groupby then 7/14/28 recent observed-event summaries",
        "audited_builder_path": str(_audited_builder_path()),
        "audited_builder_sha256": _builder_sha256(),
        "audited_builder_expected_sha256": EXPECTED_BUILDER_SHA256,
        "audited_core_path": str(_audited_core_path()),
        "audited_core_sha256": _file_sha256(_audited_core_path()),
        "audited_core_expected_sha256": EXPECTED_CORE_SHA256,
        "audited_code_hash_contract_passed": True,
        "source_diagnostics": source_diagnostics,
        "label_copy_count": int(len(files.labels)),
        "label_copy_modalities": [path.parent.name for path in files.labels],
        "label_copies_consistent": bool(labels is not None),
        "mmse_source_resolved": False,
        "mmse_source_opened": False,
        "mmse_source_used": False,
        "mmse_values_used": False,
        "cognitive_source_opened": False,
        "cognitive_label_path_resolved": False,
        "cognitive_label_copy_opened": False,
        "coverage_or_calendar_protocol_features_used": False,
        "raw_identifier_used_as_feature": False,
        "direct_diagnosis_used_as_feature": False,
    }
    return BinaryDataset(subject_ids=subject_ids, X=X, y=y, audit=audit)


def assert_official_split_contract(
    dataset: BinaryDataset,
    split: Literal["training", "validation"],
) -> dict:
    """Fail closed unless a labeled dataset matches the official split counts."""

    if split not in OFFICIAL_SPLIT_CONTRACTS:
        raise ValueError(f"Unknown official split: {split!r}")
    if dataset.y is None:
        raise AssertionError(f"{split} class-count contract requires labels")
    expected = OFFICIAL_SPLIT_CONTRACTS[split]
    target = np.asarray(dataset.y, dtype=np.int64)
    observed_subjects = int(len(dataset.subject_ids))
    if len(dataset.X) != observed_subjects or len(target) != observed_subjects:
        raise AssertionError(f"{split} rows, subject IDs, and labels are misaligned")
    if len(set(map(str, dataset.subject_ids))) != observed_subjects:
        raise AssertionError(f"{split} contains duplicate subject IDs")
    if set(np.unique(target).tolist()) != {0, 1}:
        raise AssertionError(f"{split} labels must be exactly binary {{0, 1}}")
    observed_counts = [
        int(np.sum(target == class_id))
        for class_id in range(len(CLASS_NAMES))
    ]
    if observed_subjects != int(expected["subjects"]):
        raise AssertionError(
            f"{split} subject contract failed: "
            f"expected {expected['subjects']}, observed {observed_subjects}"
        )
    if observed_counts != list(expected["class_counts"]):
        raise AssertionError(
            f"{split} class-count contract failed in {CLASS_NAMES} order: "
            f"expected {expected['class_counts']}, observed {observed_counts}"
        )
    return {
        "split": split,
        "subjects": observed_subjects,
        "class_names": list(CLASS_NAMES),
        "class_counts": observed_counts,
        "passed": True,
    }
