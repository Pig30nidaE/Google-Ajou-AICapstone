"""Load the repository's audited MMSE-free subject-level feature table.

The feature contract lives in the sibling GoogleYDF experiment.  Reusing it
keeps the Naive Bayes baseline directly comparable while avoiding a second,
slowly diverging copy of the activity/sleep parsing code.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


CLASS_NAMES = ("CN", "MCI", "DEM")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
SHARED_EXPERIMENT = (
    Path(__file__).resolve().parents[1] / "ThreeClass_GoogleYDF_CNBoost"
)
SHARED_FEATURE_SOURCE = SHARED_EXPERIMENT / "feature_engineering.py"


def _load_shared_feature_module():
    if not SHARED_FEATURE_SOURCE.is_file():
        raise FileNotFoundError(
            "Audited wearable feature builder is missing: "
            f"{SHARED_FEATURE_SOURCE}"
        )
    module_name = "threeclass_naive_bayes_shared_features"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, SHARED_FEATURE_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load feature builder: {SHARED_FEATURE_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if tuple(module.CLASS_NAMES) != CLASS_NAMES:
        raise AssertionError(
            "Shared feature builder class order changed; expected "
            f"{CLASS_NAMES}, found {tuple(module.CLASS_NAMES)}"
        )
    return module


def build_subject_dataset(split_root: str | Path, *, require_labels: bool):
    """Return one row per subject using activity/sleep features only."""

    return _load_shared_feature_module().build_subject_dataset(
        split_root,
        require_labels=require_labels,
    )


def load_aligned_labels(
    split_root: str | Path,
    subject_ids: Sequence[object],
) -> np.ndarray:
    """Read the three consistent diagnosis-label copies and align their order."""

    shared = _load_shared_feature_module()
    files = shared.discover_split_files(split_root, require_labels=True)
    labels = shared.load_consistent_labels(files.labels)
    normalized_ids = [str(value) for value in subject_ids]
    aligned = labels.reindex(normalized_ids)
    if aligned.isna().any():
        missing = [
            normalized_ids[index]
            for index, value in enumerate(aligned.isna().to_numpy())
            if value
        ]
        raise AssertionError(f"Source subject(s) have no diagnosis label: {missing[:5]}")
    return aligned.map(CLASS_TO_ID).to_numpy(dtype=np.int64)


def shared_code_manifest() -> dict[str, str]:
    """Hash the reused feature implementation for reproducibility."""

    performance_core = (
        Path(__file__).resolve().parents[1]
        / "ThreeClass_PerformanceLab"
        / "performance_lab_core.py"
    )
    paths = {
        "../ThreeClass_GoogleYDF_CNBoost/feature_engineering.py": SHARED_FEATURE_SOURCE,
        "../ThreeClass_PerformanceLab/performance_lab_core.py": performance_core,
    }
    manifest: dict[str, str] = {}
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required shared code is missing: {path}")
        manifest[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest
