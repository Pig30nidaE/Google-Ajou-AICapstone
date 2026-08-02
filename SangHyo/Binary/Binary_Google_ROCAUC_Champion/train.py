"""End-to-end two-track training, freezing, and historical evaluation.

The public launcher invokes :func:`run_experiment`.  There is intentionally no
reduced training mode: only the predeclared full and max protocols may produce
performance reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .data import (
    AccessAudit,
    MMSE_ALLOWED_SOURCE_COLUMNS,
    assert_disjoint_subjects,
    load_diagnoses,
    load_mmse_allowed,
    load_wearable_sequence,
    resolve_data_root,
)
from .evaluation import (
    FittedChampion,
    NestedCVConfig,
    fit_final_champion,
    run_repeated_nested_cv,
)
from .features import ChampionDataset, build_champion_dataset
from .metrics import evaluate_binary_scores

TRACK_ORDER = ("mmse", "wearable")


@dataclass(frozen=True)
class RunConfig:
    data_root: str
    output_dir: str
    profile: str = "default"
    seed: int = 20260728
    include_tabpfn: bool = False
    hard_runtime_seconds: int = 21_600

    def nested(self) -> NestedCVConfig:
        profile = str(self.profile).lower()
        if profile not in {"default", "max"}:
            raise ValueError("profile must be 'default' or 'max'")
        return NestedCVConfig(
            outer_repeats=5 if profile == "default" else 10,
            seed=int(self.seed),
            bootstrap_resamples=5000 if profile == "default" else 10_000,
            include_tabpfn=bool(self.include_tabpfn),
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subject_hash(subject_id: str) -> str:
    return hashlib.sha256(str(subject_id).encode("utf-8")).hexdigest()


def _dependency_versions() -> dict[str, str | None]:
    packages = (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "joblib",
        "torch",
        "tabpfn",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _prepare_training_dataset(
    data_root: Path,
    track: str,
) -> tuple[ChampionDataset, AccessAudit]:
    audit = AccessAudit()
    wearable = load_wearable_sequence(
        data_root, "train", require_labels=True, audit=audit
    )
    diagnoses = load_diagnoses(data_root, "train", audit=audit)
    mmse = (
        load_mmse_allowed(data_root, "train", track=track, audit=audit)
        if track == "mmse"
        else None
    )
    dataset = build_champion_dataset(
        wearable,
        track=track,
        audit=audit,
        mmse=mmse,
        diagnoses=diagnoses,
    )
    return dataset, audit


def _prepare_validation_dataset(
    data_root: Path,
    track: str,
) -> tuple[ChampionDataset, AccessAudit]:
    """Load only label-free Validation sources."""

    audit = AccessAudit()
    wearable = load_wearable_sequence(
        data_root, "val", require_labels=False, audit=audit
    )
    mmse = (
        load_mmse_allowed(data_root, "val", track=track, audit=audit)
        if track == "mmse"
        else None
    )
    dataset = build_champion_dataset(
        wearable,
        track=track,
        audit=audit,
        mmse=mmse,
        diagnoses=None,
    )
    if dataset.y is not None:
        raise AssertionError("Label-free Validation dataset unexpectedly has y")
    return dataset, audit


def _assert_access_contract(
    track: str,
    *audits: AccessAudit,
) -> dict[str, Any]:
    combined = [
        event for audit in audits for event in audit.to_dict()["events"]
    ]
    cognitive = [
        event for event in combined if "3.CognitiveFunction" in event["path"]
    ]
    if track == "wearable" and cognitive:
        raise RuntimeError(
            "Wearable access audit contains a CognitiveFunction read"
        )
    if track == "mmse":
        for event in cognitive:
            selected = event["selected_columns"]
            if selected != list(MMSE_ALLOWED_SOURCE_COLUMNS):
                raise RuntimeError("MMSE access was not restricted by the allow-list")
            if not str(event["path"]).endswith(("_mmse.csv",)):
                raise RuntimeError("Unexpected CognitiveFunction source")
    return {
        "track": track,
        "events": combined,
        "n_reads": len(combined),
        "cognitive_reads": cognitive,
        "contract_passed": True,
    }


def _write_nested_artifacts(
    track_dir: Path,
    dataset: ChampionDataset,
    result: Any,
) -> dict[str, Any]:
    track_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for repeat in range(result.repeat_scores.shape[0]):
        for index, identifier in enumerate(dataset.subject_ids):
            rows.append(
                {
                    "repeat": int(repeat),
                    "subject_sha256": subject_hash(str(identifier)),
                    "champion_score": float(result.repeat_scores[repeat, index]),
                    "anchor_score": float(
                        result.repeat_anchor_scores[repeat, index]
                    ),
                    "threshold_margin": float(
                        result.repeat_threshold_margins[repeat, index]
                    ),
                }
            )
    oof_path = track_dir / "oof_predictions_hashed.csv"
    pd.DataFrame(rows).to_csv(oof_path, index=False)
    summary = result.summary(np.asarray(dataset.y))
    if dataset.diagnoses is not None:
        summary["subtype_sensitivity"] = {
            "selected_policy": _subtype_evaluation(
                np.asarray(dataset.diagnoses),
                np.mean(result.repeat_scores, axis=0),
                bootstrap_resamples=result.config.bootstrap_resamples,
                seed=result.config.seed + 31,
            ),
            "anchor": _subtype_evaluation(
                np.asarray(dataset.diagnoses),
                np.mean(result.repeat_anchor_scores, axis=0),
                bootstrap_resamples=result.config.bootstrap_resamples,
                seed=result.config.seed + 41,
            ),
            "status": "secondary; primary task remains CN vs MCI+Dem",
        }
    write_json(track_dir / "nested_oof_report.json", summary)
    write_json(
        track_dir / "fold_manifests.json",
        {
            "track": dataset.track,
            "folds": result.fold_records,
            "outer_test_subjects_are_hashed": True,
        },
    )
    return {
        "summary": summary,
        "oof_file": str(oof_path),
        "oof_sha256": sha256_file(oof_path),
        "fold_manifest_file": str(track_dir / "fold_manifests.json"),
    }


def _deployment_inventory(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": str(file.relative_to(path)),
            "size_bytes": int(file.stat().st_size),
            "sha256": sha256_file(file),
        }
        for file in sorted(path.rglob("*"))
        if file.is_file()
    ]


def _write_validation_prediction(
    path: Path,
    dataset: ChampionDataset,
    scores: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    columns: dict[str, Any] = {
        "subject_sha256": [
            subject_hash(str(identifier)) for identifier in dataset.subject_ids
        ]
    }
    for name, values in scores.items():
        columns[name] = np.asarray(values, dtype=np.float64)
    frame = pd.DataFrame(columns)
    if frame["subject_sha256"].duplicated().any():
        raise AssertionError("Validation subject hashes are not unique")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "n_subjects": int(len(frame)),
        "columns": list(frame.columns),
        "contains_raw_identifier": False,
    }


@dataclass
class TwoTrackValidationFreeze:
    root: Path
    records: dict[str, dict[str, Any]]

    @classmethod
    def create(cls, root: Path) -> "TwoTrackValidationFreeze":
        return cls(root=root, records={})

    def record(self, track: str, record: dict[str, Any]) -> None:
        if track not in TRACK_ORDER or track in self.records:
            raise RuntimeError(f"Invalid/duplicate Validation freeze track: {track}")
        self.records[track] = dict(record)

    def finalize(self) -> Path:
        if set(self.records) != set(TRACK_ORDER):
            raise RuntimeError("Both tracks must be frozen before labels are opened")
        self.verify()
        manifest = self.root / "VALIDATION_PREDICTIONS_FROZEN.json"
        write_json(
            manifest,
            {
                "status": "both_tracks_frozen",
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "tracks": self.records,
                "validation_labels_opened": False,
                "rule": (
                    "Both label-free files were written and SHA-256 verified "
                    "before the first Validation label read"
                ),
            },
        )
        return manifest

    def verify(self) -> None:
        if set(self.records) != set(TRACK_ORDER):
            raise RuntimeError("Two-track freeze is incomplete")
        for track, record in self.records.items():
            path = Path(record["path"])
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"Frozen {track} prediction hash mismatch")


def _subtype_evaluation(
    diagnosis: np.ndarray,
    score: np.ndarray,
    *,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    labels = np.asarray(diagnosis).astype(str)
    values = np.asarray(score, dtype=np.float64)
    output: dict[str, Any] = {}
    for positive, key in (("MCI", "cn_vs_mci"), ("Dem", "cn_vs_dem")):
        mask = np.isin(labels, ("CN", positive))
        binary = (labels[mask] == positive).astype(np.int64)
        output[key] = evaluate_binary_scores(
            binary,
            values[mask],
            n_resamples=bootstrap_resamples,
            seed=seed + (1 if positive == "MCI" else 2),
            score_name=key,
        )
    return output


def run_experiment(config: RunConfig) -> dict[str, Any]:
    """Run both tracks and open historical Validation labels only after freeze."""

    started_monotonic = time.monotonic()
    data_root = resolve_data_root(config.data_root)
    output = Path(config.output_dir).expanduser().resolve()
    training_root = output / "training"
    output.mkdir(parents=True, exist_ok=True)
    nested_config = config.nested()
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

    write_json(
        output / "run_manifest.json",
        {
            "experiment": "Binary_Google_ROCAUC_Champion",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config),
            "nested_protocol": asdict(nested_config),
            "tracks": list(TRACK_ORDER),
            "primary_metric": "subject-level continuous-score ROC-AUC",
            "data_root": str(data_root),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": _dependency_versions(),
                "tabpfn_telemetry_disabled": (
                    os.environ.get("TABPFN_DISABLE_TELEMETRY") == "1"
                ),
            },
            "validation_label_state": "closed",
        },
    )

    training_datasets: dict[str, ChampionDataset] = {}
    training_audits: dict[str, AccessAudit] = {}
    for track in TRACK_ORDER:
        dataset, audit = _prepare_training_dataset(data_root, track)
        training_datasets[track] = dataset
        training_audits[track] = audit

    track_reports: dict[str, Any] = {}
    deployments: dict[str, FittedChampion] = {}
    for track in TRACK_ORDER:
        track_dir = training_root / track
        nested_result = run_repeated_nested_cv(
            training_datasets[track],
            nested_config,
            progress_path=track_dir / "nested_cv_progress.json",
        )
        nested_artifacts = _write_nested_artifacts(
            track_dir, training_datasets[track], nested_result
        )
        deployment = fit_final_champion(training_datasets[track], nested_config)
        deployment_dir = track_dir / "deployment"
        deployment.save(deployment_dir)
        deployments[track] = deployment
        track_reports[track] = {
            **nested_artifacts,
            "deployment_dir": str(deployment_dir),
            "deployment_inventory": _deployment_inventory(deployment_dir),
            "final_selection": deployment.selection_audit,
        }

    write_json(
        training_root / "TRAINING_COMPLETE.json",
        {
            "status": "complete",
            "tracks": list(TRACK_ORDER),
            "validation_labels_opened": False,
            "elapsed_seconds": float(time.monotonic() - started_monotonic),
        },
    )

    validation_datasets: dict[str, ChampionDataset] = {}
    validation_audits: dict[str, AccessAudit] = {}
    for track in TRACK_ORDER:
        dataset, audit = _prepare_validation_dataset(data_root, track)
        assert_disjoint_subjects(
            training_datasets[track].subject_ids, dataset.subject_ids
        )
        validation_datasets[track] = dataset
        validation_audits[track] = audit

    freeze = TwoTrackValidationFreeze.create(training_root)
    frozen_scores: dict[str, dict[str, np.ndarray]] = {}
    for track in TRACK_ORDER:
        dataset = validation_datasets[track]
        scores = deployments[track].predict_scores(dataset)
        frozen_scores[track] = scores
        prediction_path = (
            training_root
            / f"validation_predictions_label_free_hashed_{track}.csv"
        )
        record = _write_validation_prediction(prediction_path, dataset, scores)
        record["deployment_threshold"] = float(deployments[track].threshold)
        freeze.record(track, record)

    freeze_manifest = freeze.finalize()
    freeze.verify()

    # Save and reload is checked only after the original predictions are frozen,
    # but still before labels.  The frozen files themselves are never replaced.
    round_trip: dict[str, Any] = {}
    for track in TRACK_ORDER:
        deployment_dir = training_root / track / "deployment"
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        loaded = FittedChampion.load(deployment_dir, device=device)
        reloaded = loaded.predict_scores(validation_datasets[track])
        differences = {
            name: float(
                np.max(np.abs(np.asarray(reloaded[name]) - np.asarray(values)))
            )
            for name, values in frozen_scores[track].items()
        }
        tolerance = 1e-5 if config.include_tabpfn else 1e-7
        if max(differences.values()) > tolerance:
            raise RuntimeError(f"{track} deployment round-trip score mismatch")
        round_trip[track] = {
            "passed": True,
            "device": device,
            "absolute_tolerance": tolerance,
            "max_abs_difference_by_score": differences,
        }
    write_json(training_root / "deployment_round_trip.json", round_trip)
    freeze.verify()

    access_reports = {
        track: _assert_access_contract(
            track, training_audits[track], validation_audits[track]
        )
        for track in TRACK_ORDER
    }
    write_json(output / "data_access_audit.json", access_reports)

    # This is the first and only Validation-label opening operation.
    label_audit = AccessAudit()
    validation_diagnoses = load_diagnoses(
        data_root, "val", audit=label_audit
    )
    historical: dict[str, Any] = {}
    for track in TRACK_ORDER:
        dataset = validation_datasets[track]
        aligned = validation_diagnoses.reindex(dataset.subject_ids)
        if aligned.isna().any():
            raise RuntimeError("Frozen Validation subjects and labels differ")
        y = aligned.isin({"MCI", "Dem"}).to_numpy(dtype=np.int64)
        champion_score = frozen_scores[track]["champion_score"]
        historical[track] = {
            "primary_task": evaluate_binary_scores(
                y,
                champion_score,
                threshold=deployments[track].threshold,
                n_resamples=nested_config.bootstrap_resamples,
                seed=config.seed + 701,
                score_name=f"historical_validation_{track}",
            ),
            "subtypes": _subtype_evaluation(
                aligned.to_numpy(dtype=str),
                champion_score,
                bootstrap_resamples=nested_config.bootstrap_resamples,
                seed=config.seed + 801,
            ),
            "status": "historical_validation_reused_not_independent_test",
        }
    write_json(
        training_root / "historical_validation_report.json",
        {
            "labels_opened_after_freeze_manifest": str(freeze_manifest),
            "label_access_audit": label_audit.to_dict(),
            "tracks": historical,
            "prohibition": (
                "No model, branch, threshold, seed, gate, or feature may be "
                "changed in response to these historical scores"
            ),
        },
    )

    final_report = {
        "experiment": "Binary_Google_ROCAUC_Champion",
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(time.monotonic() - started_monotonic),
        "nested_training_oof": track_reports,
        "historical_validation": historical,
        "validation_freeze_manifest": str(freeze_manifest),
        "primary_conclusion_rule": (
            "Use nested Training OOF as primary; historical Validation is "
            "descriptive and may not drive another model choice"
        ),
    }
    write_json(training_root / "FINAL_REPORT.json", final_report)
    return final_report


__all__ = [
    "RunConfig",
    "TRACK_ORDER",
    "TwoTrackValidationFreeze",
    "run_experiment",
]
