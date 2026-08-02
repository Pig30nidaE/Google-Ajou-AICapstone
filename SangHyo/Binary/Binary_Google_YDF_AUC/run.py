"""Single entrypoint for the Google YDF-only CN vs MCI+Dem AUC experiment.

The experiment is deliberately non-nested: a predeclared Google YDF candidate
bank and YDF-only blends are selected on repeated subject OOF ROC-AUC.  This
permits model-selection optimism as requested, while every held-out subject is
excluded from feature screening and model fitting for the score assigned to
that subject.

No sklearn estimator fallback exists.  Sparse-oblique candidates either train
with their exact YDF split axis or fail closed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "SangHyo.Binary_Google_YDF_AUC"

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import EXPERIMENT_NAME
from .data import (
    AccessAudit,
    MMSE_ALLOWED_SOURCE_COLUMNS,
    MMSE_FORBIDDEN_SOURCE_COLUMNS,
    LeakageContractError,
    assert_disjoint_subjects,
    binary_target,
    load_diagnoses,
    load_split_sources,
    resolve_data_root,
)
from .features import SubjectTable, build_subject_table
from .models import ENGINE_NAME, YDFBinaryModel, require_ydf, ydf_runtime_info
from .selection import (
    CandidateSpec,
    ProfileSpec,
    ReferenceECDF,
    build_and_select_policies,
    make_repeated_folds,
    profile_spec,
    select_fold_columns,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = PACKAGE_ROOT / "requirements_colab.in"
DEFAULT_SEED = 20260730
EXPECTED_TRAIN_COUNTS = {"CN": 85, "MCI": 47, "Dem": 9}
EXPECTED_VALIDATION_COUNTS = {"CN": 26, "MCI": 4, "Dem": 3}
CLASS_MAPPING = {"CN": 0, "MCI_or_Dem": 1}


def write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def hash_subject(value: str, *, salt: str) -> str:
    return hashlib.sha256(f"{salt}\0{value}".encode("utf-8")).hexdigest()


def _fingerprint_table(table: SubjectTable) -> str:
    digest = hashlib.sha256("\n".join(table.feature_names).encode("utf-8"))
    values = np.asarray(table.X, dtype=np.float32)
    digest.update(np.nan_to_num(values, nan=-9999.0).tobytes())
    return digest.hexdigest()


def _status(output: Path, status: str, **extra: Any) -> None:
    write_json(
        output / "LAUNCHER_STATUS.json",
        {
            "experiment": EXPERIMENT_NAME,
            "status": status,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            **extra,
        },
    )


def resolve_paths(
    *,
    namespace: Mapping[str, Any],
    data_root: str | None,
    output_dir: str | None,
) -> tuple[Path, Path]:
    data_candidates = (
        data_root,
        namespace.get("DATA_ROOT"),
        os.environ.get("BGYA_DATA_ROOT"),
        PACKAGE_ROOT.parents[1] / "Data",
    )
    resolved_data: Path | None = None
    for candidate in data_candidates:
        if not candidate:
            continue
        try:
            resolved_data = resolve_data_root(Path(os.fspath(candidate)))
            break
        except FileNotFoundError:
            continue
    if resolved_data is None:
        raise FileNotFoundError(
            "Data root with 1.Training and 2.Validation was not found"
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
    if output_dir:
        output = Path(output_dir).expanduser().resolve()
    elif os.environ.get("BGYA_OUTPUT_ROOT"):
        output = (
            Path(os.environ["BGYA_OUTPUT_ROOT"]).expanduser().resolve() / run_id
        )
    elif Path("/content").is_dir():
        drive = Path("/content/drive/MyDrive")
        if not drive.is_dir():
            raise RuntimeError(
                "Colab detected but Google Drive is not mounted; refusing a "
                "non-persistent local result directory"
            )
        output = drive / f"{EXPERIMENT_NAME}_result" / run_id
    else:
        output = PACKAGE_ROOT / f"{EXPERIMENT_NAME}_result" / run_id
    return resolved_data, output


def ensure_dependencies(*, include_ydf: bool, skip_install: bool) -> None:
    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "sklearn",
    }
    if include_ydf:
        required["ydf"] = "ydf"
    missing = [
        distribution
        for distribution, module in required.items()
        if importlib.util.find_spec(module) is None
    ]
    if not missing:
        if include_ydf:
            require_ydf()
        return
    if skip_install:
        raise ModuleNotFoundError("Missing dependencies: " + ", ".join(missing))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            str(REQUIREMENTS_FILE),
        ],
        check=True,
    )
    importlib.invalidate_caches()
    if include_ydf:
        require_ydf()


def _load_feature_table(
    data_root: Path,
    split: str,
    audit: AccessAudit,
) -> SubjectTable:
    sources = load_split_sources(
        data_root,
        split,
        include_labels=False,
        audit=audit,
    )
    table = build_subject_table(sources)
    if table.y is not None or table.diagnoses is not None:
        raise LeakageContractError("Feature table unexpectedly contains labels")
    return table


def _aligned_target(
    data_root: Path,
    split: str,
    table: SubjectTable,
    audit: AccessAudit,
) -> tuple[np.ndarray, pd.Series]:
    diagnoses = load_diagnoses(data_root, split, audit=audit)
    ids = list(map(str, table.subject_ids))
    if set(ids) != set(map(str, diagnoses.index)):
        raise LeakageContractError(f"{split}: feature and diagnosis subjects differ")
    aligned = diagnoses.reindex(ids)
    expected = (
        EXPECTED_TRAIN_COUNTS if split == "train" else EXPECTED_VALIDATION_COUNTS
    )
    observed = aligned.value_counts().to_dict()
    if observed != expected:
        raise LeakageContractError(
            f"{split}: diagnosis contract changed: {observed} != {expected}"
        )
    y = binary_target(aligned).to_numpy(dtype=np.int64)
    return y, aligned


def feature_manifest(table: SubjectTable) -> dict[str, Any]:
    forbidden_collection_tokens = (
        "count",
        "coverage",
        "days",
        "missing",
        "observation",
    )
    collection_offenders = [
        name
        for name in table.feature_names
        if (
            "non_wear" in name.lower()
            or any(
                token in name.lower().replace("__", "_").split("_")
                for token in forbidden_collection_tokens
            )
        )
    ]
    if collection_offenders:
        raise LeakageContractError(
            "Collection-volume feature entered the model bank: "
            + ", ".join(collection_offenders[:8])
        )
    return {
        "split": table.split,
        "n_subjects": int(len(table.subject_ids)),
        "union_feature_count": len(table.feature_names),
        "feature_names": list(table.feature_names),
        "views": {
            name: {
                "count": len(indices),
                "feature_names": [
                    table.feature_names[index] for index in indices
                ],
            }
            for name, indices in table.views.items()
        },
        "table_fingerprint_sha256": _fingerprint_table(table),
        "source_contract": {
            "mmse_usecols": list(MMSE_ALLOWED_SOURCE_COLUMNS),
            "mmse_forbidden_not_read": sorted(MMSE_FORBIDDEN_SOURCE_COLUMNS),
            "diagnosis_or_admin_in_features": False,
            "identifier_in_features": False,
            "observation_or_day_count_in_features": False,
            "prior_collection_proxy_features_replaced_by": [
                "w_activity_active_fraction__mean",
                "w_activity_moderate_high_fraction__mean",
                "w_activity_high_fraction_of_active__mean",
            ],
        },
    }


def inspect_data(data_root: Path, audit: AccessAudit) -> dict[str, Any]:
    table = _load_feature_table(data_root, "train", audit)
    return {
        "stage": "inspect",
        "cohort": "1.Training label-free feature inspection",
        "labels_opened": False,
        "historical_validation_opened": False,
        "feature_manifest": feature_manifest(table),
        "source_access": audit.to_dict(),
    }


def _auc_summary(y: np.ndarray, repeated_scores: np.ndarray) -> dict[str, Any]:
    scores = np.asarray(repeated_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(y):
        raise ValueError("Repeated OOF score matrix shape mismatch")
    if not np.isfinite(scores).all():
        raise ValueError("Repeated OOF score matrix is incomplete")
    repeat_auc = [
        float(roc_auc_score(y, scores[repeat]))
        for repeat in range(scores.shape[0])
    ]
    return {
        "metric": "ROC-AUC",
        "repeat_roc_auc": repeat_auc,
        "mean_repeat_roc_auc": float(np.mean(repeat_auc)),
        "repeat_roc_auc_sd": (
            float(np.std(repeat_auc, ddof=1)) if len(repeat_auc) > 1 else 0.0
        ),
        "subject_mean_oof_roc_auc": float(
            roc_auc_score(y, scores.mean(axis=0))
        ),
    }


def _candidate_seed(
    base_seed: int,
    candidate_index: int,
    repeat: int,
    fold: int,
    bag: int,
) -> int:
    return int(
        base_seed
        + 1_000_003 * candidate_index
        + 10_007 * repeat
        + 101 * fold
        + 17 * bag
    )


def evaluate_candidates(
    table: SubjectTable,
    y: np.ndarray,
    profile: ProfileSpec,
    *,
    seed: int,
    num_threads: int,
    output: Path,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    """Fit every held-out score without using that subject in screening or fit."""

    n_subjects = len(y)
    folds = make_repeated_folds(y, repeats=profile.repeats, seed=seed)
    raw_scores = {
        candidate.candidate_id: np.full(
            (profile.repeats, n_subjects), np.nan, dtype=np.float64
        )
        for candidate in profile.candidates
    }
    ecdf_scores = {
        candidate.candidate_id: np.full(
            (profile.repeats, n_subjects), np.nan, dtype=np.float64
        )
        for candidate in profile.candidates
    }
    fold_records: list[dict[str, Any]] = []
    seen = np.zeros((profile.repeats, n_subjects), dtype=np.int64)

    for fold_position, fold in enumerate(folds, start=1):
        train = fold.train_indices
        test = fold.test_indices
        assert_disjoint_subjects(
            table.subject_ids[train],
            table.subject_ids[test],
            role=f"repeat={fold.repeat} fold={fold.fold}",
        )
        seen[fold.repeat, test] += 1
        print(
            f"[OOF {fold_position:03d}/{len(folds):03d}] "
            f"repeat={fold.repeat} fold={fold.fold}",
            flush=True,
        )
        for candidate_index, candidate in enumerate(profile.candidates):
            view = table.view_indices(candidate.view)
            selected = select_fold_columns(
                table.X[train],
                y[train],
                view,
                top_k=candidate.top_k,
                corr_threshold=candidate.corr_threshold,
            )
            selected_names = tuple(
                table.feature_names[index] for index in selected
            )
            raw_bags: list[np.ndarray] = []
            ecdf_bags: list[np.ndarray] = []
            bag_seeds: list[int] = []
            oblique_contracts: list[dict[str, Any]] = []
            for bag in range(profile.bag_seeds):
                model_seed = _candidate_seed(
                    seed,
                    candidate_index,
                    fold.repeat,
                    fold.fold,
                    bag,
                )
                bag_seeds.append(model_seed)
                model = YDFBinaryModel(
                    candidate.family,
                    candidate.params,
                    seed=model_seed,
                    num_threads=num_threads,
                ).fit(table.X[train][:, selected], y[train], selected_names)
                train_raw = model.predict_score(table.X[train][:, selected])
                test_raw = model.predict_score(table.X[test][:, selected])
                reference = ReferenceECDF().fit(train_raw)
                raw_bags.append(test_raw)
                ecdf_bags.append(reference.transform(test_raw))
                oblique_contracts.append(
                    dict(model.manifest()["oblique_contract"])
                )
            raw_scores[candidate.candidate_id][fold.repeat, test] = np.mean(
                np.vstack(raw_bags), axis=0
            )
            ecdf_scores[candidate.candidate_id][fold.repeat, test] = np.mean(
                np.vstack(ecdf_bags), axis=0
            )
            fold_records.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "family": candidate.family,
                    "repeat": fold.repeat,
                    "fold": fold.fold,
                    "train_subject_count": int(len(train)),
                    "heldout_subject_count": int(len(test)),
                    "train_positive_count": int(y[train].sum()),
                    "heldout_positive_count": int(y[test].sum()),
                    "subject_overlap_count": 0,
                    "feature_screening_scope": "current fold training subjects only",
                    "selected_feature_count": int(len(selected)),
                    "selected_global_indices": selected.tolist(),
                    "selected_feature_names": list(selected_names),
                    "bag_seeds": bag_seeds,
                    "oblique_contracts": oblique_contracts,
                }
            )

    if not np.all(seen == 1):
        raise LeakageContractError(
            "Every subject must be held out exactly once per repeat"
        )
    for candidate in profile.candidates:
        candidate_id = candidate.candidate_id
        if not np.isfinite(raw_scores[candidate_id]).all():
            raise AssertionError(f"{candidate_id}: raw OOF is incomplete")
        if not np.isfinite(ecdf_scores[candidate_id]).all():
            raise AssertionError(f"{candidate_id}: ECDF OOF is incomplete")

    candidate_family = {
        candidate.candidate_id: candidate.family
        for candidate in profile.candidates
    }
    subject_raw = {
        candidate_id: scores.mean(axis=0)
        for candidate_id, scores in raw_scores.items()
    }
    subject_ecdf = {
        candidate_id: scores.mean(axis=0)
        for candidate_id, scores in ecdf_scores.items()
    }
    chosen_policy, policies = build_and_select_policies(
        subject_raw,
        subject_ecdf,
        candidate_family,
        y,
        blend_draws=profile.blend_draws,
        seed=seed + 90_000_001,
    )
    candidate_results: dict[str, Any] = {}
    spec_by_id = {
        candidate.candidate_id: candidate for candidate in profile.candidates
    }
    for candidate_id in sorted(spec_by_id):
        candidate_results[candidate_id] = {
            "spec": spec_by_id[candidate_id].to_dict(),
            "raw_probability": _auc_summary(y, raw_scores[candidate_id]),
            "training_reference_ecdf": _auc_summary(
                y, ecdf_scores[candidate_id]
            ),
        }
    write_json(output / "FOLD_FEATURE_SELECTION.json", fold_records)
    write_json(output / "CANDIDATE_RESULTS.json", candidate_results)
    write_json(
        output / "POLICY_RESULTS.json",
        {"chosen": chosen_policy, "all_policies": policies},
    )
    np.savez_compressed(
        output / "OOF_ALL_CANDIDATES.npz",
        **{
            **{
                f"raw__{candidate_id}": scores
                for candidate_id, scores in raw_scores.items()
            },
            **{
                f"ecdf__{candidate_id}": scores
                for candidate_id, scores in ecdf_scores.items()
            },
        },
    )
    return (
        raw_scores,
        ecdf_scores,
        candidate_results,
        chosen_policy,
        fold_records,
    )


def _policy_repeated_scores(
    policy: Mapping[str, Any],
    raw_scores: Mapping[str, np.ndarray],
    ecdf_scores: Mapping[str, np.ndarray],
) -> np.ndarray:
    source = raw_scores if policy["score_space"] == "raw" else ecdf_scores
    components = list(map(str, policy["components"]))
    weights = np.asarray(policy["weights"], dtype=np.float64)
    if len(components) != len(weights) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Chosen policy components/weights are invalid")
    result = np.zeros_like(source[components[0]], dtype=np.float64)
    for candidate_id, weight in zip(components, weights):
        result += float(weight) * source[candidate_id]
    return result


def _write_champion_oof(
    path: Path,
    table: SubjectTable,
    y: np.ndarray,
    repeated_scores: np.ndarray,
    *,
    salt: str,
) -> Path:
    rows: list[dict[str, Any]] = []
    for repeat in range(repeated_scores.shape[0]):
        for index, subject_id in enumerate(table.subject_ids):
            rows.append(
                {
                    "subject_hash": hash_subject(str(subject_id), salt=salt),
                    "repeat": repeat,
                    "y_true": int(y[index]),
                    "score": float(repeated_scores[repeat, index]),
                }
            )
    pd.DataFrame.from_records(rows).to_csv(path, index=False)
    subject_mean = pd.DataFrame(
        {
            "subject_hash": [
                hash_subject(str(subject_id), salt=salt)
                for subject_id in table.subject_ids
            ],
            "y_true": y,
            "score": repeated_scores.mean(axis=0),
        }
    )
    subject_mean.to_csv(
        path.with_name("OOF_CHAMPION_SUBJECT_MEAN_HASHED.csv"),
        index=False,
    )
    return path


def fit_deployment(
    table: SubjectTable,
    y: np.ndarray,
    profile: ProfileSpec,
    policy: Mapping[str, Any],
    *,
    seed: int,
    num_threads: int,
    destination: Path,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    if destination.exists():
        raise FileExistsError(f"Deployment path already exists: {destination}")
    destination.mkdir(parents=True)
    spec_by_id = {
        candidate.candidate_id: candidate for candidate in profile.candidates
    }
    runtime_components: dict[str, list[tuple[YDFBinaryModel, ReferenceECDF]]] = {}
    component_manifest: dict[str, Any] = {}

    for component_position, candidate_id in enumerate(policy["components"]):
        candidate = spec_by_id[str(candidate_id)]
        view = table.view_indices(candidate.view)
        selected = select_fold_columns(
            table.X,
            y,
            view,
            top_k=candidate.top_k,
            corr_threshold=candidate.corr_threshold,
        )
        selected_names = tuple(
            table.feature_names[index] for index in selected
        )
        bags: list[tuple[YDFBinaryModel, ReferenceECDF]] = []
        bag_manifest: list[dict[str, Any]] = []
        for bag in range(profile.bag_seeds):
            model_seed = int(
                seed + 200_000_003 + component_position * 1_000_003 + bag * 17
            )
            model = YDFBinaryModel(
                candidate.family,
                candidate.params,
                seed=model_seed,
                num_threads=num_threads,
            ).fit(table.X[:, selected], y, selected_names)
            reference = ReferenceECDF().fit(
                model.predict_score(table.X[:, selected])
            )
            relative = (
                Path("components")
                / str(candidate_id)
                / f"bag_{bag:02d}"
            )
            model.save(destination / relative)
            bags.append((model, reference))
            bag_manifest.append(
                {
                    "bag": bag,
                    "seed": model_seed,
                    "model_path": relative.as_posix(),
                    "model_tree_sha256": _tree_sha256(destination / relative),
                    "training_reference_ecdf": reference.to_dict(),
                    "model": model.manifest(),
                }
            )
        runtime_components[str(candidate_id)] = bags
        component_manifest[str(candidate_id)] = {
            "spec": candidate.to_dict(),
            "selected_global_indices": selected.tolist(),
            "selected_feature_names": list(selected_names),
            "bags": bag_manifest,
        }

    manifest = {
        "format": 1,
        "experiment": EXPERIMENT_NAME,
        "engine": ENGINE_NAME,
        "fallback_permitted": False,
        "class_mapping": CLASS_MAPPING,
        "input_feature_names": list(table.feature_names),
        "input_feature_count": len(table.feature_names),
        "input_feature_fingerprint": _fingerprint_table(table),
        "policy": dict(policy),
        "components": component_manifest,
    }
    runtime = {"manifest": manifest, "components": runtime_components}
    before = predict_deployment(runtime, table)
    write_json(destination / "DEPLOYMENT.json", manifest)
    restored = load_deployment(destination)
    after = predict_deployment(restored, table)
    max_difference = float(np.max(np.abs(before - after)))
    if not np.allclose(before, after, rtol=0.0, atol=1e-12):
        raise AssertionError(
            f"YDF checkpoint roundtrip changed scores: max diff={max_difference}"
        )
    roundtrip = {
        "verified": True,
        "max_abs_difference": max_difference,
        "deployment_manifest_sha256": _file_sha256(
            destination / "DEPLOYMENT.json"
        ),
        "component_checkpoint_hashes_verified_on_load": True,
    }
    write_json(destination / "ROUNDTRIP.json", roundtrip)
    return restored, roundtrip, after


def load_deployment(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest = json.loads((root / "DEPLOYMENT.json").read_text(encoding="utf-8"))
    if manifest.get("engine") != ENGINE_NAME:
        raise RuntimeError("Deployment is not a Google YDF bundle")
    if manifest.get("fallback_permitted") is not False:
        raise RuntimeError("Deployment fallback contract changed")
    components: dict[str, list[tuple[YDFBinaryModel, ReferenceECDF]]] = {}
    for candidate_id, component in manifest["components"].items():
        bags: list[tuple[YDFBinaryModel, ReferenceECDF]] = []
        for bag in component["bags"]:
            model_root = root / bag["model_path"]
            expected_hash = str(bag["model_tree_sha256"])
            observed_hash = _tree_sha256(model_root)
            if observed_hash != expected_hash:
                raise RuntimeError(
                    f"YDF checkpoint checksum mismatch for {bag['model_path']}"
                )
            model = YDFBinaryModel.load(model_root)
            reference = ReferenceECDF.from_dict(
                bag["training_reference_ecdf"]
            )
            bags.append((model, reference))
        components[str(candidate_id)] = bags
    return {"manifest": manifest, "components": components}


def predict_deployment(
    runtime: Mapping[str, Any],
    table: SubjectTable,
) -> np.ndarray:
    manifest = runtime["manifest"]
    if tuple(manifest["input_feature_names"]) != tuple(table.feature_names):
        raise LeakageContractError("Deployment input feature schema changed")
    policy = manifest["policy"]
    component_scores: dict[str, np.ndarray] = {}
    for candidate_id in policy["components"]:
        component = manifest["components"][candidate_id]
        selected = np.asarray(
            component["selected_global_indices"], dtype=np.int64
        )
        bag_scores: list[np.ndarray] = []
        for model, reference in runtime["components"][candidate_id]:
            raw = model.predict_score(table.X[:, selected])
            bag_scores.append(
                raw
                if policy["score_space"] == "raw"
                else reference.transform(raw)
            )
        component_scores[candidate_id] = np.mean(
            np.vstack(bag_scores), axis=0
        )
    result = np.zeros(len(table.subject_ids), dtype=np.float64)
    for candidate_id, weight in zip(
        policy["components"], policy["weights"]
    ):
        result += float(weight) * component_scores[candidate_id]
    if not np.isfinite(result).all():
        raise RuntimeError("Deployment emitted invalid scores")
    return result


def historical_validation(
    *,
    data_root: Path,
    output: Path,
    audit: AccessAudit,
    training_subject_ids: Sequence[str],
    deployment: Mapping[str, Any],
    salt: str,
) -> dict[str, Any]:
    table = _load_feature_table(data_root, "val", audit)
    assert_disjoint_subjects(
        training_subject_ids,
        table.subject_ids,
        role="Training versus historical Validation",
    )
    scores = predict_deployment(deployment, table)
    validation_label_reads = [
        event
        for event in audit.events
        if str(event.get("purpose", "")).startswith("val ")
        and "diagnosis copy" in str(event.get("purpose", ""))
    ]
    if validation_label_reads:
        raise LeakageContractError(
            "Historical Validation labels were opened before prediction freeze"
        )
    freeze_path = output / "HISTORICAL_VALIDATION_PREDICTIONS_FROZEN.csv"
    pd.DataFrame(
        {
            "subject_hash": [
                hash_subject(str(subject_id), salt=salt)
                for subject_id in table.subject_ids
            ],
            "score": scores,
        }
    ).to_csv(freeze_path, index=False)
    freeze_hash = _file_sha256(freeze_path)
    write_json(
        output / "HISTORICAL_VALIDATION_FREEZE.json",
        {
            "prediction_file": str(freeze_path),
            "sha256_before_label_open": freeze_hash,
            "labels_opened_at_freeze": bool(validation_label_reads),
            "training_validation_subject_overlap": 0,
        },
    )
    y, _diagnoses = _aligned_target(
        data_root, "val", table, audit
    )
    report = {
        "role": "historical benchmark; not an untouched external test",
        "n_subjects": int(len(y)),
        "class_counts": {
            "CN": int((y == 0).sum()),
            "MCI_or_Dem": int(y.sum()),
        },
        "roc_auc": float(roc_auc_score(y, scores)),
        "prediction_freeze_sha256": freeze_hash,
        "labels_opened_after_prediction_freeze": True,
    }
    write_json(output / "HISTORICAL_VALIDATION_REPORT.json", report)
    return report


def train_experiment(
    *,
    data_root: Path,
    output: Path,
    audit: AccessAudit,
    profile: ProfileSpec,
    seed: int,
    num_threads: int,
    historical_eval: bool,
    artifact_salt: str,
) -> dict[str, Any]:
    # Build the entire model feature matrix before opening any target file.
    table = _load_feature_table(data_root, "train", audit)
    features = feature_manifest(table)
    write_json(output / "FEATURE_MANIFEST.json", features)
    y, diagnoses = _aligned_target(data_root, "train", table, audit)
    if int(y.sum()) != 56 or int((y == 0).sum()) != 85:
        raise LeakageContractError("Training binary class counts changed")

    (
        raw_scores,
        ecdf_scores,
        candidate_results,
        policy,
        fold_records,
    ) = evaluate_candidates(
        table,
        y,
        profile,
        seed=seed,
        num_threads=num_threads,
        output=output,
    )
    champion_scores = _policy_repeated_scores(
        policy, raw_scores, ecdf_scores
    )
    champion_auc = _auc_summary(y, champion_scores)
    if not np.isclose(
        champion_auc["subject_mean_oof_roc_auc"],
        policy["subject_mean_oof_roc_auc"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("Chosen policy reconstruction changed its OOF AUC")
    _write_champion_oof(
        output / "OOF_CHAMPION_REPEATED_HASHED.csv",
        table,
        y,
        champion_scores,
        salt=artifact_salt,
    )

    deployment, roundtrip, _train_scores = fit_deployment(
        table,
        y,
        profile,
        policy,
        seed=seed,
        num_threads=num_threads,
        destination=output / "deployment",
    )
    report: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "task": {
            "negative": "CN",
            "positive": "MCI + Dem",
            "n_subjects": int(len(y)),
            "diagnosis_counts": {
                key: int(value)
                for key, value in diagnoses.value_counts().to_dict().items()
            },
        },
        "engine_contract": {
            **ydf_runtime_info(),
            "models_in_champion": [ENGINE_NAME],
            "sklearn_estimator_in_champion": False,
            "sparse_oblique_downgrade_permitted": False,
        },
        "metric_contract": {
            "primary": "ROC-AUC",
            "candidate_selection": "subject-mean repeated OOF ROC-AUC",
            "ensemble_weight_selection": "subject-mean repeated OOF ROC-AUC",
            "other_metrics_used_for_selection": [],
            "nested": False,
            "selection_note": (
                "The predeclared model/configuration/blend winner is selected "
                "on these repeated OOF labels, so the development score is "
                "selection-optimistic. Each individual OOF score remains from "
                "a model and feature screen that excluded that subject."
            ),
        },
        "cv": {
            "folds": 5,
            "repeats": profile.repeats,
            "bag_seeds": profile.bag_seeds,
            "subject_level": True,
            "all_subject_overlap_checks_passed": all(
                record["subject_overlap_count"] == 0
                for record in fold_records
            ),
        },
        "profile": profile.to_dict(),
        "reportability": {
            "reportable_full_profile": profile.reportable,
            "smoke_is_wiring_only": not profile.reportable,
        },
        "champion_policy": {
            **dict(policy),
            "auc": champion_auc,
        },
        "candidate_results_path": str(output / "CANDIDATE_RESULTS.json"),
        "candidate_count": len(candidate_results),
        "deployment": {
            "path": str(output / "deployment"),
            "full_training_refit": True,
            "roundtrip": roundtrip,
        },
    }
    write_json(output / "FINAL_REPORT.json", report)
    if historical_eval:
        report["historical_validation"] = historical_validation(
            data_root=data_root,
            output=output,
            audit=audit,
            training_subject_ids=table.subject_ids,
            deployment=deployment,
            salt=artifact_salt,
        )
        write_json(output / "FINAL_REPORT.json", report)

    leakage_audit = {
        "direct_leakage_checks": {
            "subject_level_folds": True,
            "all_oof_subject_overlap_counts_zero": True,
            "feature_screening_fold_training_only": True,
            "heldout_labels_used_in_screen_or_fit": False,
            "diagnosis_admin_identifier_features": False,
            "mmse_source_allowlist_enforced": True,
            "observation_day_count_features": False,
            "google_ydf_only": True,
            "estimator_fallback_permitted": False,
        },
        "allowed_non_nested_optimism": {
            "candidate_selected_on_same_repeated_oof": True,
            "blend_weights_selected_on_same_repeated_oof": True,
            "reported_explicitly": True,
        },
        "source_access": audit.to_dict(),
    }
    write_json(output / "LEAKAGE_AUDIT.json", leakage_audit)
    completion = {
        "status": "complete",
        "experiment": EXPERIMENT_NAME,
        "profile": profile.name,
        "reportable": profile.reportable,
        "final_report": str(output / "FINAL_REPORT.json"),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "TRAINING_COMPLETE.json", completion)
    return report


def run_pipeline(
    *,
    namespace: Mapping[str, Any] | None = None,
    stage: str = "all",
    profile_name: str = "default",
    data_root: str | None = None,
    output_dir: str | None = None,
    historical_eval: bool = False,
    seed: int = DEFAULT_SEED,
    num_threads: int | None = None,
    skip_install: bool = False,
) -> dict[str, Any]:
    namespace = globals() if namespace is None else namespace
    profile = profile_spec(profile_name)
    resolved_data, output = resolve_paths(
        namespace=namespace,
        data_root=data_root,
        output_dir=output_dir,
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    artifact_salt = os.urandom(32).hex()
    threads = (
        max(1, min(8, os.cpu_count() or 1))
        if num_threads is None
        else max(1, int(num_threads))
    )
    write_json(
        output / "RUN_CONFIG.json",
        {
            "experiment": EXPERIMENT_NAME,
            "stage": stage,
            "profile": profile.to_dict(),
            "data_root": str(resolved_data),
            "output_dir": str(output),
            "historical_eval": historical_eval,
            "seed": seed,
            "num_threads": threads,
            "primary_metric": "ROC-AUC",
            "subject_tokenization": (
                "SHA-256 with a run-local random secret that is not persisted"
            ),
        },
    )
    _status(output, "starting", stage=stage, profile=profile.name)
    audit = AccessAudit()
    try:
        ensure_dependencies(
            include_ydf=stage in {"train", "all"},
            skip_install=skip_install,
        )
        inspection = inspect_data(resolved_data, audit)
        write_json(output / "DATA_AUDIT.json", inspection)
        if stage == "inspect":
            _status(
                output,
                "complete",
                stage=stage,
                elapsed_seconds=time.monotonic() - started,
            )
            return {"output_dir": str(output), "inspect": inspection}

        report = train_experiment(
            data_root=resolved_data,
            output=output,
            audit=audit,
            profile=profile,
            seed=seed,
            num_threads=threads,
            historical_eval=historical_eval,
            artifact_salt=artifact_salt,
        )
        write_json(
            output / "DATA_AUDIT.json",
            {
                **inspection,
                "labels_opened_after_feature_construction": True,
                "historical_validation_evaluated": historical_eval,
                "source_access": audit.to_dict(),
            },
        )
        _status(
            output,
            "complete",
            stage=stage,
            profile=profile.name,
            elapsed_seconds=time.monotonic() - started,
            final_report=str(output / "FINAL_REPORT.json"),
        )
        print("Complete:", output / "FINAL_REPORT.json", flush=True)
        return {"output_dir": str(output), "final_report": report}
    except Exception as error:
        _status(
            output,
            "failed",
            stage=stage,
            elapsed_seconds=time.monotonic() - started,
            error_type=type(error).__name__,
            error=str(error)[:1000],
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Google YDF-only CN vs MCI+Dem model; non-nested candidate "
            "selection optimizes ROC-AUC only."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("inspect", "train", "all"),
        default="all",
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "default", "max"),
        default="default",
    )
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--historical-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--skip-install", action="store_true")
    return parser


def _strip_jupyter_arguments(argv: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    index = 0
    values = list(map(str, argv))
    while index < len(values):
        token = values[index]
        if token in {"-f", "--f"} and index + 1 < len(values):
            candidate = Path(values[index + 1])
            if (
                candidate.name.startswith("kernel-")
                and candidate.suffix == ".json"
            ):
                index += 2
                continue
        if token.startswith(("-f=", "--f=")):
            candidate = Path(token.split("=", 1)[1])
            if (
                candidate.name.startswith("kernel-")
                and candidate.suffix == ".json"
            ):
                index += 1
                continue
        cleaned.append(token)
        index += 1
    return cleaned


def notebook_argv(environ: Mapping[str, str] | None = None) -> list[str]:
    environment = os.environ if environ is None else environ
    return shlex.split(
        str(environment.get("BGYA_ARGS", "--stage all --profile default"))
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    namespace: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw = list(sys.argv[1:] if argv is None else argv)
    cleaned = _strip_jupyter_arguments(raw)
    if argv is None and not cleaned and os.environ.get("BGYA_ARGS"):
        cleaned = notebook_argv()
    args = build_parser().parse_args(cleaned)
    return run_pipeline(
        namespace=namespace,
        stage=args.stage,
        profile_name=args.profile,
        data_root=args.data_root,
        output_dir=args.output_dir,
        historical_eval=args.historical_eval,
        seed=args.seed,
        num_threads=args.num_threads,
        skip_install=args.skip_install,
    )


if __name__ == "__main__":
    main()
