"""Repeated nested subject-CV and checkpointed balanced model fusion."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from .data import (
    COMPACT_DAILY_FEATURES,
    assert_disjoint_subjects,
    build_subject_dataset,
    load_validation_labels_checked,
    make_fixed_views,
)
from .eda import run_eda
from .features import FoldFeaturePipeline
from .models import (
    GOOGLE_MODELS,
    MODEL_NAMES,
    ElasticNetAdapter,
    TabNetAdapter,
    TransformerAdapter,
    YDFDailyAdapter,
    YDFSubjectAdapter,
    fit_model,
    predict_model,
    set_global_seed,
)


TARGET_ACCURACY = 0.90
TARGET_BALANCED_ACCURACY = 0.75
HISTORICAL_VALIDATION_WARNING = (
    "The released 33-person Validation split has already been inspected by "
    "multiple experiments. It is historical, not a fresh independent test."
)


@dataclass(frozen=True)
class RunConfig:
    training_root: str
    validation_root: str
    output_dir: str
    run_mode: str = "full"
    seed: int = 20260723
    outer_folds: int = 5
    outer_repeats: int = 2
    inner_folds: int = 3
    max_features: int = 24
    max_runtime_seconds: int = 20_400
    evaluate_historical_validation: bool = True


class SoftTimeBudgetExceeded(TimeoutError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subject_hash(subject_id: str) -> str:
    return hashlib.sha256(
        f"balanced-fusion-mmse-free-v1::{subject_id}".encode("utf-8")
    ).hexdigest()[:20]


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, score))
    except ValueError:
        return float("nan")


def binary_metrics(
    y_true: np.ndarray,
    risk_score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    target = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(risk_score, dtype=np.float64)
    if target.shape != score.shape or not np.isfinite(score).all():
        raise ValueError("Metric targets/scores are invalid")
    prediction = (score >= float(threshold)).astype(np.int64)
    matrix = confusion_matrix(target, prediction, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "recall_impaired": float(recall_score(target, prediction, zero_division=0)),
        "specificity_cn": float(tn / max(1, tn + fp)),
        "precision_impaired": float(
            precision_score(target, prediction, zero_division=0)
        ),
        "roc_auc": _safe_auc(target, score),
        "pr_auc": float(average_precision_score(target, score)),
        "threshold": float(threshold),
        "confusion_matrix": matrix.tolist(),
        "counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "support": {
            "CN": int(np.sum(target == 0)),
            "MCI_DEM": int(np.sum(target == 1)),
        },
    }


def _all_cn_baseline(y: np.ndarray) -> dict[str, Any]:
    target = np.asarray(y, dtype=np.int64)
    return {
        "accuracy": float(np.mean(target == 0)),
        "balanced_accuracy": 0.5,
        "correct": int(np.sum(target == 0)),
        "total": int(len(target)),
    }


def _target_check(metrics: Mapping[str, Any]) -> dict[str, Any]:
    accuracy = float(metrics["accuracy"])
    balanced = float(metrics["balanced_accuracy"])
    return {
        "target_accuracy": TARGET_ACCURACY,
        "target_balanced_accuracy": TARGET_BALANCED_ACCURACY,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "both_met": bool(
            np.isfinite(accuracy)
            and np.isfinite(balanced)
            and accuracy >= TARGET_ACCURACY
            and balanced >= TARGET_BALANCED_ACCURACY
        ),
        "note": "Targets are goals, not guarantees.",
    }


def _threshold_objective(metrics: Mapping[str, Any]) -> float:
    recall = float(metrics["recall_impaired"])
    specificity = float(metrics["specificity_cn"])
    return float(
        0.65 * float(metrics["balanced_accuracy"])
        + 0.20 * min(recall, specificity)
        + 0.15 * float(metrics["accuracy"])
        - 0.10 * abs(recall - specificity)
    )


def select_threshold(y: np.ndarray, risk: np.ndarray) -> dict[str, Any]:
    """Choose a balanced operating point using only the supplied training OOF."""

    target = np.asarray(y, dtype=np.int64)
    score = np.asarray(risk, dtype=np.float64)
    unique = np.unique(score)
    if len(unique) > 1:
        midpoints = (unique[:-1] + unique[1:]) / 2.0
    else:
        midpoints = unique
    candidates = np.unique(
        np.r_[0.05, np.linspace(0.10, 0.90, 33), midpoints, 0.95]
    )
    rows = []
    for threshold in candidates:
        metrics = binary_metrics(target, score, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "objective": _threshold_objective(metrics),
                "metrics": metrics,
            }
        )
    best_value = max(row["objective"] for row in rows)
    near = [row for row in rows if row["objective"] >= best_value - 1e-12]
    chosen = min(
        near,
        key=lambda row: (
            abs(row["metrics"]["recall_impaired"] - row["metrics"]["specificity_cn"]),
            abs(row["threshold"] - 0.5),
        ),
    )
    # Small shrinkage reduces extreme threshold transfer from tiny inner OOF sets.
    raw_threshold = float(chosen["threshold"])
    shrunk_threshold = 0.85 * raw_threshold + 0.15 * 0.5
    shrunk_metrics = binary_metrics(target, score, shrunk_threshold)
    high_accuracy = max(
        rows,
        key=lambda row: (
            row["metrics"]["accuracy"]
            if row["metrics"]["balanced_accuracy"] >= 0.65
            else -1.0,
            row["metrics"]["balanced_accuracy"],
        ),
    )
    high_raw_threshold = float(high_accuracy["threshold"])
    high_shrunk_threshold = 0.90 * high_raw_threshold + 0.10 * 0.5
    return {
        "selection_scope": "current outer-training inner OOF only",
        "objective": (
            "0.65 balanced_accuracy + 0.20 min(recall,specificity) + "
            "0.15 accuracy - 0.10 recall/specificity gap"
        ),
        "raw_balanced_threshold": raw_threshold,
        "chosen_threshold": float(shrunk_threshold),
        "metrics_before_shrink": chosen["metrics"],
        "metrics_after_shrink": shrunk_metrics,
        "high_accuracy_operating_point": {
            "selection_scope": "current outer-training inner OOF only",
            "minimum_requested_inner_balanced_accuracy": 0.65,
            "raw_threshold": high_raw_threshold,
            "chosen_threshold": high_shrunk_threshold,
            "metrics_before_shrink": high_accuracy["metrics"],
            "metrics_after_shrink": binary_metrics(
                target, score, high_shrunk_threshold
            ),
        },
    }


@dataclass
class RankNormalizer:
    """Map scores to the mean percentile of inner-fold score distributions."""

    sorted_scores_: np.ndarray | None = None
    fold_sorted_scores_: list[np.ndarray] | None = None

    def fit(
        self,
        score: np.ndarray,
        fold_ids: np.ndarray | None = None,
    ) -> "RankNormalizer":
        values = np.asarray(score, dtype=np.float64)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError("RankNormalizer requires finite one-dimensional scores")
        self.sorted_scores_ = np.sort(values)
        if fold_ids is None:
            self.fold_sorted_scores_ = [self.sorted_scores_]
        else:
            folds = np.asarray(fold_ids, dtype=np.int64)
            if folds.shape != values.shape or np.any(folds < 0):
                raise ValueError("RankNormalizer fold IDs are invalid")
            self.fold_sorted_scores_ = [
                np.sort(values[folds == fold]) for fold in np.unique(folds)
            ]
            if any(len(distribution) == 0 for distribution in self.fold_sorted_scores_):
                raise AssertionError("Empty inner-fold score distribution")
        return self

    def transform(self, score: np.ndarray) -> np.ndarray:
        if self.sorted_scores_ is None or self.fold_sorted_scores_ is None:
            raise RuntimeError("RankNormalizer must be fitted first")
        values = np.asarray(score, dtype=np.float64)
        percentiles = [
            (
                np.searchsorted(distribution, values, side="left")
                + np.searchsorted(distribution, values, side="right")
                + 1.0
            )
            / (2.0 * (len(distribution) + 1.0))
            for distribution in self.fold_sorted_scores_
        ]
        return np.clip(
            np.mean(np.stack(percentiles), axis=0),
            1e-6,
            1 - 1e-6,
        )


def _weight_grid(names: Sequence[str], units: int = 4) -> list[dict[str, float]]:
    values: list[dict[str, float]] = []
    for cuts in itertools.combinations_with_replacement(range(len(names)), units):
        counts = Counter(cuts)
        values.append(
            {name: float(counts.get(index, 0) / units) for index, name in enumerate(names)}
        )
    return values


def _apply_orientation(score: np.ndarray, inverted: bool) -> np.ndarray:
    values = np.asarray(score, dtype=np.float64)
    return 1.0 - values if inverted else values


def select_ensemble(
    y: np.ndarray,
    raw_inner_oof: Mapping[str, np.ndarray],
    inner_fold_ids: np.ndarray,
) -> tuple[dict[str, RankNormalizer], dict[str, Any]]:
    """Quality-gated non-negative stacking selected only on inner OOF."""

    target = np.asarray(y, dtype=np.int64)
    fold_ids = np.asarray(inner_fold_ids, dtype=np.int64)
    unique_folds = np.unique(fold_ids)
    normalizers: dict[str, RankNormalizer] = {}
    normalized: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    qualities: dict[str, float] = {}
    eligible: list[str] = []
    for name in MODEL_NAMES:
        raw = np.asarray(raw_inner_oof[name], dtype=np.float64)
        fold_auc_raw = [
            _safe_auc(target[fold_ids == fold], raw[fold_ids == fold])
            for fold in unique_folds
        ]
        consistent_anti_signal = bool(
            all(np.isfinite(value) and value < 0.50 for value in fold_auc_raw)
            and np.mean([1.0 - value for value in fold_auc_raw]) >= 0.55
        )
        # Every adapter already verifies that class 1 means MCI+DEM.  A model
        # that is consistently backwards is rejected by the quality gate; its
        # direction is not learned as another tuning degree of freedom.
        consistent_inverse = False
        oriented = raw
        normalizer = RankNormalizer().fit(oriented, fold_ids)
        scaled = normalizer.transform(oriented)
        threshold = select_threshold(target, scaled)
        chosen_threshold = float(threshold["chosen_threshold"])
        fold_metrics = [
            binary_metrics(
                target[fold_ids == fold],
                scaled[fold_ids == fold],
                chosen_threshold,
            )
            for fold in unique_folds
        ]
        fold_bacc = np.asarray(
            [row["balanced_accuracy"] for row in fold_metrics], dtype=float
        )
        fold_auc = np.asarray([row["roc_auc"] for row in fold_metrics], dtype=float)
        global_metrics = binary_metrics(target, scaled, chosen_threshold)
        quality = float(
            0.65 * np.mean(fold_bacc)
            + 0.25 * np.nanmean(fold_auc)
            + 0.10
            * min(
                global_metrics["recall_impaired"],
                global_metrics["specificity_cn"],
            )
            - 0.35 * np.std(fold_bacc)
        )
        accepted = bool(
            np.nanmean(fold_auc) >= 0.51
            and np.mean(fold_bacc) >= 0.515
            and quality >= 0.50
        )
        if accepted:
            eligible.append(name)
        qualities[name] = quality
        normalizers[name] = normalizer
        normalized[name] = scaled
        diagnostics[name] = {
            "raw_fold_auc": fold_auc_raw,
            "orientation_inverted": consistent_inverse,
            "consistent_anti_signal_detected": consistent_anti_signal,
            "orientation_rule": (
                "class-1 orientation is fixed as MCI+DEM; anti-signal models "
                "are quality-gated out and never probability-inverted"
            ),
            "normalized_global_metrics": global_metrics,
            "normalized_fold_metrics": fold_metrics,
            "quality": quality,
            "quality_gate_passed": accepted,
        }
    passed_quality_gate = list(eligible)
    fallback_model = None
    if not eligible:
        fallback_model = max(MODEL_NAMES, key=lambda name: qualities[name])
        eligible = [fallback_model]
    # A four-model grid is already 35 candidates at quarter steps.  If all five
    # pass, retain the four most stable inner-OOF experts for the blend search.
    eligible = sorted(eligible, key=lambda name: (-qualities[name], name))[:4]
    candidates: list[dict[str, Any]] = []
    for partial_weights in _weight_grid(eligible):
        weights = {name: float(partial_weights.get(name, 0.0)) for name in MODEL_NAMES}
        weights["prior"] = 0.0
        risk = sum(weights[name] * normalized[name] for name in MODEL_NAMES)
        threshold = select_threshold(target, risk)
        chosen_threshold = float(threshold["chosen_threshold"])
        fold_metrics = [
            binary_metrics(
                target[fold_ids == fold],
                risk[fold_ids == fold],
                chosen_threshold,
            )
            for fold in unique_folds
        ]
        fold_bacc = np.asarray(
            [row["balanced_accuracy"] for row in fold_metrics], dtype=float
        )
        fold_accuracy = np.asarray([row["accuracy"] for row in fold_metrics])
        global_metrics = binary_metrics(target, risk, chosen_threshold)
        stable_score = float(
            0.65 * np.mean(fold_bacc)
            + 0.15 * np.mean(fold_accuracy)
            + 0.20
            * min(
                global_metrics["recall_impaired"],
                global_metrics["specificity_cn"],
            )
            - 0.35 * np.std(fold_bacc)
        )
        candidates.append(
            {
                "weights": weights,
                "stable_score": stable_score,
                "threshold": threshold,
                "global_metrics": global_metrics,
                "fold_metrics": fold_metrics,
            }
        )
    # A constant prior is an explicit safety candidate.  If every learned
    # branch is anti-informative, the honest result is a balanced-accuracy 0.5
    # baseline rather than forcing a harmful model into deployment.
    prior_weights = {name: 0.0 for name in MODEL_NAMES}
    prior_weights["prior"] = 1.0
    prior_risk = np.full(len(target), 0.5, dtype=np.float64)
    prior_threshold = select_threshold(target, prior_risk)
    prior_chosen_threshold = float(prior_threshold["chosen_threshold"])
    prior_fold_metrics = [
        binary_metrics(
            target[fold_ids == fold],
            prior_risk[fold_ids == fold],
            prior_chosen_threshold,
        )
        for fold in unique_folds
    ]
    prior_global = binary_metrics(
        target, prior_risk, prior_chosen_threshold
    )
    prior_fold_bacc = np.asarray(
        [row["balanced_accuracy"] for row in prior_fold_metrics], dtype=float
    )
    prior_fold_accuracy = np.asarray(
        [row["accuracy"] for row in prior_fold_metrics], dtype=float
    )
    candidates.append(
        {
            "weights": prior_weights,
            "stable_score": float(
                0.65 * np.mean(prior_fold_bacc)
                + 0.15 * np.mean(prior_fold_accuracy)
                + 0.20
                * min(
                    prior_global["recall_impaired"],
                    prior_global["specificity_cn"],
                )
                - 0.35 * np.std(prior_fold_bacc)
            ),
            "threshold": prior_threshold,
            "global_metrics": prior_global,
            "fold_metrics": prior_fold_metrics,
        }
    )
    chosen = max(
        candidates,
        key=lambda row: (
            row["stable_score"],
            row["global_metrics"]["balanced_accuracy"],
            row["global_metrics"]["accuracy"],
            -sum(weight > 0 for weight in row["weights"].values()),
            row["weights"].get("prior", 0.0),
        ),
    )
    return normalizers, {
        "selection_scope": "current outer-training inner subject OOF only",
        "model_diagnostics": diagnostics,
        "quality_gate_passed": passed_quality_gate,
        "eligible_for_weight_grid": eligible,
        "fallback_used": fallback_model is not None,
        "fallback_model": fallback_model,
        "all_model_weights_may_be_zero": True,
        "google_model_minimum_weight": 0.0,
        "weight_grid_step": 0.25,
        "candidate_count": len(candidates),
        "chosen": chosen,
        "top_candidates": sorted(
            candidates, key=lambda row: row["stable_score"], reverse=True
        )[:10],
    }


def apply_ensemble(
    raw_scores: Mapping[str, np.ndarray],
    normalizers: Mapping[str, RankNormalizer],
    selection: Mapping[str, Any],
) -> np.ndarray:
    diagnostics = selection["model_diagnostics"]
    weights = selection["chosen"]["weights"]
    risk = np.full_like(
        np.asarray(next(iter(raw_scores.values())), dtype=np.float64),
        0.5 * float(weights.get("prior", 0.0)),
    )
    for name in MODEL_NAMES:
        oriented = _apply_orientation(
            raw_scores[name], bool(diagnostics[name]["orientation_inverted"])
        )
        risk += float(weights[name]) * normalizers[name].transform(oriented)
    return np.clip(risk, 1e-6, 1 - 1e-6)


def _release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _ensure_time(deadline: float, *, reserve: float, step: str) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= reserve:
        raise SoftTimeBudgetExceeded(
            f"Insufficient soft runtime before {step}: "
            f"remaining={remaining:.1f}s, reserve={reserve:.1f}s"
        )


def _fit_all_models(
    pipeline: FoldFeaturePipeline,
    train_views: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    fast: bool,
    device_name: str,
    deadline: float,
    reserve: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    subject_X = pipeline.transform_subject(train_views)
    temporal_X = pipeline.transform_temporal(train_views)
    models: dict[str, Any] = {}
    for model_index, model_name in enumerate(MODEL_NAMES):
        _ensure_time(
            deadline,
            reserve=reserve,
            step=f"fit {model_name}",
        )
        models[model_name] = fit_model(
            model_name,
            subject_X=subject_X,
            temporal_X=temporal_X,
            y=y,
            subject_feature_names=pipeline.subject_feature_names,
            seed=seed + model_index * 10_007,
            fast=fast,
            device_name=device_name,
        )
    return models, subject_X, temporal_X


def _predict_all(
    models: Mapping[str, Any],
    pipeline: FoldFeaturePipeline,
    views: np.ndarray,
) -> dict[str, np.ndarray]:
    subject_X = pipeline.transform_subject(views)
    temporal_X = pipeline.transform_temporal(views)
    return {
        name: predict_model(
            name,
            models[name],
            subject_X=subject_X,
            temporal_X=temporal_X,
        )
        for name in MODEL_NAMES
    }


def collect_inner_oof(
    views: np.ndarray,
    y: np.ndarray,
    *,
    inner_folds: int,
    max_features: int,
    seed: int,
    fast: bool,
    device_name: str,
    deadline: float,
    reserve: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, Any]]]:
    target = np.asarray(y, dtype=np.int64)
    splitter = StratifiedKFold(
        n_splits=inner_folds, shuffle=True, random_state=int(seed)
    )
    raw = {name: np.zeros(len(target), dtype=np.float64) for name in MODEL_NAMES}
    seen = np.zeros(len(target), dtype=np.int8)
    fold_ids = np.full(len(target), -1, dtype=np.int64)
    manifests: list[dict[str, Any]] = []
    for fold_id, (fit_index, valid_index) in enumerate(splitter.split(views, target)):
        pipeline = FoldFeaturePipeline(
            max_features=max_features,
            seed=seed + fold_id * 193,
        ).fit(views[fit_index], target[fit_index], COMPACT_DAILY_FEATURES)
        models: dict[str, Any] = {}
        try:
            models, _, _ = _fit_all_models(
                pipeline,
                views[fit_index],
                target[fit_index],
                seed=seed + fold_id * 1009,
                fast=fast,
                device_name=device_name,
                deadline=deadline,
                reserve=reserve,
            )
            predictions = _predict_all(models, pipeline, views[valid_index])
            for name in MODEL_NAMES:
                raw[name][valid_index] = predictions[name]
            seen[valid_index] += 1
            fold_ids[valid_index] = fold_id
            manifests.append(
                {
                    "fold": fold_id,
                    "fit_subjects": len(fit_index),
                    "valid_subjects": len(valid_index),
                    "feature_pipeline": pipeline.manifest(),
                }
            )
        finally:
            models.clear()
            _release_memory()
    if not np.all(seen == 1) or np.any(fold_ids < 0):
        raise AssertionError("Inner OOF subject coverage failed")
    return raw, fold_ids, manifests


def _checkpoint_files(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.name not in {"checkpoint_manifest.json", "CHECKPOINT_COMPLETE.json"}
    }


def _save_model(model_name: str, model: Any, root: Path) -> str:
    if model_name == "elastic_net":
        relative = "elastic_net/model.joblib"
        model.save(root / relative)
    elif model_name == "ydf_subject":
        relative = "ydf_subject"
        model.save(root / relative)
    elif model_name == "ydf_daily":
        relative = "ydf_daily"
        model.save(root / relative)
    elif model_name == "tabnet":
        base = root / "tabnet/model"
        stored = model.save(base)
        relative = str(stored.relative_to(root))
    elif model_name == "transformer":
        relative = "transformer/model.pt"
        model.save(root / relative)
    else:
        raise ValueError(model_name)
    return relative


def _load_model(model_name: str, root: Path, relative: str) -> Any:
    path = root / relative
    if model_name == "elastic_net":
        return ElasticNetAdapter.load(path)
    if model_name == "ydf_subject":
        return YDFSubjectAdapter.load(path)
    if model_name == "ydf_daily":
        return YDFDailyAdapter.load(path)
    if model_name == "tabnet":
        return TabNetAdapter.load(path)
    if model_name == "transformer":
        return TransformerAdapter.load(path)
    raise ValueError(model_name)


def save_checkpoint_bundle(
    path: Path,
    *,
    pipeline: FoldFeaturePipeline,
    models: Mapping[str, Any],
    normalizers: Mapping[str, RankNormalizer],
    selection: Mapping[str, Any],
    probe_views: np.ndarray,
) -> dict[str, Any]:
    """Atomically save and CPU-reload every branch before marking complete."""

    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Stale partial checkpoint: {temporary}")
    temporary.mkdir(parents=True, exist_ok=False)
    joblib.dump(pipeline, temporary / "feature_pipeline.joblib")
    joblib.dump(dict(normalizers), temporary / "rank_normalizers.joblib")
    write_json(temporary / "ensemble_selection.json", selection)
    stored_paths = {
        name: _save_model(name, models[name], temporary) for name in MODEL_NAMES
    }
    write_json(temporary / "model_paths.json", stored_paths)
    reloaded_pipeline = joblib.load(temporary / "feature_pipeline.joblib")
    reloaded_normalizers = joblib.load(temporary / "rank_normalizers.joblib")
    reloaded_selection = json.loads(
        (temporary / "ensemble_selection.json").read_text(encoding="utf-8")
    )
    reloaded_paths = json.loads(
        (temporary / "model_paths.json").read_text(encoding="utf-8")
    )
    # TransformerAdapter.save has already moved its parameters to CPU.  Keep
    # the original probe on CPU as well so the reload check measures storage,
    # not expected bfloat16-vs-float32 device arithmetic differences.
    models["transformer"].device = "cpu"
    import torch

    original_tabnet = models["tabnet"].model
    original_tabnet.device = torch.device("cpu")
    original_tabnet.network.to("cpu")
    original_tabnet.pin_memory = False
    original_raw = _predict_all(models, pipeline, probe_views)
    reloaded_models = {
        name: _load_model(name, temporary, reloaded_paths[name])
        for name in MODEL_NAMES
    }
    reloaded_raw = _predict_all(reloaded_models, reloaded_pipeline, probe_views)
    checks: dict[str, Any] = {}
    for name in MODEL_NAMES:
        maximum_error = float(
            np.max(np.abs(original_raw[name] - reloaded_raw[name]), initial=0.0)
        )
        if name == "tabnet":
            relative_tolerance, absolute_tolerance = 2e-4, 2e-5
        elif name == "transformer":
            relative_tolerance, absolute_tolerance = 1e-5, 2e-6
        else:
            relative_tolerance, absolute_tolerance = 1e-8, 1e-8
        if not np.allclose(
            original_raw[name],
            reloaded_raw[name],
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        ):
            raise AssertionError(
                f"{name} checkpoint roundtrip mismatch; max error={maximum_error}"
            )
        checks[name] = {
            "maximum_absolute_probability_error": maximum_error,
            "relative_tolerance": relative_tolerance,
            "absolute_tolerance": absolute_tolerance,
            "verified": True,
        }
    original_risk = apply_ensemble(original_raw, normalizers, selection)
    reloaded_risk = apply_ensemble(
        reloaded_raw, reloaded_normalizers, reloaded_selection
    )
    ensemble_error = float(np.max(np.abs(original_risk - reloaded_risk), initial=0.0))
    if not np.allclose(
        original_risk, reloaded_risk, rtol=2e-4, atol=2e-5
    ):
        raise AssertionError("Reloaded ensemble risk differs from original")
    verification = {
        "roundtrip_verified": True,
        "models": checks,
        "ensemble_maximum_absolute_error": ensemble_error,
    }
    write_json(temporary / "roundtrip_verification.json", verification)
    files = _checkpoint_files(temporary)
    manifest = {
        "format": "balanced-fusion-checkpoint-v1",
        "models": list(MODEL_NAMES),
        "stored_model_paths": stored_paths,
        "files_before_manifest": files,
    }
    write_json(temporary / "checkpoint_manifest.json", manifest)
    manifest_sha = sha256_file(temporary / "checkpoint_manifest.json")
    write_json(
        temporary / "CHECKPOINT_COMPLETE.json",
        {
            "complete": True,
            "roundtrip_verified": True,
            "manifest_sha256": manifest_sha,
        },
    )
    temporary.replace(path)
    reloaded_models.clear()
    _release_memory()
    return {
        "bundle": path.name,
        "complete": True,
        "roundtrip_verified": True,
        "manifest_sha256": manifest_sha,
        "files": len(files) + 2,
    }


def verify_checkpoint_tree(root: Path, expected_bundles: int) -> dict[str, Any]:
    partial = sorted(path.name for path in root.glob("*.partial"))
    if partial:
        raise AssertionError(f"Partial checkpoint bundles remain: {partial}")
    bundles = sorted(path for path in root.glob("repeat_*_fold_*") if path.is_dir())
    if len(bundles) != expected_bundles:
        raise AssertionError(
            f"Expected {expected_bundles} outer bundles, found {len(bundles)}"
        )
    rows = []
    for bundle in bundles:
        complete_path = bundle / "CHECKPOINT_COMPLETE.json"
        manifest_path = bundle / "checkpoint_manifest.json"
        if not complete_path.is_file() or not manifest_path.is_file():
            raise AssertionError(f"Incomplete checkpoint bundle: {bundle}")
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("complete") is not True or complete.get(
            "roundtrip_verified"
        ) is not True:
            raise AssertionError(f"Invalid completion marker: {bundle}")
        if complete.get("manifest_sha256") != sha256_file(manifest_path):
            raise AssertionError(f"Manifest hash mismatch: {bundle}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative, expected in manifest["files_before_manifest"].items():
            stored = bundle / relative
            if not stored.is_file():
                raise AssertionError(f"Missing checkpoint file: {stored}")
            if stored.stat().st_size != int(expected["bytes"]):
                raise AssertionError(f"Checkpoint size mismatch: {stored}")
            if sha256_file(stored) != expected["sha256"]:
                raise AssertionError(f"Checkpoint hash mismatch: {stored}")
        rows.append(
            {
                "bundle": bundle.name,
                "manifest_sha256": complete["manifest_sha256"],
                "roundtrip_verified": True,
            }
        )
    return {
        "verified": True,
        "bundle_count": len(rows),
        "expected_bundle_count": expected_bundles,
        "bundles": rows,
    }


def bootstrap_intervals(
    y: np.ndarray,
    risk: np.ndarray,
    threshold: float,
    *,
    seed: int,
    rounds: int = 2000,
) -> dict[str, Any]:
    target = np.asarray(y, dtype=np.int64)
    score = np.asarray(risk, dtype=np.float64)
    class_indices = [np.flatnonzero(target == class_id) for class_id in (0, 1)]
    rng = np.random.default_rng(seed)
    values = {
        name: []
        for name in ("accuracy", "balanced_accuracy", "recall_impaired", "specificity_cn", "roc_auc")
    }
    for _ in range(rounds):
        sampled = np.concatenate(
            [rng.choice(index, size=len(index), replace=True) for index in class_indices]
        )
        metrics = binary_metrics(target[sampled], score[sampled], threshold)
        for name in values:
            values[name].append(metrics[name])
    return {
        name: {
            "lower_95": float(np.quantile(rows, 0.025)),
            "median": float(np.quantile(rows, 0.5)),
            "upper_95": float(np.quantile(rows, 0.975)),
        }
        for name, rows in values.items()
    }


def feature_selection_stability(
    selected_by_outer_fold: Sequence[Sequence[str]],
) -> dict[str, Any]:
    sets = [set(map(str, names)) for names in selected_by_outer_fold]
    if not sets:
        raise ValueError("At least one outer feature set is required")
    frequencies = Counter(name for names in sets for name in names)
    jaccard = [
        len(left & right) / max(1, len(left | right))
        for index, left in enumerate(sets)
        for right in sets[index + 1 :]
    ]
    return {
        "outer_fold_sets": len(sets),
        "mean_selected_features": float(np.mean([len(names) for names in sets])),
        "mean_pairwise_jaccard": float(np.mean(jaccard)) if jaccard else 1.0,
        "minimum_pairwise_jaccard": float(np.min(jaccard)) if jaccard else 1.0,
        "features_selected_in_every_outer_fold": sorted(
            name for name, count in frequencies.items() if count == len(sets)
        ),
        "selection_frequency": [
            {"feature": name, "outer_folds": int(count)}
            for name, count in sorted(
                frequencies.items(), key=lambda item: (-item[1], item[0])
            )
        ],
    }


def runtime_info() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    for distribution in (
        "numpy",
        "pandas",
        "scikit-learn",
        "pytorch-tabnet",
        "ydf",
        "torch",
    ):
        try:
            payload[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            payload[distribution] = None
    try:
        import torch

        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["cuda_device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except ImportError:
        payload["cuda_available"] = False
    return payload


def code_manifest() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    files = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(root.glob("*.py"))
    }
    shared = {}
    repository = root.parents[1]
    for relative in (
        "SangHyo/Binary_Wearable_TabNet_Google/data.py",
        "SangHyo/Binary_Wearable_SequenceFusion_Google/data.py",
        "SangHyo/Binary_Wearable_SequenceFusion_Google/eda.py",
    ):
        path = repository / relative
        if not path.is_file():
            raise FileNotFoundError(f"Shared audited dependency missing: {path}")
        shared[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return {
        "experiment_root": str(root),
        "files": files,
        "transitive_shared_dependencies": shared,
    }


def run_experiment(config: RunConfig) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + float(config.max_runtime_seconds)
    fast = config.run_mode == "smoke"
    device_name = "cuda" if not fast else "cpu"
    reserve = 60.0 if fast else 900.0
    output = Path(config.output_dir).expanduser().resolve()
    training_output = output / "training"
    checkpoints = training_output / "checkpoints"
    training_output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    write_json(training_output / "run_config.json", asdict(config))
    write_json(training_output / "environment.json", runtime_info())
    write_json(training_output / "code_manifest.json", code_manifest())
    set_global_seed(config.seed)

    training_dataset = build_subject_dataset(
        config.training_root, require_labels=True, expected_split="train"
    )
    validation_dataset = build_subject_dataset(
        config.validation_root, require_labels=False, expected_split="val"
    )
    if training_dataset.y is None:
        raise AssertionError("Training labels were not loaded")
    if tuple(training_dataset.feature_names) != tuple(validation_dataset.feature_names):
        raise AssertionError("Training/Validation compact schemas differ")
    assert_disjoint_subjects(
        training_dataset.subject_ids, validation_dataset.subject_ids
    )
    write_json(training_output / "training_data_audit.json", training_dataset.audit)
    write_json(
        training_output / "validation_data_audit_label_free.json",
        validation_dataset.audit,
    )
    run_eda(training_dataset, output / "eda")
    train_views = make_fixed_views(training_dataset)
    validation_views = make_fixed_views(validation_dataset)
    y = np.asarray(training_dataset.y, dtype=np.int64)

    oof_margin_sum = np.zeros(len(y), dtype=np.float64)
    oof_high_accuracy_margin_sum = np.zeros(len(y), dtype=np.float64)
    oof_risk_sum = np.zeros(len(y), dtype=np.float64)
    oof_count = np.zeros(len(y), dtype=np.int8)
    validation_margins: list[np.ndarray] = []
    validation_high_accuracy_margins: list[np.ndarray] = []
    outer_rows: list[dict[str, Any]] = []
    selected_weight_rows: list[dict[str, float]] = []
    selected_feature_rows: list[list[str]] = []
    for repeat in range(config.outer_repeats):
        outer = StratifiedKFold(
            n_splits=config.outer_folds,
            shuffle=True,
            random_state=config.seed + repeat * 10_007,
        )
        for fold, (fit_index, test_index) in enumerate(outer.split(train_views, y)):
            fold_seed = config.seed + repeat * 100_003 + fold * 1009
            _ensure_time(
                deadline,
                reserve=reserve,
                step=f"outer repeat={repeat} fold={fold}",
            )
            inner_raw, inner_fold_ids, inner_manifests = collect_inner_oof(
                train_views[fit_index],
                y[fit_index],
                inner_folds=config.inner_folds,
                max_features=config.max_features,
                seed=fold_seed,
                fast=fast,
                device_name=device_name,
                deadline=deadline,
                reserve=reserve,
            )
            normalizers, selection = select_ensemble(
                y[fit_index], inner_raw, inner_fold_ids
            )
            pipeline = FoldFeaturePipeline(
                max_features=config.max_features,
                seed=fold_seed,
            ).fit(train_views[fit_index], y[fit_index], COMPACT_DAILY_FEATURES)
            models: dict[str, Any] = {}
            try:
                models, _, _ = _fit_all_models(
                    pipeline,
                    train_views[fit_index],
                    y[fit_index],
                    seed=fold_seed + 500_009,
                    fast=fast,
                    device_name=device_name,
                    deadline=deadline,
                    reserve=reserve,
                )
                test_raw = _predict_all(models, pipeline, train_views[test_index])
                validation_raw = _predict_all(
                    models, pipeline, validation_views
                )
                test_risk = apply_ensemble(test_raw, normalizers, selection)
                validation_risk = apply_ensemble(
                    validation_raw, normalizers, selection
                )
                threshold = float(
                    selection["chosen"]["threshold"]["chosen_threshold"]
                )
                high_accuracy_threshold = float(
                    selection["chosen"]["threshold"][
                        "high_accuracy_operating_point"
                    ]["chosen_threshold"]
                )
                test_margin = test_risk - threshold
                validation_margin = validation_risk - threshold
                test_high_accuracy_margin = test_risk - high_accuracy_threshold
                validation_high_accuracy_margin = (
                    validation_risk - high_accuracy_threshold
                )
                oof_margin_sum[test_index] += test_margin
                oof_high_accuracy_margin_sum[test_index] += (
                    test_high_accuracy_margin
                )
                oof_risk_sum[test_index] += test_risk
                oof_count[test_index] += 1
                validation_margins.append(validation_margin)
                validation_high_accuracy_margins.append(
                    validation_high_accuracy_margin
                )
                fold_metrics = binary_metrics(y[test_index], test_margin, 0.0)
                high_accuracy_fold_metrics = binary_metrics(
                    y[test_index], test_high_accuracy_margin, 0.0
                )
                bundle_name = f"repeat_{repeat:02d}_fold_{fold:02d}"
                bundle = save_checkpoint_bundle(
                    checkpoints / bundle_name,
                    pipeline=pipeline,
                    models=models,
                    normalizers=normalizers,
                    selection=selection,
                    probe_views=train_views[test_index[: min(4, len(test_index))]],
                )
                chosen_weights = {
                    name: float(selection["chosen"]["weights"][name])
                    for name in MODEL_NAMES
                }
                chosen_weights["prior"] = float(
                    selection["chosen"]["weights"].get("prior", 0.0)
                )
                selected_weight_rows.append(chosen_weights)
                selected_feature_rows.append(
                    list(pipeline.selector_.selected_feature_names_)
                )
                outer_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "fit_subjects": len(fit_index),
                        "test_subjects": len(test_index),
                        "threshold": threshold,
                        "high_accuracy_threshold": high_accuracy_threshold,
                        **{
                            f"weight_{name}": chosen_weights[name]
                            for name in (*MODEL_NAMES, "prior")
                        },
                        **{
                            f"outer_{key}": fold_metrics[key]
                            for key in (
                                "accuracy",
                                "balanced_accuracy",
                                "recall_impaired",
                                "specificity_cn",
                                "roc_auc",
                            )
                        },
                        **{
                            f"outer_high_accuracy_{key}": (
                                high_accuracy_fold_metrics[key]
                            )
                            for key in (
                                "accuracy",
                                "balanced_accuracy",
                                "recall_impaired",
                                "specificity_cn",
                                "roc_auc",
                            )
                        },
                        "selected_features": len(
                            pipeline.selector_.selected_feature_names_
                        ),
                        "feature_pipeline": pipeline.manifest(),
                        "inner_feature_manifests": inner_manifests,
                        "selection": selection,
                        "checkpoint": bundle,
                    }
                )
                write_json(
                    training_output / "nested_progress.json",
                    {
                        "completed_outer_folds": len(outer_rows),
                        "expected_outer_folds": (
                            config.outer_repeats * config.outer_folds
                        ),
                        "last_bundle": bundle_name,
                    },
                )
            finally:
                models.clear()
                _release_memory()

    expected_folds = config.outer_folds * config.outer_repeats
    if not np.all(oof_count == config.outer_repeats):
        raise AssertionError(f"Repeated OOF coverage failed: {oof_count.tolist()}")
    if len(validation_margins) != expected_folds:
        raise AssertionError("Historical Validation cross-fold coverage failed")
    if len(validation_high_accuracy_margins) != expected_folds:
        raise AssertionError("High-accuracy Validation cross-fold coverage failed")
    oof_margin = oof_margin_sum / oof_count
    oof_high_accuracy_margin = oof_high_accuracy_margin_sum / oof_count
    oof_risk = oof_risk_sum / oof_count
    oof_metrics = binary_metrics(y, oof_margin, 0.0)
    oof_high_accuracy_metrics = binary_metrics(
        y, oof_high_accuracy_margin, 0.0
    )
    oof_fixed_half = binary_metrics(y, oof_risk, 0.5)
    oof_ci = bootstrap_intervals(
        y, oof_margin, 0.0, seed=config.seed + 700_001
    )
    oof_frame = pd.DataFrame(
        {
            "subject_hash": [
                subject_hash(value) for value in training_dataset.subject_ids
            ],
            "repeat_averaged_risk": oof_risk,
            "repeat_averaged_margin": oof_margin,
            "prediction": (oof_margin >= 0.0).astype(int),
            "repeat_averaged_high_accuracy_margin": oof_high_accuracy_margin,
            "high_accuracy_prediction": (
                oof_high_accuracy_margin >= 0.0
            ).astype(int),
            "target": y,
        }
    )
    oof_frame.to_csv(
        training_output / "nested_oof_predictions_hashed.csv", index=False
    )
    pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if not isinstance(value, (dict, list))
            }
            for row in outer_rows
        ]
    ).to_csv(training_output / "outer_fold_metrics.csv", index=False)
    write_json(
        training_output / "nested_cv_report.json",
        {
            "primary_operating_point": (
                "repeat-averaged outer-fold risk margin; each fold threshold "
                "selected only from that outer-training inner OOF"
            ),
            "primary_metrics": oof_metrics,
            "secondary_high_accuracy_operating_point": {
                "metrics": oof_high_accuracy_metrics,
                "target_check": _target_check(oof_high_accuracy_metrics),
                "confidence_intervals": bootstrap_intervals(
                    y,
                    oof_high_accuracy_margin,
                    0.0,
                    seed=config.seed + 700_002,
                ),
            },
            "fixed_rank_score_0_5_diagnostic": oof_fixed_half,
            "confidence_intervals": oof_ci,
            "all_cn_baseline": _all_cn_baseline(y),
            "target_check": _target_check(oof_metrics),
            "outer_folds": outer_rows,
        },
    )
    stability_report = feature_selection_stability(selected_feature_rows)
    write_json(
        training_output / "feature_selection_stability.json",
        stability_report,
    )
    checkpoint_index = verify_checkpoint_tree(checkpoints, expected_folds)
    write_json(training_output / "checkpoint_index.json", checkpoint_index)
    write_json(
        training_output / "ENSEMBLE_DEPLOYMENT.json",
        {
            "primary_deployable_model": (
                "average the normalized risk margins from all verified outer "
                "checkpoint bundles and classify at margin >= 0"
            ),
            "secondary_operating_point": (
                "the same bundles also store an inner-OOF-selected "
                "high-accuracy threshold; average those margins separately"
            ),
            "full_refit_intentionally_not_used": True,
            "reason": (
                "a full-data refit has no inner-OOF scale/threshold matching its "
                "new model and was unstable in the prior experiment"
            ),
            "checkpoint_bundles": expected_folds,
        },
    )

    validation_margin = np.mean(np.stack(validation_margins), axis=0)
    validation_high_accuracy_margin = np.mean(
        np.stack(validation_high_accuracy_margins), axis=0
    )
    frozen_frame = pd.DataFrame(
        {
            "subject_hash": [
                subject_hash(value) for value in validation_dataset.subject_ids
            ],
            "crossfold_risk_margin": validation_margin,
            "prediction": (validation_margin >= 0.0).astype(int),
            "crossfold_high_accuracy_margin": validation_high_accuracy_margin,
            "high_accuracy_prediction": (
                validation_high_accuracy_margin >= 0.0
            ).astype(int),
        }
    )
    frozen_path = training_output / "validation_predictions_label_free_hashed.csv"
    frozen_frame.to_csv(frozen_path, index=False)
    frozen_sha = sha256_file(frozen_path)
    write_json(
        training_output / "VALIDATION_PREDICTIONS_FROZEN.json",
        {
            "labels_loaded_before_freeze": False,
            "path": frozen_path.name,
            "sha256": frozen_sha,
            "subjects": len(frozen_frame),
            "checkpoint_bundles": expected_folds,
        },
    )
    if sha256_file(frozen_path) != frozen_sha:
        raise AssertionError("Frozen Validation prediction hash changed")

    validation_report: dict[str, Any] | None = None
    if config.evaluate_historical_validation:
        # Do not move this label load above the prediction CSV and SHA freeze.
        validation_labels = load_validation_labels_checked(
            config.validation_root, validation_dataset.subject_ids
        )
        if sha256_file(frozen_path) != frozen_sha:
            raise AssertionError("Frozen Validation predictions changed after labels")
        validation_metrics = binary_metrics(
            validation_labels, validation_margin, 0.0
        )
        validation_high_accuracy_metrics = binary_metrics(
            validation_labels, validation_high_accuracy_margin, 0.0
        )
        validation_report = {
            "warning": HISTORICAL_VALIDATION_WARNING,
            "primary_crossfold_metrics": validation_metrics,
            "secondary_high_accuracy_crossfold": {
                "metrics": validation_high_accuracy_metrics,
                "confidence_intervals": bootstrap_intervals(
                    validation_labels,
                    validation_high_accuracy_margin,
                    0.0,
                    seed=config.seed + 900_002,
                ),
                "target_check": _target_check(
                    validation_high_accuracy_metrics
                ),
            },
            "confidence_intervals": bootstrap_intervals(
                validation_labels,
                validation_margin,
                0.0,
                seed=config.seed + 900_001,
            ),
            "all_cn_baseline": _all_cn_baseline(validation_labels),
            "target_check": _target_check(validation_metrics),
            "label_loading_order_verified": True,
            "frozen_prediction_sha256": frozen_sha,
        }
        write_json(training_output / "validation_report.json", validation_report)

    mean_weights = {
        name: float(np.mean([row[name] for row in selected_weight_rows]))
        for name in (*MODEL_NAMES, "prior")
    }
    selected_counts = {
        name: int(sum(row[name] > 0 for row in selected_weight_rows))
        for name in (*MODEL_NAMES, "prior")
    }
    final_report = {
        "experiment": "Binary_Wearable_BalancedFusion_Google",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_mode": config.run_mode,
        "smoke": fast,
        "elapsed_seconds": time.monotonic() - started,
        "task": "CN vs MCI+DEM",
        "modalities": ["Activity", "Sleep"],
        "mmse_or_cognitive_features": 0,
        "compact_daily_channels": len(COMPACT_DAILY_FEATURES),
        "models_trained_in_every_fold": list(MODEL_NAMES),
        "google_models": GOOGLE_MODELS,
        "model_quality_gate": (
            "each model may receive zero weight; no Google-model minimum weight"
        ),
        "mean_selected_weights": mean_weights,
        "outer_folds_with_positive_weight": selected_counts,
        "feature_selection_stability": stability_report,
        "nested_oof": {
            "primary_metrics": oof_metrics,
            "secondary_high_accuracy_metrics": oof_high_accuracy_metrics,
            "confidence_intervals": oof_ci,
            "all_cn_baseline": _all_cn_baseline(y),
            "target_check": _target_check(oof_metrics),
            "secondary_target_check": _target_check(
                oof_high_accuracy_metrics
            ),
        },
        "historical_validation": validation_report,
        "checkpoints": checkpoint_index,
        "result_interpretation": (
            "Judge generalization by repeated nested OOF first. Historical "
            "Validation is secondary and too small to prove the target."
        ),
    }
    write_json(training_output / "FINAL_REPORT.json", final_report)
    write_json(
        training_output / "TRAINING_COMPLETE.json",
        {
            "complete": True,
            "final_report_sha256": sha256_file(
                training_output / "FINAL_REPORT.json"
            ),
            "checkpoint_index_sha256": sha256_file(
                training_output / "checkpoint_index.json"
            ),
            "smoke": fast,
        },
    )
    return final_report


__all__ = [
    "RankNormalizer",
    "RunConfig",
    "apply_ensemble",
    "binary_metrics",
    "run_experiment",
    "select_ensemble",
    "select_threshold",
    "verify_checkpoint_tree",
]
