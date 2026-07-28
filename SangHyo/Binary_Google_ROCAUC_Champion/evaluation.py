"""Repeated nested subject-CV and conservative ROC-AUC ensemble selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .features import ChampionDataset
from .metrics import (
    paired_bootstrap_auc_difference,
    summarize_repeated_oof,
)
from .models import (
    FoldLocalTableModel,
    MMSEMaxAUCAnchor,
    ReferenceECDF,
    SequenceTransformerBranch,
)


@dataclass(frozen=True)
class NestedCVConfig:
    outer_splits: int = 5
    outer_repeats: int = 5
    inner_splits: int = 4
    inner_repeats: int = 2
    seed: int = 20260728
    blend_min_auc_gain: float = 0.005
    blend_min_win_fraction: float = 0.50
    max_selected_branches: int = 3
    bootstrap_resamples: int = 5000
    include_tabpfn: bool = False

    def validate(self) -> None:
        if self.outer_splits != 5:
            raise ValueError("The predeclared outer protocol requires 5 folds")
        if self.outer_repeats not in {5, 10}:
            raise ValueError("outer_repeats must be the full 5 or max 10 profile")
        if self.inner_splits != 4 or self.inner_repeats != 2:
            raise ValueError("The predeclared inner protocol is 4 folds x 2 repeats")
        if not 0.0 <= self.blend_min_auc_gain <= 0.1:
            raise ValueError("blend_min_auc_gain is outside the audited range")
        if not 0.5 <= self.blend_min_win_fraction < 1.0:
            raise ValueError("blend_min_win_fraction must be at least one half")
        if not 1 <= self.max_selected_branches <= 3:
            raise ValueError("At most three branches may be selected")
        if self.bootstrap_resamples < 1000:
            raise ValueError("At least 1000 subject bootstrap resamples are required")


@dataclass(frozen=True)
class BranchSpec:
    name: str
    kind: str
    view: str
    top_k: int | None = None

    @property
    def is_anchor(self) -> bool:
        return self.kind in {"mmse_anchor", "sequence_transformer"}


def branch_library(track: str, *, include_tabpfn: bool) -> tuple[BranchSpec, ...]:
    """Return a small, predeclared candidate library in a stable order."""

    if track == "mmse":
        specs = [
            BranchSpec("mmse_maxauc_anchor", "mmse_anchor", "mmse_anchor"),
            BranchSpec("fusion_elastic_top25", "elastic", "fusion_all", 25),
            BranchSpec("fusion_rbf_top25", "rbf_svm", "fusion_all", 25),
            BranchSpec(
                "fusion_catboost_top25", "catboost", "fusion_all", 25
            ),
        ]
        if include_tabpfn:
            specs.append(
                BranchSpec("fusion_tabpfn_top64", "tabpfn", "fusion_all", 64)
            )
        return tuple(specs)
    if track == "wearable":
        specs = [
            BranchSpec(
                "sequence_transformer_anchor",
                "sequence_transformer",
                "wearable_all",
            ),
            BranchSpec(
                "wearable_core_ridge",
                "ridge",
                "wearable_core",
                None,
            ),
            BranchSpec(
                "wearable_elastic_top25",
                "elastic",
                "wearable_all",
                25,
            ),
            BranchSpec(
                "wearable_rbf_top25",
                "rbf_svm",
                "wearable_all",
                25,
            ),
            BranchSpec(
                "wearable_catboost_top25",
                "catboost",
                "wearable_all",
                25,
            ),
        ]
        if include_tabpfn:
            specs.append(
                BranchSpec(
                    "wearable_tabpfn_top64", "tabpfn", "wearable_all", 64
                )
            )
        return tuple(specs)
    raise ValueError(f"Unknown track {track!r}")


def _branch_seed(base_seed: int, spec_name: str, stage: int) -> int:
    digest = hashlib.sha256(spec_name.encode("utf-8")).digest()
    name_offset = int.from_bytes(digest[:4], "little")
    return int((int(base_seed) + name_offset + 104729 * int(stage)) % (2**32 - 1))


def _repeated_splits(
    y: np.ndarray,
    *,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> Iterable[tuple[int, int, np.ndarray, np.ndarray]]:
    target = np.asarray(y, dtype=np.int64)
    for repeat in range(int(n_repeats)):
        splitter = StratifiedKFold(
            n_splits=int(n_splits),
            shuffle=True,
            random_state=int(seed) + repeat * 1009,
        )
        for fold, (train_index, test_index) in enumerate(
            splitter.split(np.zeros((len(target), 1)), target)
        ):
            yield repeat, fold, train_index, test_index


def _fit_branch(
    spec: BranchSpec,
    dataset: ChampionDataset,
    train_index: np.ndarray,
    *,
    seed: int,
) -> Any:
    if dataset.y is None:
        raise ValueError("Training requires labels")
    target = np.asarray(dataset.y)[train_index]
    if spec.kind == "sequence_transformer":
        model = SequenceTransformerBranch(seed=seed)
        return model.fit(
            [dataset.sequences[index] for index in train_index],
            target,
            dataset.sequence_feature_names,
            fast=False,
        )
    values = dataset.view_matrix(spec.view)[train_index]
    names = tuple(dataset.feature_names[index] for index in dataset.view_indices(spec.view))
    if spec.kind == "mmse_anchor":
        return MMSEMaxAUCAnchor(seed=seed).fit(values, target, names)
    return FoldLocalTableModel(
        kind=spec.kind,
        seed=seed,
        top_k=spec.top_k,
    ).fit(values, target, names)


def _score_branch(
    spec: BranchSpec,
    model: Any,
    dataset: ChampionDataset,
    test_index: np.ndarray,
) -> np.ndarray:
    if spec.kind == "sequence_transformer":
        score = model.score([dataset.sequences[index] for index in test_index])
    else:
        score = model.score(dataset.view_matrix(spec.view)[test_index])
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    if values.shape != (len(test_index),) or not np.isfinite(values).all():
        raise FloatingPointError(f"{spec.name} emitted invalid subject scores")
    return values


@dataclass
class InnerOOF:
    score_matrix: np.ndarray
    split_indices: tuple[tuple[int, int, tuple[int, ...]], ...]
    reference: ReferenceECDF
    aggregate_auc: float

    @property
    def aggregate_score(self) -> np.ndarray:
        return np.mean(self.score_matrix, axis=0)


def _inner_oof(
    spec: BranchSpec,
    dataset: ChampionDataset,
    outer_train_index: np.ndarray,
    config: NestedCVConfig,
    *,
    seed: int,
) -> InnerOOF:
    if dataset.y is None:
        raise ValueError("Nested CV requires labels")
    local_target = np.asarray(dataset.y)[outer_train_index]
    scores = np.full(
        (config.inner_repeats, len(outer_train_index)), np.nan, dtype=np.float64
    )
    split_indices: list[tuple[int, int, tuple[int, ...]]] = []
    for repeat, fold, local_train, local_test in _repeated_splits(
        local_target,
        n_splits=config.inner_splits,
        n_repeats=config.inner_repeats,
        seed=seed,
    ):
        global_train = outer_train_index[local_train]
        global_test = outer_train_index[local_test]
        stage = repeat * config.inner_splits + fold
        model = _fit_branch(
            spec,
            dataset,
            global_train,
            seed=_branch_seed(seed, spec.name, stage),
        )
        scores[repeat, local_test] = _score_branch(
            spec, model, dataset, global_test
        )
        split_indices.append((repeat, fold, tuple(map(int, local_test))))
    if not np.isfinite(scores).all():
        raise AssertionError(f"Incomplete inner OOF matrix for {spec.name}")
    aggregate = np.mean(scores, axis=0)
    reference = ReferenceECDF.fit(aggregate)
    return InnerOOF(
        score_matrix=scores,
        split_indices=tuple(split_indices),
        reference=reference,
        aggregate_auc=float(roc_auc_score(local_target, aggregate)),
    )


def _mean_normalized(
    selected: Sequence[str],
    inner: dict[str, InnerOOF],
) -> np.ndarray:
    normalized = [
        inner[name].reference.transform(inner[name].score_matrix)
        for name in selected
    ]
    return np.mean(np.stack(normalized, axis=0), axis=0)


def _fold_win_fraction(
    y: np.ndarray,
    current: np.ndarray,
    proposed: np.ndarray,
    split_indices: Sequence[tuple[int, int, tuple[int, ...]]],
) -> tuple[float, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    wins = 0
    for repeat, fold, test_tuple in split_indices:
        test_index = np.asarray(test_tuple, dtype=np.int64)
        current_auc = float(roc_auc_score(y[test_index], current[repeat, test_index]))
        proposed_auc = float(
            roc_auc_score(y[test_index], proposed[repeat, test_index])
        )
        won = proposed_auc > current_auc
        wins += int(won)
        records.append(
            {
                "repeat": int(repeat),
                "fold": int(fold),
                "anchor_or_current_auc": current_auc,
                "proposed_auc": proposed_auc,
                "delta": float(proposed_auc - current_auc),
                "strict_win": bool(won),
            }
        )
    return float(wins / len(records)), records


def _select_branches(
    specs: Sequence[BranchSpec],
    inner: dict[str, InnerOOF],
    y: np.ndarray,
    config: NestedCVConfig,
) -> tuple[list[str], dict[str, Any], np.ndarray]:
    if not specs or not specs[0].is_anchor:
        raise AssertionError("First branch must be the immutable anchor")
    selected = [specs[0].name]
    remaining = [spec.name for spec in specs[1:]]
    audit: dict[str, Any] = {
        "rule": (
            "greedy fixed-equal-ECDF blend; add only if aggregate inner OOF "
            "AUC gain >= gate and strict wins exceed half of inner folds"
        ),
        "minimum_auc_gain": float(config.blend_min_auc_gain),
        "minimum_win_fraction_exclusive": float(config.blend_min_win_fraction),
        "steps": [],
    }
    while remaining and len(selected) < config.max_selected_branches:
        current_matrix = _mean_normalized(selected, inner)
        current_score = np.mean(current_matrix, axis=0)
        current_auc = float(roc_auc_score(y, current_score))
        eligible: list[tuple[float, float, str, dict[str, Any], np.ndarray]] = []
        step_candidates: list[dict[str, Any]] = []
        for candidate in remaining:
            proposed_names = [*selected, candidate]
            proposed_matrix = _mean_normalized(proposed_names, inner)
            proposed_score = np.mean(proposed_matrix, axis=0)
            proposed_auc = float(roc_auc_score(y, proposed_score))
            gain = float(proposed_auc - current_auc)
            win_fraction, fold_records = _fold_win_fraction(
                y,
                current_matrix,
                proposed_matrix,
                inner[candidate].split_indices,
            )
            passes = (
                gain >= config.blend_min_auc_gain
                and win_fraction > config.blend_min_win_fraction
            )
            record = {
                "candidate": candidate,
                "current_selected": list(selected),
                "current_auc": current_auc,
                "proposed_auc": proposed_auc,
                "auc_gain": gain,
                "strict_fold_win_fraction": win_fraction,
                "passes_gate": bool(passes),
                "folds": fold_records,
            }
            step_candidates.append(record)
            if passes:
                eligible.append(
                    (gain, win_fraction, candidate, record, proposed_matrix)
                )
        if not eligible:
            audit["steps"].append(
                {
                    "decision": "anchor_or_current_fallback",
                    "candidates": step_candidates,
                }
            )
            break
        # Fixed tie-break: larger gain, then stability, then lexical branch name.
        eligible.sort(key=lambda item: (-item[0], -item[1], item[2]))
        _, _, winner, winner_record, winner_matrix = eligible[0]
        selected.append(winner)
        remaining.remove(winner)
        audit["steps"].append(
            {
                "decision": f"add:{winner}",
                "winner": winner_record,
                "candidates": step_candidates,
            }
        )
    final_matrix = _mean_normalized(selected, inner)
    audit["selected"] = list(selected)
    audit["equal_weight"] = float(1.0 / len(selected))
    audit["final_inner_oof_auc"] = float(
        roc_auc_score(y, np.mean(final_matrix, axis=0))
    )
    return selected, audit, final_matrix


def select_balanced_threshold_from_inner_oof(
    y: np.ndarray, score: np.ndarray
) -> dict[str, float]:
    """Secondary threshold, selected solely from outer-training inner OOF."""

    target = np.asarray(y, dtype=np.int64)
    values = np.asarray(score, dtype=np.float64)
    unique = np.unique(values)
    candidates = np.r_[
        unique[0] - 1e-12,
        (unique[:-1] + unique[1:]) / 2.0,
        unique[-1] + 1e-12,
    ]
    best: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        prediction = (values >= threshold).astype(np.int64)
        balanced = float(balanced_accuracy_score(target, prediction))
        accuracy = float(accuracy_score(target, prediction))
        key = (balanced, accuracy, -abs(float(threshold) - 0.5), -float(threshold))
        if best is None or key > best:
            best = key
            best_threshold = float(threshold)
    assert best is not None
    return {
        "threshold": best_threshold,
        "inner_oof_balanced_accuracy": float(best[0]),
        "inner_oof_accuracy": float(best[1]),
        "selected_without_outer_test_labels": True,
    }


@dataclass
class NestedCVResult:
    track: str
    subject_ids: np.ndarray
    repeat_scores: np.ndarray
    repeat_anchor_scores: np.ndarray
    repeat_threshold_margins: np.ndarray
    fold_records: list[dict[str, Any]]
    config: NestedCVConfig

    def summary(self, y: np.ndarray) -> dict[str, Any]:
        champion = summarize_repeated_oof(
            y,
            self.repeat_scores,
            n_resamples=self.config.bootstrap_resamples,
            seed=self.config.seed,
            score_name=f"{self.track}_nested_oof",
        )
        anchor = summarize_repeated_oof(
            y,
            self.repeat_anchor_scores,
            n_resamples=self.config.bootstrap_resamples,
            seed=self.config.seed + 1,
            score_name=f"{self.track}_anchor_nested_oof",
        )
        mean_champion = np.mean(self.repeat_scores, axis=0)
        mean_anchor = np.mean(self.repeat_anchor_scores, axis=0)
        delta = paired_bootstrap_auc_difference(
            y,
            mean_anchor,
            mean_champion,
            n_resamples=self.config.bootstrap_resamples,
            seed=self.config.seed + 2,
            reference_name=f"{self.track}_anchor",
            candidate_name=f"{self.track}_selected_policy",
        )
        selections: dict[str, int] = {}
        for record in self.fold_records:
            key = "+".join(record["selection"]["selected"])
            selections[key] = selections.get(key, 0) + 1
        margin = np.mean(self.repeat_threshold_margins, axis=0)
        threshold_prediction = (margin >= 0.0).astype(np.int64)
        return {
            "track": self.track,
            "protocol": asdict(self.config),
            "primary_selected_policy": champion,
            "immutable_anchor": anchor,
            "paired_auc_difference_selected_minus_anchor": delta,
            "fold_selection_frequency": selections,
            "secondary_cross_fitted_threshold": {
                "aggregation": (
                    "mean of outer-fold score-minus-inner-selected-threshold "
                    "margins across repeats"
                ),
                "accuracy": float(accuracy_score(y, threshold_prediction)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y, threshold_prediction)
                ),
                "all_cn_accuracy_baseline": float(np.mean(np.asarray(y) == 0)),
            },
            "estimand_warning": (
                "repeat-level mean/SD and subject-mean repeated-OOF AUC are "
                "reported separately; the bootstrap CI belongs only to the latter"
            ),
        }


def run_repeated_nested_cv(
    dataset: ChampionDataset,
    config: NestedCVConfig,
    *,
    progress_path: str | Path | None = None,
) -> NestedCVResult:
    """Evaluate the complete selection policy; never use this function on val."""

    config.validate()
    if dataset.y is None:
        raise ValueError("Nested CV requires training diagnoses")
    target = np.asarray(dataset.y, dtype=np.int64)
    specs = branch_library(dataset.track, include_tabpfn=config.include_tabpfn)
    repeat_scores = np.full(
        (config.outer_repeats, len(target)), np.nan, dtype=np.float64
    )
    repeat_anchor = np.full_like(repeat_scores, np.nan)
    repeat_margins = np.full_like(repeat_scores, np.nan)
    fold_records: list[dict[str, Any]] = []
    progress = None if progress_path is None else Path(progress_path)

    for repeat, fold, outer_train, outer_test in _repeated_splits(
        target,
        n_splits=config.outer_splits,
        n_repeats=config.outer_repeats,
        seed=config.seed,
    ):
        outer_stage = repeat * config.outer_splits + fold
        inner_seed = int(config.seed + 100_003 * (outer_stage + 1))
        inner = {
            spec.name: _inner_oof(
                spec,
                dataset,
                outer_train,
                config,
                # Every branch uses exactly the same paired subject splits.
                # Branch identity affects only the model RNG inside _inner_oof.
                seed=inner_seed,
            )
            for spec in specs
        }
        selected, selection_audit, selected_inner_matrix = _select_branches(
            specs,
            inner,
            target[outer_train],
            config,
        )
        threshold = select_balanced_threshold_from_inner_oof(
            target[outer_train], np.mean(selected_inner_matrix, axis=0)
        )
        selected_specs = [spec for spec in specs if spec.name in selected]
        outer_normalized: dict[str, np.ndarray] = {}
        model_manifests: dict[str, Any] = {}
        for spec in selected_specs:
            model = _fit_branch(
                spec,
                dataset,
                outer_train,
                seed=_branch_seed(config.seed, spec.name, outer_stage),
            )
            raw = _score_branch(spec, model, dataset, outer_test)
            outer_normalized[spec.name] = inner[spec.name].reference.transform(raw)
            model_manifests[spec.name] = model.manifest()
        selected_score = np.mean(
            np.stack([outer_normalized[name] for name in selected]), axis=0
        )
        anchor_name = specs[0].name
        repeat_scores[repeat, outer_test] = selected_score
        repeat_anchor[repeat, outer_test] = outer_normalized[anchor_name]
        repeat_margins[repeat, outer_test] = (
            selected_score - threshold["threshold"]
        )
        fold_records.append(
            {
                "repeat": int(repeat),
                "fold": int(fold),
                "outer_train_n": int(len(outer_train)),
                "outer_test_n": int(len(outer_test)),
                "outer_test_subject_hashes": [
                    hashlib.sha256(
                        str(dataset.subject_ids[index]).encode("utf-8")
                    ).hexdigest()[:16]
                    for index in outer_test
                ],
                "selection": selection_audit,
                "inner_branch_auc": {
                    name: value.aggregate_auc for name, value in inner.items()
                },
                "inner_threshold": threshold,
                "models": model_manifests,
                # Descriptive only.  It is never fed into a later selection.
                "outer_fold_auc_descriptive": float(
                    roc_auc_score(target[outer_test], selected_score)
                ),
                "outer_anchor_auc_descriptive": float(
                    roc_auc_score(
                        target[outer_test], outer_normalized[anchor_name]
                    )
                ),
            }
        )
        if progress is not None:
            progress.parent.mkdir(parents=True, exist_ok=True)
            progress.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "track": dataset.track,
                        "completed_outer_folds": len(fold_records),
                        "total_outer_folds": (
                            config.outer_splits * config.outer_repeats
                        ),
                        "last_repeat": int(repeat),
                        "last_fold": int(fold),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    for role, matrix in (
        ("selected", repeat_scores),
        ("anchor", repeat_anchor),
        ("threshold margin", repeat_margins),
    ):
        if not np.isfinite(matrix).all():
            raise AssertionError(f"Incomplete repeated OOF {role} matrix")
    if progress is not None:
        progress.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "track": dataset.track,
                    "completed_outer_folds": len(fold_records),
                    "total_outer_folds": len(fold_records),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return NestedCVResult(
        track=dataset.track,
        subject_ids=np.asarray(dataset.subject_ids),
        repeat_scores=repeat_scores,
        repeat_anchor_scores=repeat_anchor,
        repeat_threshold_margins=repeat_margins,
        fold_records=fold_records,
        config=config,
    )


@dataclass
class FittedChampion:
    track: str
    specs: tuple[BranchSpec, ...]
    models: dict[str, Any]
    references: dict[str, ReferenceECDF]
    threshold: float
    selection_audit: dict[str, Any]
    table_feature_names: tuple[str, ...]
    sequence_feature_names: tuple[str, ...]

    def predict_scores(self, dataset: ChampionDataset) -> dict[str, np.ndarray]:
        if dataset.track != self.track:
            raise ValueError("Deployment and prediction track differ")
        if tuple(dataset.feature_names) != self.table_feature_names:
            raise ValueError("Prediction table feature schema differs from training")
        if tuple(dataset.sequence_feature_names) != self.sequence_feature_names:
            raise ValueError("Prediction sequence schema differs from training")
        branch_scores: dict[str, np.ndarray] = {}
        full_index = np.arange(len(dataset.subject_ids), dtype=np.int64)
        for spec in self.specs:
            raw = _score_branch(spec, self.models[spec.name], dataset, full_index)
            branch_scores[spec.name] = self.references[spec.name].transform(raw)
        champion = np.mean(np.stack(list(branch_scores.values())), axis=0)
        return {
            **branch_scores,
            "champion_score": champion,
            "threshold_margin": champion - float(self.threshold),
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        model_entries: list[dict[str, str]] = []
        for spec in self.specs:
            model = self.models[spec.name]
            if spec.kind in {"sequence_transformer", "tabpfn"}:
                relative = f"models/{spec.name}"
            else:
                relative = f"models/{spec.name}.joblib"
            model.save(output / relative)
            model_entries.append(
                {
                    "name": spec.name,
                    "kind": spec.kind,
                    "relative_path": relative,
                }
            )
        manifest = {
            "track": self.track,
            "specs": [asdict(spec) for spec in self.specs],
            "references": {
                name: list(reference.sorted_reference)
                for name, reference in self.references.items()
            },
            "threshold": float(self.threshold),
            "selection_audit": self.selection_audit,
            "table_feature_names": list(self.table_feature_names),
            "sequence_feature_names": list(self.sequence_feature_names),
            "model_entries": model_entries,
            "model_manifests": {
                spec.name: self.models[spec.name].manifest()
                for spec in self.specs
            },
        }
        (output / "deployment.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "FittedChampion":
        source = Path(path)
        manifest = json.loads(
            (source / "deployment.json").read_text(encoding="utf-8")
        )
        specs = tuple(BranchSpec(**item) for item in manifest["specs"])
        entries = {item["name"]: item for item in manifest["model_entries"]}
        models: dict[str, Any] = {}
        for spec in specs:
            model_path = source / entries[spec.name]["relative_path"]
            if spec.kind == "sequence_transformer":
                models[spec.name] = SequenceTransformerBranch.load(
                    model_path, device=device
                )
            elif spec.kind == "mmse_anchor":
                models[spec.name] = MMSEMaxAUCAnchor.load(model_path)
            else:
                models[spec.name] = FoldLocalTableModel.load(
                    model_path, device=device
                )
        references = {
            name: ReferenceECDF(tuple(map(float, values)))
            for name, values in manifest["references"].items()
        }
        return cls(
            track=manifest["track"],
            specs=specs,
            models=models,
            references=references,
            threshold=float(manifest["threshold"]),
            selection_audit=manifest["selection_audit"],
            table_feature_names=tuple(manifest["table_feature_names"]),
            sequence_feature_names=tuple(manifest["sequence_feature_names"]),
        )


def fit_final_champion(
    dataset: ChampionDataset,
    config: NestedCVConfig,
) -> FittedChampion:
    """Select on full-training inner OOF, then refit only selected branches."""

    config.validate()
    if dataset.y is None:
        raise ValueError("Final fitting requires training labels")
    target = np.asarray(dataset.y, dtype=np.int64)
    full_index = np.arange(len(target), dtype=np.int64)
    specs = branch_library(dataset.track, include_tabpfn=config.include_tabpfn)
    inner = {
        spec.name: _inner_oof(
            spec,
            dataset,
            full_index,
            config,
            seed=config.seed + 91_117,
        )
        for spec in specs
    }
    selected, selection_audit, selected_inner_matrix = _select_branches(
        specs, inner, target, config
    )
    threshold_audit = select_balanced_threshold_from_inner_oof(
        target, np.mean(selected_inner_matrix, axis=0)
    )
    selected_specs = tuple(spec for spec in specs if spec.name in selected)
    models = {
        spec.name: _fit_branch(
            spec,
            dataset,
            full_index,
            seed=_branch_seed(config.seed + 271_828, spec.name, 0),
        )
        for spec in selected_specs
    }
    selection_audit = {
        **selection_audit,
        "final_refit_threshold": threshold_audit,
        "final_refit_uses_all_training_subjects": True,
        "validation_labels_used": False,
    }
    return FittedChampion(
        track=dataset.track,
        specs=selected_specs,
        models=models,
        references={name: inner[name].reference for name in selected},
        threshold=float(threshold_audit["threshold"]),
        selection_audit=selection_audit,
        table_feature_names=tuple(dataset.feature_names),
        sequence_feature_names=tuple(dataset.sequence_feature_names),
    )


__all__ = [
    "BranchSpec",
    "FittedChampion",
    "NestedCVConfig",
    "NestedCVResult",
    "branch_library",
    "fit_final_champion",
    "run_repeated_nested_cv",
    "select_balanced_threshold_from_inner_oof",
]
