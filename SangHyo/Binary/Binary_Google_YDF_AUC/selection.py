"""Candidate bank, fold-local feature selection, and ROC-AUC-only policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .models import YDF_FAMILIES


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    family: str
    view: str
    top_k: int
    corr_threshold: float
    params: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.family not in YDF_FAMILIES:
            raise ValueError(f"Non-YDF candidate family: {self.family}")
        if self.view not in {"mmse39", "mmse_all", "all151"}:
            raise ValueError(f"Unknown candidate view: {self.view}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "view": self.view,
            "top_k": int(self.top_k),
            "corr_threshold": float(self.corr_threshold),
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    repeats: int
    bag_seeds: int
    blend_draws: int
    candidates: tuple[CandidateSpec, ...]
    reportable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "outer_folds": 5,
            "outer_repeats": self.repeats,
            "bag_seeds": self.bag_seeds,
            "blend_draws": self.blend_draws,
            "reportable": self.reportable,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class FoldSpec:
    repeat: int
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray


def _gbt(
    *,
    trees: int,
    depth: int,
    minimum: int,
    shrinkage: float,
    subsample: float,
    attributes: float,
    l2: float,
) -> dict[str, Any]:
    return {
        "num_trees": trees,
        "max_depth": depth,
        "min_examples": minimum,
        "shrinkage": shrinkage,
        "subsample": subsample,
        "num_candidate_attributes_ratio": attributes,
        "l2_regularization": l2,
    }


def _oblique(**kwargs: Any) -> dict[str, Any]:
    return {
        **_gbt(**kwargs),
        "sparse_oblique_normalization": "STANDARD_DEVIATION",
        "sparse_oblique_num_projections_exponent": 1.5,
        "sparse_oblique_projection_density_factor": 3.0,
    }


def _rf(*, trees: int, depth: int, minimum: int, attributes: float) -> dict[str, Any]:
    return {
        "num_trees": trees,
        "max_depth": depth,
        "min_examples": minimum,
        "num_candidate_attributes_ratio": attributes,
    }


def candidate_bank() -> tuple[CandidateSpec, ...]:
    """Small predeclared YDF-only bank, including the prior best oblique setup."""

    return (
        CandidateSpec(
            "axis_mmse39_d3",
            "axis_gbt",
            "mmse39",
            0,
            1.01,
            _gbt(
                trees=400,
                depth=3,
                minimum=8,
                shrinkage=0.05,
                subsample=0.8,
                attributes=0.7,
                l2=1.0,
            ),
        ),
        CandidateSpec(
            "axis_mmse_all_d2",
            "axis_gbt",
            "mmse_all",
            0,
            0.99,
            _gbt(
                trees=600,
                depth=2,
                minimum=8,
                shrinkage=0.05,
                subsample=0.8,
                attributes=0.7,
                l2=0.5,
            ),
        ),
        CandidateSpec(
            "axis_all151_top25",
            "axis_gbt",
            "all151",
            25,
            0.95,
            _gbt(
                trees=600,
                depth=3,
                minimum=12,
                shrinkage=0.05,
                subsample=0.7,
                attributes=0.5,
                l2=1.0,
            ),
        ),
        CandidateSpec(
            "axis_all151_top70",
            "axis_gbt",
            "all151",
            70,
            0.99,
            _gbt(
                trees=600,
                depth=5,
                minimum=20,
                shrinkage=0.08,
                subsample=0.6,
                attributes=0.3,
                l2=0.0,
            ),
        ),
        CandidateSpec(
            "oblique_mmse39_d3",
            "sparse_oblique_gbt",
            "mmse39",
            0,
            1.01,
            _oblique(
                trees=400,
                depth=3,
                minimum=8,
                shrinkage=0.05,
                subsample=0.8,
                attributes=0.7,
                l2=0.5,
            ),
        ),
        CandidateSpec(
            "oblique_mmse_all_d2",
            "sparse_oblique_gbt",
            "mmse_all",
            0,
            0.99,
            _oblique(
                trees=600,
                depth=2,
                minimum=8,
                shrinkage=0.05,
                subsample=0.8,
                attributes=0.7,
                l2=0.5,
            ),
        ),
        CandidateSpec(
            "oblique_all151_top25",
            "sparse_oblique_gbt",
            "all151",
            25,
            0.95,
            _oblique(
                trees=600,
                depth=3,
                minimum=12,
                shrinkage=0.05,
                subsample=0.7,
                attributes=0.5,
                l2=0.5,
            ),
        ),
        CandidateSpec(
            # Exact strongest YDF branch from Binary_Google_MaxAUC_Tuned.
            "oblique_all151_prior_best",
            "sparse_oblique_gbt",
            "all151",
            70,
            0.99,
            _oblique(
                trees=600,
                depth=5,
                minimum=20,
                shrinkage=0.08,
                subsample=0.6,
                attributes=0.3,
                l2=0.0,
            ),
        ),
        CandidateSpec(
            "rf_mmse39_d6",
            "rf",
            "mmse39",
            0,
            1.01,
            _rf(trees=800, depth=6, minimum=5, attributes=0.5),
        ),
        CandidateSpec(
            "rf_mmse_all_d8",
            "rf",
            "mmse_all",
            0,
            0.99,
            _rf(trees=800, depth=8, minimum=5, attributes=0.5),
        ),
        CandidateSpec(
            "rf_all151_top25",
            "rf",
            "all151",
            25,
            0.95,
            _rf(trees=1_000, depth=8, minimum=5, attributes=0.3),
        ),
        CandidateSpec(
            "rf_all151_top70",
            "rf",
            "all151",
            70,
            0.99,
            _rf(trees=1_200, depth=12, minimum=8, attributes=0.3),
        ),
    )


def profile_spec(profile: str) -> ProfileSpec:
    name = str(profile).strip().lower()
    bank = candidate_bank()
    if name == "smoke":
        # One representative per required Google YDF family.  Smoke validates
        # wiring only and is explicitly non-reportable.
        small = (
            CandidateSpec(
                "smoke_axis",
                "axis_gbt",
                "mmse39",
                0,
                1.01,
                _gbt(
                    trees=30,
                    depth=2,
                    minimum=8,
                    shrinkage=0.08,
                    subsample=0.8,
                    attributes=0.7,
                    l2=1.0,
                ),
            ),
            CandidateSpec(
                "smoke_oblique",
                "sparse_oblique_gbt",
                "mmse39",
                0,
                1.01,
                _oblique(
                    trees=30,
                    depth=2,
                    minimum=8,
                    shrinkage=0.08,
                    subsample=0.8,
                    attributes=0.7,
                    l2=0.5,
                ),
            ),
            CandidateSpec(
                "smoke_rf",
                "rf",
                "mmse39",
                0,
                1.01,
                _rf(trees=50, depth=5, minimum=5, attributes=0.5),
            ),
        )
        return ProfileSpec(name, 1, 1, 128, small, False)
    if name == "default":
        selected_ids = {
            "axis_mmse39_d3",
            "axis_mmse_all_d2",
            "axis_all151_top25",
            "oblique_mmse39_d3",
            "oblique_all151_top25",
            "oblique_all151_prior_best",
            "rf_mmse39_d6",
            "rf_mmse_all_d8",
            "rf_all151_top25",
        }
        selected = tuple(
            candidate for candidate in bank if candidate.candidate_id in selected_ids
        )
        return ProfileSpec(name, 5, 3, 4_096, selected, True)
    if name == "max":
        return ProfileSpec(name, 10, 5, 20_000, bank, True)
    raise ValueError("profile must be smoke, default, or max")


def make_repeated_folds(
    y: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> tuple[FoldSpec, ...]:
    target = np.asarray(y, dtype=np.int64)
    indices = np.arange(len(target))
    folds: list[FoldSpec] = []
    for repeat in range(int(repeats)):
        splitter = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=int(seed) + repeat,
        )
        for fold, (train_indices, test_indices) in enumerate(
            splitter.split(indices, target)
        ):
            folds.append(
                FoldSpec(
                    repeat=repeat,
                    fold=fold,
                    train_indices=np.asarray(train_indices, dtype=np.int64),
                    test_indices=np.asarray(test_indices, dtype=np.int64),
                )
            )
    return tuple(folds)


def direction_free_auc(y: np.ndarray, column: np.ndarray) -> float:
    """Direction-free AUC used only for fold-training feature screening."""

    target = np.asarray(y, dtype=np.int64)
    values = np.asarray(column, dtype=np.float64)
    mask = np.isfinite(values)
    if mask.sum() < max(8, int(np.ceil(0.4 * len(target)))):
        return 0.5
    if len(np.unique(target[mask])) < 2:
        return 0.5
    auc = float(roc_auc_score(target[mask], values[mask]))
    return max(auc, 1.0 - auc)


def select_fold_columns(
    X_train: np.ndarray,
    y_train: np.ndarray,
    view_columns: Sequence[int],
    *,
    top_k: int,
    corr_threshold: float,
    minimum_finite_ratio: float = 0.5,
) -> np.ndarray:
    """Select global columns using only the current fold-training subjects."""

    matrix = np.asarray(X_train, dtype=np.float64)
    target = np.asarray(y_train, dtype=np.int64)
    view = np.asarray(view_columns, dtype=np.int64)
    if matrix.shape[0] != len(target):
        raise ValueError("Fold-training matrix/target length mismatch")
    local = matrix[:, view]
    finite = np.isfinite(local)
    ratios = finite.mean(axis=0)
    spread = np.zeros(local.shape[1], dtype=np.float64)
    for index in np.where(finite.sum(axis=0) >= 2)[0]:
        spread[index] = np.std(local[finite[:, index], index])
    usable = np.where(
        (ratios >= float(minimum_finite_ratio)) & (spread > 1e-10)
    )[0]
    if usable.size == 0:
        raise ValueError("No usable feature in fold-training view")
    scores = np.asarray(
        [direction_free_auc(target, local[:, index]) for index in usable]
    )
    order = usable[np.argsort(-scores, kind="stable")]

    if float(corr_threshold) < 1.0:
        kept: list[int] = []
        for candidate in order:
            redundant = False
            for previous in kept:
                left = local[:, candidate]
                right = local[:, previous]
                mask = np.isfinite(left) & np.isfinite(right)
                if mask.sum() < 8:
                    continue
                correlation = np.corrcoef(left[mask], right[mask])[0, 1]
                if np.isfinite(correlation) and abs(correlation) >= float(
                    corr_threshold
                ):
                    redundant = True
                    break
            if not redundant:
                kept.append(int(candidate))
        order = np.asarray(kept, dtype=np.int64)
    if int(top_k) > 0:
        order = order[: int(top_k)]
    if order.size == 0:
        raise ValueError("Fold-local feature selection returned no columns")
    return np.sort(view[order]).astype(np.int64)


class ReferenceECDF:
    """Training-reference empirical CDF; never ranks the held-out batch."""

    def fit(self, reference_scores: np.ndarray) -> "ReferenceECDF":
        values = np.asarray(reference_scores, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("Reference scores must be a finite vector")
        self.reference_ = np.sort(values)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if not hasattr(self, "reference_"):
            raise RuntimeError("ReferenceECDF is not fitted")
        values = np.asarray(scores, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Scores to transform must be finite")
        left = np.searchsorted(self.reference_, values, side="left")
        right = np.searchsorted(self.reference_, values, side="right")
        return (left + right) / (2.0 * len(self.reference_))

    def to_dict(self) -> dict[str, Any]:
        return {"sorted_reference_scores": self.reference_.tolist()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReferenceECDF":
        instance = cls()
        instance.reference_ = np.asarray(
            payload["sorted_reference_scores"], dtype=np.float64
        )
        if (
            instance.reference_.ndim != 1
            or instance.reference_.size == 0
            or not np.isfinite(instance.reference_).all()
        ):
            raise ValueError("Stored ECDF reference is invalid")
        return instance


def auc(y: np.ndarray, scores: np.ndarray) -> float:
    target = np.asarray(y, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != target.shape or not np.isfinite(values).all():
        raise ValueError("AUC score vector is invalid")
    return float(roc_auc_score(target, values))


def optimize_auc_weights(
    score_matrix: np.ndarray,
    y: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Choose simplex weights using ROC-AUC and no secondary metric."""

    matrix = np.asarray(score_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(y):
        raise ValueError("Blend score matrix shape mismatch")
    n_models = matrix.shape[1]
    if n_models == 1:
        return np.ones(1), auc(y, matrix[:, 0])
    rng = np.random.default_rng(int(seed))
    candidates = [
        *np.eye(n_models, dtype=np.float64),
        np.full(n_models, 1.0 / n_models, dtype=np.float64),
        *rng.dirichlet(np.ones(n_models), size=int(draws)),
    ]
    best_weights = np.asarray(candidates[0], dtype=np.float64)
    best_auc = -np.inf
    for weights in candidates:
        value = auc(y, matrix @ np.asarray(weights, dtype=np.float64))
        if value > best_auc:
            best_auc = value
            best_weights = np.asarray(weights, dtype=np.float64)
    return best_weights / best_weights.sum(), float(best_auc)


def build_and_select_policies(
    candidate_raw: Mapping[str, np.ndarray],
    candidate_ecdf: Mapping[str, np.ndarray],
    candidate_family: Mapping[str, str],
    y: np.ndarray,
    *,
    blend_draws: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate single and ensemble policies by subject-mean OOF ROC-AUC only."""

    ids = sorted(candidate_raw)
    if set(ids) != set(candidate_ecdf) or set(ids) != set(candidate_family):
        raise ValueError("Candidate score maps are not aligned")
    policies: list[dict[str, Any]] = []

    def add_policy(
        name: str,
        score_space: str,
        components: Sequence[str],
        weights: Sequence[float],
        selection_type: str,
    ) -> None:
        source = candidate_raw if score_space == "raw" else candidate_ecdf
        component_ids = list(components)
        vector = np.column_stack([source[item] for item in component_ids]) @ np.asarray(
            weights, dtype=np.float64
        )
        policies.append(
            {
                "name": name,
                "selection_type": selection_type,
                "score_space": score_space,
                "components": component_ids,
                "weights": list(map(float, weights)),
                "subject_mean_oof_roc_auc": auc(y, vector),
                "_score": vector,
            }
        )

    for candidate_id in ids:
        add_policy(
            f"single_raw::{candidate_id}",
            "raw",
            [candidate_id],
            [1.0],
            "single",
        )
        add_policy(
            f"single_ecdf::{candidate_id}",
            "ecdf",
            [candidate_id],
            [1.0],
            "single",
        )

    for score_space, source in (
        ("raw", candidate_raw),
        ("ecdf", candidate_ecdf),
    ):
        score_order = sorted(ids, key=lambda item: -auc(y, source[item]))
        top_two = score_order[: min(2, len(score_order))]
        if len(top_two) >= 2:
            add_policy(
                f"equal_top2_{score_space}",
                score_space,
                top_two,
                [1.0 / len(top_two)] * len(top_two),
                "equal_top2",
            )
            top_weights, _ = optimize_auc_weights(
                np.column_stack([source[item] for item in top_two]),
                y,
                draws=blend_draws,
                seed=seed + (11 if score_space == "raw" else 13),
            )
            add_policy(
                f"auc_simplex_top2_{score_space}",
                score_space,
                top_two,
                top_weights,
                "auc_optimized_simplex_top2",
            )

        family_winners: list[str] = []
        for family in YDF_FAMILIES:
            members = [
                item for item in ids if candidate_family[item] == family
            ]
            if members:
                family_winners.append(
                    max(members, key=lambda item: auc(y, source[item]))
                )
        if len(family_winners) >= 2:
            matrix = np.column_stack([source[item] for item in family_winners])
            equal = np.full(len(family_winners), 1.0 / len(family_winners))
            add_policy(
                f"equal_family_winners_{score_space}",
                score_space,
                family_winners,
                equal,
                "equal_family_winners",
            )
            weights, _ = optimize_auc_weights(
                matrix,
                y,
                draws=blend_draws,
                seed=seed + (17 if score_space == "raw" else 19),
            )
            add_policy(
                f"auc_simplex_family_winners_{score_space}",
                score_space,
                family_winners,
                weights,
                "auc_optimized_simplex",
            )

        all_weights, _ = optimize_auc_weights(
            np.column_stack([source[item] for item in ids]),
            y,
            draws=blend_draws,
            seed=seed + (23 if score_space == "raw" else 29),
        )
        add_policy(
            f"auc_simplex_all_{score_space}",
            score_space,
            ids,
            all_weights,
            "auc_optimized_simplex_all_candidates",
        )

    chosen = max(
        policies,
        key=lambda policy: (
            policy["subject_mean_oof_roc_auc"],
            -len(policy["components"]),
            policy["name"],
        ),
    )
    public_policies: list[dict[str, Any]] = []
    for policy in policies:
        public_policies.append(
            {key: value for key, value in policy.items() if key != "_score"}
        )
    public_chosen = {key: value for key, value in chosen.items() if key != "_score"}
    return public_chosen, public_policies


__all__ = [
    "CandidateSpec",
    "FoldSpec",
    "ProfileSpec",
    "ReferenceECDF",
    "auc",
    "build_and_select_policies",
    "candidate_bank",
    "direction_free_auc",
    "make_repeated_folds",
    "optimize_auc_weights",
    "profile_spec",
    "select_fold_columns",
]
