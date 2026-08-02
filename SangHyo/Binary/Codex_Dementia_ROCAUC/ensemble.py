"""OOF-only probability/rank blending with a prespecified anchor safeguard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .metrics import safe_roc_auc


def ecdf_transform(reference: Sequence[float], values: Sequence[float]) -> np.ndarray:
    """Map scores through a training-OOF empirical CDF.

    Outer-validation scores are never ranked against one another, which would
    leak validation-set composition into each prediction.
    """

    ref = np.sort(np.asarray(reference, dtype=np.float64))
    query = np.asarray(values, dtype=np.float64)
    if ref.ndim != 1 or len(ref) == 0 or query.ndim != 1:
        raise ValueError("ECDF inputs must be non-empty one-dimensional arrays")
    if not np.isfinite(ref).all() or not np.isfinite(query).all():
        raise ValueError("ECDF inputs contain non-finite scores")
    left = np.searchsorted(ref, query, side="left")
    right = np.searchsorted(ref, query, side="right")
    return (left + right) / (2.0 * max(1, len(ref)))


def rank_matrix(
    reference_by_model: Mapping[str, Sequence[float]],
    score_by_model: Mapping[str, Sequence[float]],
    members: Sequence[str],
) -> np.ndarray:
    return np.column_stack(
        [
            ecdf_transform(reference_by_model[name], score_by_model[name])
            for name in members
        ]
    )


@dataclass(frozen=True)
class BlendPolicy:
    members: tuple[str, ...]
    weights: tuple[float, ...]
    mode: str
    anchor: str
    base_model: str
    inner_oof_auc: float
    anchor_inner_oof_auc: float
    selection_trace: tuple[dict, ...]
    reference_scores: dict[str, tuple[float, ...]]

    def predict(self, score_by_model: Mapping[str, Sequence[float]]) -> np.ndarray:
        missing = set(self.members) - set(score_by_model)
        if missing:
            raise KeyError(f"Blend prediction lacks members: {sorted(missing)}")
        if self.mode == "probability":
            matrix = np.column_stack(
                [np.asarray(score_by_model[name], dtype=float) for name in self.members]
            )
        elif self.mode == "rank_ecdf":
            matrix = rank_matrix(
                self.reference_scores,
                score_by_model,
                self.members,
            )
        else:
            raise ValueError(f"Unknown blend mode: {self.mode}")
        result = matrix @ np.asarray(self.weights, dtype=np.float64)
        return np.clip(result, 1e-7, 1.0 - 1e-7)

    def to_dict(self) -> dict:
        return {
            "members": list(self.members),
            "weights": list(self.weights),
            "mode": self.mode,
            "anchor": self.anchor,
            "base_model": self.base_model,
            "inner_oof_auc": self.inner_oof_auc,
            "anchor_inner_oof_auc": self.anchor_inner_oof_auc,
            "inner_metric_is_selection_only_not_performance_claim": True,
            "selection_trace": list(self.selection_trace),
            "outer_scores_ranked_against_outer_subjects": False,
        }


def _equal_score(
    oof_by_model: Mapping[str, np.ndarray],
    members: Sequence[str],
    *,
    mode: str,
) -> np.ndarray:
    if mode == "probability":
        matrix = np.column_stack([oof_by_model[name] for name in members])
    elif mode == "rank_ecdf":
        matrix = rank_matrix(oof_by_model, oof_by_model, members)
    else:
        raise ValueError(mode)
    return matrix.mean(axis=1)


def _repeat_wins(
    y: np.ndarray,
    current: np.ndarray,
    candidate: np.ndarray,
) -> tuple[int, int]:
    if current.ndim == 1:
        current = current[None, :]
    if candidate.ndim == 1:
        candidate = candidate[None, :]
    if current.shape != candidate.shape or current.shape[1] != len(y):
        raise ValueError("Repeat-level blend predictions are misaligned")
    wins = 0
    for repeat in range(current.shape[0]):
        wins += int(
            safe_roc_auc(y, candidate[repeat])
            > safe_roc_auc(y, current[repeat]) + 1e-12
        )
    return wins, current.shape[0]


def _required_repeat_wins(repeat_count: int) -> int:
    """Require a strict majority, including when the repeat count is even."""

    if int(repeat_count) < 1:
        raise ValueError("repeat_count must be positive")
    return int(repeat_count) // 2 + 1


def _repeat_blend(
    repeat_oof_by_model: Mapping[str, np.ndarray],
    members: Sequence[str],
    *,
    mode: str,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    arrays = [np.asarray(repeat_oof_by_model[name], dtype=float) for name in members]
    shape = arrays[0].shape
    if len(shape) != 2 or any(array.shape != shape for array in arrays):
        raise ValueError("Model repeat OOF matrices are misaligned")
    resolved_weights = (
        np.full(len(members), 1.0 / len(members))
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    output = np.zeros(shape, dtype=float)
    for repeat in range(shape[0]):
        if mode == "probability":
            matrix = np.column_stack([array[repeat] for array in arrays])
        elif mode == "rank_ecdf":
            per_model = {
                name: repeat_oof_by_model[name][repeat] for name in members
            }
            matrix = rank_matrix(per_model, per_model, members)
        else:
            raise ValueError(mode)
        output[repeat] = matrix @ resolved_weights
    return output


def _optimize_weights(
    matrix: np.ndarray,
    y: np.ndarray,
    *,
    trials: int,
    seed: int,
    required_position: int,
) -> tuple[np.ndarray, float]:
    n_models = matrix.shape[1]
    if n_models == 1:
        return np.ones(1), safe_roc_auc(y, matrix[:, 0])
    rng = np.random.default_rng(seed)
    candidates = [np.full(n_models, 1.0 / n_models)]
    # Bounded random search is deterministic and deliberately low-dimensional.
    for _ in range(max(0, int(trials))):
        raw = rng.dirichlet(np.full(n_models, 2.0))
        if raw[required_position] < 0.05:
            other = np.arange(n_models) != required_position
            raw[other] *= 0.95 / raw[other].sum()
            raw[required_position] = 0.05
        candidates.append(raw)
    best_weight = candidates[0]
    best_auc = safe_roc_auc(y, matrix @ best_weight)
    for weight in candidates[1:]:
        auc = safe_roc_auc(y, matrix @ weight)
        # Prefer the less concentrated ensemble on exact AUC ties.
        better_tie = (
            abs(auc - best_auc) <= 1e-12
            and float(np.square(weight).sum()) < float(np.square(best_weight).sum())
        )
        if auc > best_auc + 1e-12 or better_tie:
            best_weight, best_auc = weight, auc
    return best_weight, best_auc


def fit_blend_policy(
    y: Sequence[int],
    oof_by_model: Mapping[str, Sequence[float]],
    *,
    anchor: str,
    max_members: int,
    minimum_auc_gain: float,
    weight_trials: int,
    seed: int,
    repeat_oof_by_model: Mapping[str, np.ndarray] | None = None,
) -> BlendPolicy:
    """Select members/mode/weights using inner OOF predictions only."""

    target = np.asarray(y, dtype=np.int64)
    scores = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in oof_by_model.items()
    }
    if anchor not in scores:
        raise KeyError(f"Immutable anchor {anchor!r} lacks inner OOF scores")
    if any(values.shape != target.shape for values in scores.values()):
        raise ValueError("Inner OOF model predictions are misaligned")
    if any(not np.isfinite(values).all() for values in scores.values()):
        raise ValueError("Inner OOF model predictions contain non-finite scores")

    if repeat_oof_by_model is None:
        repeat_scores = {
            name: values[None, :] for name, values in scores.items()
        }
    else:
        repeat_scores = {
            name: np.asarray(repeat_oof_by_model[name], dtype=float)
            for name in scores
        }
        if any(
            values.ndim != 2 or values.shape[1] != len(target)
            for values in repeat_scores.values()
        ):
            raise ValueError("Repeat OOF matrices have invalid shapes")
        if len({values.shape for values in repeat_scores.values()}) != 1:
            raise ValueError("Repeat OOF matrices have inconsistent repeat counts")
        if any(
            not np.isfinite(values).all() for values in repeat_scores.values()
        ):
            raise ValueError("Repeat OOF matrices contain non-finite scores")

    current = scores[anchor]
    current_auc = safe_roc_auc(target, current)
    anchor_auc = current_auc
    base_model = anchor
    current_repeat = repeat_scores[anchor]
    trace: list[dict] = [
        {
            "step": 0,
            "action": "anchor",
            "members": [anchor],
            "auc": current_auc,
        }
    ]
    best_individual = min(
        scores,
        key=lambda name: (-safe_roc_auc(target, scores[name]), name),
    )
    if best_individual != anchor:
        best_auc = safe_roc_auc(target, scores[best_individual])
        wins, repeat_count = _repeat_wins(
            target,
            repeat_scores[anchor],
            repeat_scores[best_individual],
        )
        promoted = (
            best_auc >= anchor_auc + float(minimum_auc_gain)
            and wins >= _required_repeat_wins(repeat_count)
        )
        trace.append(
            {
                "step": 0,
                "action": "compare_best_individual_to_anchor",
                "candidate": best_individual,
                "auc_gain": best_auc - anchor_auc,
                "repeat_wins": wins,
                "required_repeat_wins": _required_repeat_wins(repeat_count),
                "promoted": promoted,
            }
        )
        if promoted:
            base_model = best_individual
            current = scores[best_individual]
            current_auc = best_auc
            current_repeat = repeat_scores[best_individual]
    members = [base_model]
    base_auc = current_auc
    base_repeat = current_repeat.copy()
    while len(members) < min(max(1, int(max_members)), len(scores)):
        options: list[
            tuple[float, int, str, str, np.ndarray, np.ndarray, int]
        ] = []
        for candidate_name in sorted(set(scores) - set(members)):
            trial_members = [*members, candidate_name]
            for mode in ("probability", "rank_ecdf"):
                candidate_score = _equal_score(scores, trial_members, mode=mode)
                candidate_repeat = _repeat_blend(
                    repeat_scores,
                    trial_members,
                    mode=mode,
                )
                auc = safe_roc_auc(target, candidate_score)
                wins, repeat_count = _repeat_wins(
                    target,
                    current_repeat,
                    candidate_repeat,
                )
                gain = auc - current_auc
                options.append(
                    (
                        gain,
                        wins,
                        candidate_name,
                        mode,
                        candidate_score,
                        candidate_repeat,
                        repeat_count,
                    )
                )
        options.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
        (
            gain,
            wins,
            candidate_name,
            mode,
            candidate_score,
            candidate_repeat,
            repeat_count,
        ) = options[0]
        required_wins = _required_repeat_wins(repeat_count)
        accepted = gain >= float(minimum_auc_gain) and wins >= required_wins
        trace.append(
            {
                "step": len(members),
                "candidate": candidate_name,
                "mode": mode,
                "auc_gain": float(gain),
                "repeat_wins": int(wins),
                "required_repeat_wins": int(required_wins),
                "accepted": bool(accepted),
            }
        )
        if not accepted:
            break
        members.append(candidate_name)
        current = candidate_score
        current_repeat = candidate_repeat
        current_auc = safe_roc_auc(target, current)

    best: tuple[float, str, np.ndarray] | None = None
    for mode in ("probability", "rank_ecdf"):
        if mode == "probability":
            matrix = np.column_stack([scores[name] for name in members])
        else:
            matrix = rank_matrix(scores, scores, members)
        weights, auc = _optimize_weights(
            matrix,
            target,
            trials=weight_trials,
            seed=seed + (0 if mode == "probability" else 7919),
            required_position=members.index(base_model),
        )
        if best is None or auc > best[0] + 1e-12:
            best = (auc, mode, weights)
    assert best is not None
    best_auc, best_mode, best_weights = best
    final_repeat = _repeat_blend(
        repeat_scores,
        members,
        mode=best_mode,
        weights=best_weights,
    )
    final_wins, final_repeat_count = _repeat_wins(
        target,
        base_repeat,
        final_repeat,
    )
    # The stable base is a hard fallback: do not accept a tuned blend that fails
    # to clear the prespecified gain and repeat-win requirements.
    if (
        best_auc < base_auc + float(minimum_auc_gain)
        or final_wins < _required_repeat_wins(final_repeat_count)
    ):
        members = [base_model]
        best_weights = np.ones(1)
        best_mode = "probability"
        best_auc = base_auc
        trace.append(
            {
                "action": "fallback_to_stable_base",
                "base_model": base_model,
                "repeat_wins": final_wins,
                "required_repeat_wins": _required_repeat_wins(
                    final_repeat_count
                ),
                "reason": (
                    "optimized blend did not clear inner-OOF gain/repeat stability"
                ),
            }
        )
    return BlendPolicy(
        members=tuple(members),
        weights=tuple(map(float, best_weights)),
        mode=best_mode,
        anchor=anchor,
        base_model=base_model,
        inner_oof_auc=float(best_auc),
        anchor_inner_oof_auc=float(anchor_auc),
        selection_trace=tuple(trace),
        reference_scores={
            name: tuple(map(float, scores[name])) for name in members
        },
    )


def stacking_applicability(y: Sequence[int]) -> dict[str, str | bool | int]:
    """Explain why a free meta-model is disabled for the nine-positive cohort."""

    positive = int(np.asarray(y, dtype=np.int64).sum())
    enabled = positive >= 20
    return {
        "enabled": enabled,
        "n_positive": positive,
        "reason": (
            "at least 20 positive development subjects available"
            if enabled
            else "disabled a priori: a learned stacking meta-model is unstable "
            "with fewer than 20 positive development subjects"
        ),
    }


__all__ = [
    "BlendPolicy",
    "ecdf_transform",
    "fit_blend_policy",
    "rank_matrix",
    "stacking_applicability",
]
