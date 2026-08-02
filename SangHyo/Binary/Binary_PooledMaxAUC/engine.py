"""Repeated subject-level OOF evaluation, fold-local screening and ensembling.

The direct-leakage boundary lives entirely in :func:`_run_fold`:

* imputation medians, winsorization limits and standardization statistics are
  computed on the training fold only;
* univariate screening and correlation pruning see the training fold only, and
  :func:`assert_screening_is_train_local` proves it at the call site;
* score normalization uses the training fold's ECDF as the reference, so a
  held-out subject's rank never depends on the other held-out subjects.

Everything *outside* that boundary is deliberately optimistic and disclosed:
candidates, screening size and ensemble weights are chosen on the same repeated
OOF that is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .leakage import (
    LeakageAudit,
    LeakageError,
    assert_fold_disjoint,
    assert_finite_scores,
    assert_screening_is_train_local,
)
from .models import Candidate, fit_predict

__all__ = [
    "CandidateResult",
    "SplitPlan",
    "build_split_plan",
    "evaluate_candidate",
    "search_ensemble",
    "roc_auc",
    "binary_metrics",
]


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U), tie-safe."""

    from scipy.stats import rankdata

    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def directionless_auc(y_true: np.ndarray, values: np.ndarray) -> float:
    """AUC folded to >= 0.5 so screening ranks by strength, not direction."""

    auc = roc_auc(y_true, values)
    return float("nan") if not np.isfinite(auc) else max(auc, 1.0 - auc)


def binary_metrics(y_true: Sequence[int], scores: Sequence[float], *, threshold: float = 0.5) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef

    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=float)
    predicted = (s >= threshold).astype(np.int64)
    positives, negatives = int(y.sum()), int((y == 0).sum())
    tp = int(np.sum((predicted == 1) & (y == 1)))
    tn = int(np.sum((predicted == 0) & (y == 0)))
    return {
        "n": int(len(y)),
        "n_positive": positives,
        "n_negative": negatives,
        "roc_auc": roc_auc(y, s),
        "pr_auc": float(average_precision_score(y, s)),
        "threshold": float(threshold),
        "recall_sensitivity": float(tp / positives) if positives else float("nan"),
        "specificity": float(tn / negatives) if negatives else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, predicted)) if len(set(predicted.tolist())) > 1 else 0.0,
        "accuracy": float(np.mean(predicted == y)),
        "all_negative_baseline_accuracy": float(negatives / len(y)) if len(y) else float("nan"),
    }


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FoldSplit:
    split_id: str
    repeat: int
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray


@dataclass(frozen=True)
class SplitPlan:
    n_subjects: int
    n_splits: int
    n_repeats: int
    seed: int
    records: tuple[FoldSplit, ...]
    plan_hash: str


def build_split_plan(
    y: Sequence[int],
    subject_ids: Sequence[str],
    *,
    n_splits: int,
    n_repeats: int,
    seed: int,
    min_positive_per_validation_fold: int = 1,
) -> SplitPlan:
    """Repeated StratifiedKFold. One row per subject => inherently subject-level."""

    import hashlib
    import json

    from sklearn.model_selection import StratifiedKFold

    target = np.asarray(y, dtype=np.int64)
    subjects = np.asarray([str(v) for v in subject_ids], dtype=str)
    if target.shape != subjects.shape:
        raise LeakageError("Target and subject arrays are not aligned")
    if len(set(subjects.tolist())) != len(subjects):
        raise LeakageError("Subject ids must be unique for a subject-level split")

    counts = np.bincount(target, minlength=2)
    if int(counts.min()) < int(n_splits):
        raise LeakageError(
            f"{n_splits} folds exceed the minority class size {int(counts.min())}"
        )

    records: list[FoldSplit] = []
    for repeat in range(max(1, int(n_repeats))):
        splitter = StratifiedKFold(
            n_splits=int(n_splits), shuffle=True, random_state=int(seed) + repeat * 1009
        )
        seen = np.zeros(len(target), dtype=np.int64)
        for fold, (train_index, test_index) in enumerate(
            splitter.split(np.zeros((len(target), 1)), target)
        ):
            context = f"repeat={repeat}/fold={fold}"
            assert_fold_disjoint(subjects[train_index], subjects[test_index], context=context)
            positives = int(target[test_index].sum())
            if positives < int(min_positive_per_validation_fold):
                raise LeakageError(f"{context}: only {positives} positive subject(s) held out")
            seen[test_index] += 1
            records.append(
                FoldSplit(
                    split_id=f"r{repeat:02d}_f{fold:02d}",
                    repeat=repeat,
                    fold=fold,
                    train_indices=np.asarray(train_index, dtype=np.int64),
                    test_indices=np.asarray(test_index, dtype=np.int64),
                )
            )
        if np.any(seen != 1):
            raise LeakageError(f"repeat={repeat}: each subject must be held out exactly once")

    material = json.dumps(
        [{"id": r.split_id, "test": sorted(r.test_indices.tolist())} for r in records],
        sort_keys=True,
    )
    return SplitPlan(
        n_subjects=len(target),
        n_splits=int(n_splits),
        n_repeats=max(1, int(n_repeats)),
        seed=int(seed),
        records=tuple(records),
        plan_hash=hashlib.sha256(material.encode("utf-8")).hexdigest()[:16],
    )


# --------------------------------------------------------------------------- #
# fold-local preprocessing + screening
# --------------------------------------------------------------------------- #
def _fit_preprocessor(
    X_train: np.ndarray, *, winsorize_quantile: float, scale: bool
) -> dict[str, np.ndarray]:
    """All statistics come from the training fold only."""

    medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(X_train), X_train, medians)
    q = float(winsorize_quantile)
    lower = np.quantile(filled, q, axis=0) if q > 0 else filled.min(axis=0)
    upper = np.quantile(filled, 1.0 - q, axis=0) if q > 0 else filled.max(axis=0)
    clipped = np.clip(filled, lower, upper)
    if scale:
        centre = clipped.mean(axis=0)
        spread = clipped.std(axis=0)
        spread = np.where(spread > 1e-12, spread, 1.0)
    else:
        centre = np.zeros(clipped.shape[1])
        spread = np.ones(clipped.shape[1])
    return {"medians": medians, "lower": lower, "upper": upper, "centre": centre, "spread": spread}


def _apply_preprocessor(X: np.ndarray, state: Mapping[str, np.ndarray]) -> np.ndarray:
    filled = np.where(np.isfinite(X), X, state["medians"])
    clipped = np.clip(filled, state["lower"], state["upper"])
    return (clipped - state["centre"]) / state["spread"]


def _screen_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    top_k: int,
    correlation_threshold: float,
) -> np.ndarray:
    """Direction-free univariate AUC ranking + correlation pruning, train-only."""

    n_features = X_train.shape[1]
    scores = np.asarray(
        [directionless_auc(y_train, X_train[:, j]) for j in range(n_features)], dtype=float
    )
    scores = np.where(np.isfinite(scores), scores, 0.0)
    order = np.argsort(-scores, kind="mergesort")

    selected: list[int] = []
    for index in order:
        if len(selected) >= int(top_k):
            break
        column = X_train[:, index]
        if np.std(column) < 1e-12:
            continue
        redundant = False
        for chosen in selected:
            other = X_train[:, chosen]
            if np.std(other) < 1e-12:
                continue
            corr = np.corrcoef(column, other)[0, 1]
            if np.isfinite(corr) and abs(corr) > float(correlation_threshold):
                redundant = True
                break
        if not redundant:
            selected.append(int(index))
    if not selected:  # degenerate fold: keep the single strongest column
        selected = [int(order[0])]
    return np.asarray(selected, dtype=np.int64)


def _ecdf_transform(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Percentile of ``values`` within the TRAINING fold's score distribution.

    Using the training reference (rather than ranking the held-out batch among
    itself) keeps each held-out subject's transformed score independent of the
    other held-out subjects - i.e. avoids transductive normalization.
    """

    ordered = np.sort(np.asarray(reference, dtype=float))
    if ordered.size == 0:
        return np.zeros_like(values, dtype=float)
    positions = np.searchsorted(ordered, np.asarray(values, dtype=float), side="right")
    return positions / float(ordered.size)


# --------------------------------------------------------------------------- #
# candidate evaluation
# --------------------------------------------------------------------------- #
@dataclass
class CandidateResult:
    candidate: Candidate
    oof_by_repeat: np.ndarray  # (n_repeats, n_subjects) ECDF-normalized scores
    subject_mean_oof: np.ndarray
    subject_mean_auc: float
    repeat_aucs: list[float] = field(default_factory=list)
    mean_repeat_auc: float = float("nan")
    sd_repeat_auc: float = float("nan")
    n_features_used: list[int] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.candidate.name,
            "family": self.candidate.family,
            "view": self.candidate.view,
            "params": dict(self.candidate.params),
            "top_k": self.candidate.top_k,
            "subject_mean_oof_roc_auc": self.subject_mean_auc,
            "mean_repeat_roc_auc": self.mean_repeat_auc,
            "sd_repeat_roc_auc": self.sd_repeat_auc,
            "repeat_roc_auc": self.repeat_aucs,
            "median_features_used": (
                int(np.median(self.n_features_used)) if self.n_features_used else None
            ),
            "error": self.error,
        }


def _run_fold(
    candidate: Candidate,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: tuple[str, ...],
    record: FoldSplit,
    *,
    seed: int,
    winsorize_quantile: float,
    correlation_threshold: float,
    balanced: bool,
    audit: LeakageAudit | None,
) -> tuple[np.ndarray, int]:
    """Fit one fold under the fold-local contract; return held-out ECDF scores."""

    train_idx, test_idx = record.train_indices, record.test_indices
    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train = y[train_idx]

    state = _fit_preprocessor(
        X_train_raw, winsorize_quantile=winsorize_quantile, scale=candidate.requires_scaling
    )
    X_train = _apply_preprocessor(X_train_raw, state)
    X_test = _apply_preprocessor(X_test_raw, state)

    columns = np.arange(X.shape[1], dtype=np.int64)
    if candidate.top_k is not None and candidate.top_k < X.shape[1]:
        # Proves screening saw exactly the training fold and nothing else.
        assert_screening_is_train_local(
            X_train.shape[0], len(train_idx), context=f"{candidate.name}/{record.split_id}"
        )
        columns = _screen_features(
            X_train,
            y_train,
            top_k=candidate.top_k,
            correlation_threshold=correlation_threshold,
        )
        if audit is not None and record.split_id.endswith("f00") and record.repeat == 0:
            audit.record(
                "screening_is_fold_local",
                True,
                f"{candidate.name}: screened on {X_train.shape[0]} training rows only",
            )

    selected_names = tuple(feature_names[i] for i in columns)
    train_scores, test_scores = fit_predict(
        candidate,
        X_train[:, columns],
        y_train,
        X_test[:, columns],
        seed=seed,
        balanced=balanced,
        feature_names=selected_names,
    )
    normalized = _ecdf_transform(train_scores, test_scores)
    assert_finite_scores(normalized, context=f"{candidate.name}/{record.split_id}")
    return normalized, len(columns)


def evaluate_candidate(
    candidate: Candidate,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: tuple[str, ...],
    plan: SplitPlan,
    *,
    seed: int,
    winsorize_quantile: float,
    correlation_threshold: float,
    balanced: bool,
    audit: LeakageAudit | None = None,
) -> CandidateResult:
    """Full repeated OOF for one candidate."""

    n_subjects = len(y)
    oof = np.full((plan.n_repeats, n_subjects), np.nan, dtype=float)
    n_features_used: list[int] = []

    try:
        for record in plan.records:
            scores, n_used = _run_fold(
                candidate,
                X,
                y,
                feature_names,
                record,
                seed=seed + record.fold,
                winsorize_quantile=winsorize_quantile,
                correlation_threshold=correlation_threshold,
                balanced=balanced,
                audit=audit,
            )
            oof[record.repeat, record.test_indices] = scores
            n_features_used.append(n_used)
    except Exception as error:  # noqa: BLE001 - a broken candidate must not kill the run
        return CandidateResult(
            candidate=candidate,
            oof_by_repeat=oof,
            subject_mean_oof=np.full(n_subjects, np.nan),
            subject_mean_auc=float("nan"),
            error=f"{type(error).__name__}: {error}"[:400],
        )

    if np.isnan(oof).any():
        raise LeakageError(f"{candidate.name}: {int(np.isnan(oof).sum())} subject-repeats unscored")

    repeat_aucs = [float(roc_auc(y, oof[r])) for r in range(plan.n_repeats)]
    subject_mean = oof.mean(axis=0)
    return CandidateResult(
        candidate=candidate,
        oof_by_repeat=oof,
        subject_mean_oof=subject_mean,
        subject_mean_auc=float(roc_auc(y, subject_mean)),
        repeat_aucs=repeat_aucs,
        mean_repeat_auc=float(np.mean(repeat_aucs)),
        sd_repeat_auc=float(np.std(repeat_aucs, ddof=1)) if len(repeat_aucs) > 1 else 0.0,
        n_features_used=n_features_used,
    )


# --------------------------------------------------------------------------- #
# ensemble search (non-nested by design, disclosed)
# --------------------------------------------------------------------------- #
def search_ensemble(
    results: Sequence[CandidateResult],
    y: np.ndarray,
    *,
    n_top: int,
    n_draws: int,
    seed: int,
    include_structured: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Weighted rank blend over the strongest candidates.

    Weights are selected on the same OOF that is reported. That is selection
    optimism, permitted here and disclosed; it is not direct leakage, because no
    held-out subject's label ever entered a fit or a screening step.
    """

    usable = [r for r in results if r.error is None and np.isfinite(r.subject_mean_auc)]
    if not usable:
        return {"enabled": False, "reason": "no usable candidates"}
    usable.sort(key=lambda r: -r.subject_mean_auc)

    # Prefer diversity: at most 2 entries per family among the top slots.
    chosen: list[CandidateResult] = []
    per_family: dict[str, int] = {}
    for result in usable:
        family = result.candidate.family
        if per_family.get(family, 0) >= 2:
            continue
        chosen.append(result)
        per_family[family] = per_family.get(family, 0) + 1
        if len(chosen) >= int(n_top):
            break
    if len(chosen) < 2:
        chosen = usable[: max(2, min(len(usable), int(n_top)))]

    matrix = np.vstack([r.subject_mean_oof for r in chosen])  # (n_models, n_subjects)
    names = [r.candidate.name for r in chosen]

    best_weights = np.zeros(len(chosen))
    best_weights[0] = 1.0
    best_auc = float(roc_auc(y, matrix[0]))
    best_label = f"single:{names[0]}"

    if include_structured:
        equal = np.ones(len(chosen)) / len(chosen)
        equal_auc = float(roc_auc(y, equal @ matrix))
        if equal_auc > best_auc:
            best_auc, best_weights, best_label = equal_auc, equal, "equal_weight"

    rng = np.random.default_rng(int(seed))
    for _ in range(int(n_draws)):
        weights = rng.dirichlet(np.ones(len(chosen)))
        auc = float(roc_auc(y, weights @ matrix))
        if auc > best_auc:
            best_auc, best_weights, best_label = auc, weights, "simplex_search"
    if progress:
        progress(f"[ensemble] best={best_label} auc={best_auc:.6f} over {len(chosen)} members")

    return {
        "enabled": True,
        "members": names,
        "weights": [float(w) for w in best_weights],
        "selection_kind": best_label,
        "subject_mean_oof_roc_auc": best_auc,
        "blended_scores": (best_weights @ matrix).tolist(),
        "n_simplex_draws": int(n_draws),
        "selection_note": (
            "Members and weights were chosen on the same repeated OOF that this "
            "score reports (non-nested). Treat as a development score."
        ),
    }
