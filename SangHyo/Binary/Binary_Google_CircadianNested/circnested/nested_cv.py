"""Repeated nested subject-level cross-validation engine.

Structure (per outer repeat, per outer fold):

    outer-train subjects
        inner CV (inner_k folds x inner_repeats) evaluates EVERY candidate
        -> candidate selected by mean inner AUC with a simplicity-biased
           tolerance rule (never sees the outer-test subjects)
        -> decision threshold chosen from the selected candidate's inner OOF
    outer-test subjects
        scored once by the model refit on the full outer-train

The engine also stores every candidate's own outer-test predictions (one extra
fit per candidate per fold).  Those per-candidate OOF tracks serve two audit
purposes: they quantify selection optimism (best single candidate chosen ON the
report metric vs the honest nested track), and they provide the pre-registered
paired contrasts (each fusion candidate vs its MMSE-only twin).

Because the feature matrix has exactly one row per subject, a stratified
K-fold over rows IS a subject-level group split; ``assert_fold_partition``
still verifies disjointness and coverage explicitly.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import Candidate, Profile, SELECTION_TOLERANCE, VIEWS
from .evaluation import pick_threshold_balanced, roc_auc_safe, thresholded_metrics
from .features import select_view
from .modeling import build_model


def derive_seed(*parts: int) -> int:
    """Deterministically mix seed components into the sklearn/numpy range."""

    value = 0x9E3779B9
    for part in parts:
        value = (value * 1_000_003 + int(part) + 0x7F4A7C15) % (2**31 - 1)
    return int(value)


def assert_fold_partition(n_subjects: int, folds: list[tuple[np.ndarray, np.ndarray]]) -> None:
    seen = np.zeros(n_subjects, dtype=int)
    for train_index, test_index in folds:
        if np.intersect1d(train_index, test_index).size:
            raise AssertionError("A subject appears in both train and test of one fold")
        seen[test_index] += 1
    if not np.all(seen == 1):
        raise AssertionError("Outer folds do not partition the subjects exactly once")


def stratified_subject_folds(diag: np.ndarray, k: int, seed: int
                             ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Subject-level folds stratified on the 3-class diagnosis.

    Stratifying on CN/MCI/Dem (rather than the binary target) keeps the nine
    Dem subjects spread across folds, which stabilizes fold AUCs.
    """

    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = [
        (np.asarray(tr, dtype=int), np.asarray(te, dtype=int))
        for tr, te in splitter.split(np.zeros(len(diag)), diag)
    ]
    assert_fold_partition(len(diag), folds)
    return folds


def select_candidate(inner_mean_auc: dict[str, float], candidates: list[Candidate],
                     tolerance: float = SELECTION_TOLERANCE) -> str:
    """Highest inner AUC wins, except that any candidate within ``tolerance``
    of the best with LOWER complexity is preferred (simplicity bias)."""

    best_value = max(inner_mean_auc[c.candidate_id] for c in candidates)
    eligible = [
        c for c in candidates
        if inner_mean_auc[c.candidate_id] >= best_value - tolerance
    ]
    winner = min(eligible, key=lambda c: c.complexity)
    return winner.candidate_id


def _inner_evaluation(
    view_frames: dict[str, pd.DataFrame],
    y: np.ndarray,
    diag: np.ndarray,
    train_index: np.ndarray,
    candidates: list[Candidate],
    profile: Profile,
    seed: int,
) -> dict[str, dict]:
    """Evaluate every candidate on inner CV inside one outer-training set.

    Returns, per candidate id: ``inner_mean_auc`` (mean of per-inner-repeat
    pooled AUCs) and ``inner_oof`` (per-subject scores averaged over inner
    repeats; used only for threshold selection).
    """

    y_train = y[train_index]
    diag_train = diag[train_index]
    n_train = len(train_index)
    results = {
        c.candidate_id: {"repeat_aucs": [], "oof_sum": np.zeros(n_train),
                         "oof_count": np.zeros(n_train)}
        for c in candidates
    }

    for inner_repeat in range(profile.inner_repeats):
        inner_seed = derive_seed(seed, 131, inner_repeat)
        splitter = StratifiedKFold(
            n_splits=profile.inner_k, shuffle=True, random_state=inner_seed
        )
        fold_pairs = list(splitter.split(np.zeros(n_train), diag_train))
        repeat_scores = {c.candidate_id: np.full(n_train, np.nan) for c in candidates}
        for inner_fold, (inner_tr, inner_te) in enumerate(fold_pairs):
            fit_seed = derive_seed(inner_seed, 17, inner_fold)
            for candidate in candidates:
                frame = view_frames[candidate.view]
                X_tr = frame.iloc[train_index[inner_tr]]
                X_te = frame.iloc[train_index[inner_te]]
                model = build_model(candidate, fit_seed,
                                    ydf_trees_override=profile.ydf_trees_override)
                model.fit(X_tr, y_train[inner_tr])
                repeat_scores[candidate.candidate_id][inner_te] = model.predict_score(X_te)
        for candidate_id, scores in repeat_scores.items():
            results[candidate_id]["repeat_aucs"].append(roc_auc_safe(y_train, scores))
            observed = np.isfinite(scores)
            results[candidate_id]["oof_sum"][observed] += scores[observed]
            results[candidate_id]["oof_count"][observed] += 1

    out: dict[str, dict] = {}
    for candidate_id, payload in results.items():
        counts = np.maximum(payload["oof_count"], 1)
        out[candidate_id] = {
            "inner_mean_auc": float(np.nanmean(payload["repeat_aucs"])),
            "inner_repeat_aucs": [float(v) for v in payload["repeat_aucs"]],
            "inner_oof": payload["oof_sum"] / counts,
        }
    return out


def run_repeated_nested_cv(
    features: pd.DataFrame,
    y: np.ndarray,
    diag: np.ndarray,
    selection_candidates: list[Candidate],
    audit_candidates: list[Candidate],
    profile: Profile,
    seed: int,
    log: Callable[[str], None] = print,
) -> dict:
    """Run the full protocol and return raw per-fold records plus OOF tracks.

    ``selection_candidates`` compete inside inner CV; ``audit_candidates`` are
    additionally evaluated on the same folds (fixed arms) without competing.
    Tracks returned in ``oof_scores``: ``"nested"`` plus every candidate id.
    """

    every_candidate = list(selection_candidates) + [
        c for c in audit_candidates
        if c.candidate_id not in {s.candidate_id for s in selection_candidates}
    ]
    view_frames = {name: select_view(features, columns) for name, columns in VIEWS.items()
                   if name in {c.view for c in every_candidate}}

    n_subjects = len(features)
    track_ids = ["nested"] + [c.candidate_id for c in every_candidate]
    oof_scores = {track: np.full((profile.outer_repeats, n_subjects), np.nan)
                  for track in track_ids}
    nested_flags = np.zeros((profile.outer_repeats, n_subjects), dtype=int)

    fold_records: list[dict] = []
    start = time.monotonic()

    for outer_repeat in range(profile.outer_repeats):
        repeat_seed = derive_seed(seed, 1000, outer_repeat)
        folds = stratified_subject_folds(diag, profile.outer_k, repeat_seed)
        for fold_index, (train_index, test_index) in enumerate(folds):
            fold_seed = derive_seed(repeat_seed, 31, fold_index)
            # Fixed audit arms never compete, so only selection candidates pay
            # the inner-CV cost.
            inner = _inner_evaluation(
                view_frames, y, diag, train_index, selection_candidates, profile, fold_seed
            )
            inner_mean = {cid: inner[cid]["inner_mean_auc"] for cid in inner}
            selected_id = select_candidate(inner_mean, selection_candidates)

            # One refit per candidate on the full outer-training set.
            candidate_test_scores: dict[str, np.ndarray] = {}
            for candidate in every_candidate:
                frame = view_frames[candidate.view]
                model = build_model(candidate, fold_seed,
                                    ydf_trees_override=profile.ydf_trees_override)
                model.fit(frame.iloc[train_index], y[train_index])
                scores = model.predict_score(frame.iloc[test_index])
                candidate_test_scores[candidate.candidate_id] = scores
                oof_scores[candidate.candidate_id][outer_repeat, test_index] = scores

            nested_test_scores = candidate_test_scores[selected_id]
            oof_scores["nested"][outer_repeat, test_index] = nested_test_scores
            nested_flags[outer_repeat, test_index] = 1

            threshold = pick_threshold_balanced(
                y[train_index], inner[selected_id]["inner_oof"]
            )
            fold_records.append(
                {
                    "outer_repeat": outer_repeat,
                    "outer_fold": fold_index,
                    "n_train": int(len(train_index)),
                    "n_test": int(len(test_index)),
                    "selected_candidate": selected_id,
                    "selected_inner_mean_auc": inner_mean[selected_id],
                    "inner_mean_auc_by_candidate": {
                        cid: float(value) for cid, value in inner_mean.items()
                    },
                    "outer_test_auc_selected": roc_auc_safe(y[test_index], nested_test_scores),
                    "threshold_from_inner_oof": float(threshold),
                    "outer_test_thresholded": thresholded_metrics(
                        y[test_index], nested_test_scores, threshold
                    ),
                }
            )
        elapsed = time.monotonic() - start
        log(
            f"  [nested-cv] repeat {outer_repeat + 1}/{profile.outer_repeats} done "
            f"({elapsed:6.1f}s elapsed)"
        )

    for track, matrix in oof_scores.items():
        if np.isnan(matrix).any():
            raise AssertionError(f"OOF track {track!r} has unscored subjects")
    if not np.all(nested_flags == 1):
        raise AssertionError("Nested OOF does not cover every subject exactly once per repeat")

    return {"oof_scores": oof_scores, "fold_records": fold_records}


def select_and_fit_deployment(
    features: pd.DataFrame,
    y: np.ndarray,
    diag: np.ndarray,
    selection_candidates: list[Candidate],
    profile: Profile,
    seed: int,
) -> dict:
    """Apply the identical inner-CV selection to ALL training subjects, then
    refit the winner on the full training cohort (for the frozen validation
    pass).  No validation subject or label is visible here."""

    view_frames = {name: select_view(features, columns) for name, columns in VIEWS.items()
                   if name in {c.view for c in selection_candidates}}
    all_index = np.arange(len(features))
    inner = _inner_evaluation(
        view_frames, y, diag, all_index, selection_candidates, profile, derive_seed(seed, 7, 3)
    )
    inner_mean = {cid: inner[cid]["inner_mean_auc"] for cid in inner}
    selected_id = select_candidate(inner_mean, selection_candidates)
    selected = next(c for c in selection_candidates if c.candidate_id == selected_id)

    model = build_model(selected, derive_seed(seed, 7, 5),
                        ydf_trees_override=profile.ydf_trees_override)
    model.fit(view_frames[selected.view], y)
    threshold = pick_threshold_balanced(y, inner[selected_id]["inner_oof"])
    return {
        "model": model,
        "candidate": selected,
        "threshold": float(threshold),
        "inner_mean_auc_by_candidate": {k: float(v) for k, v in inner_mean.items()},
    }


__all__ = [
    "assert_fold_partition", "run_repeated_nested_cv", "select_and_fit_deployment",
    "select_candidate", "stratified_subject_folds",
]
