"""Repeated nested subject-level CV with fold-local SSL pretraining.

Per outer repeat, per outer fold:

    outer-train subjects (~113)
        1. MAE pretraining on THEIR days only (labels unused; outer-test
           subjects contribute zero minutes) + fold-local channel stats
        2. frozen encoder -> per-subject mean+std embedding view
        3. inner CV (inner_k x inner_repeats) evaluates every candidate and
           every pre-registered probe config on the embedding / FE views
           -> per candidate: best config by mean inner AUC
           -> across candidates: simplicity-biased tolerance selection
           -> decision threshold from the selected candidate's inner OOF
    outer-test subjects (~28)
        scored once by every candidate refit on the full outer-train
        (per-candidate audit tracks) and once by the selected candidate
        (the honest "nested" track)

Fold construction (StratifiedKFold on the 3-class diagnosis, seed 20260813)
is verbatim from Binary_Google_TabFM_Nested, so outer folds are paired
repeat-by-repeat across the two experiments.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import (
    Candidate,
    PretrainBudget,
    Profile,
    SELECTION_TOLERANCE,
    ProbeConfig,
)
from .evaluation import pick_threshold_balanced, roc_auc_safe, thresholded_metrics
from .grids import DayBank
from .probes import BlendProbe, LinearProbe

# NOTE: ``.pretrain`` (and through it torch) is imported lazily inside the two
# engine functions, so the fold/selection utilities in this module stay usable
# in torch-free environments (local contract tests).

HEARTBEAT_SECONDS = 60.0


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
    """Subject-level folds stratified on CN/MCI/Dem (keeps Dem spread out)."""

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


# ------------------------------------------------------------ fit machinery --
def _fit_and_score(candidate: Candidate, config: ProbeConfig | None,
                   fe: pd.DataFrame, emb: pd.DataFrame, y: np.ndarray,
                   train_rows: np.ndarray, test_rows: np.ndarray,
                   seed: int) -> np.ndarray:
    """Fit one candidate on train rows, return scores on test rows."""

    y_train = y[train_rows]
    if candidate.view == "blend":
        model = BlendProbe(seed).fit(fe.iloc[train_rows], emb.iloc[train_rows], y_train)
        return model.predict_score(fe.iloc[test_rows], emb.iloc[test_rows])
    frame = fe if candidate.view == "fe" else emb
    if config is None:
        raise ValueError(f"candidate {candidate.candidate_id} requires a config")
    model = LinearProbe(config, seed).fit(frame.iloc[train_rows], y_train)
    return model.predict_score(frame.iloc[test_rows])


def _inner_evaluation(
    fe: pd.DataFrame,
    emb: pd.DataFrame,
    y: np.ndarray,
    diag: np.ndarray,
    train_index: np.ndarray,
    candidates: list[Candidate],
    profile: Profile,
    seed: int,
    on_step: Callable[[], None] | None = None,
) -> dict[str, dict]:
    """Inner CV over candidates x configs inside one outer-training set.

    Returns per candidate id: chosen config, its mean inner AUC, per-config
    AUC table, and the chosen config's inner OOF scores (threshold source).
    """

    y_train = y[train_index]
    diag_train = diag[train_index]
    n_train = len(train_index)

    def _config_list(candidate: Candidate) -> list[ProbeConfig | None]:
        return list(candidate.configs) if candidate.configs else [None]

    accumulator: dict[tuple[str, str], dict] = {}
    for candidate in candidates:
        for config in _config_list(candidate):
            key = (candidate.candidate_id, config.key() if config else "fixed")
            accumulator[key] = {"repeat_aucs": [], "oof_sum": np.zeros(n_train),
                               "oof_count": np.zeros(n_train)}

    for inner_repeat in range(profile.inner_repeats):
        inner_seed = derive_seed(seed, 131, inner_repeat)
        splitter = StratifiedKFold(
            n_splits=profile.inner_k, shuffle=True, random_state=inner_seed
        )
        fold_pairs = list(splitter.split(np.zeros(n_train), diag_train))
        repeat_scores = {key: np.full(n_train, np.nan) for key in accumulator}
        for inner_fold, (inner_tr, inner_te) in enumerate(fold_pairs):
            fit_seed = derive_seed(inner_seed, 17, inner_fold)
            for candidate in candidates:
                for config in _config_list(candidate):
                    key = (candidate.candidate_id, config.key() if config else "fixed")
                    scores = _fit_and_score(
                        candidate, config, fe, emb, y,
                        train_index[inner_tr], train_index[inner_te], fit_seed,
                    )
                    repeat_scores[key][inner_te] = scores
                    if on_step is not None:
                        on_step()
        for key, scores in repeat_scores.items():
            accumulator[key]["repeat_aucs"].append(roc_auc_safe(y_train, scores))
            observed = np.isfinite(scores)
            accumulator[key]["oof_sum"][observed] += scores[observed]
            accumulator[key]["oof_count"][observed] += 1

    out: dict[str, dict] = {}
    for candidate in candidates:
        per_config: dict[str, float] = {}
        payloads: dict[str, dict] = {}
        for config in _config_list(candidate):
            config_key = config.key() if config else "fixed"
            payload = accumulator[(candidate.candidate_id, config_key)]
            per_config[config_key] = float(np.nanmean(payload["repeat_aucs"]))
            payloads[config_key] = payload
        # Best config; ties broken toward the earlier (pre-registered simpler)
        # entry because dict order follows the config tuple order.
        best_key = max(per_config, key=lambda k: (per_config[k],
                                                  -list(per_config).index(k)))
        chosen = payloads[best_key]
        counts = np.maximum(chosen["oof_count"], 1)
        configs = _config_list(candidate)
        chosen_config = configs[list(per_config).index(best_key)]
        out[candidate.candidate_id] = {
            "chosen_config": best_key,
            "chosen_config_object": chosen_config,
            "inner_mean_auc": per_config[best_key],
            "inner_auc_by_config": per_config,
            "inner_repeat_aucs": [float(v) for v in chosen["repeat_aucs"]],
            "inner_oof": chosen["oof_sum"] / counts,
        }
    return out


# ---------------------------------------------------------------- main loop --
def run_repeated_nested_cv(
    bank: DayBank,
    fe: pd.DataFrame,
    y: np.ndarray,
    diag: np.ndarray,
    candidates: list[Candidate],
    profile: Profile,
    budget: PretrainBudget,
    seed: int,
    device: str,
    log: Callable[[str], None] = print,
    on_fold: Callable[[dict], None] | None = None,
) -> dict:
    """Full protocol; returns OOF tracks, fold records and pretrain diagnostics."""

    from .pretrain import embed_all_subjects, pretrain_encoder

    n_subjects = len(bank.subject_ids)
    if list(fe.index) != list(bank.subject_ids):
        raise AssertionError("FE frame and DayBank subject order disagree")

    track_ids = ["nested"] + [c.candidate_id for c in candidates]
    oof_scores = {track: np.full((profile.outer_repeats, n_subjects), np.nan)
                  for track in track_ids}
    nested_flags = np.zeros((profile.outer_repeats, n_subjects), dtype=int)

    fold_records: list[dict] = []
    pretrain_records: list[dict] = []
    start = time.monotonic()
    total_folds = profile.outer_k * profile.outer_repeats
    folds_done = 0
    last_heartbeat = [time.monotonic()]

    def _heartbeat() -> None:
        now = time.monotonic()
        if now - last_heartbeat[0] < HEARTBEAT_SECONDS:
            return
        last_heartbeat[0] = now
        elapsed = now - start
        log(f"    [alive] fold {folds_done + 1}/{total_folds} in progress | "
            f"elapsed {elapsed / 60:5.1f} min")

    for outer_repeat in range(profile.outer_repeats):
        repeat_seed = derive_seed(seed, 1000, outer_repeat)
        folds = stratified_subject_folds(diag, profile.outer_k, repeat_seed)
        for fold_index, (train_index, test_index) in enumerate(folds):
            fold_started = time.monotonic()
            fold_seed = derive_seed(repeat_seed, 31, fold_index)

            # 1) fold-local SSL pretraining (outer-train subjects only).
            pretrained = pretrain_encoder(
                bank, train_index, profile.model_variant, budget,
                seed=derive_seed(fold_seed, 71), device=device, log=log,
                heartbeat=_heartbeat,
            )
            pretrain_records.append(
                {
                    "outer_repeat": outer_repeat, "outer_fold": fold_index,
                    "best_epoch": pretrained["best_epoch"],
                    "best_val_mse": pretrained["best_val_mse"],
                    "epochs_ran": pretrained["epochs_ran"],
                    "n_pretrain_days": pretrained["n_pretrain_days"],
                    "n_val_days": pretrained["n_val_days"],
                    "seconds": pretrained["seconds"],
                    "loss_curve_every_10": pretrained["history"][::10],
                }
            )

            # 2) frozen embeddings for everyone (inference only on outer-test).
            emb = embed_all_subjects(
                bank, pretrained["model"], pretrained["channel_mean"],
                pretrained["channel_std"], device,
            )

            # 3) inner CV -> config choice per candidate + candidate selection.
            inner = _inner_evaluation(
                fe, emb, y, diag, train_index, candidates, profile, fold_seed,
                on_step=_heartbeat,
            )
            inner_mean = {cid: inner[cid]["inner_mean_auc"] for cid in inner}
            selected_id = select_candidate(inner_mean, candidates)

            # 4) refit every candidate on the full outer-train, score the test.
            candidate_test_scores: dict[str, np.ndarray] = {}
            for candidate in candidates:
                config = inner[candidate.candidate_id]["chosen_config_object"]
                scores = _fit_and_score(
                    candidate, config, fe, emb, y, train_index, test_index,
                    derive_seed(fold_seed, 53),
                )
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
                    "selected_config": inner[selected_id]["chosen_config"],
                    "selected_inner_mean_auc": inner_mean[selected_id],
                    "inner_mean_auc_by_candidate": {
                        cid: float(value) for cid, value in inner_mean.items()
                    },
                    "inner_auc_by_config": {
                        cid: inner[cid]["inner_auc_by_config"] for cid in inner
                    },
                    "outer_test_auc_selected": roc_auc_safe(y[test_index], nested_test_scores),
                    "threshold_from_inner_oof": float(threshold),
                    "outer_test_thresholded": thresholded_metrics(
                        y[test_index], nested_test_scores, threshold
                    ),
                    "pretrain_best_val_mse": pretrained["best_val_mse"],
                }
            )
            folds_done += 1
            elapsed = time.monotonic() - start
            eta = elapsed / folds_done * (total_folds - folds_done)
            log(
                f"  [nested-cv] fold {folds_done}/{total_folds} "
                f"(repeat {outer_repeat + 1}/{profile.outer_repeats}, "
                f"fold {fold_index + 1}/{profile.outer_k}) "
                f"took {time.monotonic() - fold_started:6.1f}s "
                f"(pretrain {pretrained['seconds']:6.1f}s) | "
                f"elapsed {elapsed / 60:5.1f} min | ETA {eta / 60:5.1f} min | "
                f"picked {selected_id}"
            )
            if on_fold is not None:
                on_fold(
                    {
                        "folds_done": folds_done,
                        "total_folds": total_folds,
                        "elapsed_minutes": round(elapsed / 60.0, 2),
                        "eta_minutes": round(eta / 60.0, 2),
                        "last_selected_candidate": selected_id,
                        "last_outer_test_auc": fold_records[-1]["outer_test_auc_selected"],
                        "last_pretrain_val_mse": pretrained["best_val_mse"],
                    }
                )

    for track, matrix in oof_scores.items():
        if np.isnan(matrix).any():
            raise AssertionError(f"OOF track {track!r} has unscored subjects")
    if not np.all(nested_flags == 1):
        raise AssertionError("Nested OOF does not cover every subject exactly once per repeat")

    return {
        "oof_scores": oof_scores,
        "fold_records": fold_records,
        "pretrain_records": pretrain_records,
    }


def select_and_fit_deployment(
    bank: DayBank,
    fe: pd.DataFrame,
    y: np.ndarray,
    diag: np.ndarray,
    candidates: list[Candidate],
    profile: Profile,
    budget: PretrainBudget,
    seed: int,
    device: str,
    log: Callable[[str], None] = print,
) -> dict:
    """Identical selection applied to ALL training subjects, then one refit.

    Pretrains a deployment encoder on the full 141-subject training cohort
    (still zero validation minutes), runs the same inner CV to pick candidate
    + config, refits on everyone.  No validation subject or label is visible.
    """

    from .pretrain import embed_all_subjects, pretrain_encoder

    all_index = np.arange(len(bank.subject_ids))
    pretrained = pretrain_encoder(
        bank, all_index, profile.model_variant, budget,
        seed=derive_seed(seed, 7, 1), device=device, log=log,
    )
    emb = embed_all_subjects(
        bank, pretrained["model"], pretrained["channel_mean"],
        pretrained["channel_std"], device,
    )
    inner = _inner_evaluation(
        fe, emb, y, diag, all_index, candidates, profile, derive_seed(seed, 7, 3)
    )
    inner_mean = {cid: inner[cid]["inner_mean_auc"] for cid in inner}
    selected_id = select_candidate(inner_mean, candidates)
    selected = next(c for c in candidates if c.candidate_id == selected_id)
    config = inner[selected_id]["chosen_config_object"]

    fit_seed = derive_seed(seed, 7, 5)
    if selected.view == "blend":
        model: BlendProbe | LinearProbe = BlendProbe(fit_seed).fit(fe, emb, y)
    else:
        frame = fe if selected.view == "fe" else emb
        model = LinearProbe(config, fit_seed).fit(frame, y)
    threshold = pick_threshold_balanced(y, inner[selected_id]["inner_oof"])
    return {
        "pretrained": pretrained,
        "embedding_frame": emb,
        "model": model,
        "candidate": selected,
        "config": config,
        "threshold": float(threshold),
        "inner_mean_auc_by_candidate": {k: float(v) for k, v in inner_mean.items()},
        "inner_auc_by_config": {cid: inner[cid]["inner_auc_by_config"] for cid in inner},
    }


__all__ = [
    "assert_fold_partition", "derive_seed", "run_repeated_nested_cv",
    "select_and_fit_deployment", "select_candidate", "stratified_subject_folds",
]
