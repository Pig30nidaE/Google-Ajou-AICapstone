"""Single entry point for Binary_Google_SensorFM_Nested.

Colab / base.ipynb (Cell 2)
---------------------------
    USER_FOLDER = "SangHyo"
    RUN_FILE    = "Binary/Binary_Google_SensorFM_Nested/run.py"

    # optional, before Cell 5:
    # import os; os.environ["BGSFM_ARGS"] = "--profile quick"

Runtime: **GPU required for formal profiles** (A100/T4).  The MAE pretraining
runs once per outer fold (50 pretrainings on the default profile), so right
after the day grids are built the script measures real training steps on the
real data and prints a projected total; an infeasible runtime is visible in
the first minutes, not hours in.

Recommended order: ``--profile quick`` first (2 outer repeats, 40-epoch
budget, feasibility + projection), then ``default`` (10 repeats, primary).

Pipeline stages (in order):
    1. environment check: torch present, device resolved (no silent CPU run
       of a formal profile), versions recorded
    2. labels + day grids (CONVERT intraday strings -> 1440x8 minute grids)
       + the paper's engineered-feature baseline matrix (subject-local)
    3. measured micro-benchmark of real pretraining steps -> projected total
    4. repeated nested subject-level CV with FOLD-LOCAL SSL pretraining
       (primary evidence; PROGRESS.json updated after every outer fold)
    5. deployment: pretrain on all 141, inner-CV selection, refit
    6. frozen prediction of the 33 historical validation subjects
       (SHA-256 freeze BEFORE validation labels are opened) + encoder/probe
       round-trip reload check
    7. one-shot validation scoring + FINAL_REPORT.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_NAME = "Binary_Google_SensorFM_Nested"

# ``base.ipynb`` executes this file with ``runpy`` (no package context), so the
# experiment folder itself is put on ``sys.path`` and any stale module from a
# previous in-process run is dropped.
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
for _name in [m for m in list(sys.modules)
              if m == "sensorfmnested" or m.startswith("sensorfmnested.")]:
    sys.modules.pop(_name, None)


# ----------------------------------------------------------- environment ----
def _in_notebook_host() -> bool:
    """True inside a notebook kernel (runpy keeps the kernel's ``-f ...``
    argv; parsing it as experiment args was the CircadianNested post-mortem)."""

    ipython = sys.modules.get("IPython")
    if ipython is None:
        return False
    try:
        return ipython.get_ipython() is not None
    except Exception:  # pragma: no cover - defensive
        return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--profile", default="default",
                        choices=("smoke", "quick", "default", "max"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None,
                        help="cuda | mps | cpu (default: auto)")
    argv = [] if _in_notebook_host() else sys.argv[1:]
    if not argv and os.environ.get("BGSFM_ARGS"):
        argv = shlex.split(os.environ["BGSFM_ARGS"])
    return parser.parse_args(argv)


def _ensure_torch() -> None:
    """PyTorch is the SSL engine; fail-closed with a clear instruction.

    Colab ships torch preinstalled.  We deliberately do NOT pip-install a
    multi-GB wheel behind the user's back.
    """

    import importlib.util

    if importlib.util.find_spec("torch") is None:
        raise RuntimeError(
            "PyTorch is not importable. This experiment re-implements the "
            "SensorFM MAE in torch (no public checkpoint exists to download). "
            "On Colab use a GPU runtime (torch is preinstalled); elsewhere "
            "install torch>=2.1 first."
        )


def _resolve_data_root(namespace: dict, explicit: str | None) -> Path:
    candidates: list[Path] = []
    for value in (explicit, namespace.get("DATA_ROOT"), os.environ.get("BGSFM_DATA_ROOT")):
        if value:
            candidates.append(Path(os.fspath(value)).expanduser())
    project_root = namespace.get("PROJECT_ROOT")
    if project_root:
        candidates.append(Path(os.fspath(project_root)) / "Data")
    candidates += [
        EXPERIMENT_ROOT.parents[2] / "Data",
        Path("/content/drive/Shareddrives/GoogleAI_contest/Data"),
        Path("/content/drive/MyDrive/GoogleAI_contest/Data"),
    ]
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "1.Training").is_dir() and (resolved / "2.Validation").is_dir():
            return resolved
    raise FileNotFoundError(
        "Data root with 1.Training and 2.Validation not found. Checked: "
        + ", ".join(str(c) for c in candidates)
    )


def _resolve_output_dir(explicit: str | None, run_id: str) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if os.environ.get("BGSFM_OUTPUT_DIR"):
        return Path(os.environ["BGSFM_OUTPUT_DIR"]).expanduser().resolve() / run_id
    drive = Path("/content/drive/MyDrive")
    if drive.is_dir():
        return drive / f"{EXPERIMENT_NAME}_result" / run_id
    return EXPERIMENT_ROOT / f"{EXPERIMENT_NAME}_result" / run_id


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                    encoding="utf-8")


def _sha256_of(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------- projection ------
def _measure_and_project(log, bank, profile, budget, device, seed) -> dict:
    """Run a few REAL pretraining steps and project the whole run's cost.

    Lesson from previous long runs: fit-count ETAs undercount row-scaled
    engines, so the projection here is measured seconds-per-step on the real
    tensors, times the planned step count (upper bound: no early stopping).
    """

    import numpy as np

    from sensorfmnested.config import MODEL_VARIANTS
    from sensorfmnested.grids import DayBank  # noqa: F401 (typing only)
    from sensorfmnested.mae import (
        SensorFMMae, sample_artificial_masks, token_observed_mask,
    )
    from sensorfmnested.pretrain import _to_device  # shared tensor path
    import torch

    variant = MODEL_VARIANTS[profile.model_variant]
    model = SensorFMMae(variant).to(device)
    n_parameters = model.parameter_count()
    optimizer = torch.optim.AdamW(model.parameters(), lr=budget.base_lr)
    n_days_total = int(bank.day_subject.size)
    batch = np.arange(min(budget.batch_size, n_days_total))
    mean, std = bank.fold_channel_stats(np.arange(len(bank.subject_ids)))
    generator = torch.Generator().manual_seed(seed)

    warm_steps, timed_steps = 1, 3
    times = []
    for step in range(warm_steps + timed_steps):
        minutes, element_mask, meta = _to_device(bank, batch, mean, std, device)
        observed = token_observed_mask(torch.from_numpy(bank.mask[batch]))
        artificial = sample_artificial_masks(observed, generator).to(device)
        started = time.monotonic()
        out = model(minutes, element_mask, meta, artificial)
        optimizer.zero_grad(set_to_none=True)
        out["loss"].backward()
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        if step >= warm_steps:
            times.append(time.monotonic() - started)
    seconds_per_step = float(np.median(times))

    outer_train_days = n_days_total * (1 - 1 / profile.outer_k)
    steps_per_epoch = math.ceil(outer_train_days * 0.9 / budget.batch_size)
    embed_steps = math.ceil(n_days_total / budget.batch_size)
    folds = profile.outer_k * profile.outer_repeats
    # per fold: pretrain epochs (train + val ~ +12%) + one full embedding pass
    fold_steps = budget.epochs * steps_per_epoch * 1.12 + embed_steps
    deploy_steps = budget.epochs * math.ceil(n_days_total * 0.9 / budget.batch_size) * 1.12
    total_steps = folds * fold_steps + deploy_steps + embed_steps
    projected = total_steps * seconds_per_step

    del model, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()

    log(f"[probe] variant {variant.name} ({n_parameters:,} params) "
        f"| measured {seconds_per_step * 1000:.0f} ms/step "
        f"(batch {len(batch)} days, device {device})")
    log(f"[probe] planned ~{int(total_steps):,} steps over {folds} folds "
        f"-> PROJECTED TOTAL {projected / 60:.0f} min ({projected / 3600:.1f} h) "
        f"upper bound (early stopping shortens it)")
    if projected > 6 * 3600:
        log("[probe] WARNING: projection exceeds the 6-hour budget. Use "
            "--profile quick, a stronger GPU, or reduce epochs before a full run.")
    return {
        "seconds_per_step": round(seconds_per_step, 4),
        "planned_steps_upper_bound": int(total_steps),
        "projected_minutes_upper_bound": round(projected / 60.0, 1),
        "device": device,
    }


# ------------------------------------------------------------ aggregation ----
def _aggregate_tracks(oof_scores, y, diag, n_bootstrap, seed):
    import numpy as np

    from sensorfmnested.evaluation import (
        cn_vs_mci_auc, roc_auc_safe, score_metrics, subject_bootstrap_auc_ci,
    )

    tracks = {}
    subject_means = {}
    for track, matrix in oof_scores.items():
        repeat_aucs = [roc_auc_safe(y, matrix[r]) for r in range(matrix.shape[0])]
        mean_scores = matrix.mean(axis=0)
        subject_means[track] = mean_scores
        tracks[track] = {
            "per_repeat_pooled_roc_auc": [float(v) for v in repeat_aucs],
            "repeat_roc_auc_mean": float(np.nanmean(repeat_aucs)),
            "repeat_roc_auc_sd": float(np.nanstd(repeat_aucs, ddof=0)),
            "subject_mean_oof": score_metrics(y, mean_scores),
            "subject_mean_oof_cn_vs_mci_auc": float(cn_vs_mci_auc(diag, mean_scores)),
            "subject_mean_oof_bootstrap": subject_bootstrap_auc_ci(
                y, mean_scores, n_boot=n_bootstrap, seed=seed
            ),
        }
    return tracks, subject_means


def _nested_threshold_summary(fold_records, outer_repeats):
    import numpy as np

    per_repeat = []
    for repeat in range(outer_repeats):
        rows = [r["outer_test_thresholded"] for r in fold_records
                if r["outer_repeat"] == repeat]
        tp = sum(r["tp"] for r in rows)
        tn = sum(r["tn"] for r in rows)
        fp = sum(r["fp"] for r in rows)
        fn = sum(r["fn"] for r in rows)
        sensitivity = tp / max(1, tp + fn)
        specificity = tn / max(1, tn + fp)
        per_repeat.append(
            {
                "repeat": repeat, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
                "sensitivity_recall": sensitivity,
                "specificity": specificity,
                "balanced_accuracy": 0.5 * (sensitivity + specificity),
            }
        )
    return {
        "per_repeat": per_repeat,
        "balanced_accuracy_mean": float(
            np.mean([r["balanced_accuracy"] for r in per_repeat])
        ),
        "sensitivity_mean": float(np.mean([r["sensitivity_recall"] for r in per_repeat])),
        "specificity_mean": float(np.mean([r["specificity"] for r in per_repeat])),
    }


# ------------------------------------------------------------------- main ----
def main(namespace: dict) -> None:
    arguments = _parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
    output = _resolve_output_dir(arguments.output_dir, run_id)
    output.mkdir(parents=True, exist_ok=True)
    training_dir = output / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        print(message, flush=True)

    status = {
        "experiment": EXPERIMENT_NAME, "run_id": run_id, "status": "starting",
        "profile": arguments.profile, "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output / "LAUNCHER_STATUS.json", status)

    try:
        _ensure_torch()

        import numpy as np
        import pandas as pd
        import sklearn
        import torch

        from sensorfmnested import config as C
        from sensorfmnested.data import assert_disjoint_splits, load_labels, subject_hash
        from sensorfmnested.evaluation import (
            cn_vs_mci_auc, paired_bootstrap_auc_diff, score_metrics,
            thresholded_metrics,
        )
        from sensorfmnested.fe_features import build_fe_features
        from sensorfmnested.grids import (
            bank_summary, build_day_bank, grid_fingerprint, load_split_grids,
        )
        from sensorfmnested.mae import SensorFMMae
        from sensorfmnested.nested_cv import (
            run_repeated_nested_cv, select_and_fit_deployment,
        )
        from sensorfmnested.pretrain import embed_all_subjects, resolve_device

        profile = C.PROFILES[arguments.profile]
        budget = C.PRETRAIN_BUDGETS[arguments.profile]
        seed = int(arguments.seed) if arguments.seed is not None else C.SEED
        data_root = _resolve_data_root(namespace, arguments.data_root)
        device = resolve_device(arguments.device)
        if device == "cpu" and profile.name in ("default", "max"):
            raise RuntimeError(
                "Formal profiles require a GPU (cuda). A silent CPU run would "
                "blow the 6-hour budget; use --profile smoke/quick on CPU or "
                "switch the Colab runtime to GPU."
            )

        candidates = [
            c for c in C.CANDIDATES if c.candidate_id in set(profile.candidate_ids)
        ]

        run_config = {
            "experiment": EXPERIMENT_NAME,
            "task": C.TASK_DESCRIPTION,
            "run_id": run_id,
            "profile": profile.name,
            "profile_detail": {
                "outer_k": profile.outer_k, "outer_repeats": profile.outer_repeats,
                "inner_k": profile.inner_k, "inner_repeats": profile.inner_repeats,
                "n_bootstrap": profile.n_bootstrap,
                "model_variant": profile.model_variant,
                "max_days_per_subject": profile.max_days_per_subject,
                "candidate_ids": list(profile.candidate_ids),
            },
            "pretrain_budget": budget.__dict__,
            "seed": seed,
            "selection_tolerance_auc": C.SELECTION_TOLERANCE,
            "data_root": str(data_root),
            "output_dir": str(output),
            "modality_contract": "wearable-only (MMSE source file never opened)",
            "google_technology": {
                "engine": "SensorFM recipe (Google Research / Google DeepMind)",
                "role": (
                    "no public checkpoint or code exists (verified 2026-08); this "
                    "run re-implements the published architecture + training "
                    "recipe (ViT-1D MAE, AIM masking, patch [20,1], Table ED.4 "
                    f"variant {profile.model_variant}, M.3.4 downstream protocol) "
                    "and pretrains it from scratch inside each outer fold"
                ),
                "paper": C.SENSORFM_PAPER,
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sklearn": sklearn.__version__,
                "torch": torch.__version__,
                "device": device,
                "cuda_device": (torch.cuda.get_device_name(0)
                                if device == "cuda" else None),
                "platform": platform.platform(),
            },
            "benchmark_anchors": C.BENCHMARK,
        }

        log("=" * 78)
        log(f"{EXPERIMENT_NAME}  |  run {run_id}  |  profile {profile.name}")
        log(f"task            : {C.TASK_DESCRIPTION}")
        log(f"data root       : {data_root}")
        log(f"output          : {output}")
        log(f"seed            : {seed} (outer-fold parity with TabFM/CircadianNested)")
        log(f"outer CV        : {profile.outer_k} folds x {profile.outer_repeats} repeats"
            f" | inner {profile.inner_k} x {profile.inner_repeats}")
        log(f"model variant   : SensorFM-{profile.model_variant} (Table ED.4), "
            f"pretrained per outer fold, device {device}")
        log(f"candidates      : {', '.join(profile.candidate_ids)}")
        log("=" * 78)
        if profile.name == "smoke":
            log("!! SMOKE RUN: wiring check only; numbers are NOT performance !!")

        # ------------------------------------------------ data + grids -------
        stage_start = time.monotonic()
        train_labels = load_labels(data_root, "train")
        log(f"[data] training subjects: {len(train_labels)} "
            f"({train_labels['diag'].value_counts().to_dict()})")

        grids = load_split_grids(
            data_root, "train", max_days_per_subject=profile.max_days_per_subject
        )
        bank = build_day_bank(grids, list(train_labels.index))
        summary = bank_summary(bank)
        fingerprint = grid_fingerprint(bank)
        log(f"[grids] {summary['n_days_total']} admitted days "
            f"({summary['days_per_subject_min']}-{summary['days_per_subject_max']} "
            f"per subject, median {summary['days_per_subject_median']:.0f}) | "
            f"observed cells {summary['observed_cell_fraction']:.3f} | "
            f"values {summary['memory_mb_values']} MB | fingerprint {fingerprint} | "
            f"{time.monotonic() - stage_start:.1f}s")

        fe_start = time.monotonic()
        fe = build_fe_features(bank)
        forbidden = [c for c in fe.columns
                     if any(t in c.lower() for t in C.FORBIDDEN_SUBSTRINGS)
                     or any(t in c.lower() for t in C.FORBIDDEN_FEATURE_TOKENS)]
        if forbidden:
            raise AssertionError(f"Forbidden feature names built: {forbidden[:5]}")
        log(f"[features] FE baseline matrix {fe.shape[0]} x {fe.shape[1]} "
            f"in {time.monotonic() - fe_start:.1f}s")

        y = train_labels["y"].to_numpy(int)
        diag = train_labels["diag"].to_numpy(object)

        eda = {
            "n_subjects": int(len(train_labels)),
            "class_distribution": train_labels["diag"].value_counts().to_dict(),
            "binary_positive_rate": float(y.mean()),
            "grid_summary": summary,
            "grid_fingerprint_train": fingerprint,
            "n_fe_features": int(fe.shape[1]),
        }
        _write_json(output / "eda" / "summary.json", eda)

        # --------------------------------------------- measured projection ---
        probe = _measure_and_project(log, bank, profile, budget, device, seed)

        # -------------------------------------------------- nested CV --------
        log(f"[nested-cv] starting: {len(candidates)} candidates, "
            f"fold-local {profile.model_variant} pretraining")
        cv_start = time.monotonic()

        def _persist_progress(payload: dict) -> None:
            _write_json(
                output / "PROGRESS.json",
                {"run_id": run_id, "profile": profile.name,
                 "updated_utc": datetime.now(timezone.utc).isoformat(), **payload},
            )

        nested = run_repeated_nested_cv(
            bank, fe, y, diag, candidates, profile, budget, seed, device,
            log=log, on_fold=_persist_progress,
        )
        log(f"[nested-cv] finished in {(time.monotonic() - cv_start) / 60:.1f} min")

        tracks, subject_means = _aggregate_tracks(
            nested["oof_scores"], y, diag, profile.n_bootstrap, seed
        )

        from collections import Counter
        selected_counts = Counter(
            record["selected_candidate"] for record in nested["fold_records"]
        )
        candidate_tracks = {
            track: entry for track, entry in tracks.items() if track != "nested"
        }
        best_single_id = max(
            candidate_tracks,
            key=lambda track: candidate_tracks[track]["subject_mean_oof"]["roc_auc"],
        )
        optimism = (
            candidate_tracks[best_single_id]["subject_mean_oof"]["roc_auc"]
            - tracks["nested"]["subject_mean_oof"]["roc_auc"]
        )

        contrasts = {}
        for track_a, track_b in C.PAIRED_CONTRASTS:
            if track_a in subject_means and track_b in subject_means:
                contrasts[f"{track_a}__minus__{track_b}"] = paired_bootstrap_auc_diff(
                    y, subject_means[track_a], subject_means[track_b],
                    n_boot=profile.n_bootstrap, seed=seed + 17,
                )

        threshold_summary = _nested_threshold_summary(
            nested["fold_records"], profile.outer_repeats
        )

        _write_json(training_dir / "fold_results.json",
                    {"fold_records": nested["fold_records"]})
        _write_json(training_dir / "pretrain_records.json",
                    {"pretrain_records": nested["pretrain_records"]})

        oof_frame = pd.DataFrame(
            {
                "subject_hash": [subject_hash(s) for s in train_labels.index],
                "diag": train_labels["diag"].to_numpy(),
                "y": y,
            }
        )
        for track, scores in subject_means.items():
            oof_frame[f"score_mean__{track}"] = scores
        oof_frame.to_csv(training_dir / "oof_predictions_hashed.csv", index=False)

        log(
            "[result] nested OOF: repeat AUC "
            f"{tracks['nested']['repeat_roc_auc_mean']:.4f} "
            f"+- {tracks['nested']['repeat_roc_auc_sd']:.4f} | subject-mean AUC "
            f"{tracks['nested']['subject_mean_oof']['roc_auc']:.4f} | CN-vs-MCI "
            f"{tracks['nested']['subject_mean_oof_cn_vs_mci_auc']:.4f}"
        )
        log(f"[result] selection counts: {dict(selected_counts)}")
        log(f"[result] optimism (best single {best_single_id} - nested): {optimism:+.4f}")

        # -------------------------------------- deployment + validation ------
        log("[deploy] pretraining deployment encoder on all 141 training subjects ...")
        deployment = select_and_fit_deployment(
            bank, fe, y, diag, candidates, profile, budget, seed, device, log=log,
        )
        deployed = deployment["candidate"]
        log(f"[deploy] selected {deployed.describe()} "
            f"config {deployment['config'].key() if deployment['config'] else 'fixed'} "
            f"(threshold {deployment['threshold']:.4f})")

        # Save + round-trip reload check (encoder and probe must reproduce).
        encoder_path = training_dir / "deployment_encoder_state.pt"
        torch.save(
            {
                "variant": profile.model_variant,
                "state_dict": deployment["pretrained"]["model"].state_dict(),
                "channel_mean": deployment["pretrained"]["channel_mean"],
                "channel_std": deployment["pretrained"]["channel_std"],
            },
            encoder_path,
        )

        val_grids = load_split_grids(
            data_root, "val", max_days_per_subject=profile.max_days_per_subject
        )
        val_label_free_ids = sorted(val_grids.keys())
        val_bank = build_day_bank(val_grids, val_label_free_ids)
        val_emb = embed_all_subjects(
            val_bank, deployment["pretrained"]["model"],
            deployment["pretrained"]["channel_mean"],
            deployment["pretrained"]["channel_std"], device,
        )
        val_fe = build_fe_features(val_bank)
        val_fe = val_fe.reindex(columns=fe.columns)

        if deployed.view == "blend":
            validation_scores = deployment["model"].predict_score(val_fe, val_emb)
        else:
            frame = val_fe if deployed.view == "fe" else val_emb
            validation_scores = deployment["model"].predict_score(frame)

        # Round-trip: fresh model object + saved state must reproduce scores.
        checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)
        reloaded = SensorFMMae(C.MODEL_VARIANTS[checkpoint["variant"]]).to(device)
        reloaded.load_state_dict(checkpoint["state_dict"])
        reloaded.eval()
        val_emb_reloaded = embed_all_subjects(
            val_bank, reloaded, checkpoint["channel_mean"],
            checkpoint["channel_std"], device,
        )
        if not np.allclose(val_emb.to_numpy(), val_emb_reloaded.to_numpy(),
                           atol=1e-4):
            raise AssertionError("Encoder round-trip reload changed embeddings")
        log("[check] encoder round-trip reload reproduces validation embeddings")

        freeze_frame = pd.DataFrame(
            {
                "subject_hash": [subject_hash(s) for s in val_label_free_ids],
                "score": validation_scores,
                "predicted_label": (
                    validation_scores >= deployment["threshold"]
                ).astype(int),
            }
        )
        freeze_path = training_dir / "validation_predictions_label_free_hashed.csv"
        freeze_frame.to_csv(freeze_path, index=False)
        freeze_hash = _sha256_of(freeze_path)
        _write_json(
            training_dir / "VALIDATION_PREDICTIONS_FROZEN.json",
            {
                "sha256": freeze_hash,
                "file": freeze_path.name,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "deployed_candidate": deployed.candidate_id,
                "threshold": deployment["threshold"],
                "note": "written BEFORE any validation label file was opened",
            },
        )
        log(f"[freeze] validation predictions frozen (sha256 {freeze_hash[:16]}...)")

        if _sha256_of(freeze_path) != freeze_hash:
            raise AssertionError("Frozen validation predictions changed after freeze")
        validation_labels = load_labels(data_root, "val")
        assert_disjoint_splits(train_labels.index, validation_labels.index)
        missing_grid = sorted(set(validation_labels.index) - set(val_label_free_ids))
        if missing_grid:
            raise AssertionError(
                f"{len(missing_grid)} validation subjects lack admitted wearable days"
            )
        extra_grid = sorted(set(val_label_free_ids) - set(validation_labels.index))
        if extra_grid:
            raise AssertionError(
                f"{len(extra_grid)} wearable subjects absent from validation labels"
            )
        validation_labels = validation_labels.reindex(val_label_free_ids)
        y_val = validation_labels["y"].to_numpy(int)

        validation_report = {
            "disclaimer": (
                "The 33-subject validation split is a historical benchmark reused "
                "by many prior experiments (all-CN accuracy 26/33 = 0.788). It is "
                "scored exactly once here and was never used for any selection. "
                "Do not treat it as an independent cohort."
            ),
            "n_subjects": int(len(validation_labels)),
            "class_distribution": validation_labels["diag"].value_counts().to_dict(),
            "deployed_candidate": deployed.candidate_id,
            "deployment_inner_auc_by_candidate": deployment["inner_mean_auc_by_candidate"],
            "prediction_freeze_sha256": freeze_hash,
            "labels_opened_after_prediction_freeze": True,
            "metrics": {
                **score_metrics(y_val, validation_scores),
                "cn_vs_mci_auc": float(
                    cn_vs_mci_auc(validation_labels["diag"].to_numpy(object),
                                  validation_scores)
                ),
                "at_deployment_threshold": thresholded_metrics(
                    y_val, validation_scores, deployment["threshold"]
                ),
            },
        }
        _write_json(training_dir / "validation_report.json", validation_report)
        log(
            "[validation] AUC "
            f"{validation_report['metrics']['roc_auc']:.4f} | balanced acc "
            f"{validation_report['metrics']['at_deployment_threshold']['balanced_accuracy']:.4f}"
            " (historical benchmark, scored once)"
        )

        # ---------------------------------------------------- final report ---
        wearable_history = C.BENCHMARK["wearable_only_history"]
        final_report = {
            "config": run_config,
            "eda": eda,
            "runtime_projection_probe": probe,
            "primary_metric": (
                "nested track: pooled OOF ROC-AUC per outer repeat, "
                "mean +- sd across repeats (subject-level, fold-local SSL "
                "pretraining, selection inside inner CV)"
            ),
            "nested_oof": tracks["nested"],
            "nested_thresholded_summary": threshold_summary,
            "candidate_oof_tracks": candidate_tracks,
            "selection_counts": dict(selected_counts),
            "selection_optimism": {
                "best_single_candidate_on_report_metric": best_single_id,
                "best_single_subject_mean_auc": candidate_tracks[best_single_id][
                    "subject_mean_oof"]["roc_auc"],
                "nested_subject_mean_auc": tracks["nested"]["subject_mean_oof"]["roc_auc"],
                "optimism_estimate": float(optimism),
            },
            "paired_contrasts_subject_mean": contrasts,
            "wearable_only_history_delta": {
                "history": wearable_history,
                "nested_minus_best_history": float(
                    tracks["nested"]["subject_mean_oof"]["roc_auc"]
                    - wearable_history["Binary_Wearable_SequenceFusion_Google"]
                ),
            },
            "pretraining_summary": {
                "n_folds": len(nested["pretrain_records"]),
                "mean_best_val_mse": float(np.mean(
                    [r["best_val_mse"] for r in nested["pretrain_records"]]
                )),
                "mean_epochs_ran": float(np.mean(
                    [r["epochs_ran"] for r in nested["pretrain_records"]]
                )),
                "mean_seconds": float(np.mean(
                    [r["seconds"] for r in nested["pretrain_records"]]
                )),
            },
            "deployment": {
                "candidate": deployed.candidate_id,
                "view": deployed.view,
                "config": (deployment["config"].key()
                           if deployment["config"] else "fixed"),
                "threshold": deployment["threshold"],
                "inner_mean_auc_by_candidate": deployment["inner_mean_auc_by_candidate"],
                "encoder_checkpoint": encoder_path.name,
                "encoder_round_trip_ok": True,
            },
            "validation": validation_report,
            "smoke_disclaimer": (
                "SMOKE RUN - wiring check only, not a performance measurement"
                if profile.name == "smoke" else None
            ),
        }
        _write_json(training_dir / "FINAL_REPORT.json", final_report)
        _write_json(
            training_dir / "TRAINING_COMPLETE.json",
            {"completed_utc": datetime.now(timezone.utc).isoformat(),
             "run_id": run_id, "profile": profile.name,
             "engine": f"SensorFM-recipe {profile.model_variant} (from scratch)"},
        )
        status.update(
            {"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat()}
        )
        _write_json(output / "LAUNCHER_STATUS.json", status)
        log(f"[done] FINAL_REPORT.json written to {training_dir}")

    except BaseException as error:
        status.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "failed_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_json(output / "LAUNCHER_STATUS.json", status)
        raise


if __name__ == "__main__":
    main(dict(globals()))
