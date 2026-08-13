"""Single entry point for Binary_Google_CircadianNested.

Colab / base.ipynb (Cell 2)
---------------------------
    USER_FOLDER = "SangHyo"
    RUN_FILE    = "Binary/Binary_Google_CircadianNested/run.py"

    # optional, before Cell 5:
    # import os; os.environ["BGCN_ARGS"] = "--profile default"

Runtime: **CPU** (High-RAM not required).  Every learner is Google YDF
gradient boosted trees or a regularized logistic regression; no GPU code path
exists.  The default profile is designed to finish well under an hour.

Local shell:
    python run.py --profile smoke     # wiring check only, never a result
    python run.py                     # default profile

Pipeline stages (in order):
    1. environment check (ydf==0.16.1 is installed on demand, fail-closed)
    2. label + feature build (subject-local, leakage-audited)
    3. EDA summary
    4. repeated nested subject-level CV (primary evidence)
    5. deployment selection + fit on the 141 training subjects
    6. frozen prediction of the 33 historical validation subjects
       (SHA-256 freeze written BEFORE validation labels are opened)
    7. one-shot validation scoring + FINAL_REPORT.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_NAME = "Binary_Google_CircadianNested"

# ``base.ipynb`` executes this file with ``runpy`` (no package context), so the
# experiment folder itself is put on ``sys.path`` and any stale module from a
# previous in-process run is dropped.
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
for _name in [m for m in list(sys.modules) if m == "circnested" or m.startswith("circnested.")]:
    sys.modules.pop(_name, None)


# ----------------------------------------------------------- environment ----
YDF_PIN = "0.16.1"


def _installed_ydf_version() -> str | None:
    import importlib.metadata
    try:
        return importlib.metadata.version("ydf")
    except importlib.metadata.PackageNotFoundError:
        return None


def _ensure_ydf() -> None:
    """Install the pinned Google YDF wheel, fail-closed.

    The Colab image can ship its own ydf (run 20260813_060156_utc silently used
    0.15.0 because an earlier version of this function only installed when the
    import failed).  Tree structure and defaults can move between minor
    versions, so the pin is enforced rather than merely requested; if pip
    cannot deliver it the run continues on the installed version but the
    deviation is printed and recorded in FINAL_REPORT.
    """

    current = _installed_ydf_version()
    if current == YDF_PIN:
        return
    if current is None:
        print(f"[env] ydf not found; installing ydf=={YDF_PIN} ...", flush=True)
    else:
        print(f"[env] ydf {current} present but pin is {YDF_PIN}; installing the pin ...",
              flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", f"ydf=={YDF_PIN}"],
        check=False,
    )
    import importlib
    importlib.invalidate_caches()
    if result.returncode != 0 or _installed_ydf_version() != YDF_PIN:
        if current is None:
            raise RuntimeError(
                f"pip install ydf=={YDF_PIN} failed. Google YDF is the core engine "
                "of this experiment and has no fallback."
            )
        print(
            f"[env] WARNING: could not install the pin; continuing on ydf {current}. "
            "Results are valid Google-YDF results but not version-pinned.",
            flush=True,
        )
    try:
        import ydf  # noqa: F401
    except ImportError as error:  # pragma: no cover - broken install
        raise RuntimeError("Google YDF is unimportable after installation") from error


def _in_notebook_host() -> bool:
    """True when this file is being executed from inside a notebook kernel.

    ``base.ipynb`` runs it with ``runpy.run_path``, which rewrites
    ``sys.argv[0]`` to this file but leaves the *kernel's* own arguments
    (``-f .../kernel-<uuid>.json``) in place.  Those are not ours, and parsing
    them aborts the run, so in a notebook host sys.argv is ignored entirely and
    BGCN_ARGS becomes the only argument channel.
    """

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
                        choices=("smoke", "default", "max"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    argv = [] if _in_notebook_host() else sys.argv[1:]
    # base.ipynb runs this file through runpy inside an IPython kernel, so it
    # passes no arguments of its own; BGCN_ARGS is the notebook's channel.
    if not argv and os.environ.get("BGCN_ARGS"):
        argv = shlex.split(os.environ["BGCN_ARGS"])
    return parser.parse_args(argv)


def _resolve_data_root(namespace: dict, explicit: str | None) -> Path:
    candidates: list[Path] = []
    for value in (explicit, namespace.get("DATA_ROOT"), os.environ.get("BGCN_DATA_ROOT")):
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
    if os.environ.get("BGCN_OUTPUT_DIR"):
        return Path(os.environ["BGCN_OUTPUT_DIR"]).expanduser().resolve() / run_id
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


# ------------------------------------------------------------ aggregation ----
def _aggregate_tracks(oof_scores, y, diag, n_bootstrap, seed):
    """Per-track: per-repeat pooled AUC mean/sd + subject-mean OOF metrics."""

    import numpy as np

    from circnested.evaluation import (
        cn_vs_mci_auc, roc_auc_safe, score_metrics, subject_bootstrap_auc_ci,
    )

    tracks = {}
    subject_means = {}
    for track, matrix in oof_scores.items():
        repeat_aucs = [roc_auc_safe(y, matrix[r]) for r in range(matrix.shape[0])]
        mean_scores = matrix.mean(axis=0)
        subject_means[track] = mean_scores
        entry = {
            "per_repeat_pooled_roc_auc": [float(v) for v in repeat_aucs],
            "repeat_roc_auc_mean": float(np.nanmean(repeat_aucs)),
            "repeat_roc_auc_sd": float(np.nanstd(repeat_aucs, ddof=0)),
            "subject_mean_oof": score_metrics(y, mean_scores),
            "subject_mean_oof_cn_vs_mci_auc": float(cn_vs_mci_auc(diag, mean_scores)),
            "subject_mean_oof_bootstrap": subject_bootstrap_auc_ci(
                y, mean_scores, n_boot=n_bootstrap, seed=seed
            ),
        }
        tracks[track] = entry
    return tracks, subject_means


def _nested_threshold_summary(fold_records, outer_repeats):
    """Aggregate the fold-local thresholded confusion counts per repeat."""

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
        _ensure_ydf()

        import numpy as np
        import pandas as pd

        from circnested import config as C
        from circnested.data import assert_disjoint_splits, load_labels, subject_hash
        from circnested.evaluation import paired_bootstrap_auc_diff, thresholded_metrics
        from circnested.features import (
            assert_no_forbidden, build_split_features, drop_degenerate_columns,
            feature_fingerprint,
        )
        from circnested.modeling import ydf_runtime_info
        from circnested.nested_cv import run_repeated_nested_cv, select_and_fit_deployment

        profile = C.PROFILES[arguments.profile]
        seed = int(arguments.seed) if arguments.seed is not None else C.SEED
        data_root = _resolve_data_root(namespace, arguments.data_root)

        import sklearn

        run_config = {
            "experiment": EXPERIMENT_NAME,
            "task": C.TASK_DESCRIPTION,
            "run_id": run_id,
            "profile": profile.name,
            "profile_detail": {
                "outer_k": profile.outer_k, "outer_repeats": profile.outer_repeats,
                "inner_k": profile.inner_k, "inner_repeats": profile.inner_repeats,
                "n_bootstrap": profile.n_bootstrap,
                "candidate_ids": list(profile.candidate_ids),
                "ydf_trees_override": profile.ydf_trees_override,
            },
            "seed": seed,
            "selection_tolerance_auc": C.SELECTION_TOLERANCE,
            "data_root": str(data_root),
            "output_dir": str(output),
            "google_technology": {
                "engine": "Yggdrasil Decision Forests (YDF), Google",
                "role": (
                    "core learner of 6/9 nested candidates (axis-aligned GBT, "
                    "sparse-oblique GBT, and the YDF half of both rank blends) "
                    "plus the circadian-only diagnostic arm; no fallback permitted"
                ),
                "runtime": None,  # filled after import check
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sklearn": sklearn.__version__,
                "platform": platform.platform(),
            },
            "benchmark_anchors": C.BENCHMARK,
        }
        run_config["google_technology"]["runtime"] = ydf_runtime_info()
        run_config["google_technology"]["version_pin"] = YDF_PIN
        run_config["google_technology"]["version_pin_honored"] = (
            run_config["google_technology"]["runtime"]["version"] == YDF_PIN
        )

        log("=" * 78)
        log(f"{EXPERIMENT_NAME}  |  run {run_id}  |  profile {profile.name}")
        log(f"task            : {C.TASK_DESCRIPTION}")
        log(f"data root       : {data_root}")
        log(f"output          : {output}")
        log(f"seed            : {seed}")
        log(f"outer CV        : {profile.outer_k} folds x {profile.outer_repeats} repeats "
            f"(subject-level, stratified on CN/MCI/Dem)")
        log(f"inner CV        : {profile.inner_k} folds x {profile.inner_repeats} repeats "
            f"(model+view selection, thresholds)")
        log(f"candidates      : {', '.join(profile.candidate_ids)}")
        log(f"google engine   : {run_config['google_technology']['runtime']}")
        log("=" * 78)
        if profile.name == "smoke":
            log("!! SMOKE RUN: wiring check only; numbers are NOT performance !!")

        # ------------------------------------------------ data + features ----
        stage_start = time.monotonic()
        train_labels = load_labels(data_root, "train")
        log(f"[data] training subjects: {len(train_labels)} "
            f"({train_labels['diag'].value_counts().to_dict()})")

        train_features = build_split_features(data_root, "train")
        degenerate = drop_degenerate_columns(train_features)
        if degenerate:
            train_features = train_features.drop(columns=degenerate)
        assert_no_forbidden(train_features.columns)
        train_features = train_features.reindex(train_labels.index)
        if train_features.isna().all(axis=1).any():
            raise AssertionError("A training subject has no features at all")
        log(f"[features] built {train_features.shape[1]} features for "
            f"{train_features.shape[0]} subjects in {time.monotonic() - stage_start:.1f}s "
            f"(dropped degenerate: {len(degenerate)})")

        # Views must survive the degenerate filter (registration items are
        # constant and expected to drop; they are excluded from views here).
        available = set(train_features.columns)
        views_effective = {
            name: tuple(column for column in columns if column in available)
            for name, columns in C.VIEWS.items()
        }
        removed_from_views = {
            name: sorted(set(columns) - set(views_effective[name]))
            for name, columns in C.VIEWS.items()
        }
        C.VIEWS.clear()
        C.VIEWS.update(views_effective)

        y = train_labels["y"].to_numpy(int)
        diag = train_labels["diag"].to_numpy(object)

        eda = {
            "n_subjects": int(len(train_labels)),
            "class_distribution": train_labels["diag"].value_counts().to_dict(),
            "binary_positive_rate": float(y.mean()),
            "n_features_built": int(train_features.shape[1]),
            "view_sizes": {name: len(cols) for name, cols in C.VIEWS.items()},
            "view_columns": {name: list(cols) for name, cols in C.VIEWS.items()},
            "degenerate_columns_dropped": degenerate,
            "view_columns_dropped_as_degenerate": removed_from_views,
            "missingness_top10": train_features.isna().mean().sort_values(
                ascending=False).head(10).round(4).to_dict(),
            "feature_fingerprint_train": feature_fingerprint(train_features),
        }
        _write_json(output / "eda" / "summary.json", eda)
        log(f"[eda] view sizes: {eda['view_sizes']} | fingerprint "
            f"{eda['feature_fingerprint_train']}")

        # -------------------------------------------------- nested CV --------
        selection_candidates = [
            c for c in C.CANDIDATES if c.candidate_id in set(profile.candidate_ids)
        ]
        audit_candidates = list(C.FIXED_ARMS.values()) if profile.name != "smoke" else []
        log(f"[nested-cv] starting: {len(selection_candidates)} selection candidates, "
            f"{len(audit_candidates)} fixed audit arms")
        cv_start = time.monotonic()
        nested = run_repeated_nested_cv(
            train_features, y, diag, selection_candidates, audit_candidates,
            profile, seed, log=log,
        )
        log(f"[nested-cv] finished in {(time.monotonic() - cv_start) / 60:.1f} min")

        tracks, subject_means = _aggregate_tracks(
            nested["oof_scores"], y, diag, profile.n_bootstrap, seed
        )

        # Selection stability and optimism diagnostics.
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

        # Pre-registered paired contrasts on the SAME subject-mean OOF scores.
        contrasts = {}
        contrast_pairs = [("nested", "lr_mmse_c001")]
        for fusion_id, mmse_id in (
            ("lr_fusion_c001", "lr_mmse_c001"), ("gbt_fusion", "gbt_mmse"),
            ("obl_fusion", "obl_mmse"), ("blend_fusion", "blend_mmse"),
        ):
            if fusion_id in subject_means and mmse_id in subject_means:
                contrast_pairs.append((fusion_id, mmse_id))
        for track_a, track_b in contrast_pairs:
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
        log("[deploy] inner-CV selection on all training subjects ...")
        deployment = select_and_fit_deployment(
            train_features, y, diag, selection_candidates, profile, seed
        )
        deployed = deployment["candidate"]
        log(f"[deploy] selected {deployed.describe()} "
            f"(threshold {deployment['threshold']:.4f})")

        # Validation features are built WITHOUT opening any validation label
        # file; the freeze file and its SHA-256 are written before labels load.
        validation_features = build_split_features(data_root, "val")
        validation_features = validation_features.reindex(
            columns=train_features.columns
        )
        from circnested.features import select_view
        validation_view = select_view(validation_features, C.VIEWS[deployed.view])
        validation_scores = deployment["model"].predict_score(validation_view)

        validation_ids = list(validation_features.index)
        freeze_frame = pd.DataFrame(
            {
                "subject_hash": [subject_hash(s) for s in validation_ids],
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

        # Only now may validation labels be read -- exactly once.
        if _sha256_of(freeze_path) != freeze_hash:
            raise AssertionError("Frozen validation predictions changed after freeze")
        validation_labels = load_labels(data_root, "val")
        assert_disjoint_splits(train_labels.index, validation_labels.index)
        validation_labels = validation_labels.reindex(validation_features.index)
        y_val = validation_labels["y"].to_numpy(int)

        from circnested.evaluation import cn_vs_mci_auc, score_metrics
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
        final_report = {
            "config": run_config,
            "eda": {k: v for k, v in eda.items() if k != "view_columns"},
            "primary_metric": (
                "nested track: pooled OOF ROC-AUC per outer repeat, "
                "mean +- sd across repeats (subject-level, selection inside inner CV)"
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
                "note": (
                    "picking the best single candidate on the report metric is "
                    "selection ON the OOF; the nested track is the honest number"
                ),
            },
            "paired_contrasts_subject_mean": contrasts,
            "deployment": {
                "candidate": deployed.candidate_id,
                "view": deployed.view,
                "threshold": deployment["threshold"],
                "inner_mean_auc_by_candidate": deployment["inner_mean_auc_by_candidate"],
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
             "run_id": run_id, "profile": profile.name},
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
