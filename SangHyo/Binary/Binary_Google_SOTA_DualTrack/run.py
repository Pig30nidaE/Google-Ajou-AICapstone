"""base.ipynb entry point.

    USER_FOLDER = "SangHyo"
    RUN_FILE    = "Binary_Google_SOTA_DualTrack/run.py"

Reproduces the ensemble described in ``report.docx.pdf`` (V26/V29/V41) with
Google YDF learners, after removing the leakage paths documented in
``LEAKAGE_AUDIT_KO.md``.

Runtime
-------
**CPU, High-RAM. No GPU.**  YDF is a multi-threaded CPU library, so core count
is the useful knob; an A100 session would sit idle.

Modes (``--mode``)::

    smoke      ~2 min    wiring check only -- metrics are meaningless
    standard   ~30 min   2 repeats, reduced search budget
    full       ~2 h      default: 3 repeats, full search budget

Tracks (``--track``)::

    wearable      Activity + Sleep only; 3.CognitiveFunction is never opened
    mmse_fusion   adds MMSE item scores (DIAG_NM/DIAG_SEQ/MMSE_NUM/MMSE_KIND
                  dropped fail-closed).  NOT label-independent -- see README.
    both          run each track through the identical folds and compare

Validation protocol: predictions for the 33 held-out people are written and
SHA-256 stamped *before* their labels are read.  The labels are then opened
exactly once and never fed back into any choice.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # allow `python run.py` as well as `-m`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "Binary_Google_SOTA_DualTrack"

from .data import (  # noqa: E402
    DIAG_COLUMN, LABEL_COLUMN, PERSON_KEY, align_features_and_labels,
    assert_no_mmse_features, assert_person_disjoint, load_labels, load_mmse,
    load_wearable, resolve_roots,
)
from .eda import run_eda  # noqa: E402
from .features import build_person_features  # noqa: E402
from .metrics import bootstrap_interval, classification_metrics  # noqa: E402
from .models import engine_report  # noqa: E402
from .train import PipelineConfig, VARIANTS, fit_final, nested_cv, predict_fold  # noqa: E402

EXPERIMENT_NAME = "Binary_Google_SOTA_DualTrack"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_COLAB_RESULTS_ROOT = Path(f"/content/drive/MyDrive/{EXPERIMENT_NAME}_result")

MODE_SETTINGS = {
    "smoke": dict(repeats=1, outer_k=3, inner_k=3, search_budget=2, use_search=False,
                  max_candidates=8, max_selected=4, bootstrap_resamples=200,
                  deadline_seconds=None),
    "standard": dict(repeats=2, outer_k=5, inner_k=5, search_budget=8, use_search=True,
                     max_candidates=25, max_selected=15, bootstrap_resamples=2000,
                     deadline_seconds=3 * 3600),
    "full": dict(repeats=3, outer_k=5, inner_k=5, search_budget=12, use_search=True,
                 max_candidates=40, max_selected=15, bootstrap_resamples=2000,
                 deadline_seconds=5 * 3600 + 45 * 60),
}


# ------------------------------------------------------------------ plumbing --
def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
                    encoding="utf-8")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def resolve_output_root(run_id: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve() / run_id
    if Path("/content/drive/MyDrive").exists():
        return DEFAULT_COLAB_RESULTS_ROOT / run_id
    if Path("/content").exists():
        raise RuntimeError(
            "Running on Colab but /content/drive/MyDrive is not mounted. "
            "Mount Drive first -- refusing to write results to ephemeral local disk."
        )
    return EXPERIMENT_ROOT / "outputs" / run_id


# ------------------------------------------------------------------- dataset --
def build_split(data_root: Path, split: str, track: str) -> tuple[pd.DataFrame, dict]:
    activity, sleep = load_wearable(data_root, split, track=track)
    mmse = load_mmse(data_root, split, track=track) if track == "mmse_fusion" else None
    features, audit = build_person_features(activity, sleep, mmse)
    if track == "wearable":
        assert_no_mmse_features(features.columns)
    audit["track"] = track
    audit["split"] = split
    return features, audit


def prepare_track(data_root: Path, track: str) -> dict:
    train_features, train_audit = build_split(data_root, "train", track)
    val_features, val_audit = build_split(data_root, "validation", track)

    train_labels = load_labels(data_root, "train")
    assert_person_disjoint(train_features[PERSON_KEY], val_features[PERSON_KEY])

    X_train_frame, y_train, train_ids = align_features_and_labels(train_features, train_labels)

    # Validation features are aligned to the Training column order; unseen
    # columns are dropped and missing ones filled with NaN.  No Validation
    # statistic is used to build the Training matrix.
    columns = list(X_train_frame.columns)
    val_indexed = val_features.set_index(PERSON_KEY)
    val_aligned = val_indexed.reindex(columns=columns)
    val_ids = np.asarray(val_aligned.index)

    return {
        "track": track,
        "X_train": X_train_frame,
        "y_train": y_train,
        "train_ids": train_ids,
        "diagnoses": train_labels.set_index(PERSON_KEY).loc[train_ids, DIAG_COLUMN],
        "X_val": val_aligned,
        "val_ids": val_ids,
        "audit": {"train": train_audit, "validation": val_audit,
                  "n_shared_columns": int(val_indexed.columns.isin(columns).sum()),
                  "n_train_columns": len(columns)},
    }


# ----------------------------------------------------------------- one track --
def run_track(bundle: dict, config: PipelineConfig, output_root: Path,
              data_root: Path, *, verbose: bool = True) -> dict:
    track = bundle["track"]
    track_dir = output_root / track
    track_dir.mkdir(parents=True, exist_ok=True)

    X_train_frame: pd.DataFrame = bundle["X_train"]
    y_train: np.ndarray = bundle["y_train"]
    feature_names = list(X_train_frame.columns)
    X_train = X_train_frame.to_numpy(dtype=np.float64)

    print(f"\n=== track={track}: {X_train.shape[0]} people x {X_train.shape[1]} features ===")

    eda = run_eda(X_train_frame, y_train, bundle["train_ids"], bundle["diagnoses"])
    write_json(track_dir / "eda_report.json", eda)

    # -- 1. nested CV on Training only ---------------------------------------
    started = time.time()
    cv_results = nested_cv(X_train, y_train, config, feature_names, verbose=verbose)
    cv_results["elapsed_seconds"] = round(time.time() - started, 1)
    cv_results["track"] = track
    write_json(track_dir / "nested_cv_report.json", cv_results)

    # -- 2. refit on all Training people, then FREEZE the Validation forecast --
    artifacts, final_summary = fit_final(X_train, y_train, config, feature_names)
    X_val = bundle["X_val"].to_numpy(dtype=np.float64)
    val_scores = predict_fold(artifacts, X_val, config)

    frozen = pd.DataFrame({PERSON_KEY: bundle["val_ids"]})
    for name, values in val_scores.items():
        frozen[f"score_{name}"] = values
    for variant in VARIANTS:
        threshold = artifacts.thresholds[variant]
        frozen[f"pred_{variant}"] = (val_scores[variant] >= threshold).astype(int)

    frozen_path = track_dir / "validation_predictions_frozen.csv"
    frozen.to_csv(frozen_path, index=False)
    freeze_stamp = {
        "file": frozen_path.name,
        "sha256": sha256_of(frozen_path),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": final_summary["thresholds"],
        "n_rows": int(len(frozen)),
        "note": "Written and hashed BEFORE any Validation label was read.",
    }
    write_json(track_dir / "validation_freeze.json", freeze_stamp)
    write_json(track_dir / "final_model_summary.json", final_summary)
    print(f"[run] frozen Validation predictions: {frozen_path.name} "
          f"sha256={freeze_stamp['sha256'][:16]}...")

    # -- 3. only now open the Validation labels, exactly once ------------------
    val_labels = load_labels(data_root, "validation")
    label_map = val_labels.set_index(PERSON_KEY)[LABEL_COLUMN]
    missing = [p for p in bundle["val_ids"] if p not in label_map.index]
    if missing:
        raise RuntimeError(f"{len(missing)} Validation person(s) have no label: {missing[:3]}")
    y_val = label_map.loc[list(bundle["val_ids"])].to_numpy(dtype=np.int64)

    validation_metrics = {}
    for variant in VARIANTS:
        threshold = artifacts.thresholds[variant]
        metrics = classification_metrics(y_val, val_scores[variant], threshold)
        metrics["bootstrap"] = bootstrap_interval(
            y_val, val_scores[variant], threshold,
            n_resamples=config.bootstrap_resamples, seed=config.seed,
        )
        validation_metrics[variant] = metrics
    for kind in config.kinds:
        validation_metrics[kind] = classification_metrics(y_val, val_scores[kind], 0.5)

    payload = {
        "track": track,
        "frozen": freeze_stamp,
        "validation_metrics": validation_metrics,
        "note": ("Threshold and every model choice were fixed on Training only. "
                 "This Validation set is evaluated once; do not re-tune against it."),
    }
    write_json(track_dir / "validation_report.json", payload)

    return {
        "track": track,
        "nested_cv": cv_results["metrics"],
        "feature_stability": cv_results["feature_stability"],
        "completed_repeats": cv_results["completed_repeats"],
        "validation": validation_metrics,
        "n_features": int(X_train.shape[1]),
        "n_selected_final": final_summary["n_selected"],
        "elapsed_seconds": cv_results["elapsed_seconds"],
    }


# ---------------------------------------------------------------------- main --
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--mode", choices=tuple(MODE_SETTINGS),
                        default=os.environ.get("SOTA_MODE", "full"))
    parser.add_argument("--track", choices=("wearable", "mmse_fusion", "both"),
                        default=os.environ.get("SOTA_TRACK", "both"))
    parser.add_argument("--smote", choices=("borderline", "plain", "none"),
                        default=os.environ.get("SOTA_SMOTE", "borderline"))
    parser.add_argument("--data-root", default=os.environ.get("SOTA_DATA_ROOT"))
    parser.add_argument("--output-root", default=os.environ.get("SOTA_OUTPUT_ROOT"))
    parser.add_argument("--seed", type=int,
                        default=int(os.environ.get("SOTA_SEED", 20260728)))
    parser.add_argument("--quiet", action="store_true")

    # base.ipynb runs this file with runpy inside a Jupyter kernel, so sys.argv
    # still carries the kernel's own "-f .../kernel-xxx.json".  Ignore unknown
    # arguments rather than aborting; in the notebook the SOTA_* environment
    # variables are the control surface.
    arguments, _unknown = parser.parse_known_args(argv)
    return arguments


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = MODE_SETTINGS[args.mode]
    config = PipelineConfig(seed=args.seed, smote_kind=args.smote, **settings)

    # base.ipynb injects PROJECT_ROOT / DATA_ROOT / USER_ROOT as globals via
    # runpy's init_globals; honour DATA_ROOT when no explicit override is given.
    injected_data_root = globals().get("DATA_ROOT")
    project_root, data_root = resolve_roots(args.data_root or injected_data_root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = resolve_output_root(run_id, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    engine = engine_report()
    print(f"[run] {EXPERIMENT_NAME} mode={args.mode} track={args.track} run_id={run_id}")
    print(f"[run] data_root   = {data_root}")
    print(f"[run] output_root = {output_root}")
    print(f"[run] engine      = {engine['engine']} ({engine['note']})")
    if args.mode == "smoke":
        print("[run] SMOKE MODE -- wiring check only. Metrics are NOT results.")

    tracks = ("wearable", "mmse_fusion") if args.track == "both" else (args.track,)
    summaries = []
    for track in tracks:
        bundle = prepare_track(data_root, track)
        write_json(output_root / track / "data_audit.json", bundle["audit"])
        summaries.append(run_track(bundle, config, output_root, data_root,
                                   verbose=not args.quiet))

    manifest = {
        "experiment": EXPERIMENT_NAME,
        "run_id": run_id,
        "mode": args.mode,
        "smoke_mode": args.mode == "smoke",
        "tracks": list(tracks),
        "config": config.to_dict(),
        "engine": engine,
        "project_root": str(project_root),
        "data_root": str(data_root),
        "summaries": summaries,
        "caveats": [
            "Nested OOF is an honest INTERNAL estimate, not an independent-cohort estimate: "
            "candidate features and models were chosen with knowledge of earlier experiments "
            "on this same 141-person cohort.",
            "The mmse_fusion track is NOT label-independent. MMSE is the cognitive instrument "
            "behind DIAG_NM and lives in the same file as it; treat its lift as an upper bound.",
            "Validation is 33 people (7 MCI+DEM). Its confidence intervals are very wide.",
        ],
    }
    write_json(output_root / "run_manifest.json", manifest)

    print("\n=== summary (nested OOF, Training) ===")
    for summary in summaries:
        for variant in VARIANTS:
            metrics = summary["nested_cv"][variant]
            print(f"  {summary['track']:<12} {variant:<12} "
                  f"AUC={metrics['roc_auc']:.4f}  "
                  f"BalAcc={metrics['balanced_accuracy']:.4f}  "
                  f"Recall={metrics['recall_mci_dem']:.4f}  "
                  f"Spec={metrics['specificity_cn']:.4f}")
    print("\n=== Validation (33 people, evaluated once) ===")
    for summary in summaries:
        for variant in VARIANTS:
            metrics = summary["validation"][variant]
            print(f"  {summary['track']:<12} {variant:<12} "
                  f"AUC={metrics['roc_auc']:.4f}  "
                  f"BalAcc={metrics['balanced_accuracy']:.4f}  "
                  f"Acc={metrics['accuracy']:.4f} "
                  f"(all-CN baseline {metrics['all_cn_baseline_accuracy']:.4f})")
    print(f"\n[run] results written to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
