#!/usr/bin/env python
"""Merge single-seed runs of one experiment into one results directory (A26).

python merge_runs.py --config configs/nested_cv.yaml --runs results/nested_cv_seed42 ... --out results/nested_cv
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR / "src"))
from cheon.manifest import write_json  # noqa: E402
from cheon.pipeline import write_results  # noqa: E402

CSVS = ["fold_metrics.csv", "oof_predictions.csv", "split_summary.csv", "inner_selection.csv", "selected_features_by_fold.csv",
        "selected_hyperparameters_by_fold.csv", "feature_selection.csv", "shap_importance.csv"]
JSON_LISTS = ["leakage_assertions.json", "selected_features.json", "feature_selection_diagnostic.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg_text = (EXP_DIR / a.config).read_text()
    cfg = yaml.safe_load(cfg_text)
    seeds = [int(s) for s in cfg["seeds"]]
    runs = [EXP_DIR / r for r in a.runs]
    out = EXP_DIR / a.out
    out.mkdir(parents=True, exist_ok=True)
    manifests = [json.loads((r / "run_manifest.json").read_text()) for r in runs]
    for key in ("code_fingerprint_sha256", "config_fingerprint_sha256", "data_fingerprint_sha256"):
        vals = {m[key] for m in manifests}
        if len(vals) != 1:
            raise RuntimeError(f"runs differ in {key}: {vals}")
    run_seeds = [m["seeds"] for m in manifests]
    if sorted(s for ss in run_seeds for s in ss) != sorted(seeds):
        raise RuntimeError(f"run seeds {run_seeds} do not cover config seeds {seeds}")

    def remap(df: pd.DataFrame) -> pd.DataFrame:
        if "seed" in df.columns:
            df = df.copy()
            df["repeat"] = df["seed"].map({s: i for i, s in enumerate(seeds)}).astype(int)
        return df

    tables = {}
    for name in CSVS:
        parts = [pd.read_csv(r / name) for r in runs if (r / name).exists()]
        if parts:
            tables[name] = remap(pd.concat(parts, ignore_index=True)).sort_values(["repeat"], kind="mergesort")
    for name in JSON_LISTS:
        items = []
        for r in runs:
            if (r / name).exists():
                items += json.loads((r / name).read_text())
        if items:
            for it in items:
                if "seed" in it:
                    it["repeat"] = seeds.index(int(it["seed"]))
            write_json(out / name, sorted(items, key=lambda d: (d.get("repeat", 0), d.get("outer_fold", 0))))
    for name, df in tables.items():
        if name not in ("fold_metrics.csv", "oof_predictions.csv", "split_summary.csv"):
            df.to_csv(out / name, index=False)
    write_results(out, cfg["experiment"], False, tables["fold_metrics.csv"], tables["oof_predictions.csv"], tables["split_summary.csv"], cfg)

    timing = {}
    for r in runs:
        for k, v in json.loads((r / "timing.json").read_text()).items():
            t = timing.setdefault(k, {"n": 0, "total_s": 0.0})
            t["n"] += v["n"]; t["total_s"] += v["total_s"]
    for v in timing.values():
        v["mean_s"] = v["total_s"] / v["n"] if v["n"] else None
    write_json(out / "timing.json", timing)

    starts = [datetime.fromisoformat(m["start_utc"]) for m in manifests]
    ends = [datetime.fromisoformat(m["end_utc"]) for m in manifests]
    merged = {**manifests[0], "seeds": seeds, "output_dir": str(out),
              "start_utc": min(starts).isoformat(), "end_utc": max(ends).isoformat(),
              "runtime_seconds": (max(ends) - min(starts)).total_seconds(),
              "sum_of_process_runtime_seconds": sum(m["runtime_seconds"] for m in manifests),
              "execution": "merged from single-seed parallel processes (A26)",
              "per_seed_runs": [{"seeds": m["seeds"], "output_dir": m["output_dir"], "start_utc": m["start_utc"],
                                 "end_utc": m["end_utc"], "runtime_seconds": m["runtime_seconds"]} for m in manifests]}
    merged["config"]["seeds"] = seeds
    merged["config"]["output_dir"] = cfg["output_dir"]
    write_json(out / "run_manifest.json", merged)
    envs = {(r / "environment.txt").read_text() for r in runs}
    if len(envs) != 1:
        raise RuntimeError("environment.txt differs between runs")
    shutil.copy(runs[0] / "environment.txt", out / "environment.txt")
    with open(out / "stdout.log", "w") as f:
        for r in runs:
            f.write(f"##### {r.name}\n")
            f.write((r / "stdout.log").read_text())
    print(f"merged {len(runs)} runs -> {out}")


if __name__ == "__main__":
    main()
