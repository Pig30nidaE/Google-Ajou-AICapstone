from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT_PATH.parents[1] / "src"))

from xai_paper_reproduction import (
    ProjectPaths,
    compare_against_paper,
    ensure_dir,
    evaluate_cv_model,
    load_dataset_outputs,
    model_registry,
    plot_model_comparison,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper baseline comparison across seven classifiers.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--grouped", action="store_true", help="Diagnostic subject-group CV instead of paper row-level CV.")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Optional comma-separated model names for smoke tests. Default runs all seven paper models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_script(SCRIPT_PATH, output_dir=args.output_dir)
    out_dir = ensure_dir(paths.output_dir / "baselines")
    df, feature_cols = load_dataset_outputs(paths.output_dir)

    X = df[feature_cols]
    y = df["binary_class"].astype(int)
    groups = df["patient_id"]

    rows = []
    raw_metrics = {}
    registry = model_registry(random_state=args.random_state)
    selected_models = list(registry)
    if args.models:
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = sorted(set(requested) - set(registry))
        if unknown:
            raise ValueError(f"Unknown model names: {unknown}. Available: {list(registry)}")
        selected_models = requested

    for name in selected_models:
        estimator = registry[name]
        print(f"[model] {name}")
        metrics = evaluate_cv_model(
            estimator,
            X,
            y,
            groups=groups,
            n_splits=args.n_splits,
            random_state=args.random_state,
            grouped=args.grouped,
        )
        raw_metrics[name] = metrics
        rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "roc_auc": metrics["roc_auc"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "precision_positive": metrics["precision_positive"],
                "recall_positive": metrics["recall_positive"],
                "f1_positive": metrics["f1_positive"],
            }
        )
        print(rows[-1])

    metrics_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(out_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    compare_against_paper(metrics_df).to_csv(out_dir / "model_comparison_against_paper.csv", index=False, encoding="utf-8-sig")
    save_json(raw_metrics, out_dir / "model_comparison_raw.json")
    plot_model_comparison(metrics_df, out_dir / "model_comparison_auc.png")

    print("[saved]", out_dir / "model_comparison.csv")
    print("[best]", metrics_df.iloc[0].to_dict())


if __name__ == "__main__":
    main()
