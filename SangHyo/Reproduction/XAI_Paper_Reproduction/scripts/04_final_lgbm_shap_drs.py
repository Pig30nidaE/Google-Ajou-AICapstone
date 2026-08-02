from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT_PATH.parents[1] / "src"))

from xai_paper_reproduction import (
    PAPER_FINAL_PARAMS,
    PAPER_SELECTED_FEATURE_COUNT,
    ProjectPaths,
    default_lgbm_grid,
    ensure_dir,
    evaluate_cv_model,
    final_shap_and_drs,
    grid_search_lgbm,
    lgbm_pipeline,
    load_dataset_outputs,
    load_json,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final LightGBM, compute SHAP values, and derive DRS.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--grouped", action="store_true", help="Diagnostic subject-group CV instead of paper row-level CV.")
    parser.add_argument("--grid-search", action="store_true", help="Run a compact grid around paper params before final fitting.")
    parser.add_argument("--use-paper-params", action="store_true", help="Use Table 2 params directly. This is the default if --grid-search is absent.")
    parser.add_argument(
        "--n-estimators-override",
        type=int,
        default=None,
        help="Override LightGBM n_estimators for local smoke tests. Leave unset for paper reproduction.",
    )
    parser.add_argument(
        "--max-drs-rows",
        type=int,
        default=None,
        help="Compute final SHAP/DRS outputs on a stratified row sample for local smoke tests. Leave unset for all rows.",
    )
    parser.add_argument(
        "--feature-source",
        choices=["paper_top40", "best_reproduction"],
        default="paper_top40",
        help="Use paper's top-40 convention or the best k found by this reproduction.",
    )
    return parser.parse_args()


def resolve_selected_features(output_dir: Path, all_features: list[str], feature_source: str) -> list[str]:
    path = output_dir / "feature_selection" / "selected_features.json"
    if not path.exists():
        print("[warn] selected_features.json not found; falling back to first 40 raw feature columns.")
        return all_features[:PAPER_SELECTED_FEATURE_COUNT]
    data = load_json(path)
    if feature_source == "best_reproduction":
        return data["selected_features"]
    return data["paper_top40_features"]


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_script(SCRIPT_PATH, output_dir=args.output_dir)
    out_dir = ensure_dir(paths.output_dir / "final")
    df, feature_cols = load_dataset_outputs(paths.output_dir)
    selected_features = resolve_selected_features(paths.output_dir, feature_cols, args.feature_source)

    X = df[feature_cols]
    y = df["binary_class"].astype(int)
    groups = df["patient_id"]
    meta = df[["patient_id", "sample_date", "split", "diagnosis", "binary_class"]].copy()

    if args.grid_search:
        print("[step] grid search around paper params")
        grid_metrics = grid_search_lgbm(
            X,
            y,
            feature_cols=selected_features,
            grid=default_lgbm_grid(),
            n_splits=args.n_splits,
            random_state=args.random_state,
            groups=groups,
            grouped=args.grouped,
        )
        grid_metrics.to_csv(out_dir / "grid_search_metrics.csv", index=False, encoding="utf-8-sig")
        best = grid_metrics.iloc[0].to_dict()
        params = {
            "num_leaves": int(best["num_leaves"]),
            "min_child_samples": int(best["min_child_samples"]),
            "learning_rate": float(best["learning_rate"]),
            "n_estimators": int(best["n_estimators"]),
        }
    else:
        params = dict(PAPER_FINAL_PARAMS)
    if args.n_estimators_override is not None:
        params["n_estimators"] = args.n_estimators_override

    print("[params]", params)
    cv_metrics = evaluate_cv_model(
        lgbm_pipeline(params=params, random_state=args.random_state),
        X[selected_features],
        y,
        groups=groups,
        n_splits=args.n_splits,
        random_state=args.random_state,
        grouped=args.grouped,
    )
    save_json(
        {
            "params": params,
            "feature_source": args.feature_source,
            "selected_feature_count": len(selected_features),
            "selected_features": selected_features,
            "cv_metrics": {k: v for k, v in cv_metrics.items() if k not in {"oof_prediction", "oof_probability"}},
        },
        out_dir / "final_cv_metrics.json",
    )

    final_result = final_shap_and_drs(
        X[selected_features],
        y,
        meta,
        feature_cols=selected_features,
        params=params,
        output_dir=out_dir,
        random_state=args.random_state,
        max_drs_rows=args.max_drs_rows,
    )
    save_json(final_result, out_dir / "final_training_shap_drs_result.json")

    print("[saved]", out_dir / "final_cv_metrics.json")
    print("[saved]", out_dir / "dementia_risk_scores.csv")
    print("[daily_t_test]", final_result["daily_one_sided_t_test"])


if __name__ == "__main__":
    main()
