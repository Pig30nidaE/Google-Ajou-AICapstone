from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT_PATH.parents[1] / "src"))

from xai_paper_reproduction import (
    PAPER_FINAL_PARAMS,
    PAPER_SELECTED_FEATURE_COUNT,
    ProjectPaths,
    compute_oof_shap_importance,
    ensure_dir,
    load_dataset_outputs,
    plot_forward_selection,
    run_forward_selection,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHAP-importance forward feature selection for LightGBM.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=80)
    parser.add_argument("--sample-per-fold", type=int, default=None, help="Optional SHAP speed limit per validation fold.")
    parser.add_argument("--grouped", action="store_true", help="Diagnostic subject-group CV instead of paper row-level CV.")
    parser.add_argument(
        "--use-paper-params-for-ranking",
        action="store_true",
        help="Use final paper LightGBM params when computing SHAP ranking. Default uses untuned LightGBM.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_script(SCRIPT_PATH, output_dir=args.output_dir)
    out_dir = ensure_dir(paths.output_dir / "feature_selection")
    df, feature_cols = load_dataset_outputs(paths.output_dir)

    X = df[feature_cols]
    y = df["binary_class"].astype(int)
    groups = df["patient_id"]
    ranking_params = PAPER_FINAL_PARAMS if args.use_paper_params_for_ranking else None

    print("[step] computing OOF SHAP importance")
    importance = compute_oof_shap_importance(
        X,
        y,
        params=ranking_params,
        groups=groups,
        n_splits=args.n_splits,
        random_state=args.random_state,
        grouped=args.grouped,
        sample_per_fold=args.sample_per_fold,
    )
    importance.to_csv(out_dir / "shap_importance_full.csv", index=False, encoding="utf-8-sig")

    ranked = importance["feature"].tolist()
    print("[step] forward selection")
    metrics = run_forward_selection(
        X,
        y,
        ranked,
        params=None,
        max_features=args.max_features,
        n_splits=args.n_splits,
        random_state=args.random_state,
        groups=groups,
        grouped=args.grouped,
    )
    metrics.to_csv(out_dir / "forward_selection_metrics.csv", index=False, encoding="utf-8-sig")
    plot_forward_selection(metrics, out_dir / "forward_selection_auc.png")

    best = metrics.sort_values("roc_auc", ascending=False).iloc[0].to_dict()
    selected_count = int(best["n_features"])
    selected = ranked[:selected_count]
    paper_top40 = ranked[: min(PAPER_SELECTED_FEATURE_COUNT, len(ranked))]
    save_json(
        {
            "best_by_reproduction": best,
            "selected_features": selected,
            "paper_top40_features": paper_top40,
            "note": "Paper reports top 40 as best; use paper_top40_features for strict paper reproduction.",
        },
        out_dir / "selected_features.json",
    )

    print("[saved]", out_dir / "shap_importance_full.csv")
    print("[saved]", out_dir / "forward_selection_metrics.csv")
    print("[best]", best)
    print("[paper_top40_first5]", paper_top40[:5])


if __name__ == "__main__":
    main()
