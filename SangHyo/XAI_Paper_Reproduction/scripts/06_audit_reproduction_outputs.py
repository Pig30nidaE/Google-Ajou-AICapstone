from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT_PATH.parents[1] / "src"))

from xai_paper_reproduction import (
    PAPER_FINAL_PARAMS,
    PAPER_MODEL_METRICS,
    PAPER_SELECTED_FEATURE_COUNT,
    ProjectPaths,
    ensure_dir,
    load_json,
    save_json,
)


PAPER_COUNTS = {
    "rows": 12183,
    "subjects": 174,
    "class_counts": {"0": 7737, "1": 4446},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit generated outputs against paper-reproduction requirements.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="Permit reduced smoke-test outputs such as one-model baselines or sampled DRS rows.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return load_json(path)


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def check_exists(checks: list[dict[str, Any]], path: Path) -> bool:
    exists = path.exists()
    add_check(checks, f"exists:{path.name}", exists, str(path))
    return exists


def audit_outputs(output_dir: Path, *, allow_smoke: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    required_paths = [
        output_dir / "data" / "daily_binary_lifelog.csv",
        output_dir / "data" / "feature_columns.json",
        output_dir / "data" / "preprocess_summary.json",
        output_dir / "baselines" / "model_comparison.csv",
        output_dir / "baselines" / "model_comparison_against_paper.csv",
        output_dir / "feature_selection" / "shap_importance_full.csv",
        output_dir / "feature_selection" / "forward_selection_metrics.csv",
        output_dir / "feature_selection" / "selected_features.json",
        output_dir / "final" / "final_cv_metrics.json",
        output_dir / "final" / "dementia_risk_scores.csv",
        output_dir / "final" / "dementia_risk_score_summary.json",
        output_dir / "final" / "shap_importance_positive.csv",
        output_dir / "reproduction_report.md",
    ]
    for path in required_paths:
        check_exists(checks, path)

    preprocess = read_json(output_dir / "data" / "preprocess_summary.json")
    features = read_json(output_dir / "data" / "feature_columns.json")
    if preprocess:
        add_check(checks, "paper_row_count", preprocess.get("rows") == PAPER_COUNTS["rows"], str(preprocess.get("rows")))
        add_check(
            checks,
            "paper_subject_count",
            preprocess.get("subjects") == PAPER_COUNTS["subjects"],
            str(preprocess.get("subjects")),
        )
        add_check(
            checks,
            "paper_class_counts",
            preprocess.get("class_counts") == PAPER_COUNTS["class_counts"],
            str(preprocess.get("class_counts")),
        )
        add_check(checks, "nonzero_feature_count", int(preprocess.get("feature_count", 0)) > 0, str(preprocess.get("feature_count")))
        if isinstance(features, list):
            add_check(
                checks,
                "feature_column_count_matches_summary",
                len(features) == preprocess.get("feature_count"),
                f"{len(features)} vs {preprocess.get('feature_count')}",
            )

    baselines = read_csv(output_dir / "baselines" / "model_comparison.csv")
    if baselines is not None:
        expected_models = set(PAPER_MODEL_METRICS)
        observed_models = set(baselines.get("model", []))
        has_all_models = expected_models.issubset(observed_models)
        add_check(
            checks,
            "all_paper_models_evaluated",
            allow_smoke or has_all_models,
            f"observed={sorted(observed_models)}",
        )
        if not has_all_models:
            warnings.append(
                {
                    "name": "baseline_smoke_only",
                    "detail": "Not all seven paper baseline models were found. This is acceptable only with --allow-smoke.",
                }
            )

    selected = read_json(output_dir / "feature_selection" / "selected_features.json")
    if selected:
        top40 = selected.get("paper_top40_features", [])
        add_check(
            checks,
            "paper_top40_feature_count",
            len(top40) == PAPER_SELECTED_FEATURE_COUNT,
            str(len(top40)),
        )

    final_cv = read_json(output_dir / "final" / "final_cv_metrics.json")
    if final_cv:
        params = final_cv.get("params", {})
        paper_params_match = all(params.get(k) == v for k, v in PAPER_FINAL_PARAMS.items())
        add_check(checks, "final_params_match_paper", allow_smoke or paper_params_match, str(params))
        add_check(
            checks,
            "final_selected_feature_count",
            final_cv.get("selected_feature_count") == PAPER_SELECTED_FEATURE_COUNT,
            str(final_cv.get("selected_feature_count")),
        )
        metrics = final_cv.get("cv_metrics", {})
        add_check(checks, "final_cv_has_roc_auc", "roc_auc" in metrics, str(metrics.get("roc_auc")))
        if "roc_auc" in metrics and metrics["roc_auc"] < 0.8:
            warnings.append({"name": "low_final_cv_roc_auc", "detail": f"ROC-AUC={metrics['roc_auc']:.4f}"})

    drs_summary = read_json(output_dir / "final" / "dementia_risk_score_summary.json")
    risk = read_csv(output_dir / "final" / "dementia_risk_scores.csv")
    if drs_summary:
        row_sample = drs_summary.get("drs_row_sample", {})
        sampled = bool(row_sample.get("sampled", False))
        add_check(checks, "drs_not_sampled_for_full_run", allow_smoke or not sampled, str(row_sample))
        daily_test = drs_summary.get("daily_one_sided_t_test", {})
        mean_ok = daily_test.get("impaired_mean", 0) > daily_test.get("cn_mean", float("inf"))
        p_ok = daily_test.get("p_value", 1) < 0.05
        add_check(checks, "drs_impaired_mean_greater_than_cn", mean_ok, str(daily_test))
        add_check(checks, "drs_one_sided_t_test_significant", p_ok, str(daily_test))
    if risk is not None:
        expected_rows = PAPER_COUNTS["rows"]
        add_check(checks, "drs_row_count", allow_smoke or len(risk) == expected_rows, str(len(risk)))
        add_check(checks, "drs_contains_both_classes", set(risk.get("binary_class", [])) == {0, 1}, str(sorted(set(risk.get("binary_class", [])))))

    return checks, warnings


def write_markdown(output_dir: Path, checks: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> None:
    lines = ["# XAI Reproduction Output Audit", ""]
    lines.append("| Check | Status | Detail |")
    lines.append("| --- | --- | --- |")
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        detail = str(check["detail"]).replace("|", "\\|")
        lines.append(f"| {check['name']} | {status} | {detail} |")
    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning['name']}: {warning['detail']}")
    else:
        lines.append("_None._")
    (output_dir / "reproduction_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_script(SCRIPT_PATH, output_dir=args.output_dir)
    output_dir = ensure_dir(paths.output_dir)

    checks, warnings = audit_outputs(output_dir, allow_smoke=args.allow_smoke)
    passed = all(check["passed"] for check in checks)
    result = {
        "passed": passed,
        "allow_smoke": args.allow_smoke,
        "checks": checks,
        "warnings": warnings,
    }
    save_json(result, output_dir / "reproduction_audit.json")
    write_markdown(output_dir, checks, warnings)

    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"[{status}] {check['name']}: {check['detail']}")
    for warning in warnings:
        print(f"[WARN] {warning['name']}: {warning['detail']}")
    print("[saved]", output_dir / "reproduction_audit.json")
    print("[saved]", output_dir / "reproduction_audit.md")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
