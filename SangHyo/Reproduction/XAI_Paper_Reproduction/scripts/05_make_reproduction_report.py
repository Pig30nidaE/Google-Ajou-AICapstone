from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT_PATH.parents[1] / "src"))

from xai_paper_reproduction import PAPER_FINAL_PARAMS, PAPER_MODEL_METRICS, ProjectPaths, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a markdown summary report from reproduction outputs.")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def markdown_table(df: pd.DataFrame, cols: list[str] | None = None, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_Not available yet._"
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    if max_rows:
        df = df.head(max_rows)
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    for col in df.columns:
        df[col] = df[col].map(format_cell)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in df.astype(str).to_numpy()]
    return "\n".join([header, sep, *rows])


def format_cell(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def paper_metric_table() -> pd.DataFrame:
    return pd.DataFrame(
        [{"model": model, **metrics} for model, metrics in PAPER_MODEL_METRICS.items()]
    )


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_script(SCRIPT_PATH, output_dir=args.output_dir)
    out = ensure_dir(paths.output_dir)

    preprocess = read_json(out / "data" / "preprocess_summary.json")
    baselines = read_csv(out / "baselines" / "model_comparison.csv")
    baseline_delta = read_csv(out / "baselines" / "model_comparison_against_paper.csv")
    fs_metrics = read_csv(out / "feature_selection" / "forward_selection_metrics.csv")
    selected = read_json(out / "feature_selection" / "selected_features.json")
    final_cv = read_json(out / "final" / "final_cv_metrics.json")
    drs = read_json(out / "final" / "dementia_risk_score_summary.json")
    final_importance = read_csv(out / "final" / "shap_importance_positive.csv")

    lines = [
        "# XAI Paper Reproduction Report",
        "",
        "Target paper: `설명가능 인공지능을 활용한 라이프로그 기반 치매 위험도 산정 방법에 관한 연구.pdf`",
        "",
        "## Paper Reference Metrics",
        "",
        markdown_table(paper_metric_table()),
        "",
        "## Preprocessing",
        "",
    ]
    if preprocess:
        lines.extend(
            [
                "```json",
                json.dumps(preprocess, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    else:
        lines.extend(["_Not available yet. Run `01_preprocess_daily_binary.py`._", ""])

    lines.extend(["## Baseline Model Comparison", "", markdown_table(baselines), ""])
    lines.extend(["## Baseline Delta Against Paper", "", markdown_table(baseline_delta), ""])

    lines.extend(["## SHAP Forward Selection", ""])
    if fs_metrics is not None and not fs_metrics.empty:
        best = fs_metrics.sort_values("roc_auc", ascending=False).iloc[0]
        paper40 = fs_metrics.loc[fs_metrics["n_features"] == 40]
        lines.append(f"- Best reproduction k: `{int(best['n_features'])}`, ROC-AUC: `{best['roc_auc']:.4f}`")
        if not paper40.empty:
            lines.append(f"- Paper top-40 ROC-AUC in this run: `{paper40.iloc[0]['roc_auc']:.4f}`")
        lines.append("")
        lines.append(markdown_table(fs_metrics.sort_values("roc_auc", ascending=False), max_rows=10))
    else:
        lines.append("_Not available yet. Run `03_shap_forward_selection.py`._")
    lines.append("")

    lines.extend(["## Final LightGBM", ""])
    lines.append("Paper parameters:")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(PAPER_FINAL_PARAMS, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    if final_cv:
        lines.append("Reproduction final CV:")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(final_cv, ensure_ascii=False, indent=2))
        lines.append("```")
    else:
        lines.append("_Not available yet. Run `04_final_lgbm_shap_drs.py`._")
    lines.append("")

    lines.extend(["## Dementia Risk Score", ""])
    if drs:
        lines.append("```json")
        lines.append(json.dumps(drs, ensure_ascii=False, indent=2))
        lines.append("```")
    else:
        lines.append("_Not available yet._")
    lines.append("")

    lines.extend(["## Top Final SHAP Features", "", markdown_table(final_importance, max_rows=20), ""])

    if selected:
        lines.extend(
            [
                "## Selected Feature Lists",
                "",
                f"- Best reproduction feature count: `{len(selected.get('selected_features', []))}`",
                f"- Paper top-40 feature count: `{len(selected.get('paper_top40_features', []))}`",
                "",
            ]
        )

    report_path = out / "reproduction_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("[saved]", report_path)


if __name__ == "__main__":
    main()
