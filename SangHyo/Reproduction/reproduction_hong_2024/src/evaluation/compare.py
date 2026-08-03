"""The three result tables from the spec, plus the estimand-aware deltas.

Every table keeps the paper's numbers in their own column and never averages them
with ours.  Table 3 is the one that stops a reader from comparing the wrong pair:
it states, per validation scheme, whether subjects repeat across the split and
what population the number actually describes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..utils.io import read_json

#: Hong et al. (2024) Table 5.
PAPER_RESULTS: dict[str, dict[str, float]] = {
    "lstm_3day": {"sensitivity": 0.87, "specificity": 0.74, "roc_auc": 0.88,
                  "accuracy": 0.81, "precision": 0.77, "f1": 0.82},
    "lstm_4day": {"sensitivity": 0.89, "specificity": 0.77, "roc_auc": 0.91,
                  "accuracy": 0.83, "precision": 0.80, "f1": 0.84},
    "lstm_5day": {"sensitivity": 0.89, "specificity": 0.80, "roc_auc": 0.92,
                  "accuracy": 0.85, "precision": 0.82, "f1": 0.85},
    "xgboost": {"sensitivity": 0.68, "specificity": 0.76, "roc_auc": 0.81,
                "accuracy": 0.72, "precision": 0.74, "f1": 0.71},
    "random_forest": {"sensitivity": 0.67, "specificity": 0.77, "roc_auc": 0.81,
                      "accuracy": 0.72, "precision": 0.75, "f1": 0.71},
    "logistic_regression": {"sensitivity": 0.59, "specificity": 0.60, "roc_auc": 0.63,
                            "accuracy": 0.60, "precision": 0.60, "f1": 0.60},
    "svm": {"sensitivity": 0.62, "specificity": 0.59, "roc_auc": 0.64,
            "accuracy": 0.61, "precision": 0.60, "f1": 0.61},
}

PAPER_PRECISION_AT_100 = 0.96

EXPERIMENT_COLUMNS = (
    ("paper_temporal_reconstruction", "원 시간분할 재구현"),
    ("strict_same_subject_temporal", "Strict temporal"),
    ("fixed_subject_independent", "Subject-wise fixed"),
    ("nested_subject_independent", "Nested Group CV"),
)

#: What each validation scheme actually estimates.  This is Table 3.
VALIDATION_PROPERTIES = [
    {
        "scheme": "원 논문 시간분할 (paper_temporal_reconstruction)",
        "subject_overlap": "있음 (설계상)",
        "date_boundary_shared": "없음 (날짜 먼저 분할)",
        "model_selection_independent": "아니오 (논문 보고 설정 고정)",
        "estimand": "A: 이미 학습된 피험자의 미래 기간",
    },
    {
        "scheme": "논문 문언 그대로 (paper_literal_variant)",
        "subject_overlap": "있음",
        "date_boundary_shared": "있음 (경계 윈도우 존재)",
        "model_selection_independent": "아니오",
        "estimand": "A + 누수 (성능 주장 불가, 진단 전용)",
    },
    {
        "scheme": "엄격 시간분할 (strict_same_subject_temporal)",
        "subject_overlap": "있음 (설계상)",
        "date_boundary_shared": "없음 (embargo 적용)",
        "model_selection_independent": "예 (test 1회만 사용)",
        "estimand": "A: 이미 학습된 피험자의 미래 기간",
    },
    {
        "scheme": "피험자 독립 고정 (fixed_subject_independent)",
        "subject_overlap": "없음",
        "date_boundary_shared": "없음",
        "model_selection_independent": "예 (논문 설정 고정, 재선택 없음)",
        "estimand": "B: 신규 피험자",
    },
    {
        "scheme": "Nested Group CV (nested_subject_independent)",
        "subject_overlap": "없음",
        "date_boundary_shared": "없음",
        "model_selection_independent": "예 (inner CV에서만 선택)",
        "estimand": "B: 신규 피험자, 모델 선택 비용 포함",
    },
]


def load_reports(paths: dict[str, str | Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        path = Path(path)
        if path.is_file():
            out[name] = read_json(path)
    return out


def _value(block: dict[str, Any], unit: str, metric: str) -> Any:
    unit_block = block.get(unit)
    if not unit_block:
        return None
    value = unit_block.get(metric)
    return None if value is None or (isinstance(value, float) and not np.isfinite(value)) else value


def _lookup(report: dict[str, Any], model: str, length: int, unit: str, metric: str) -> Any:
    """Read a *full* per-length estimate, never a partial subset.

    The nested experiment also emits ``{model}_L{length}`` blocks, but those are
    the subsets of outer folds whose inner CV happened to pick that length -- in
    the 2026-08-02 run, ``lstm_L5`` was one fold and 35 subjects.  Filling the
    per-length rows from them would put "nested LSTM 5-day = 0.78" next to the
    paper's 0.92, which is exactly the pick-the-best-length bias experiment C
    exists to avoid.  Partial blocks are reported separately, in table 5.
    """
    block = report.get("results", {}).get(f"{model}_L{length}")
    if not block or block.get("is_partial_subset"):
        return None
    return _value(block, unit, metric)


def _nested_estimate(report: dict[str, Any], model: str) -> dict[str, Any] | None:
    """The one honest nested number per model: all outer folds, length chosen inside."""
    block = report.get("results", {}).get(f"{model}_Lnested")
    if not block:
        return None
    subject = block.get("subject_level") or {}
    ci = block.get("subject_bootstrap_ci") or {}
    selection = block.get("selection") or {}
    return {
        "model": model,
        "roc_auc": _value(block, "subject_level", "roc_auc"),
        "pr_auc": _value(block, "subject_level", "pr_auc"),
        "balanced_accuracy": _value(block, "subject_level", "balanced_accuracy"),
        "n_subjects": subject.get("n_subjects"),
        "n_folds": block.get("n_folds"),
        "ci_lower": ci.get("ci_lower"),
        "ci_upper": ci.get("ci_upper"),
        "chosen_sequence_lengths": selection.get("chosen_sequence_lengths", {}),
        "sequence_roc_auc": _value(block, "sequence_level", "roc_auc"),
        "threshold_source": block.get("threshold_source"),
    }


def _partial_diagnostics(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-length subsets from the nested run, labelled as the fragments they are."""
    out = []
    for key, block in (report.get("results") or {}).items():
        if not block.get("is_partial_subset"):
            continue
        subject = block.get("subject_level") or {}
        out.append(
            {
                "key": key,
                "model": block.get("model"),
                "sequence_length": block.get("sequence_length"),
                "n_folds": block.get("n_folds"),
                "n_subjects": subject.get("n_subjects"),
                "roc_auc": _value(block, "subject_level", "roc_auc"),
            }
        )
    return sorted(out, key=lambda r: (str(r["model"]), str(r["sequence_length"])))


def build_comparison(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Assemble tables 1-3 from whichever experiment reports exist."""
    table1, table2 = [], []
    for length in (3, 4, 5):
        paper = PAPER_RESULTS[f"lstm_{length}day"]
        row: dict[str, Any] = {
            "model": "LSTM",
            "sequence_length": length,
            "paper_roc_auc": paper["roc_auc"],
            "paper_sensitivity": paper["sensitivity"],
            "paper_specificity": paper["specificity"],
        }
        row2: dict[str, Any] = {"model": "LSTM", "sequence_length": length,
                                "paper_roc_auc": paper["roc_auc"]}
        for key, _label in EXPERIMENT_COLUMNS:
            report = reports.get(key)
            unit = "sequence_level" if key.startswith(("paper_", "strict_")) else "subject_level"
            row[key] = _lookup(report or {}, "lstm", length, unit, "roc_auc")
            row2[key] = row[key]
            row[f"{key}__unit"] = unit
        table1.append(row)
        table2.append(row2)

    nested_report = reports.get("nested_subject_independent") or {}
    nested_models = sorted(
        {
            key[: -len("_Lnested")]
            for key in (nested_report.get("results") or {})
            if key.endswith("_Lnested")
        }
    )
    nested_estimates = [
        estimate
        for estimate in (_nested_estimate(nested_report, m) for m in nested_models)
        if estimate is not None
    ]

    deltas = _deltas(table1, nested_estimates)
    return {
        "table1_by_sequence_length": table1,
        "table2_auc_summary": table2,
        "table3_validation_properties": VALIDATION_PROPERTIES,
        "table4_nested_selected": nested_estimates,
        "table5_nested_partial_subsets": _partial_diagnostics(nested_report),
        "paper_results": PAPER_RESULTS,
        "deltas": deltas,
        "available_reports": sorted(reports),
    }


def _deltas(
    table1: list[dict[str, Any]], nested_estimates: list[dict[str, Any]]
) -> dict[str, Any]:
    """The comparisons the spec asks for, each labelled with what it means."""
    def diff(row: dict[str, Any], left: str, right: str) -> float | None:
        a, b = row.get(left), row.get(right)
        return None if a is None or b is None else float(a - b)

    out: dict[str, Any] = {"per_sequence_length": []}
    for row in table1:
        out["per_sequence_length"].append(
            {
                "sequence_length": row["sequence_length"],
                "reconstruction_minus_paper": diff(row, "paper_temporal_reconstruction", "paper_roc_auc"),
                "strict_minus_reconstruction": diff(row, "strict_same_subject_temporal", "paper_temporal_reconstruction"),
                "subjectwise_minus_strict": diff(row, "fixed_subject_independent", "strict_same_subject_temporal"),
            }
        )

    # The nested arm produces one estimate per model, not one per length -- the
    # length is what it selected.  Comparing it against the non-nested value at a
    # fixed length is therefore a comparison across different granularities, and
    # it is written out that way rather than folded into the per-length rows.
    lstm_nested = next((e for e in nested_estimates if e["model"] == "lstm"), None)
    out["nested_vs_nonnested"] = [
        {
            "sequence_length": row["sequence_length"],
            "nonnested_fixed_subject": row.get("fixed_subject_independent"),
            "nested_selected": lstm_nested["roc_auc"] if lstm_nested else None,
            "nested_minus_nonnested": (
                None
                if lstm_nested is None or row.get("fixed_subject_independent") is None
                else float(lstm_nested["roc_auc"] - row["fixed_subject_independent"])
            ),
        }
        for row in table1
    ]

    out["interpretation"] = {
        "reconstruction_minus_paper": "구현 차이. 같은 estimand 안에서의 재현 오차다.",
        "strict_minus_reconstruction": "경계 시퀀스와 전처리 누수를 제거한 효과. estimand는 A로 동일하다.",
        "subjectwise_minus_strict": (
            "estimand A에서 B로 바뀐 차이다. 누수 제거 효과와 질문 자체가 바뀐 효과가 "
            "함께 들어 있으므로 이 값을 '누수 크기'라고 부르면 안 된다."
        ),
        "nested_minus_nonnested": (
            "모델 선택 비용. 음수면 non-nested 값이 낙관적이었다는 뜻이다. "
            "nested 쪽은 길이를 inner CV에서 고른 단일 추정치이므로, 고정 길이 "
            "non-nested 값과는 입도가 다르다는 점을 함께 적어야 한다."
        ),
    }
    return out


def verify_paper_arithmetic() -> dict[str, Any]:
    """Sanity-check Table 5 against itself before trusting it as a target.

    Sensitivity, specificity, precision and prevalence over-determine accuracy and
    F1, so an internally inconsistent row would mean the target numbers cannot all
    be hit at once.  F1 is checked because it follows from sensitivity and
    precision alone, with no prevalence term.
    """
    checks = []
    for name, row in PAPER_RESULTS.items():
        sensitivity, precision, f1 = row["sensitivity"], row["precision"], row["f1"]
        implied_f1 = (
            2 * precision * sensitivity / (precision + sensitivity)
            if (precision + sensitivity)
            else float("nan")
        )
        checks.append(
            {
                "model": name,
                "reported_f1": f1,
                "implied_f1_from_precision_and_sensitivity": round(float(implied_f1), 4),
                "abs_error": round(abs(implied_f1 - f1), 4),
                "consistent": bool(abs(implied_f1 - f1) <= 0.015),
            }
        )
    return {
        "checks": checks,
        "all_models_consistent": all(c["consistent"] for c in checks),
        "note": (
            "F1은 precision과 sensitivity만으로 결정된다. 불일치는 반올림이거나 "
            "표의 값들이 서로 다른 실행에서 왔음을 뜻한다."
        ),
    }


def render_comparison_markdown(comparison: dict[str, Any]) -> str:
    def cell(value: Any) -> str:
        return "–" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))

    lines = [
        "# 결과 비교표",
        "",
        "표의 빈 칸은 아직 실행하지 않은 실험이다. 논문 열과 재구현 열을 같은 의미의",
        "일반화 성능으로 읽지 않는다. 각 열이 무엇을 추정하는지는 표 3에 있다.",
        "",
        "## 표 1. 시퀀스 길이별 ROC-AUC",
        "",
        "| 모델 | 길이 | 원 논문 | 원 시간분할 재구현 | Strict temporal | Subject-wise fixed | Nested Group CV |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison["table1_by_sequence_length"]:
        lines.append(
            f"| {row['model']} | {row['sequence_length']}일 | {cell(row['paper_roc_auc'])} | "
            + " | ".join(cell(row.get(key)) for key, _ in EXPERIMENT_COLUMNS)
            + " |"
        )
    lines += [
        "",
        "앞의 두 열은 **시퀀스 단위**, 뒤의 두 열은 **피험자 단위** 지표다 "
        "(각 행의 `__unit` 필드에 기록된다).",
        "",
        "**Nested 열이 비어 있는 것은 정상이다.** Nested 실험은 길이를 inner CV에서 "
        "고르므로 고정 길이별 추정치를 만들지 않는다. 그 결과는 표 4에 있다.",
        "",
        "## 표 2. AUC 요약",
        "",
        "| 모델 | 길이 | 원 논문 AUC | 원 재구현 AUC | 신규 피험자 AUC | Nested AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison["table2_auc_summary"]:
        lines.append(
            f"| {row['model']} | {row['sequence_length']}일 | {cell(row['paper_roc_auc'])} | "
            f"{cell(row.get('paper_temporal_reconstruction'))} | "
            f"{cell(row.get('fixed_subject_independent'))} | "
            f"{cell(row.get('nested_subject_independent'))} |"
        )

    lines += [
        "",
        "## 표 3. 검증방식의 성질",
        "",
        "| 검증방식 | 동일 피험자 train/test 중복 | 날짜 경계 공유 | 모델 선택 독립 | 평가대상 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in comparison["table3_validation_properties"]:
        lines.append(
            f"| {row['scheme']} | {row['subject_overlap']} | {row['date_boundary_shared']} | "
            f"{row['model_selection_independent']} | {row['estimand']} |"
        )

    lines += ["", "## 차이값과 그 해석", ""]
    for row in comparison["deltas"]["per_sequence_length"]:
        lines.append(f"- **{row['sequence_length']}일**: " + ", ".join(
            f"{k} = {cell(v)}" for k, v in row.items() if k != "sequence_length"
        ))
    lines += [""]
    for key, text in comparison["deltas"]["interpretation"].items():
        lines.append(f"- `{key}`: {text}")
    return "\n".join(lines) + "\n"
