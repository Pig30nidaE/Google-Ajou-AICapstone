"""The three result tables from the spec, plus the estimand-aware deltas.

Every table keeps the paper's numbers in their own column and never averages them
with ours.  Table 3 is the one that stops a reader from comparing the wrong pair:
it states, per validation scheme, whether subjects repeat across the split and
what population the number actually describes.
"""

from __future__ import annotations

from copy import deepcopy
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
TABLE5_METRICS = ("sensitivity", "specificity", "roc_auc", "accuracy", "precision", "f1")
REPRODUCTION_AUDIT_TOLERANCE = 0.03

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


def load_reports(
    paths: dict[str, str | Path | list[str | Path] | tuple[str | Path, ...]]
) -> dict[str, dict[str, Any]]:
    """Load reports, merging per-length runs that belong to one experiment."""
    out: dict[str, dict[str, Any]] = {}
    for name, value in paths.items():
        candidates = list(value) if isinstance(value, (list, tuple)) else [value]
        parts: list[tuple[Path, dict[str, Any]]] = []
        for candidate in candidates:
            path = Path(candidate)
            if path.is_file():
                parts.append((path, read_json(path)))
        if not parts:
            continue
        out[name] = _merge_report_parts(name, parts)
    return out


def _merge_report_parts(
    name: str, parts: list[tuple[Path, dict[str, Any]]]
) -> dict[str, Any]:
    if len(parts) == 1:
        return parts[0][1]

    merged = dict(parts[0][1])
    reference_path, reference = parts[0]

    def config_except_length(report: dict[str, Any]) -> dict[str, Any] | None:
        raw = report.get("resolved_config")
        if not isinstance(raw, dict):
            return None
        normalised = deepcopy(raw)
        sequence = normalised.get("sequence")
        if isinstance(sequence, dict):
            sequence.pop("lengths", None)
        return normalised

    compatibility_fields = ("estimand", "seed", "data", "models", "runtime_signature_context")
    reference_config = config_except_length(reference)
    results: dict[str, Any] = {}
    models: set[str] = set()
    lengths: set[int] = set()
    partial = False
    for path, report in parts:
        if report.get("experiment") not in (None, name):
            raise ValueError(
                f"cannot merge {path}: experiment={report.get('experiment')!r}, "
                f"expected {name!r}"
            )
        for field in compatibility_fields:
            if report.get(field) != reference.get(field):
                raise ValueError(
                    f"cannot merge {path}: {field} differs from {reference_path}"
                )
        current_config = config_except_length(report)
        if reference_config is None or current_config is None:
            raise ValueError(
                "per-length report merging requires resolved_config in every source"
            )
        if current_config != reference_config:
            raise ValueError(
                f"cannot merge {path}: non-length config differs from {reference_path}"
            )
        overlap = set(results) & set(report.get("results") or {})
        if overlap:
            raise ValueError(f"duplicate result keys while merging {path}: {sorted(overlap)}")
        results.update(report.get("results") or {})
        models.update(map(str, report.get("models") or []))
        lengths.update(int(value) for value in report.get("sequence_lengths") or [])
        partial |= bool((report.get("run_scope") or {}).get("is_partial_run", False))

    if name == "paper_temporal_reconstruction" and lengths != {3, 4, 5}:
        partial = True

    source_metadata = [
        {
            "report_path": str(path),
            "config_path": report.get("config_path"),
            "run_signature": report.get("run_signature"),
            "artifacts": report.get("artifacts"),
            "resolved_config": report.get("resolved_config"),
        }
        for path, report in parts
    ]
    for misleading_single_source_field in (
        "artifacts", "config_path", "resolved_config", "run_signature"
    ):
        merged.pop(misleading_single_source_field, None)

    merged.update(
        {
            "experiment": name,
            "results": results,
            "models": sorted(models),
            "sequence_lengths": sorted(lengths),
            "source_reports": [str(path) for path, _ in parts],
            "source_metadata": source_metadata,
            "combined_from_per_length_runs": True,
            "all_audits_passed": all(
                bool(report.get("all_audits_passed", False)) for _, report in parts
            ),
            "n_audit_warnings": sum(
                int(report.get("n_audit_warnings", 0)) for _, report in parts
            ),
            "run_scope": {
                "is_partial_run": partial,
                "combined_from_per_length_runs": True,
                "source_count": len(parts),
            },
        }
    )
    return merged


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
    if (report.get("run_scope") or {}).get("is_partial_run", False):
        return None
    block = report.get("results", {}).get(f"{model}_L{length}")
    if not block or block.get("is_partial_subset"):
        return None
    return _value(block, unit, metric)


def _nested_estimate(report: dict[str, Any], model: str) -> dict[str, Any] | None:
    """The one honest nested number per model: all outer folds, length chosen inside."""
    if (report.get("run_scope") or {}).get("is_partial_run", False):
        return None
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
        # A partial block covers exactly one length; the engine stores it as a
        # one-element list, so unwrap it for display.
        length = block.get("sequence_length")
        if isinstance(length, list) and len(length) == 1:
            length = length[0]
        out.append(
            {
                "key": key,
                "model": block.get("model"),
                "sequence_length": length,
                "n_folds": block.get("n_folds"),
                "n_subjects": subject.get("n_subjects"),
                "roc_auc": _value(block, "subject_level", "roc_auc"),
            }
        )
    return sorted(out, key=lambda r: (str(r["model"]), str(r["sequence_length"])))


def _implied_prevalence(row: dict[str, float]) -> float | None:
    """Infer class prevalence from sensitivity, specificity and precision."""
    sensitivity = row["sensitivity"]
    specificity = row["specificity"]
    precision = row["precision"]
    denominator = sensitivity * (1 - precision) + precision * (1 - specificity)
    if denominator <= 0:
        return None
    return float(precision * (1 - specificity) / denominator)


def _paper_key(model: str, length: int) -> str | None:
    if model == "lstm":
        return f"lstm_{length}day"
    return model if model in PAPER_RESULTS else None


def _paper_metric_diagnostics(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare every reported Table 5 metric, not only the headline AUC."""
    if (report.get("run_scope") or {}).get("is_partial_run", False):
        return []
    rows: list[dict[str, Any]] = []
    for key, block in (report.get("results") or {}).items():
        if block.get("is_partial_subset") or block.get("result_kind") == "leakage_diagnostic":
            continue
        model = str(block.get("model", ""))
        length_value = block.get("sequence_length")
        if isinstance(length_value, list):
            if len(length_value) != 1:
                continue
            length_value = length_value[0]
        if length_value is None:
            continue
        length = int(length_value)
        target_key = _paper_key(model, length)
        target = PAPER_RESULTS.get(target_key or "")
        actual = block.get("sequence_level") or {}
        if not target or not actual:
            continue

        comparisons = {
            metric: {
                "paper": target[metric],
                "reproduction": actual.get(metric),
                "delta": (
                    None
                    if actual.get(metric) is None
                    else float(actual[metric] - target[metric])
                ),
                "within_abs_0_03": (
                    False
                    if actual.get(metric) is None
                    else abs(float(actual[metric] - target[metric]))
                    <= REPRODUCTION_AUDIT_TOLERANCE
                ),
            }
            for metric in TABLE5_METRICS
        }

        fit = block.get("fit") or {}
        fidelity = block.get("method_fidelity") or {}
        training = fit.get("training") or {}
        sequence_pk = block.get("precision_at_k") or {}
        n_positive = int(actual.get("n_positive", 0))
        effective_k = min(100, int(actual.get("n", 0)))
        max_sequence_pk = (
            float(min(n_positive, effective_k) / effective_k) if effective_k else None
        )
        subject_pk = block.get("subject_precision_at_k") or {}

        rows.append(
            {
                "key": key,
                "model": model,
                "sequence_length": length,
                "n_sequences": actual.get("n"),
                "n_subjects": (block.get("subject_level") or {}).get("n_subjects"),
                "backend": fit.get("backend"),
                "representation": fit.get("representation"),
                "method_fidelity": fidelity,
                "early_stopping_applied": bool(
                    training.get("early_stopping_applied")
                    or (
                        training.get("used_validation")
                        and (fit.get("params") or {}).get("early_stopping")
                    )
                ),
                "observed_sequence_prevalence": actual.get("prevalence"),
                "paper_implied_prevalence": _implied_prevalence(target),
                "metrics": comparisons,
                "all_table5_metrics_within_abs_0_03": all(
                    item["within_abs_0_03"] for item in comparisons.values()
                ),
                "sequence_precision_at_100": sequence_pk.get("precision_at_k"),
                "sequence_precision_at_100_max_possible": sequence_pk.get(
                    "max_possible_precision_at_k", max_sequence_pk
                ),
                "subject_precision_at_100": subject_pk.get("precision_at_k"),
                "subject_precision_at_100_max_possible": subject_pk.get(
                    "max_possible_precision_at_k"
                ),
                "paper_precision_at_100": (
                    PAPER_PRECISION_AT_100 if model == "lstm" and length == 5 else None
                ),
                "paper_precision_at_100_attainable_on_sequence_set": (
                    max_sequence_pk is not None
                    and PAPER_PRECISION_AT_100 <= max_sequence_pk
                    if model == "lstm" and length == 5
                    else None
                ),
            }
        )

    order = {"lstm": 0, "xgboost": 1, "random_forest": 2,
             "logistic_regression": 3, "svm": 4}
    return sorted(rows, key=lambda row: (row["sequence_length"], order.get(row["model"], 99)))


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
        "table6_paper_metric_reproduction": _paper_metric_diagnostics(
            reports.get("paper_temporal_reconstruction") or {}
        ),
        "paper_results": PAPER_RESULTS,
        "paper_precision_at_100": PAPER_PRECISION_AT_100,
        "reproduction_audit_tolerance": REPRODUCTION_AUDIT_TOLERANCE,
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

    nested = comparison.get("table4_nested_selected") or []
    lines += [
        "",
        "## 표 4. Nested Group CV (길이를 inner CV에서 선택한 단일 추정치)",
        "",
        "이 표가 실험 C의 **정식 결과**다. 모델마다 값이 하나뿐인 이유는 시퀀스 길이가 "
        "결과가 아니라 inner CV가 고른 선택이기 때문이다.",
        "",
    ]
    if nested:
        lines += [
            "| 모델 | 피험자 AUC | 95% CI | PR-AUC | Balanced Acc. | 피험자 수 | fold | 선택된 길이 |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in nested:
            ci = (
                f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
                if row.get("ci_lower") is not None else "–"
            )
            chosen = ", ".join(
                f"{k}일×{v}" for k, v in sorted((row.get("chosen_sequence_lengths") or {}).items())
            ) or "–"
            lines.append(
                f"| {row['model']} | {cell(row['roc_auc'])} | {ci} | {cell(row['pr_auc'])} | "
                f"{cell(row['balanced_accuracy'])} | {row.get('n_subjects')} | "
                f"{row.get('n_folds')} | {chosen} |"
            )
    else:
        lines.append("아직 nested 실험 결과가 없다.")

    partials = comparison.get("table5_nested_partial_subsets") or []
    if partials:
        lines += [
            "",
            "## 표 5. Nested 부분집합 진단 (성능 주장 불가)",
            "",
            "> **경고.** 아래 값은 inner CV가 그 길이를 고른 **일부 fold의 일부 피험자**에서만 "
            "계산된 것이다. 174명 전체 추정치가 아니며, 이 중 가장 높은 값을 골라 "
            "\"nested에서 N일이 가장 좋았다\"고 보고하는 것은 실험 C가 막으려는 바로 그 "
            "선택 편향이다. 길이별 비교가 필요하면 `fixed_subject_independent`(실험 B2)를 "
            "쓴다. 그쪽은 세 길이를 모두 전체 피험자에서 평가한다.",
            "",
            "| 모델 | 길이 | fold 수 | 피험자 수 | 피험자 AUC |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in partials:
            lines.append(
                f"| {row['model']} | {row['sequence_length']}일 | {row['n_folds']} | "
                f"{row['n_subjects']} | {cell(row['roc_auc'])} |"
            )

    diagnostics = comparison.get("table6_paper_metric_reproduction") or []
    lines += [
        "",
        "## 표 6. 논문 Table 5 전체 지표 재현 진단",
        "",
        "각 셀은 `논문 → 재구현 (차이)`다. ±0.03은 누락을 드러내기 위한 감사 띠이며, "
        "통계적 동등성 판정은 아니다.",
        "",
    ]
    if diagnostics:
        lines += [
            "| 모델 | 길이 | backend | n(seq/subj) | Sens. | Spec. | AUC | Acc. | Prec. | F1 | 전체 ±.03 |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]

        def metric_cell(row: dict[str, Any], metric: str) -> str:
            item = row["metrics"][metric]
            return (
                f"{cell(item['paper'])} → {cell(item['reproduction'])} "
                f"({cell(item['delta'])})"
            )

        for row in diagnostics:
            lines.append(
                f"| {row['model']} | {row['sequence_length']}일 | {row.get('backend') or '–'} | "
                f"{row.get('n_sequences')}/{row.get('n_subjects')} | "
                f"{metric_cell(row, 'sensitivity')} | {metric_cell(row, 'specificity')} | "
                f"{metric_cell(row, 'roc_auc')} | {metric_cell(row, 'accuracy')} | "
                f"{metric_cell(row, 'precision')} | {metric_cell(row, 'f1')} | "
                f"{'예' if row['all_table5_metrics_within_abs_0_03'] else '아니오'} |"
            )

        for row in diagnostics:
            if row.get("paper_precision_at_100") is None:
                continue
            lines += [
                "",
                f"- **LSTM {row['sequence_length']}일 P@100**: 논문 "
                f"{cell(row['paper_precision_at_100'])}, 재구현 sequence "
                f"{cell(row.get('sequence_precision_at_100'))}, 현재 평가군의 이론상 최대 "
                f"{cell(row.get('sequence_precision_at_100_max_possible'))}. "
                f"논문값 달성 가능: "
                f"{'예' if row.get('paper_precision_at_100_attainable_on_sequence_set') else '아니오'}.",
                f"- 관측 sequence prevalence {cell(row.get('observed_sequence_prevalence'))}, "
                f"논문 Table 5의 sensitivity/specificity/precision이 암시하는 prevalence "
                f"{cell(row.get('paper_implied_prevalence'))}.",
                f"- 같은 평가군의 subject P@100 "
                f"{cell(row.get('subject_precision_at_100'))}, 이론상 최대 "
                f"{cell(row.get('subject_precision_at_100_max_possible'))}. "
                "논문의 P@100 단위가 보고되지 않아 sequence 값과 섞지 않는다.",
                f"- early stopping 실제 적용: {'예' if row.get('early_stopping_applied') else '아니오'}.",
                "- method fidelity: " + str(row.get("method_fidelity") or {}) + ".",
            ]
    else:
        lines.append("비교 가능한 paper-temporal 결과가 없다.")

    lines += ["", "## 차이값과 그 해석", ""]
    for row in comparison["deltas"]["per_sequence_length"]:
        lines.append(f"- **{row['sequence_length']}일**: " + ", ".join(
            f"{k} = {cell(v)}" for k, v in row.items() if k != "sequence_length"
        ))
    for row in comparison["deltas"].get("nested_vs_nonnested", []):
        if row.get("nested_minus_nonnested") is not None:
            lines.append(
                f"- **nested(선택) vs non-nested {row['sequence_length']}일**: "
                f"{cell(row['nested_selected'])} − {cell(row['nonnested_fixed_subject'])} = "
                f"{cell(row['nested_minus_nonnested'])}"
            )
    lines += [""]
    for key, text in comparison["deltas"]["interpretation"].items():
        lines.append(f"- `{key}`: {text}")
    return "\n".join(lines) + "\n"
