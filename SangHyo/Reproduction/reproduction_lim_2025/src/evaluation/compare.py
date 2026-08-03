"""The cross-experiment comparison table (task brief §10).

Reads the ``FINAL_REPORT.json`` of each experiment and produces the paper-vs-us
table, the deltas, the rank changes and the fold-to-fold spread.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

#: Table 13/14 of the thesis = Table 10/11 of the journal article.
#: The concluding tables (17 / 14) round these to two decimals; we use the
#: higher-precision source.
PAPER_REPORTED = {
    "random_forest": {"accuracy": 0.6970, "roc_auc": 0.7225, "f1": 0.50, "sensitivity": 0.7143},
    "xgboost": {"accuracy": 0.6970, "roc_auc": 0.6374, "f1": 0.50, "sensitivity": 0.7143},
    "lstm": {"accuracy": 0.818182, "roc_auc": 0.527473, "f1": 0.25, "sensitivity": 0.1428},
    "bilstm": {"accuracy": 0.818182, "roc_auc": 0.390110, "f1": 0.25, "sensitivity": 0.1428},
    "cnn1d": {"accuracy": 0.787879, "roc_auc": 0.810440, "f1": 0.63, "sensitivity": 0.8571},
}

#: Confusion matrices recovered from the reported metrics (reproduction_spec.md §2).
PAPER_RECONSTRUCTED_CONFUSION = {
    "random_forest": {"tp": 5, "fn": 2, "fp": 8, "tn": 18},
    "xgboost": {"tp": 5, "fn": 2, "fp": 8, "tn": 18},
    "lstm": {"tp": 1, "fn": 6, "fp": 0, "tn": 26},
    "bilstm": {"tp": 1, "fn": 6, "fp": 0, "tn": 26},
    "cnn1d": {"tp": 6, "fn": 1, "fp": 6, "tn": 20},
}

PAPER_EVAL_SET = {"n": 33, "n_positive": 7, "n_negative": 26}

MODEL_ORDER = ("random_forest", "xgboost", "lstm", "bilstm", "cnn1d")
MODEL_LABELS = {
    "random_forest": "RF",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
    "bilstm": "Bi-LSTM",
    "cnn1d": "1D-CNN",
}

EXPERIMENT_COLUMNS = (
    ("paper_reported_reconstruction", "원 방법 재구현"),
    ("leakage_controlled_non_nested", "누수 통제 non-nested"),
    ("nested_subject_independent", "Nested Group CV"),
)

# Operational criteria for the paper's *central qualitative conclusion*, not an
# assertion that exact numeric reproduction is possible.  With only seven
# positives, allowing one fewer detected case than the reported 6/7 is already a
# 14.3 percentage-point tolerance.  Architecture, seed and threshold are absent
# from both papers, so exact reproduction remains unassessable regardless.
CORE_CNN_CRITERIA = {
    "minimum_roc_auc": 0.75,
    "minimum_f1": 0.55,
    "minimum_sensitivity": 5 / 7,
}


def verify_paper_arithmetic() -> dict[str, Any]:
    """Re-derive the reported metrics from the recovered confusion matrices.

    This is the evidence behind ``reproduction_spec.md`` §2 and it is recomputed on
    every run so the claim can never silently rot.
    """
    results: dict[str, Any] = {}
    all_ok = True
    for model, cm in PAPER_RECONSTRUCTED_CONFUSION.items():
        tp, fn, fp, tn = cm["tp"], cm["fn"], cm["fp"], cm["tn"]
        n = tp + fn + fp + tn
        accuracy = (tp + tn) / n
        sensitivity = tp / (tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
        reported = PAPER_REPORTED[model]
        checks = {
            "n_equals_33": n == PAPER_EVAL_SET["n"],
            "positives_equal_7": (tp + fn) == PAPER_EVAL_SET["n_positive"],
            "accuracy_matches": abs(accuracy - reported["accuracy"]) < 5e-4,
            "sensitivity_matches": abs(sensitivity - reported["sensitivity"]) < 5e-4,
            "f1_matches": abs(f1 - reported["f1"]) < 5e-3,
        }
        ok = all(checks.values())
        all_ok = all_ok and ok
        results[model] = {
            "confusion": cm,
            "derived": {
                "accuracy": round(accuracy, 6),
                "sensitivity": round(sensitivity, 6),
                "f1": round(f1, 6),
            },
            "reported": reported,
            "checks": checks,
            "consistent": ok,
        }
    return {
        "all_models_consistent": all_ok,
        "evaluation_set": PAPER_EVAL_SET,
        "per_model": results,
        "conclusion": (
            "Every reported Accuracy is k/33 and every Recall is k/7, and the implied "
            "confusion matrices reproduce the reported F1 exactly. The evaluation unit "
            "is the subject and the test set is the 33-subject AI-Hub Validation split."
        ) if all_ok else "Arithmetic reconstruction FAILED -- re-check the paper values.",
    }


def _headline(report: Mapping[str, Any], model: str) -> dict[str, Any] | None:
    """Pull one model's primary subject-level metrics out of a FINAL_REPORT."""
    models = report.get("models", {})
    entry = models.get(model)
    if not entry:
        return None
    return entry.get("subject_level") or entry.get("primary")


def build_comparison(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Assemble the §10 table from ``{experiment_name: FINAL_REPORT dict}``."""
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        row: dict[str, Any] = {
            "model": model,
            "label": MODEL_LABELS[model],
            "paper": dict(PAPER_REPORTED[model]),
        }
        for experiment, _ in EXPERIMENT_COLUMNS:
            report = reports.get(experiment)
            row[experiment] = _headline(report, model) if report else None
        rows.append(row)

    deltas = _deltas(rows)
    return {
        "rows": rows,
        "paper_evaluation_set": PAPER_EVAL_SET,
        "paper_arithmetic_check": verify_paper_arithmetic(),
        "deltas": deltas,
        "rankings": _rankings(rows),
        "reproduction_assessment": _assess_reconstruction(
            rows, reports.get("paper_reported_reconstruction")
        ),
        "interpretation_notes": [
            "논문 열은 33명 · 양성 7명의 단일 고정 분할에서 나온 값이다.",
            "원 방법 재구현 열은 exact reproduction이 아니라 reported-method "
            "reconstruction이다. 딥러닝 구조와 RF 하이퍼파라미터가 미보고이기 때문이다.",
            "논문이 다섯 모델 중 최고인 1D-CNN을 사후 채택했으므로, 논문 열의 1D-CNN "
            "AUC에는 모델 선택 편향이 포함되어 있다.",
            "Nested 열만이 모델 선택과 최종 평가가 분리된 추정치다.",
        ],
    }


def _assess_reconstruction(
    rows: list[dict[str, Any]],
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Judge whether the paper's headline 1D-CNN finding was reconstructed.

    Accuracy alone is deliberately absent: an all-negative classifier obtains
    26/33 = 0.787879 here, exactly the paper's reported CNN accuracy.
    """
    reconstruction = {
        row["model"]: row.get("paper_reported_reconstruction") for row in rows
    }
    available = {
        model: block for model, block in reconstruction.items() if block is not None
    }
    cnn = available.get("cnn1d")
    if cnn is None:
        return {
            "status": "not_assessed",
            "exact_reproduction_possible_from_papers": False,
            "reason": "paper_reported_reconstruction의 1D-CNN 결과가 없다.",
            "criteria": {},
        }

    auc_scores = {
        model: float(block["roc_auc"])
        for model, block in available.items()
        if block.get("roc_auc") is not None and np.isfinite(float(block["roc_auc"]))
    }
    f1_scores = {
        model: float(block["f1"])
        for model, block in available.items()
        if block.get("f1") is not None and np.isfinite(float(block["f1"]))
    }
    def same_counts(block: Mapping[str, Any]) -> bool:
        try:
            return (
                int(block.get("n")) == PAPER_EVAL_SET["n"]
                and int(block.get("n_positive")) == PAPER_EVAL_SET["n_positive"]
                and int(block.get("n_negative")) == PAPER_EVAL_SET["n_negative"]
            )
        except (TypeError, ValueError):
            return False

    same_cohort_counts = (
        set(available) == set(MODEL_ORDER)
        and all(same_counts(block) for block in available.values())
    )
    report_block = report or {}
    config_block = report_block.get("config") or {}
    split_block = config_block.get("split") or {}
    preprocessing_block = config_block.get("preprocessing") or {}
    split_mode = split_block.get("mode")
    try:
        current_artifact_schema = (
            int(report_block.get("artifact_schema_version", -1)) == 2
        )
    except (TypeError, ValueError):
        current_artifact_schema = False
    complete_auc = set(auc_scores) == set(MODEL_ORDER)
    complete_f1 = set(f1_scores) == set(MODEL_ORDER)
    predicted_rate = cnn.get("positive_rate_predicted")
    criteria = {
        "current_artifact_schema": current_artifact_schema,
        "run_fingerprint_present": bool(report_block.get("run_fingerprint")),
        "all_audits_passed": report_block.get("all_audits_passed") is True,
        "no_degenerate_training_models": not bool(
            report_block.get("degenerate_training_models")
        ),
        "reported_method_reconstruction_class": (
            report_block.get("reproduction_class")
            == "reported-method reconstruction"
        ),
        "all_five_models_present": set(available) == set(MODEL_ORDER),
        "official_partition_used": split_mode == "official_partition",
        "train_only_preprocessing": (
            preprocessing_block.get("scaler_scope") == "train_only"
        ),
        "not_leakage_diagnostic": not bool(
            split_block.get("leakage_diagnostic_only", False)
        ),
        "same_33_subject_7_positive_counts": same_cohort_counts,
        "all_five_models_have_finite_auc": complete_auc,
        "all_five_models_have_finite_f1": complete_f1,
        "cnn_not_single_class_at_fixed_threshold": (
            predicted_rate is not None and 0.0 < float(predicted_rate) < 1.0
        ),
        "cnn_roc_auc_at_least_0_75": (
            cnn.get("roc_auc") is not None
            and float(cnn["roc_auc"]) >= CORE_CNN_CRITERIA["minimum_roc_auc"]
        ),
        "cnn_f1_at_least_0_55": (
            cnn.get("f1") is not None
            and float(cnn["f1"]) >= CORE_CNN_CRITERIA["minimum_f1"]
        ),
        "cnn_detects_at_least_5_of_7_positives": (
            cnn.get("sensitivity") is not None
            and float(cnn["sensitivity"])
            >= CORE_CNN_CRITERIA["minimum_sensitivity"] - 1e-12
        ),
        "cnn_has_highest_auc": bool(
            complete_auc
            and "cnn1d" in auc_scores
            and auc_scores["cnn1d"] >= max(auc_scores.values())
        ),
        "cnn_has_highest_f1": bool(
            complete_f1
            and "cnn1d" in f1_scores
            and f1_scores["cnn1d"] >= max(f1_scores.values())
        ),
    }
    passed = all(criteria.values())
    paper_cnn = PAPER_REPORTED["cnn1d"]
    return {
        "status": (
            "core_conclusion_reproduced" if passed
            else "core_conclusion_not_reproduced"
        ),
        "exact_reproduction_possible_from_papers": False,
        "reproduction_class": "reported-method reconstruction",
        "headline_model": "cnn1d",
        "paper_cnn_metrics": dict(paper_cnn),
        "observed_cnn_metrics": {
            metric: cnn.get(metric)
            for metric in ("accuracy", "roc_auc", "f1", "sensitivity")
        },
        "cnn_metric_deltas": {
            metric: (
                None if cnn.get(metric) is None
                else round(float(cnn[metric]) - float(paper_cnn[metric]), 6)
            )
            for metric in ("accuracy", "roc_auc", "f1", "sensitivity")
        },
        "paper_cnn_confusion_matrix": dict(
            PAPER_RECONSTRUCTED_CONFUSION["cnn1d"]
        ),
        "observed_cnn_confusion_matrix": cnn.get("confusion_matrix"),
        "criteria": criteria,
        "failed_criteria": [name for name, ok in criteria.items() if not ok],
        "note": (
            "두 논문이 딥러닝 구조·학습 설정·임계값·seed를 보고하지 않아 exact "
            "reproduction은 판정할 수 없다. 이 판정은 1D-CNN이 가장 강하고 양성 "
            "대부분을 검출한다는 핵심 결론의 재구성 여부만 평가한다. 0.75/0.55/5명 "
            "기준은 논문 정의가 아니라, 보고값 0.810/0.63/6명과 양성 7명의 작은 "
            "표본을 고려해 코드에 명시한 운영 기준이다. 직접 지표 차이와 혼동행렬을 "
            "우선 확인해야 한다."
        ),
    }


def _deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Absolute differences the brief asks for, per model."""
    out: dict[str, Any] = {}
    for row in rows:
        model = row["model"]
        entry: dict[str, Any] = {}
        paper_auc = row["paper"]["roc_auc"]

        for experiment, _ in EXPERIMENT_COLUMNS:
            block = row.get(experiment)
            if block and block.get("roc_auc") is not None and np.isfinite(block["roc_auc"]):
                entry[f"{experiment}_minus_paper_roc_auc"] = round(
                    float(block["roc_auc"]) - paper_auc, 6
                )

        recon = row.get("paper_reported_reconstruction")
        non_nested = row.get("leakage_controlled_non_nested")
        nested = row.get("nested_subject_independent")

        if recon and non_nested and _has_auc(recon) and _has_auc(non_nested):
            entry["subjectwise_minus_reconstruction_roc_auc"] = round(
                float(non_nested["roc_auc"]) - float(recon["roc_auc"]), 6
            )
        if non_nested and nested and _has_auc(non_nested) and _has_auc(nested):
            optimism = float(non_nested["roc_auc"]) - float(nested["roc_auc"])
            entry["non_nested_minus_nested_roc_auc"] = round(optimism, 6)
            entry["optimism_note"] = (
                "non-nested 값은 성능 주장이 아니라 과대평가 크기 진단이다."
            )
        out[model] = entry
    return out


def _has_auc(block: Mapping[str, Any]) -> bool:
    value = block.get("roc_auc")
    return value is not None and np.isfinite(float(value))


def _rankings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Model ranking under each column, plus whether the ordering moved."""
    rankings: dict[str, Any] = {}
    columns = [("paper", lambda r: r["paper"]["roc_auc"])]
    columns += [
        (experiment, lambda r, e=experiment: (r.get(e) or {}).get("roc_auc"))
        for experiment, _ in EXPERIMENT_COLUMNS
    ]

    for name, getter in columns:
        scored = [
            (row["model"], getter(row))
            for row in rows
            if getter(row) is not None and np.isfinite(float(getter(row)))
        ]
        if not scored:
            rankings[name] = None
            continue
        scored.sort(key=lambda pair: pair[1], reverse=True)
        rankings[name] = [model for model, _ in scored]

    paper_rank = rankings.get("paper") or []
    changes: dict[str, Any] = {}
    for experiment, _ in EXPERIMENT_COLUMNS:
        other = rankings.get(experiment)
        if not other:
            continue
        shared = [m for m in paper_rank if m in other]
        changes[experiment] = {
            "order": other,
            "top_model_changed": bool(shared and other[0] != shared[0]),
            "position_shift": {
                m: shared.index(m) - other.index(m) for m in shared
            },
        }
    rankings["changes_vs_paper"] = changes
    return rankings


def render_comparison_markdown(comparison: Mapping[str, Any]) -> str:
    """Render the §10 table as markdown."""
    lines = [
        "# 결과 비교표",
        "",
        "| 모델 | 원 논문 보고값 (Acc / AUC / F1) | 원 방법 재구현 | 누수 통제 non-nested | Nested Group CV |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in comparison["rows"]:
        paper = row["paper"]
        cells = [
            f"{paper['accuracy']:.3f} / {paper['roc_auc']:.3f} / {paper['f1']:.2f}"
        ]
        for experiment, _ in EXPERIMENT_COLUMNS:
            block = row.get(experiment)
            if not block:
                cells.append("—")
                continue
            auc = block.get("roc_auc")
            acc = block.get("accuracy")
            f1 = block.get("f1")
            cells.append(
                "—" if auc is None else
                f"{_fmt(acc)} / {_fmt(auc)} / {_fmt(f1, 2)}"
            )
        lines.append(f"| {row['label']} | " + " | ".join(cells) + " |")

    check = comparison["paper_arithmetic_check"]
    lines += [
        "",
        "형식: Accuracy / ROC-AUC / F1. 모든 값은 **피험자 단위**다.",
        "",
        "## 논문 평가집단 산술 검증",
        "",
        f"- 다섯 모델 모두 정합: **{check['all_models_consistent']}**",
        f"- 복원된 평가집단: {check['evaluation_set']}",
        f"- {check['conclusion']}",
        "",
        "## 해석 주의",
        "",
    ]
    lines += [f"- {note}" for note in comparison["interpretation_notes"]]
    assessment = comparison["reproduction_assessment"]
    lines += [
        "",
        "## 재현 판정",
        "",
        f"- 상태: **{assessment['status']}**",
        f"- Exact reproduction 가능: **{assessment['exact_reproduction_possible_from_papers']}**",
    ]
    if assessment.get("criteria"):
        lines.append("- 핵심 1D-CNN 기준:")
        for name, passed in assessment["criteria"].items():
            lines.append(f"  - {'PASS' if passed else 'FAIL'}: `{name}`")
    if assessment.get("note"):
        lines.append(f"- {assessment['note']}")

    reconstruction_present = any(
        row.get("paper_reported_reconstruction") for row in comparison["rows"]
    )
    if reconstruction_present:
        lines += [
            "",
            "## 논문 대비 원 방법 재구현 차이",
            "",
            "| 모델 | Δ Accuracy | Δ ROC-AUC | Δ F1 | Δ Recall |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in comparison["rows"]:
            block = row.get("paper_reported_reconstruction")
            if not block:
                continue
            paper = row["paper"]
            lines.append(
                f"| {row['label']} | "
                f"{_signed_delta(block.get('accuracy'), paper['accuracy'])} | "
                f"{_signed_delta(block.get('roc_auc'), paper['roc_auc'])} | "
                f"{_signed_delta(block.get('f1'), paper['f1'])} | "
                f"{_signed_delta(block.get('sensitivity'), paper['sensitivity'])} |"
            )
    return "\n".join(lines) + "\n"


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(number) else f"{number:.{digits}f}"


def _signed_delta(value: Any, reference: Any) -> str:
    if value is None or reference is None:
        return "—"
    try:
        delta = float(value) - float(reference)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(delta) else f"{delta:+.3f}"


def load_reports(paths: Mapping[str, str | Path]) -> dict[str, dict[str, Any]]:
    from ..utils.io import read_json

    reports: dict[str, dict[str, Any]] = {}
    for experiment, path in paths.items():
        candidate = Path(path)
        if candidate.is_file():
            reports[experiment] = read_json(candidate)
    return reports
