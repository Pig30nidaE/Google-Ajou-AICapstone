"""실험 A·B·C의 결과를 하나의 비교표로 묶는다 (사용자 지시 12절).

실험마다 **주 평가단위가 다르다**는 점이 이 모듈의 핵심 제약이다.

* 실험 A는 논문과 동일하게 **기록(일별 행) 단위**로 평가한다 — 논문 표 6과 직접 비교하기 위해서.
* 실험 B·C는 **피험자 단위**가 주 평가다.

따라서 교차 실험 비교표는 **피험자 단위로 통일**해서 만들고, 논문 대조표는 기록 단위로
따로 만든다. 둘을 한 표에 섞으면 "누수 통제로 성능이 떨어졌다"는 결론이
평가단위 차이 때문인지 누수 통제 때문인지 구분할 수 없게 된다.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..data.paper_reference import NOT_REPORTED
from ..evaluation.tables import (
    DEM_SUBJECT_CAVEAT,
    MAIN_TABLE_MODELS,
    build_main_comparison_table,
    build_rank_change_table,
    build_validation_comparison_table,
    compute_deltas,
)
from ..utils.io import save_json, save_table

log = logging.getLogger(__name__)

__all__ = ["assemble_comparison", "CrossExperimentResults"]


class CrossExperimentResults:
    """세 실험의 결과를 모으는 컨테이너.

    각 실험은 선택적이다 — 일부만 실행했어도 있는 것만으로 표를 만든다.
    """

    def __init__(self) -> None:
        #: {(experiment, model, augmentation): metrics}  — 피험자 단위
        self.subject_level: dict[tuple[str, str, str], dict] = {}
        #: {(experiment, model, augmentation): metrics}  — 기록 단위
        self.record_level: dict[tuple[str, str, str], dict] = {}
        #: 실험 C처럼 모델이 fold마다 선택되는 경우의 전체 결과
        self.pooled: dict[str, dict] = {}
        #: 실험 C가 fold마다 고른 파이프라인
        self.selected_pipelines: list[dict] = []
        self.notes: list[str] = []

    # ------------------------------------------------------------------
    def add_experiment_a(self, result: dict) -> None:
        for (model, aug), metrics in result.get("results", {}).items():
            self.record_level[("A", model, aug)] = metrics
        for (model, aug), metrics in result.get("results_subject", {}).items():
            self.subject_level[("A", model, aug)] = metrics
        self.notes.append(
            "실험 A5는 논문의 기록 단위 평가 틀을 따르되, 미보고 VAE 입력 척도와 "
            "손실 축척에는 결과 감사 후 교정한 가정을 사용했다. 원저자 설정의 정확 재현이 아니다. "
            "교차 비교를 위해 피험자 단위 집계도 함께 산출했다."
        )

    def add_experiment_b(self, result: dict) -> None:
        for (model, aug), metrics in result.get("results", {}).items():
            self.subject_level[("B", model, aug)] = metrics

    def add_experiment_c(self, result: dict, *, model: str | None = None,
                         augmentation: str | None = None) -> None:
        """실험 C 결과를 넣는다.

        Args:
            model: per-cell nested 실행이면 고정했던 분류기 이름.
                ``None``이면 분류기 자체를 inner CV가 골랐다는 뜻이므로
                모델별 칸이 아니라 ``pooled``에 들어간다.
            augmentation: per-cell nested 실행이면 고정했던 증강 조건.
        """
        metrics = result.get("results", {})
        self.selected_pipelines.extend(result.get("selected", []))
        if model is not None and augmentation is not None:
            self.subject_level[("C", model, augmentation)] = metrics
        else:
            self.pooled["C_free_selection"] = metrics
            self.notes.append(
                "실험 C는 분류기와 증강 여부까지 inner CV가 선택했다. "
                "따라서 모델별·증강별 칸을 채우지 않고 전체 결과 1건으로 보고한다. "
                "모델×증강 칸을 채우려면 run.nested_per_cell: true로 실행하라."
            )


def assemble_comparison(
    results: CrossExperimentResults,
    *,
    out_root: str,
    metric: str = "macro_f1",
) -> dict:
    """비교표 일체를 만들고 저장한다.

    Returns:
        생성한 표의 경로와 요약.
    """
    from pathlib import Path

    out = Path(out_root) / "COMPARISON"
    saved: dict[str, str] = {}

    # ---- 1) 주 비교표 (피험자 단위로 통일)
    main = build_main_comparison_table(results.subject_level, metric=metric)
    main.attrs["paper_note"] = (
        "'원 논문 보고값'은 논문이 보고한 macro F1이며 **기록 단위**다 "
        "(Wide & Deep은 표 6, 나머지 3모델은 그림 3). "
        "'증거 기반 교정 A5'·'누수 통제'·'Nested' 열은 모두 **피험자 단위**로 통일했다 — "
        "평가단위를 섞으면 성능 변화의 원인을 구분할 수 없기 때문이다. "
        "논문과 직접 비교 가능한 기록 단위 수치는 paper_comparison_record_level.csv에 있다. "
        "A5의 scaled VAE와 KL mean은 논문 미보고 사항에 대한 교정 가정이며 원저자 설정이 아니다. "
        "'not_reported'는 논문이 그 조합을 보고하지 않았다는 뜻이다."
    )
    saved["main_comparison"] = str(save_table(main, out / "main_comparison_subject_level.csv"))

    # ---- 2) 논문 직접 대조표 (기록 단위 — 실험 A만)
    if results.record_level:
        rows = []
        for (exp, model, aug), m in sorted(results.record_level.items()):
            label = dict(MAIN_TABLE_MODELS).get(model, model)
            rows.append(
                {
                    "모델": label,
                    "증강": "없음" if aug == "none" else "VAE",
                    "원 논문 보고값(기록 단위)": _paper_macro_f1(model, aug),
                    "교정 A5(기록 단위)": round(float(m.get(metric, float("nan"))), 4),
                    "Dem F1": round(float(m.get("dem_f1", float("nan"))), 4),
                    "Dem recall": round(float(m.get("dem_recall", float("nan"))), 4),
                    "평가 행 수": m.get("n"),
                    "n_dem_subjects": m.get("n_dem_subjects_eval"),
                }
            )
        rec = pd.DataFrame(rows)
        rec.attrs["caveat"] = DEM_SUBJECT_CAVEAT
        rec.attrs["paper_note"] = (
            "논문 표 6의 증강 전/후는 서로 다른 평가셋(N=1097 vs 1095)에서 측정되었다 "
            "(report_inconsistencies.md I-4). A5는 동일 split·동일 seed에서 비교하며, "
            "scaled VAE와 KL mean은 논문 미보고 사항에 대한 교정 가정이다."
        )
        saved["paper_comparison"] = str(
            save_table(rec, out / "paper_comparison_record_level.csv")
        )

    # ---- 3) 검증방식 비교표
    validation: dict[str, dict] = {}
    for exp, name in (("A", "증거 기반 교정 A5 (행 단위 분할·전처리 누수)"),
                      ("B", "누수 통제 non-nested (피험자 독립)")):
        best = _best_cell(results.subject_level, exp, metric)
        if best:
            validation[name] = best
    if "C_free_selection" in results.pooled:
        validation["Nested Group CV (파이프라인 선택 포함)"] = results.pooled["C_free_selection"]
    if validation:
        val = build_validation_comparison_table(validation)
        val.attrs["paper_note"] = (
            "실험 A·B는 각 실험에서 macro F1이 가장 높았던 모델×증강 조합이다. "
            "실험 C는 fold마다 파이프라인이 달라지므로 전체 out-of-fold 결과 1건이다."
        )
        saved["validation_comparison"] = str(
            save_table(val, out / "validation_comparison.csv")
        )

    # ---- 4) delta / 순위 변화
    deltas = compute_deltas(results.subject_level)
    if len(deltas):
        deltas.attrs["caveat"] = (
            DEM_SUBJECT_CAVEAT
            + " A→B delta는 행 분할→피험자 분할, all-data→train-only fit, "
              "평가 이상치 보존 등 검증 프로토콜 변경의 결합 차이다. "
              "단일 누수 요인의 인과효과로 해석할 수 없다."
        )
        saved["deltas"] = str(save_table(deltas, out / "deltas.csv"))
    ranks = build_rank_change_table(results.subject_level, metric=metric)
    if len(ranks):
        saved["rank_changes"] = str(save_table(ranks, out / "rank_changes.csv"))

    # ---- 5) 실험 C가 고른 파이프라인
    if results.selected_pipelines:
        sel = pd.DataFrame(results.selected_pipelines)
        saved["selected_pipelines"] = str(save_table(sel, out / "selected_pipelines.csv"))

    summary = {
        "metric": metric,
        "n_cells_subject_level": len(results.subject_level),
        "n_cells_record_level": len(results.record_level),
        "experiments_present": sorted({k[0] for k in results.subject_level} |
                                      set(results.pooled)),
        "notes": results.notes,
        "caveat": DEM_SUBJECT_CAVEAT,
        "saved_tables": saved,
    }
    save_json(summary, out / "comparison_summary.json")
    log.info("비교표 -> %s", out)
    return summary


def _paper_macro_f1(model: str, aug: str):
    from ..evaluation.tables import paper_value

    try:
        return paper_value(model, aug)
    except KeyError:
        return NOT_REPORTED


def _best_cell(cells: dict, experiment: str, metric: str) -> dict | None:
    """한 실험에서 metric이 가장 높은 모델×증강 결과."""
    candidates = [
        (m.get(metric, float("-inf")), m)
        for (exp, _model, _aug), m in cells.items()
        if exp == experiment
    ]
    candidates = [(s, m) for s, m in candidates if s == s and s != float("-inf")]
    if not candidates:
        return None
    return max(candidates, key=lambda t: t[0])[1]
