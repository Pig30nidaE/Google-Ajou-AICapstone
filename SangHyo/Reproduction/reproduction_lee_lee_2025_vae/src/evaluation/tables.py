"""결과표 생성 (사용자 지시 12절).

모든 표에 ``n_dem_subjects`` 열을 필수로 넣는다 — 12명 제한을 결과에서 지울 수 없게 한다
(synthetic_data_risk.md §5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.paper_reference import (
    FIGURE3_F1,
    NOT_REPORTED,
    TABLE6_WIDE_DEEP,
)

__all__ = [
    "MAIN_TABLE_MODELS",
    "build_main_comparison_table",
    "build_validation_comparison_table",
    "compute_deltas",
    "build_rank_change_table",
    "fold_variability",
    "DEM_SUBJECT_CAVEAT",
]

DEM_SUBJECT_CAVEAT = (
    "본 실험의 Dem 클래스는 독립 피험자 12명에서 유래한다. "
    "합성 Dem 행은 이 12명(각 fold에서는 8명)의 기록 분포에서 생성된 것이며 "
    "새로운 피험자를 의미하지 않는다."
)

MAIN_TABLE_MODELS: tuple[tuple[str, str], ...] = (
    ("xgboost", "XGBoost"),
    ("dnn", "DNN"),
    ("tabnet", "TabNet"),
    ("wide_deep", "Wide & Deep"),
)

_PAPER_KEY = {"xgboost": "XGBoost", "dnn": "DNN", "tabnet": "TabNet", "wide_deep": "WideDeep"}


def paper_value(model: str, augmentation: str, metric: str = "macro_f1") -> object:
    """논문 보고값을 꺼낸다. 보고되지 않은 조합은 ``NOT_REPORTED``.

    논문은 Wide & Deep만 증강 전/후를 보고했고 (표 6), 그림 3의 나머지 3개 모델은
    증강 **후** 조건으로 판단된다 (report_inconsistencies.md I-12).
    """
    key = _PAPER_KEY[model]
    if model == "wide_deep":
        col = "without_augmentation" if augmentation == "none" else "with_vae"
        return TABLE6_WIDE_DEEP[col]["macro_avg"]["f1" if metric == "macro_f1" else metric]
    if augmentation == "none":
        return NOT_REPORTED           # 논문이 보고하지 않은 칸
    return FIGURE3_F1[key]["macro_avg"]


def build_main_comparison_table(
    results: dict[tuple[str, str, str], dict],
    *,
    metric: str = "macro_f1",
) -> pd.DataFrame:
    """4모델 × 증강유무 × 3실험 비교표.

    Args:
        results: ``{(experiment, model, augmentation): metrics_dict}``.
            ``experiment``는 ``"A"`` / ``"B"`` / ``"C"``.

    Returns:
        사용자 지시 12절의 첫 번째 표 형식.
    """
    rows = []
    for model_key, model_label in MAIN_TABLE_MODELS:
        for aug_key, aug_label in (("none", "없음"), ("vae", "VAE")):
            row = {
                "모델": model_label,
                "증강": aug_label,
                "원 논문 보고값": paper_value(model_key, aug_key, metric),
            }
            n_dem = None
            for exp, col in (
                ("A", "원 방법 재구현"),
                ("B", "누수 통제 non-nested"),
                ("C", "Nested Group CV"),
            ):
                m = results.get((exp, model_key, aug_key))
                row[col] = round(float(m[metric]), 4) if m and metric in m else None
                if m is not None and n_dem is None:
                    n_dem = m.get("n_Dem")
            # Dem 12명 제한을 결과표에서 지울 수 없게 항상 마지막 열로 붙인다.
            row["n_dem_subjects"] = n_dem
            rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["caveat"] = DEM_SUBJECT_CAVEAT
    df.attrs["metric"] = metric
    df.attrs["paper_note"] = (
        "'원 논문 보고값'은 macro F1이다. Wide & Deep은 표 6, 나머지 3모델은 그림 3 "
        "(증강 후 조건으로 판단). 'not_reported'는 논문이 해당 조합을 보고하지 않았다는 뜻이다."
    )
    return df


def build_validation_comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """검증방식별 요약표 (사용자 지시 12절의 두 번째 표).

    Args:
        results: ``{검증방식 이름: metrics_dict}``.
    """
    rows = []
    for name, m in results.items():
        rows.append(
            {
                "검증방식": name,
                "Macro-F1": round(m.get("macro_f1", float("nan")), 4),
                "Balanced Acc": round(m.get("balanced_accuracy", float("nan")), 4),
                "Macro ROC-AUC": round(m.get("macro_roc_auc_ovr", float("nan")), 4),
                "Dem Recall": round(m.get("dem_recall", float("nan")), 4),
                "Dem F1": round(m.get("dem_f1", float("nan")), 4),
                "평가단위": m.get("unit", "?"),
                "n": m.get("n"),
                "n_dem_subjects": m.get("n_Dem"),
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["caveat"] = DEM_SUBJECT_CAVEAT
    return df


def compute_deltas(
    results: dict[tuple[str, str, str], dict],
    *,
    metrics: tuple[str, ...] = ("macro_f1", "balanced_accuracy", "dem_recall", "dem_f1"),
) -> pd.DataFrame:
    """세 종류의 delta를 계산한다.

    * ``vae_effect``       Δ(VAE − none), 실험별
    * ``leakage_control``  Δ(B − A), 증강조건별
    * ``nesting``          Δ(C − B), 증강조건별
    """
    rows = []
    models = [m for m, _ in MAIN_TABLE_MODELS]

    for exp in ("A", "B", "C"):
        for model in models:
            a = results.get((exp, model, "none"))
            b = results.get((exp, model, "vae"))
            if a and b:
                for met in metrics:
                    rows.append(
                        {
                            "delta_type": "vae_effect",
                            "scope": f"experiment_{exp}",
                            "model": model,
                            "metric": met,
                            "baseline": round(a.get(met, float("nan")), 4),
                            "compared": round(b.get(met, float("nan")), 4),
                            "delta": round(b.get(met, float("nan")) - a.get(met, float("nan")), 4),
                        }
                    )

    for label, (lo, hi) in (("leakage_control", ("A", "B")), ("nesting", ("B", "C"))):
        for model in models:
            for aug in ("none", "vae"):
                a = results.get((lo, model, aug))
                b = results.get((hi, model, aug))
                if a and b:
                    for met in metrics:
                        rows.append(
                            {
                                "delta_type": label,
                                "scope": f"{lo}->{hi}",
                                "model": model,
                                "metric": met,
                                "baseline": round(a.get(met, float("nan")), 4),
                                "compared": round(b.get(met, float("nan")), 4),
                                "delta": round(
                                    b.get(met, float("nan")) - a.get(met, float("nan")), 4
                                ),
                            }
                        )
    return pd.DataFrame(rows)


def build_rank_change_table(
    results: dict[tuple[str, str, str], dict], *, metric: str = "macro_f1"
) -> pd.DataFrame:
    """실험별 모델 순위와 그 변화."""
    rows = []
    for exp in ("A", "B", "C"):
        for aug in ("none", "vae"):
            scored = [
                (model, results[(exp, model, aug)].get(metric))
                for model, _ in MAIN_TABLE_MODELS
                if (exp, model, aug) in results
            ]
            scored = [(m, v) for m, v in scored if v is not None and np.isfinite(v)]
            if not scored:
                continue
            for rank, (model, val) in enumerate(sorted(scored, key=lambda t: -t[1]), start=1):
                rows.append(
                    {
                        "experiment": exp,
                        "augmentation": aug,
                        "model": model,
                        metric: round(float(val), 4),
                        "rank": rank,
                    }
                )
    return pd.DataFrame(rows)


def fold_variability(fold_metrics: list[dict], *, metrics: tuple[str, ...] = ("macro_f1", "dem_recall")) -> pd.DataFrame:
    """outer fold 간 변동성 (평균·표준편차·최소·최대)."""
    rows = []
    for met in metrics:
        vals = np.array(
            [f.get(met, float("nan")) for f in fold_metrics if np.isfinite(f.get(met, float("nan")))]
        )
        if len(vals) == 0:
            continue
        rows.append(
            {
                "metric": met,
                "n_folds": len(vals),
                "mean": round(float(vals.mean()), 4),
                "std": round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 4),
                "min": round(float(vals.min()), 4),
                "max": round(float(vals.max()), 4),
                "range": round(float(vals.max() - vals.min()), 4),
            }
        )
    return pd.DataFrame(rows)
