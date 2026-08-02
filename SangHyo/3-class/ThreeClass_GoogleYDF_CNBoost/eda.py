"""Training-only, subject-level EDA for CN-focused wearable patterns.

This script never opens Validation and never resolves the MMSE source.  It
writes only aggregate statistics and figures; subject-level rows and identifiers
are not persisted.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feature_engineering import CLASS_NAMES, build_subject_dataset, feature_family


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def finite(values: pd.Series | np.ndarray) -> np.ndarray:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def cliffs_delta(group: np.ndarray, reference: np.ndarray) -> float:
    left = finite(group)
    right = finite(reference)
    if not len(left) or not len(right):
        return float("nan")
    differences = left[:, None] - right[None, :]
    return float((np.sum(differences > 0) - np.sum(differences < 0)) / differences.size)


def robust_smd(group: np.ndarray, reference: np.ndarray) -> float:
    left = finite(group)
    right = finite(reference)
    if not len(left) or not len(right):
        return float("nan")
    pooled = np.median(
        [
            np.subtract(*np.quantile(left, [0.75, 0.25])),
            np.subtract(*np.quantile(right, [0.75, 0.25])),
        ]
    )
    if not np.isfinite(pooled) or abs(pooled) < 1e-12:
        pooled = np.nanstd(np.concatenate([left, right]))
    if not np.isfinite(pooled) or abs(pooled) < 1e-12:
        return 0.0
    return float((np.nanmedian(left) - np.nanmedian(right)) / pooled)


def direction_free_auc(values: np.ndarray, binary_target: np.ndarray) -> float:
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan)
    filled = series.fillna(series.median() if series.notna().any() else 0.0).to_numpy()
    if np.unique(filled).size < 2 or np.unique(binary_target).size < 2:
        return 0.5
    target = np.asarray(binary_target, dtype=np.int64)
    positive = target == 1
    negative = ~positive
    ranks = pd.Series(filled).rank(method="average").to_numpy(dtype=float)
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    auc = float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )
    return float(max(auc, 1.0 - auc))


def comparison_rows(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    comparisons = (
        ("MCI_vs_CN", y == 1, y == 0),
        ("DEM_vs_CN", y == 2, y == 0),
        ("IMPAIRED_vs_CN", y != 0, y == 0),
    )
    rows: list[dict[str, Any]] = []
    for feature in X.columns:
        values = pd.to_numeric(X[feature], errors="coerce").to_numpy(dtype=float)
        for comparison, group_mask, reference_mask in comparisons:
            group = values[group_mask]
            reference = values[reference_mask]
            binary = np.zeros(len(y), dtype=np.int64)
            binary[group_mask] = 1
            eligible = group_mask | reference_mask
            rows.append(
                {
                    "feature": feature,
                    "family": feature_family(feature),
                    "comparison": comparison,
                    "group_n": int(group_mask.sum()),
                    "reference_n": int(reference_mask.sum()),
                    "group_median": float(np.nanmedian(group)),
                    "reference_median": float(np.nanmedian(reference)),
                    "cliffs_delta_group_minus_cn": cliffs_delta(group, reference),
                    "robust_smd_group_minus_cn": robust_smd(group, reference),
                    "direction_free_univariate_auc": direction_free_auc(
                        values[eligible], binary[eligible]
                    ),
                    "missing_fraction": float(np.mean(~np.isfinite(values))),
                }
            )
    frame = pd.DataFrame(rows)
    frame["abs_cliffs_delta"] = frame["cliffs_delta_group_minus_cn"].abs()
    return frame


def bootstrap_top_cn(
    X: pd.DataFrame,
    y: np.ndarray,
    effects: pd.DataFrame,
    *,
    rounds: int,
    seed: int,
    top_n: int = 60,
) -> pd.DataFrame:
    top = (
        effects.loc[effects["comparison"] == "IMPAIRED_vs_CN"]
        .sort_values(
            ["direction_free_univariate_auc", "abs_cliffs_delta"],
            ascending=False,
        )
        .head(top_n)
        .copy()
    )
    rng = np.random.default_rng(seed)
    cn_indices = np.flatnonzero(y == 0)
    impaired_indices = np.flatnonzero(y != 0)
    intervals: list[dict[str, Any]] = []
    for row in top.itertuples(index=False):
        values = pd.to_numeric(X[row.feature], errors="coerce").to_numpy(dtype=float)
        deltas = []
        for _ in range(rounds):
            cn_sample = rng.choice(cn_indices, size=len(cn_indices), replace=True)
            impaired_sample = rng.choice(
                impaired_indices, size=len(impaired_indices), replace=True
            )
            deltas.append(cliffs_delta(values[impaired_sample], values[cn_sample]))
        delta_values = np.asarray(deltas, dtype=float)
        finite_delta = delta_values[np.isfinite(delta_values)]
        point = float(row.cliffs_delta_group_minus_cn)
        intervals.append(
            {
                "feature": row.feature,
                "bootstrap_delta_ci_low": float(np.quantile(finite_delta, 0.025)),
                "bootstrap_delta_ci_high": float(np.quantile(finite_delta, 0.975)),
                "bootstrap_sign_consistency": float(
                    np.mean(np.sign(finite_delta) == np.sign(point))
                ),
            }
        )
    return top.merge(pd.DataFrame(intervals), on="feature", how="left")


def family_summary(effects: pd.DataFrame) -> pd.DataFrame:
    impaired = effects.loc[effects["comparison"] == "IMPAIRED_vs_CN"].copy()
    return (
        impaired.groupby("family", as_index=False)
        .agg(
            feature_count=("feature", "nunique"),
            median_abs_delta=("abs_cliffs_delta", "median"),
            max_abs_delta=("abs_cliffs_delta", "max"),
            median_direction_free_auc=("direction_free_univariate_auc", "median"),
            max_direction_free_auc=("direction_free_univariate_auc", "max"),
        )
        .sort_values("max_direction_free_auc", ascending=False)
    )


def plot_class_counts(y: np.ndarray, output: Path) -> None:
    counts = [int(np.sum(y == index)) for index in range(3)]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(CLASS_NAMES, counts, color=["#4C78A8", "#F2CF5B", "#E45756"])
    ax.set_title("Training subjects by class")
    ax.set_ylabel("Subjects")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 1, str(count), ha="center")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_top_effects(top: pd.DataFrame, output: Path) -> None:
    shown = top.head(20).sort_values("cliffs_delta_group_minus_cn")
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#4C78A8" if value < 0 else "#E45756" for value in shown["cliffs_delta_group_minus_cn"]]
    ax.barh(range(len(shown)), shown["cliffs_delta_group_minus_cn"], color=colors)
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels([str(value)[:78] for value in shown["feature"]], fontsize=8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Cliff's delta: impaired minus CN")
    ax.set_title("Largest Training-only CN separation signals")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int) -> list[str]:
    headers = columns
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in frame.head(limit).itertuples(index=False):
        values = []
        record = row._asdict()
        for column in columns:
            value = record[column]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(f"`{value}`" if column == "feature" else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def build_report(
    audit: dict[str, Any],
    effects: pd.DataFrame,
    top_cn: pd.DataFrame,
    families: pd.DataFrame,
) -> str:
    mci = effects.loc[effects["comparison"] == "MCI_vs_CN"].sort_values(
        "abs_cliffs_delta", ascending=False
    )
    dem = effects.loc[effects["comparison"] == "DEM_vs_CN"].sort_values(
        "abs_cliffs_delta", ascending=False
    )
    strong_cn = int((top_cn["abs_cliffs_delta"] >= 0.33).sum())
    lines = [
        "# Google YDF CNBoost — Training-only EDA",
        "",
        "이 보고서는 Training 사람 단위 집계만 사용했습니다. Validation은 열지 않았고 MMSE 파일은 경로조차 찾지 않았습니다.",
        "",
        "## 먼저 볼 결론",
        "",
        f"- Training은 CN {audit['class_counts']['CN']}명, MCI {audit['class_counts']['MCI']}명, DEM {audit['class_counts']['DEM']}명입니다.",
        f"- 모델 후보로 들어가는 웨어러블 요약은 {audit['feature_contract']['feature_count']}개입니다.",
        f"- CN 대 비CN에서 |Cliff's delta| 0.33 이상인 상위 후보는 {strong_cn}개입니다. 단일 변수 하나로 해결되기보다 여러 작은 패턴을 묶어야 한다는 뜻입니다.",
        "- 이전 실험에서 강했던 관측일 수, 첫 날짜, mask, calendar gap은 수집 방식의 차이일 수 있어 주 모델에서 차단했습니다.",
        "- 아래 순위는 이해를 위한 EDA입니다. 실제 학습 변수 선택은 각 CV 학습 fold 안에서 다시 계산됩니다.",
        "",
        "## CN 대 비CN 상위 패턴",
        "",
    ]
    lines.extend(
        markdown_table(
            top_cn,
            [
                "feature",
                "cliffs_delta_group_minus_cn",
                "direction_free_univariate_auc",
                "bootstrap_delta_ci_low",
                "bootstrap_delta_ci_high",
                "bootstrap_sign_consistency",
            ],
            15,
        )
    )
    lines.extend(["", "## MCI 대 CN", ""])
    lines.extend(
        markdown_table(
            mci,
            ["feature", "cliffs_delta_group_minus_cn", "direction_free_univariate_auc"],
            12,
        )
    )
    lines.extend(["", "## DEM 대 CN", ""])
    lines.extend(
        markdown_table(
            dem,
            ["feature", "cliffs_delta_group_minus_cn", "direction_free_univariate_auc"],
            12,
        )
    )
    lines.extend(["", "## 패턴 묶음", ""])
    lines.extend(
        markdown_table(
            families,
            [
                "family",
                "feature_count",
                "median_abs_delta",
                "max_abs_delta",
                "max_direction_free_auc",
            ],
            20,
        )
    )
    lines.extend(
        [
            "",
            "## 학습 설계에 반영한 점",
            "",
            "1. 직접 3-class YDF, CN 대 비CN을 먼저 보는 계층형 YDF, 확률형 YDF Random Forest를 고정 후보로 둡니다.",
            "2. 수면/활동 한쪽에만 변수가 몰리지 않도록 fold 안에서 두 modality의 최소 후보 수를 보장합니다.",
            "3. 중앙값뿐 아니라 IQR/MAD, 상태 전이, 엔트로피, 최근 변화, 수면 시각의 원형 표현을 유지합니다.",
            "4. feature 선택·모델 조정·blend 결정은 모두 nested CV의 학습 부분 안에서만 합니다.",
            "5. Validation은 label-free 예측을 먼저 저장한 뒤 역사적 benchmark로 한 번 평가합니다.",
            "",
            "## 주의할 점",
            "",
            "- DEM은 9명뿐이므로 한두 사람만 달라져도 점수가 크게 바뀝니다.",
            "- MCI와 CN의 웨어러블 차이는 대체로 작습니다. 높은 accuracy만 보고 전부 CN에 가깝게 예측하는 모델을 선택하면 안 됩니다.",
            "- 효과크기는 연관성이지 원인이나 임상적 진단 기준이 아닙니다.",
            "- 이 EDA 결과 자체로 feature를 고정하면 전체 Training 라벨을 미리 본 셈이 되므로 학습 코드는 EDA 순위를 직접 가져다 쓰지 않습니다.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(training_root: str, output_dir: str, bootstrap_rounds: int, seed: int) -> None:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = build_subject_dataset(training_root, require_labels=True)
    if data.y is None:
        raise AssertionError("Training labels are absent")
    effects = comparison_rows(data.X, data.y)
    top_cn = bootstrap_top_cn(
        data.X,
        data.y,
        effects,
        rounds=bootstrap_rounds,
        seed=seed,
    )
    families = family_summary(effects)

    effects.to_csv(output / "feature_effects.csv", index=False)
    top_cn.to_csv(output / "cn_top_features_bootstrap.csv", index=False)
    families.to_csv(output / "feature_family_summary.csv", index=False)
    plot_class_counts(data.y, output / "class_counts.png")
    plot_top_effects(top_cn, output / "cn_top_effects.png")
    write_json(
        output / "eda_audit.json",
        {
            **data.audit,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "bootstrap_rounds": int(bootstrap_rounds),
            "validation_opened": False,
            "subject_level_rows_persisted": False,
            "raw_or_hashed_subject_identifiers_persisted": False,
            "eda_ranking_consumed_by_training_code": False,
        },
    )
    (output / "EDA_REPORT_KO.md").write_text(
        build_report(data.audit, effects, top_cn, families),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-rounds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()
    if args.bootstrap_rounds < 100:
        raise ValueError("Use at least 100 bootstrap rounds for a useful interval")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.training_root,
        arguments.output_dir,
        arguments.bootstrap_rounds,
        arguments.seed,
    )
