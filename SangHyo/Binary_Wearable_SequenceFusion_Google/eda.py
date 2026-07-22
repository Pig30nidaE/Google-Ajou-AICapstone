"""Training-only EDA for the leakage-conscious binary sequence dataset.

All exported tables are aggregate or feature-level artifacts.  Subject
identifiers (raw or hashed) are never written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:  # Package import.
    from .data import SubjectSequenceDataset, build_subject_dataset
except ImportError:  # Direct ``python eda.py`` execution.
    from data import SubjectSequenceDataset, build_subject_dataset


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _quantile(values: Sequence[float], q: float) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.quantile(array, q)) if len(array) else np.nan


def _sequence_length_summary(lengths: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = (("all", np.ones(len(y), dtype=bool)), ("CN", y == 0), ("MCI+DEM", y == 1))
    for name, selected in groups:
        values = lengths[selected]
        rows.append(
            {
                "group": name,
                "n_subjects": int(len(values)),
                "minimum": int(values.min()),
                "p10": _quantile(values, 0.10),
                "q25": _quantile(values, 0.25),
                "median": _quantile(values, 0.50),
                "q75": _quantile(values, 0.75),
                "p90": _quantile(values, 0.90),
                "maximum": int(values.max()),
            }
        )
    return pd.DataFrame(rows)


def _subject_medians(dataset: SubjectSequenceDataset) -> np.ndarray:
    # Pandas median avoids NumPy's all-NaN slice warning while keeping the same
    # per-subject, one-vote-per-feature analysis contract.
    return np.vstack(
        [pd.DataFrame(sequence).median(axis=0, skipna=True).to_numpy(float) for sequence in dataset.sequences]
    )


def _feature_quality(
    daily_values: np.ndarray,
    subject_values: np.ndarray,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(feature_names):
        daily = daily_values[:, index]
        subject = subject_values[:, index]
        finite_daily = daily[np.isfinite(daily)]
        family = feature.split("__", 1)[0]
        if len(finite_daily):
            q25, median, q75 = np.quantile(finite_daily, [0.25, 0.50, 0.75])
            minimum, maximum = float(finite_daily.min()), float(finite_daily.max())
            unique_finite = int(pd.Series(finite_daily).nunique(dropna=True))
            zero_variance = bool(np.nanmax(finite_daily) == np.nanmin(finite_daily))
        else:
            q25 = median = q75 = minimum = maximum = np.nan
            unique_finite = 0
            zero_variance = True
        rows.append(
            {
                "feature": feature,
                "family": family,
                "daily_missing_rate": float(np.mean(~np.isfinite(daily))),
                "subject_missing_rate": float(np.mean(~np.isfinite(subject))),
                "finite_unique_values": unique_finite,
                "zero_variance": zero_variance,
                "minimum": minimum,
                "q25": float(q25),
                "median": float(median),
                "q75": float(q75),
                "maximum": maximum,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["subject_missing_rate", "daily_missing_rate", "feature"],
        ascending=[False, False, True],
    )


def _rank_auc(negative: np.ndarray, positive: np.ndarray) -> float:
    negative = negative[np.isfinite(negative)]
    positive = positive[np.isfinite(positive)]
    if len(negative) == 0 or len(positive) == 0:
        return np.nan
    values = np.r_[negative, positive]
    ranks = pd.Series(values).rank(method="average").to_numpy(float)
    positive_ranks = ranks[len(negative) :]
    u_statistic = positive_ranks.sum() - len(positive) * (len(positive) + 1) / 2.0
    return float(u_statistic / (len(negative) * len(positive)))


def _class_feature_effects(
    subject_values: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(feature_names):
        cn_all = subject_values[y == 0, index]
        impaired_all = subject_values[y == 1, index]
        cn = cn_all[np.isfinite(cn_all)]
        impaired = impaired_all[np.isfinite(impaired_all)]
        auc = _rank_auc(cn, impaired)
        rows.append(
            {
                "feature": feature,
                "family": feature.split("__", 1)[0],
                "n_cn_finite": int(len(cn)),
                "n_impaired_finite": int(len(impaired)),
                "cn_subject_median": float(np.median(cn)) if len(cn) else np.nan,
                "impaired_subject_median": (
                    float(np.median(impaired)) if len(impaired) else np.nan
                ),
                "impaired_minus_cn_median": (
                    float(np.median(impaired) - np.median(cn))
                    if len(cn) and len(impaired)
                    else np.nan
                ),
                "cliffs_delta": float(2.0 * auc - 1.0) if np.isfinite(auc) else np.nan,
                "direction_free_auc": (
                    float(max(auc, 1.0 - auc)) if np.isfinite(auc) else np.nan
                ),
                "cn_subject_missing_rate": float(np.mean(~np.isfinite(cn_all))),
                "impaired_subject_missing_rate": float(
                    np.mean(~np.isfinite(impaired_all))
                ),
                "absolute_missing_rate_difference": float(
                    abs(
                        np.mean(~np.isfinite(impaired_all))
                        - np.mean(~np.isfinite(cn_all))
                    )
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["direction_free_auc", "absolute_missing_rate_difference", "feature"],
        ascending=[False, False, True],
        na_position="last",
    )


def _feature_family_summary(
    quality: pd.DataFrame,
    effects: pd.DataFrame,
) -> pd.DataFrame:
    merged = quality.merge(
        effects[["feature", "direction_free_auc", "absolute_missing_rate_difference"]],
        on="feature",
        how="left",
        validate="one_to_one",
    )
    return (
        merged.groupby("family", as_index=False)
        .agg(
            n_features=("feature", "size"),
            mean_daily_missing_rate=("daily_missing_rate", "mean"),
            maximum_subject_missing_rate=("subject_missing_rate", "max"),
            zero_variance_features=("zero_variance", "sum"),
            median_direction_free_auc=("direction_free_auc", "median"),
            maximum_direction_free_auc=("direction_free_auc", "max"),
            maximum_missing_rate_difference=(
                "absolute_missing_rate_difference",
                "max",
            ),
        )
        .sort_values("family")
    )


def _save_overview_plot(
    output_path: Path,
    lengths: np.ndarray,
    y: np.ndarray,
    quality: pd.DataFrame,
    effects: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].bar(["CN", "MCI+DEM"], [int(np.sum(y == 0)), int(np.sum(y == 1))])
    axes[0, 0].set_title("Training subject class balance")
    axes[0, 0].set_ylabel("Subjects")

    boxplot_values = [lengths[y == 0], lengths[y == 1]]
    boxplot_labels = ["CN", "MCI+DEM"]
    try:
        # ``tick_labels`` replaced ``labels`` in Matplotlib 3.9.
        axes[0, 1].boxplot(
            boxplot_values, tick_labels=boxplot_labels, showmeans=True
        )
    except TypeError as error:
        if "tick_labels" not in str(error):
            raise
        axes[0, 1].boxplot(boxplot_values, labels=boxplot_labels, showmeans=True)
    axes[0, 1].axhline(28, color="tab:red", linestyle="--", linewidth=1, label="28-day crop")
    axes[0, 1].set_title("Aligned observations per subject")
    axes[0, 1].set_ylabel("Observed days")
    axes[0, 1].legend()

    axes[1, 0].hist(quality["daily_missing_rate"], bins=20, color="tab:purple", alpha=0.85)
    axes[1, 0].set_title("Daily feature missingness")
    axes[1, 0].set_xlabel("Missing rate")
    axes[1, 0].set_ylabel("Features")

    top = effects.dropna(subset=["direction_free_auc"]).head(15).iloc[::-1]
    colors = ["tab:orange" if value >= 0 else "tab:blue" for value in top["cliffs_delta"]]
    axes[1, 1].barh(top["feature"], top["direction_free_auc"], color=colors)
    axes[1, 1].axvline(0.5, color="black", linewidth=1)
    axes[1, 1].set_xlim(0.48, max(0.7, float(top["direction_free_auc"].max()) + 0.02))
    axes[1, 1].set_title("Top subject-level direction-free AUC")
    axes[1, 1].set_xlabel("AUC after choosing direction")

    fig.suptitle("Training-only wearable sequence EDA", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(
    dataset: SubjectSequenceDataset,
    length_summary: pd.DataFrame,
    quality: pd.DataFrame,
    effects: pd.DataFrame,
    family_summary: pd.DataFrame,
) -> str:
    y = np.asarray(dataset.y, dtype=int)
    top = effects.head(15)
    lines = [
        "# Binary Wearable SequenceFusion training-only EDA",
        "",
        "이 보고서는 source training subjects만 사용했습니다. Validation source/labels는 EDA, "
        "feature 선택, threshold 선택에 사용하지 않았습니다.",
        "",
        "## 입력 계약",
        "",
        f"- Subjects: {len(y)} (CN={int(np.sum(y == 0))}, MCI+DEM={int(np.sum(y == 1))})",
        f"- Daily features: {len(dataset.feature_names)}",
        f"- Aligned subject-days: {int(sum(len(sequence) for sequence in dataset.sequences))}",
        "- Activity date: local `activity_day_start` date",
        "- Sleep date: local `sleep_bedtime_end` date; longest valid sleep selected",
        "- Model features exclude IDs, diagnosis, absolute dates, acquisition order/period, "
        "observation count/coverage/mask, and non-wear",
        "- MMSE values are not loaded or resolved",
        "",
        "## Sequence length",
        "",
        "| Group | N | Min | P10 | Median | P90 | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in length_summary.itertuples(index=False):
        lines.append(
            f"| {row.group} | {row.n_subjects} | {row.minimum} | {row.p10:.1f} | "
            f"{row.median:.1f} | {row.p90:.1f} | {row.maximum} |"
        )
    lines.extend(
        [
            "",
            "모든 subject가 28개 이상의 aligned observation을 가져 padding 없이 고정 길이 crop을 "
            "만들 수 있습니다. 관측일 수 자체는 모델 feature로 전달하지 않습니다.",
            "",
            "## Feature quality",
            "",
            f"- Zero-variance features: {int(quality['zero_variance'].sum())}",
            f"- Daily missing rate ≥ 20%: {int((quality['daily_missing_rate'] >= 0.20).sum())}",
            f"- Subject missing rate > 0%: {int((quality['subject_missing_rate'] > 0).sum())}",
            "",
            "| Family | Features | Mean daily missing | Max subject missing | Median direction-free AUC |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in family_summary.itertuples(index=False):
        lines.append(
            f"| {row.family} | {row.n_features} | {row.mean_daily_missing_rate:.3f} | "
            f"{row.maximum_subject_missing_rate:.3f} | {row.median_direction_free_auc:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Strongest univariate subject-level effects",
            "",
            "각 subject의 일별 값을 먼저 median으로 요약한 뒤 한 subject당 한 표만 사용했습니다. "
            "이 순위는 EDA이며 validation을 보지 않은 고정 feature 계약을 변경하지 않습니다.",
            "",
            "| Feature | Direction-free AUC | Cliff's delta | CN median | MCI+DEM median |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in top.itertuples(index=False):
        lines.append(
            f"| `{row.feature}` | {row.direction_free_auc:.3f} | {row.cliffs_delta:.3f} | "
            f"{row.cn_subject_median:.4g} | {row.impaired_subject_median:.4g} |"
        )
    lines.extend(
        [
            "",
            "## 해석 주의",
            "",
            "- Accuracy 0.9는 보장값이 아니라 목표입니다. 반복 subject-level nested CV와 새로운 "
            "외부/후속 holdout에서 확인해야 합니다.",
            "- 일별 관측치는 같은 subject 안에서 상관되어 있으므로 day row를 독립 표본처럼 해석하지 않습니다.",
            "- 결측값 대체와 scaling은 반드시 각 CV fold의 training subjects에서만 학습해야 합니다.",
            "- 현재 validation split은 이전 실험에서 이미 관찰된 historical validation입니다.",
            "",
        ]
    )
    return "\n".join(lines)


def run_training_eda(
    dataset: SubjectSequenceDataset,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run and save training-only EDA without exporting subject identifiers."""

    if dataset.y is None:
        raise ValueError("Training EDA requires labels")
    if str(dataset.audit.get("split")) != "train":
        raise ValueError("EDA is locked to the training split")
    y = np.asarray(dataset.y, dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training EDA requires both binary classes")

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lengths = np.asarray([len(sequence) for sequence in dataset.sequences], dtype=int)
    daily_values = np.concatenate(dataset.sequences, axis=0).astype(float)
    subject_values = _subject_medians(dataset)
    length_summary = _sequence_length_summary(lengths, y)
    quality = _feature_quality(daily_values, subject_values, dataset.feature_names)
    effects = _class_feature_effects(subject_values, y, dataset.feature_names)
    family_summary = _feature_family_summary(quality, effects)

    length_path = output / "sequence_length_summary.csv"
    quality_path = output / "feature_quality.csv"
    effects_path = output / "class_feature_effects.csv"
    family_path = output / "feature_family_summary.csv"
    figure_path = output / "eda_overview.png"
    report_path = output / "EDA_REPORT_KO.md"
    summary_path = output / "eda_summary.json"
    length_summary.to_csv(length_path, index=False)
    quality.to_csv(quality_path, index=False)
    effects.to_csv(effects_path, index=False)
    family_summary.to_csv(family_path, index=False)
    _save_overview_plot(figure_path, lengths, y, quality, effects)
    report_path.write_text(
        _markdown_report(dataset, length_summary, quality, effects, family_summary),
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "scope": "training_only",
        "subject_ids_exported": False,
        "n_subjects": int(len(y)),
        "class_counts": {"CN": int(np.sum(y == 0)), "MCI+DEM": int(np.sum(y == 1))},
        "n_aligned_subject_days": int(len(daily_values)),
        "n_features": int(len(dataset.feature_names)),
        "sequence_length": {
            "minimum": int(lengths.min()),
            "median": float(np.median(lengths)),
            "maximum": int(lengths.max()),
        },
        "feature_quality": {
            "zero_variance": int(quality["zero_variance"].sum()),
            "daily_missing_rate_ge_0_20": int((quality["daily_missing_rate"] >= 0.20).sum()),
            "subject_missing_rate_gt_0": int((quality["subject_missing_rate"] > 0).sum()),
        },
        "maximum_direction_free_auc": float(effects["direction_free_auc"].max()),
        "dataset_audit": dict(dataset.audit),
        "artifacts": {
            "sequence_length_summary": length_path.name,
            "feature_quality": quality_path.name,
            "class_feature_effects": effects_path.name,
            "feature_family_summary": family_path.name,
            "overview_figure": figure_path.name,
            "report": report_path.name,
        },
    }
    summary_path.write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


__all__ = ["run_training_eda"]
