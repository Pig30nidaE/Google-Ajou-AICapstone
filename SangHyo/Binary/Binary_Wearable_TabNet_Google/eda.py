"""Training-only EDA for raw daily and subject-level wearable features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from SangHyo.Binary.Binary_Wearable_SequenceFusion_Google.eda import run_training_eda

from .data import SubjectSequenceDataset
from .features import SubjectFeatureTable


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _rank_auc(negative: np.ndarray, positive: np.ndarray) -> float:
    negative = negative[np.isfinite(negative)]
    positive = positive[np.isfinite(positive)]
    if not len(negative) or not len(positive):
        return np.nan
    values = np.r_[negative, positive]
    ranks = pd.Series(values).rank(method="average").to_numpy(float)
    u = ranks[len(negative) :].sum() - len(positive) * (len(positive) + 1) / 2.0
    return float(u / (len(negative) * len(positive)))


def _subject_effects(table: SubjectFeatureTable) -> pd.DataFrame:
    if table.y is None:
        raise ValueError("EDA requires Training labels")
    rows: list[dict[str, Any]] = []
    for name in table.X.columns:
        values = pd.to_numeric(table.X[name], errors="coerce").to_numpy(float)
        negative, positive = values[table.y == 0], values[table.y == 1]
        auc = _rank_auc(negative, positive)
        rows.append(
            {
                "feature": name,
                "modality": name.split("__", 1)[0],
                "missing_rate": float(np.mean(~np.isfinite(values))),
                "cn_median": float(np.nanmedian(negative)) if np.isfinite(negative).any() else np.nan,
                "mci_dem_median": float(np.nanmedian(positive)) if np.isfinite(positive).any() else np.nan,
                "direction_free_auc": float(max(auc, 1.0 - auc)) if np.isfinite(auc) else np.nan,
                "cliffs_delta": float(2.0 * auc - 1.0) if np.isfinite(auc) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["direction_free_auc", "missing_rate", "feature"],
        ascending=[False, True, True],
        na_position="last",
    )


def _save_plot(path: Path, table: SubjectFeatureTable, effects: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assert table.y is not None
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    axes[0].bar(["CN", "MCI+DEM"], [int((table.y == 0).sum()), int((table.y == 1).sum())])
    axes[0].set_title("Training subjects")
    axes[0].set_ylabel("N")
    axes[1].hist(effects["missing_rate"], bins=25, color="tab:purple")
    axes[1].set_title("Subject-feature missingness")
    axes[1].set_xlabel("Missing fraction")
    top = effects.dropna(subset=["direction_free_auc"]).head(15).iloc[::-1]
    axes[2].barh(top["feature"], top["direction_free_auc"], color="tab:orange")
    axes[2].axvline(0.5, color="black", linewidth=1)
    axes[2].set_title("Top training-only univariate AUC")
    axes[2].set_xlabel("Direction-free AUC")
    fig.suptitle("MMSE-free binary TabNet feature EDA")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_eda(
    sequence_dataset: SubjectSequenceDataset,
    feature_table: SubjectFeatureTable,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Export aggregate EDA without raw/hashed subject identifiers."""

    if sequence_dataset.y is None or feature_table.y is None:
        raise ValueError("Training-only EDA requires labels")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    daily_summary = run_training_eda(sequence_dataset, output / "daily_sequence")
    effects = _subject_effects(feature_table)
    effects.to_csv(output / "subject_feature_effects.csv", index=False)
    modality = (
        effects.groupby("modality", as_index=False)
        .agg(
            features=("feature", "size"),
            mean_missing_rate=("missing_rate", "mean"),
            median_direction_free_auc=("direction_free_auc", "median"),
            maximum_direction_free_auc=("direction_free_auc", "max"),
        )
        .sort_values("modality")
    )
    modality.to_csv(output / "subject_feature_family_summary.csv", index=False)
    _save_plot(output / "subject_feature_eda.png", feature_table, effects)
    report_lines = [
        "# Binary Wearable TabNet training-only EDA",
        "",
        "Validation 데이터와 Validation 정답은 EDA 및 특징 선택에 사용하지 않았습니다.",
        "",
        f"- Subjects: {len(feature_table.X)} (CN={int((feature_table.y == 0).sum())}, MCI+DEM={int((feature_table.y == 1).sum())})",
        f"- Daily wearable channels: {len(sequence_dataset.feature_names)}",
        f"- Subject-level candidates: {feature_table.X.shape[1]}",
        f"- Overall candidate missing fraction: {feature_table.X.isna().to_numpy().mean():.4f}",
        "- Inputs: Activity and Sleep only",
        "- Cognitive test files and values are not resolved or opened",
        "- Each subject contributes exactly one model row",
        "- Observation count, coverage, identifier, date, mask, and non-wear are not model features",
        "",
        "## Top univariate effects (Training only)",
        "",
        "| Feature | Direction-free AUC | Missing |",
        "| --- | ---: | ---: |",
    ]
    for row in effects.head(20).itertuples(index=False):
        report_lines.append(
            f"| `{row.feature}` | {row.direction_free_auc:.3f} | {row.missing_rate:.3f} |"
        )
    report_lines += [
        "",
        "단변량 AUC는 탐색적 수치이며 일반화 성능이 아닙니다. 실제 선택·대체·스케일링은 nested CV의 각 training fold 안에서 다시 학습됩니다.",
        "Accuracy 0.9는 목표값이지 보장값이 아니며, 이미 여러 번 확인된 historical Validation은 새로운 독립 test가 아닙니다.",
        "",
    ]
    (output / "EDA_REPORT_KO.md").write_text("\n".join(report_lines), encoding="utf-8")
    summary = {
        "scope": "training_only",
        "subject_identifiers_exported": False,
        "daily_sequence": daily_summary,
        "subject_features": feature_table.audit,
        "maximum_direction_free_auc": float(effects["direction_free_auc"].max()),
        "files": {
            "report": "EDA_REPORT_KO.md",
            "effects": "subject_feature_effects.csv",
            "family_summary": "subject_feature_family_summary.csv",
            "figure": "subject_feature_eda.png",
        },
    }
    (output / "eda_summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


__all__ = ["run_eda"]
