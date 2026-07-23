"""Training-only EDA wrapper for the compact wearable representation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from SangHyo.Binary_Wearable_SequenceFusion_Google.eda import run_training_eda

from .data import SubjectSequenceDataset, make_fixed_views
from .features import ValuePreprocessor, build_multiscale_summaries


def run_eda(dataset: SubjectSequenceDataset, output_dir: str | Path) -> dict[str, Any]:
    if dataset.y is None:
        raise ValueError("EDA is Training-only and requires Training labels")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    daily = run_training_eda(dataset, output / "daily_sequence")
    views = make_fixed_views(dataset)
    # This extra EDA transform is label-free and is not reused by CV models.
    transformed = ValuePreprocessor(
        fit_scope="EDA Training cohort only; never reused by a model fold"
    ).fit(views, dataset.feature_names).transform(views)
    summaries, names = build_multiscale_summaries(transformed, dataset.feature_names)
    y = np.asarray(dataset.y, dtype=np.int64)
    effects: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        negative = summaries[y == 0, index]
        positive = summaries[y == 1, index]
        # Probability that a random positive value exceeds a random negative.
        auc = float(
            (
                (positive[:, None] > negative[None, :]).mean()
                + 0.5 * (positive[:, None] == negative[None, :]).mean()
            )
        )
        effects.append(
            {
                "feature": name,
                "direction_free_training_auc": max(auc, 1.0 - auc),
                "cn_median": float(np.median(negative)),
                "mci_dem_median": float(np.median(positive)),
            }
        )
    effects.sort(
        key=lambda row: (-row["direction_free_training_auc"], row["feature"])
    )
    import pandas as pd

    pd.DataFrame(effects).to_csv(
        output / "multiscale_training_effects.csv", index=False
    )
    summary = {
        "scope": "Training only; Validation data and labels unused",
        "subject_count": len(dataset.subject_ids),
        "class_counts": {
            "CN": int(np.sum(y == 0)),
            "MCI_DEM": int(np.sum(y == 1)),
        },
        "compact_daily_channels": len(dataset.feature_names),
        "fixed_view_observations": views.shape[1],
        "multiscale_candidate_features": len(names),
        "candidate_features_are_not_final_model_features": True,
        "final_selection_is_repeated_inside_each_CV_training_fold": True,
        "maximum_direction_free_training_auc": effects[0][
            "direction_free_training_auc"
        ],
        "cognitive_or_mmse_feature_count": 0,
        "daily_sequence_report": daily,
    }
    (output / "eda_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# BalancedFusion Training-only EDA",
        "",
        "이 문서는 Training만 사용합니다. Validation 데이터와 정답은 EDA에 사용하지 않습니다.",
        "",
        f"- 사람 수: {len(y)} (CN {int(np.sum(y == 0))}, MCI+DEM {int(np.sum(y == 1))})",
        f"- 사전 고정 웨어러블 채널: {len(dataset.feature_names)}개",
        f"- 모든 사람에게 동일한 마지막 관측: {views.shape[1]}개",
        f"- 7/14/35 관측 요약 후보: {len(names)}개",
        "- 실제 모델은 각 CV training fold 안에서 최대 24개 요약만 다시 선택",
        "- Activity/Sleep만 사용; MMSE·인지검사 특징 0개",
        "",
        "## Training에서 차이가 커 보인 요약 20개",
        "",
        "| 요약 특징 | 방향 무관 AUC |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| `{row['feature']}` | {row['direction_free_training_auc']:.3f} |"
        for row in effects[:20]
    )
    lines += [
        "",
        "위 AUC는 탐색용입니다. 일반화 성능이 아니며, 전체 Training에서 본 순위를 "
        "모델 특징으로 고정하지 않습니다.",
        "",
    ]
    (output / "EDA_REPORT_KO.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


__all__ = ["run_eda"]

