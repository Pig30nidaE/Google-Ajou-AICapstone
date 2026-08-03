"""Cross-fold classification must retain each fold's inner-CV threshold."""

from __future__ import annotations

import pandas as pd

from src.engine import FoldResult, aggregate_model


def _fold(
    fold: int,
    threshold: float,
    rows: list[dict[str, float | int | str]],
) -> FoldResult:
    return FoldResult(
        model="random_forest",
        repeat=0,
        fold=fold,
        subject_predictions=pd.DataFrame(rows),
        subject_metrics={"roc_auc": 1.0},
        threshold=threshold,
        threshold_source="inner_cv",
    )


def test_aggregate_uses_each_folds_selected_threshold() -> None:
    folds = [
        _fold(
            0,
            0.8,
            [
                {"subject_id": "a", "probability": 0.7, "label": 0},
                {"subject_id": "b", "probability": 0.9, "label": 1},
            ],
        ),
        _fold(
            1,
            0.2,
            [
                {"subject_id": "c", "probability": 0.1, "label": 0},
                {"subject_id": "d", "probability": 0.3, "label": 1},
            ],
        ),
    ]

    result = aggregate_model("random_forest", folds, seed=0)

    # Reusing fold 0's 0.8 globally would misclassify subject d.  Fold-specific
    # decisions are all correct and cannot be represented by one shared cutoff.
    assert result["subject_level"]["accuracy"] == 1.0
    assert result["subject_level"]["classification_source"] == "fold_specific_decisions"
    assert result["threshold"] is None
    assert result["threshold_source"] == "fold_specific_inner_cv"
    assert [item["threshold"] for item in result["thresholds_by_fold"]] == [0.8, 0.2]
