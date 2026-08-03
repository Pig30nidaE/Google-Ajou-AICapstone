"""결과표가 평가 행 수와 피험자 수를 혼동하지 않는지 검증한다."""

from __future__ import annotations

import pandas as pd

from src.experiments.compare import CrossExperimentResults, assemble_comparison


def test_record_comparison_uses_actual_evaluated_dem_subject_count(tmp_path):
    results = CrossExperimentResults()
    results.record_level[("A", "xgboost", "vae")] = {
        "macro_f1": 0.5,
        "dem_f1": 0.4,
        "dem_recall": 0.3,
        "n": 1096,
        "n_dem_subjects_eval": 8,
    }
    results.subject_level[("A", "xgboost", "vae")] = {
        "macro_f1": 0.5,
        "balanced_accuracy": 0.4,
        "macro_roc_auc_ovr": 0.6,
        "dem_recall": 0.3,
        "dem_f1": 0.4,
        "unit": "subject",
        "n": 167,
        "n_Dem": 8,
    }

    summary = assemble_comparison(results, out_root=str(tmp_path))
    comparison = pd.read_csv(summary["saved_tables"]["paper_comparison"])
    assert comparison.loc[0, "n_dem_subjects"] == 8
    assert comparison.loc[0, "평가 행 수"] == 1096
