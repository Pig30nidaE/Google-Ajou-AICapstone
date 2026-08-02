"""The reconstruction of the paper's evaluation set (reproduction_spec.md §2).

The papers never state the split unit or the evaluation unit.  Every reported
metric nonetheless decomposes over exactly 33 subjects with 7 positives, which is
the AI-Hub Validation partition.  These tests pin that derivation down so it
cannot rot, and check it against the real data when it is available.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from src.data import schema
from src.evaluation.compare import (
    PAPER_EVAL_SET,
    PAPER_RECONSTRUCTED_CONFUSION,
    PAPER_REPORTED,
    verify_paper_arithmetic,
)


def test_every_model_confusion_matrix_reproduces_its_reported_metrics() -> None:
    result = verify_paper_arithmetic()
    assert result["all_models_consistent"], result
    for model, block in result["per_model"].items():
        assert block["consistent"], f"{model}: {block['checks']}"


def test_reported_accuracies_are_thirty_thirds() -> None:
    for model, reported in PAPER_REPORTED.items():
        fraction = Fraction(reported["accuracy"]).limit_denominator(40)
        assert fraction.denominator in (33, 11, 3), (
            f"{model} accuracy {reported['accuracy']} is not k/33"
        )


def test_reported_recalls_are_sevenths() -> None:
    for model, reported in PAPER_REPORTED.items():
        fraction = Fraction(reported["sensitivity"]).limit_denominator(10)
        assert fraction.denominator in (7, 1), (
            f"{model} recall {reported['sensitivity']} is not k/7"
        )


def test_confusion_matrices_sum_to_the_reconstructed_evaluation_set() -> None:
    for model, cm in PAPER_RECONSTRUCTED_CONFUSION.items():
        assert sum(cm.values()) == PAPER_EVAL_SET["n"], model
        assert cm["tp"] + cm["fn"] == PAPER_EVAL_SET["n_positive"], model
        assert cm["tn"] + cm["fp"] == PAPER_EVAL_SET["n_negative"], model


def test_lstm_family_predicted_almost_no_positives() -> None:
    """The 0.818 accuracy is 27/33 with a single true positive and no false ones."""
    for model in ("lstm", "bilstm"):
        cm = PAPER_RECONSTRUCTED_CONFUSION[model]
        assert (cm["tp"], cm["fp"], cm["fn"], cm["tn"]) == (1, 0, 6, 26)


def test_reported_aucs_have_denominator_7_times_26() -> None:
    """AUC = U / (n_pos * n_neg); integral U is independent evidence for 33 subjects."""
    for model in ("lstm", "bilstm"):
        auc = PAPER_REPORTED[model]["roc_auc"]
        u_statistic = auc * PAPER_EVAL_SET["n_positive"] * PAPER_EVAL_SET["n_negative"]
        assert abs(u_statistic - round(u_statistic)) < 0.01, (
            f"{model}: AUC {auc} does not give an integer U over 7x26"
        )


def test_row_level_evaluation_is_ruled_out() -> None:
    """A 20% row split would score ~2,437 rows; k/33 could not arise."""
    n_rows_test = int(12183 * 0.2)
    assert n_rows_test > 2000
    for reported in PAPER_REPORTED.values():
        scaled = reported["accuracy"] * n_rows_test
        assert abs(scaled - round(scaled)) > 1e-6 or n_rows_test % 33 == 0


def test_real_validation_split_matches_the_reconstruction(real_data_root) -> None:
    import pandas as pd

    mmse = pd.read_csv(
        real_data_root / "2.Validation/SourceData/3.CognitiveFunction/val_mmse.csv"
    )
    positives = int(mmse[schema.DIAGNOSIS_COL].isin(schema.POSITIVE_DIAGNOSES).sum())
    assert len(mmse) == PAPER_EVAL_SET["n"] == 33
    assert positives == PAPER_EVAL_SET["n_positive"] == 7


def test_real_cohort_matches_the_papers_class_counts(real_data_root) -> None:
    import pandas as pd

    frames = [
        pd.read_csv(real_data_root / "1.Training/SourceData/3.CognitiveFunction/train_mmse.csv"),
        pd.read_csv(real_data_root / "2.Validation/SourceData/3.CognitiveFunction/val_mmse.csv"),
    ]
    combined = pd.concat(frames)
    assert combined[schema.LABEL_PERSON_KEY].nunique() == 174, "papers report 174 subjects"
    positives = int(combined[schema.DIAGNOSIS_COL].isin(schema.POSITIVE_DIAGNOSES).sum())
    assert positives == 63, "papers report 63 dementia-side subjects"


def test_daily_row_count_disagrees_with_the_papers(real_data_root) -> None:
    """Both papers say 12,184; the distributed CSVs contain 12,183."""
    import pandas as pd

    n_rows = sum(
        len(pd.read_csv(real_data_root / path, usecols=[0]))
        for path in (
            "1.Training/SourceData/1.Gait/train_activity.csv",
            "2.Validation/SourceData/1.Gait/val_activity.csv",
        )
    )
    assert n_rows == 12183, f"expected the measured 12,183 rows, got {n_rows}"
    assert n_rows != 12184, "the papers' figure would now agree -- update the docs"
