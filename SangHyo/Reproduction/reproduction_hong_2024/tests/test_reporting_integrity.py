"""Regressions for two defects found in the 2026-08-02 nested run.

1. The nested arm emits ``{model}_L{length}`` blocks that cover only the outer
   folds whose inner CV chose that length.  Those reached the per-length rows of
   table 1, so the published table read "Nested LSTM 5-day = 0.782" -- one fold,
   35 subjects -- while the real nested estimate (0.533, 174 subjects) appeared
   nowhere.  That is the pick-the-best-length bias experiment C exists to avoid.

2. ``choose_threshold`` accepted a threshold below every score, which predicts
   the positive class for everything: sensitivity 1.0, specificity 0.0, Youden
   exactly 0.  On a chance-level model no real candidate beats 0, so the
   degenerate point won by default in 6 of 15 folds.
"""

from __future__ import annotations

import numpy as np

from src.evaluation import metrics
from src.evaluation.compare import build_comparison, render_comparison_markdown


def _nested_report() -> dict:
    """A miniature of the real report: one honest block plus three partials."""
    def block(auc, n_subjects, n_folds, *, partial, lengths):
        return {
            "model": "lstm",
            "n_folds": n_folds,
            "sequence_length": lengths,
            "is_partial_subset": partial,
            "subject_level": {"roc_auc": auc, "pr_auc": 0.4,
                              "balanced_accuracy": 0.51, "n_subjects": n_subjects},
            "sequence_level": {"roc_auc": auc - 0.05},
            "subject_bootstrap_ci": {"ci_lower": auc - 0.09, "ci_upper": auc + 0.09},
            "selection": {"chosen_sequence_lengths": {"3": 3, "4": 1, "5": 1}},
        }

    return {
        "experiment": "nested_subject_independent",
        "estimand": "B",
        "results": {
            "lstm_Lnested": block(0.533, 174, 5, partial=False, lengths=[3, 4, 5]),
            "lstm_L3": block(0.492, 104, 3, partial=True, lengths=[3]),
            "lstm_L4": block(0.579, 35, 1, partial=True, lengths=[4]),
            "lstm_L5": block(0.782, 35, 1, partial=True, lengths=[5]),
        },
    }


# --- defect 1: partial subsets must not become headline numbers ---------------

def test_partial_subsets_never_fill_the_per_length_rows():
    comparison = build_comparison({"nested_subject_independent": _nested_report()})
    for row in comparison["table1_by_sequence_length"]:
        assert row["nested_subject_independent"] is None, (
            f"{row['sequence_length']}일 행이 부분집합 값으로 채워졌다"
        )


def test_the_real_nested_estimate_is_reported():
    comparison = build_comparison({"nested_subject_independent": _nested_report()})
    nested = comparison["table4_nested_selected"]
    assert len(nested) == 1
    assert nested[0]["model"] == "lstm"
    assert nested[0]["roc_auc"] == 0.533
    assert nested[0]["n_subjects"] == 174
    assert nested[0]["chosen_sequence_lengths"] == {"3": 3, "4": 1, "5": 1}


def test_partials_are_kept_but_labelled_with_their_size():
    comparison = build_comparison({"nested_subject_independent": _nested_report()})
    partials = {row["key"]: row for row in comparison["table5_nested_partial_subsets"]}
    assert set(partials) == {"lstm_L3", "lstm_L4", "lstm_L5"}
    assert partials["lstm_L5"]["n_subjects"] == 35
    assert partials["lstm_L5"]["n_folds"] == 1


def test_rendered_table_shows_the_nested_estimate_not_the_best_partial():
    text = render_comparison_markdown(
        build_comparison({"nested_subject_independent": _nested_report()})
    )
    table1 = text.split("## 표 2.")[0]
    assert "0.782" not in table1, "부분집합 값이 표 1에 노출되었다"
    assert "0.533" in text and "표 4." in text
    # The partial section must carry its warning, not sit there as a bare number.
    partial_section = text.split("## 표 5.")[1]
    assert "성능 주장" in partial_section and "0.782" in partial_section


def test_full_per_length_reports_still_populate_the_rows():
    """A non-nested report is per-length by construction and must still show up."""
    report = {
        "experiment": "fixed_subject_independent",
        "results": {
            f"lstm_L{length}": {
                "model": "lstm", "n_folds": 5, "is_partial_subset": False,
                "subject_level": {"roc_auc": 0.5 + length / 100, "n_subjects": 174},
            }
            for length in (3, 4, 5)
        },
    }
    comparison = build_comparison({"fixed_subject_independent": report})
    values = [row["fixed_subject_independent"] for row in comparison["table1_by_sequence_length"]]
    assert values == [0.53, 0.54, 0.55]


# --- defect 2: degenerate operating points ------------------------------------

def test_all_positive_threshold_is_rejected():
    """A chance-level model must not be handed a predict-everything threshold."""
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 30 + [1] * 20)
    y_score = rng.uniform(0, 1, size=50)          # no signal at all

    threshold, report = metrics.choose_threshold_with_report(
        y_true, y_score, policy="youden", fixed=0.5
    )
    chosen = metrics.binary_metrics(y_true, y_score, threshold=threshold)
    assert chosen["specificity"] > 0.0 or report["fallback_to_fixed"]
    assert chosen["sensitivity"] > 0.0 or report["fallback_to_fixed"]
    assert report["n_degenerate_skipped"] >= 1


def test_saturated_scores_do_not_produce_a_zero_threshold():
    """The real failure mode: LSTM scores piled at 0.0 and 1.0."""
    y_true = np.array([0] * 25 + [1] * 25)
    y_score = np.concatenate([
        np.full(20, 0.0), np.full(5, 1.0),        # negatives, mostly 0
        np.full(15, 0.0), np.full(10, 1.0),       # positives, mostly 0 too
    ])
    threshold, report = metrics.choose_threshold_with_report(
        y_true, y_score, policy="youden", fixed=0.5
    )
    result = metrics.binary_metrics(y_true, y_score, threshold=threshold)
    matrix = result["confusion_matrix"]
    assert (matrix["tp"] + matrix["fp"]) > 0
    assert (matrix["tn"] + matrix["fn"]) > 0


def test_threshold_falls_back_when_nothing_separates():
    """Identical scores everywhere: every split is degenerate, so use the fixed one."""
    y_true = np.array([0, 1] * 10)
    y_score = np.full(20, 0.7)
    threshold, report = metrics.choose_threshold_with_report(
        y_true, y_score, policy="youden", fixed=0.5
    )
    assert threshold == 0.5
    assert report["fallback_to_fixed"] is True


def test_a_separable_problem_still_gets_a_useful_threshold():
    """The guard must not break the case where a good operating point exists."""
    y_true = np.array([0] * 25 + [1] * 25)
    y_score = np.concatenate([np.linspace(0.0, 0.4, 25), np.linspace(0.6, 1.0, 25)])
    threshold, report = metrics.choose_threshold_with_report(
        y_true, y_score, policy="youden", fixed=0.5
    )
    assert report["fallback_to_fixed"] is False
    result = metrics.binary_metrics(y_true, y_score, threshold=threshold)
    assert result["sensitivity"] == 1.0 and result["specificity"] == 1.0


def test_fixed_policy_is_untouched():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.2, 0.8])
    threshold, report = metrics.choose_threshold_with_report(
        y_true, y_score, policy="fixed", fixed=0.5
    )
    assert threshold == 0.5
    assert report["fallback_to_fixed"] is False


def test_choose_threshold_wrapper_matches():
    y_true = np.array([0] * 10 + [1] * 10)
    y_score = np.concatenate([np.linspace(0, 0.45, 10), np.linspace(0.55, 1, 10)])
    assert metrics.choose_threshold(y_true, y_score, policy="youden") == (
        metrics.choose_threshold_with_report(y_true, y_score, policy="youden")[0]
    )
