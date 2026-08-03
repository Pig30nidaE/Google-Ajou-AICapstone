"""Diagnostics that decide whether the paper's *conclusions* reproduced.

The 2026-08-03 experiment-A runs matched the paper's AUC to within 0.02-0.07 but
missed sensitivity by 0.15-0.25 and lost the LSTM-over-baselines gap almost
entirely.  Three questions had to be separated to say what that means:

* is the evaluation set composed differently?  (prevalence)
* is the published operating point even on our ROC curve?  (threshold vs skill)
* did the paper's actual conclusion survive?  (LSTM vs baselines, length trend)

Each has a function now, and each is pinned here.
"""

from __future__ import annotations

import numpy as np

from src.evaluation import metrics
from src.evaluation.compare import build_comparison, render_comparison_markdown


# --- implied prevalence -------------------------------------------------------

def test_every_paper_table5_row_implies_a_balanced_evaluation_set():
    """All 7 rows solve to ~0.50, which the faithful split (~0.32) is not."""
    rows = {
        "lstm_3day": (0.87, 0.74, 0.77), "lstm_4day": (0.89, 0.77, 0.80),
        "lstm_5day": (0.89, 0.80, 0.82), "xgboost": (0.68, 0.76, 0.74),
        "random_forest": (0.67, 0.77, 0.75), "logistic_regression": (0.59, 0.60, 0.60),
        "svm": (0.62, 0.59, 0.60),
    }
    for name, (sensitivity, specificity, precision) in rows.items():
        prevalence = metrics.implied_prevalence(sensitivity, specificity, precision)
        assert prevalence is not None, name
        assert 0.49 <= prevalence <= 0.51, f"{name}: {prevalence}"


def test_implied_prevalence_round_trips():
    """Build a confusion matrix at a known prevalence and recover it."""
    prevalence, sensitivity, specificity = 0.30, 0.80, 0.70
    n = 10_000
    n_pos = int(n * prevalence)
    tp, fn = sensitivity * n_pos, (1 - sensitivity) * n_pos
    tn, fp = specificity * (n - n_pos), (1 - specificity) * (n - n_pos)
    precision = tp / (tp + fp)
    recovered = metrics.implied_prevalence(sensitivity, specificity, precision)
    assert abs(recovered - prevalence) < 1e-6
    assert fn > 0 and tn > 0


def test_implied_prevalence_returns_none_when_undetermined():
    """specificity 1.0 means no false positives, so precision is 1 at every p."""
    assert metrics.implied_prevalence(0.5, 1.0, 0.5) is None
    assert metrics.implied_prevalence(0.5, 1.0, 1.0) is None
    # Nothing predicted positive: the equation says nothing about prevalence.
    assert metrics.implied_prevalence(0.0, 0.8, 0.5) is None


def test_implied_prevalence_accepts_a_low_but_reachable_row():
    """A very low precision is legitimate when prevalence is very low."""
    value = metrics.implied_prevalence(0.9, 0.1, 0.01)
    assert value is not None and 0.0 < value < 0.05


# --- prevalence matching ------------------------------------------------------

def _scores(n_pos: int, n_neg: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.array([1] * n_pos + [0] * n_neg)
    s = np.concatenate([rng.normal(0.65, 0.2, n_pos), rng.normal(0.35, 0.2, n_neg)])
    return y, np.clip(s, 0, 1)


def test_prevalence_matching_leaves_auc_alone_but_moves_precision():
    """The whole point: AUC is prevalence-invariant, precision is not."""
    y, s = _scores(100, 300)
    full = metrics.binary_metrics(y, s, threshold=0.5)
    matched = metrics.prevalence_matched_metrics(y, s, target_prevalence=0.5, n_repeats=200)

    assert matched["available"]
    assert abs(matched["achieved_prevalence"] - 0.5) < 0.01
    assert abs(matched["metrics"]["roc_auc"]["mean"] - full["roc_auc"]) < 0.02
    # More positives in the mix can only help precision.
    assert matched["metrics"]["precision"]["mean"] > full["precision"]
    # Sensitivity and specificity are prevalence-invariant too.
    assert abs(matched["metrics"]["sensitivity"]["mean"] - full["sensitivity"]) < 0.02
    assert abs(matched["metrics"]["specificity"]["mean"] - full["specificity"]) < 0.02


def test_prevalence_matching_keeps_the_largest_possible_subsample():
    y, s = _scores(94, 200)
    matched = metrics.prevalence_matched_metrics(y, s, target_prevalence=0.5, n_repeats=20)
    assert matched["n_positive"] == 94 and matched["n_negative"] == 94


def test_prevalence_matching_refuses_a_single_class():
    y = np.ones(20, dtype=int)
    result = metrics.prevalence_matched_metrics(y, np.linspace(0, 1, 20))
    assert result["available"] is False


# --- operating point ----------------------------------------------------------

def test_operating_point_detects_a_point_above_our_curve():
    """A weak model cannot reach a strong published (sens, spec) at any threshold."""
    y, s = _scores(100, 100, seed=3)
    result = metrics.operating_point_comparison(
        y, s, target_sensitivity=0.89, target_specificity=0.80
    )
    assert result["available"]
    assert result["paper_point_above_our_curve"] is True
    assert result["at_paper_sensitivity"]["specificity"] < 0.80


def test_operating_point_accepts_a_reachable_point():
    """A near-perfect model reaches the target, so the gap is only a threshold."""
    y = np.array([1] * 50 + [0] * 50)
    s = np.concatenate([np.linspace(0.6, 1.0, 50), np.linspace(0.0, 0.4, 50)])
    result = metrics.operating_point_comparison(
        y, s, target_sensitivity=0.89, target_specificity=0.80, threshold=0.99
    )
    assert result["paper_point_above_our_curve"] is False
    assert result["at_paper_sensitivity"]["specificity"] >= 0.80


# --- central claim and trend --------------------------------------------------

def _temporal_report(
    lstm: dict[int, float], baselines: dict[str, dict[int, float]]
) -> dict:
    results = {}
    for length, auc in lstm.items():
        results[f"lstm_L{length}"] = {
            "model": "lstm", "sequence_length": length, "is_partial_subset": False,
            "sequence_level": {"roc_auc": auc, "n": 300, "n_positive": 100},
        }
        for name, per_length in baselines.items():
            if length not in per_length:
                continue
            results[f"{name}_L{length}"] = {
                "model": name, "sequence_length": length, "is_partial_subset": False,
                "sequence_level": {"roc_auc": per_length[length], "n": 300, "n_positive": 100},
            }
    return {"experiment": "paper_temporal_reconstruction", "results": results}


#: The measured 2026-08-03 experiment-A numbers.
REAL_LSTM = {3: 0.859, 4: 0.862, 5: 0.849}
REAL_BASELINES = {
    "xgboost": {3: 0.852, 4: 0.855, 5: 0.833},
    "random_forest": {3: 0.827, 4: 0.830, 5: 0.820},
    "svm": {3: 0.815, 4: 0.812, 5: 0.792},
    "logistic_regression": {3: 0.668, 4: 0.688, 5: 0.660},
}


def test_central_claim_fails_when_the_gap_collapses():
    """The 2026-08-03 shape: LSTM still ahead, but by 0.006-0.016, not 0.07-0.11."""
    report = _temporal_report(REAL_LSTM, REAL_BASELINES)
    claim = build_comparison({"paper_temporal_reconstruction": report})["table7_central_claim"]
    assert claim["claim_reproduced"] is False
    assert claim["n_lstm_ahead"] == claim["n_comparisons"]        # still ahead everywhere...
    assert claim["n_gap_at_least_0_05"] < claim["n_comparisons"]  # ...but rarely by much
    xgb5 = next(r for r in claim["rows"] if r["sequence_length"] == 5 and r["baseline"] == "xgboost")
    assert xgb5["paper_gap"] > 0.10 and xgb5["reproduction_gap"] < 0.03
    assert xgb5["gap_shrinkage"] < 0


def test_central_claim_passes_when_the_gap_holds():
    report = _temporal_report(
        {3: 0.88, 4: 0.91, 5: 0.92},
        {
            "xgboost": {3: 0.81, 4: 0.81, 5: 0.81},
            "random_forest": {3: 0.81, 4: 0.81, 5: 0.81},
            "svm": {3: 0.64, 4: 0.64, 5: 0.64},
            "logistic_regression": {3: 0.63, 4: 0.63, 5: 0.63},
        },
    )
    claim = build_comparison({"paper_temporal_reconstruction": report})["table7_central_claim"]
    assert claim["claim_reproduced"] is True


def test_length_trend_flags_a_non_monotone_reproduction():
    report = _temporal_report(REAL_LSTM, {})
    trend = build_comparison({"paper_temporal_reconstruction": report})["table8_sequence_length_trend"]
    assert trend["paper_monotone_increasing"] is True
    assert trend["reproduction_monotone_increasing"] is False
    assert trend["complete"] is True


def test_length_trend_needs_all_three_lengths():
    report = _temporal_report({5: 0.849}, {})
    trend = build_comparison({"paper_temporal_reconstruction": report})["table8_sequence_length_trend"]
    assert trend["complete"] is False


def test_rendered_report_states_both_verdicts():
    report = _temporal_report(REAL_LSTM, REAL_BASELINES)
    text = render_comparison_markdown(build_comparison({"paper_temporal_reconstruction": report}))
    assert "표 7." in text and "재현되지 않음" in text
    assert "표 8." in text
