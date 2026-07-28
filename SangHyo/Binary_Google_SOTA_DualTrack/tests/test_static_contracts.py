"""Static contract tests -- no training, no Data/ access required.

Run with::

    python -m pytest SangHyo/Binary_Google_SOTA_DualTrack/tests -q
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Binary_Google_SOTA_DualTrack import data, features, metrics, preprocessing  # noqa: E402
from Binary_Google_SOTA_DualTrack.data import LeakageGuardError  # noqa: E402


# ------------------------------------------------------------ leakage guards --
def test_wearable_track_cannot_open_cognitive_function():
    with pytest.raises(LeakageGuardError):
        data.load_mmse(Path("/nonexistent"), "train", track="wearable")


def test_guarded_read_blocks_cognitive_path_for_wearable():
    path = Path("/x/1.Training/SourceData/3.CognitiveFunction/train_mmse.csv")
    with pytest.raises(LeakageGuardError):
        data._guarded_read_csv(path, track="wearable")


def test_diagnosis_columns_are_on_the_ban_list():
    for column in ("DIAG_NM", "DIAG_SEQ", "MMSE_NUM", "MMSE_KIND"):
        assert column in data.MMSE_BANNED_COLUMNS


def test_feature_name_guard_rejects_label_and_count_proxies():
    for bad in ["diag_nm", "act_n_days", "day_count_mean", "target_leak", "n_obs_std"]:
        with pytest.raises(LeakageGuardError):
            data.assert_feature_names_clean([bad])
    data.assert_feature_names_clean(["slp_hr5min_mean", "act_class_frac_rest"])


def test_feature_name_guard_allows_genuine_oura_channels():
    """Regression: 'meet_daily_targets' is an Oura score, not the ML target."""

    data.assert_feature_names_clean([
        "act_activity_score_meet_daily_targets_mean",
        "act_activity_score_meet_daily_targets_cv",
        "act_activity_inactivity_alerts_std",
        "slp_sleep_score_alignment_median",
        "mmse_total",
    ])


def test_wearable_track_rejects_mmse_columns():
    with pytest.raises(LeakageGuardError):
        data.assert_no_mmse_features(["slp_hr5min_mean", "mmse_total"])
    data.assert_no_mmse_features(["slp_hr5min_mean"])


def test_person_overlap_is_fatal():
    with pytest.raises(LeakageGuardError):
        data.assert_person_disjoint(["a@x", "b@x"], ["b@x", "c@x"])
    data.assert_person_disjoint(["a@x"], ["b@x"])


# ------------------------------------------------------------------ parsing --
def test_parse_slash_array_basic():
    assert np.allclose(features.parse_slash_array("1/2/3"), [1.0, 2.0, 3.0])
    assert features.parse_slash_array("").size == 0
    assert features.parse_slash_array("...").size == 0
    assert features.parse_slash_array(np.nan).size == 0


def test_zero_sentinel_becomes_nan_for_hr_and_rmssd():
    """The report averages the sensor's 0 sentinel in; here it must be NaN."""

    parsed = features.parse_slash_array("60/0/62", zero_is_missing=True)
    assert np.isnan(parsed[1])
    assert np.allclose(parsed[[0, 2]], [60.0, 62.0])
    assert np.nanmean(parsed) == pytest.approx(61.0)


def test_transition_features_are_rates_not_counts():
    short = features._transition_features(np.array([1.0, 2.0, 1.0]), {1: "a", 2: "b"}, "p")
    doubled = features._transition_features(
        np.array([1.0, 2.0, 1.0, 2.0, 1.0]), {1: "a", 2: "b"}, "p"
    )
    # Same alternating pattern, different length -> same rate.
    assert short["p_trans_rate"] == pytest.approx(doubled["p_trans_rate"])


def test_summarize_handles_all_nan():
    out = features.summarize(np.array([np.nan, np.nan]), "x")
    assert all(np.isnan(v) for v in out.values())


def test_coefficient_of_variation_is_row_wise():
    frame = pd.DataFrame({"a_mean": [2.0, 4.0], "a_std": [1.0, 2.0]})
    out = features.add_coefficient_of_variation(frame)
    assert np.allclose(out["a_cv"], [0.5, 0.5])


# ------------------------------------------------------------------ metrics --
def test_roc_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    score = rng.random(200)
    assert metrics.roc_auc(y, score) == pytest.approx(roc_auc_score(y, score))


def test_roc_auc_handles_ties_and_degenerate_labels():
    assert metrics.roc_auc([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
    assert metrics.roc_auc([1, 1, 1], [0.1, 0.2, 0.3]) == 0.5


def test_metrics_bundle_reports_baseline_and_balanced_accuracy():
    y = np.array([0] * 26 + [1] * 7)
    all_cn = np.zeros(33)
    out = metrics.classification_metrics(y, all_cn, 0.5)
    assert out["accuracy"] == pytest.approx(26 / 33)
    assert out["all_cn_baseline_accuracy"] == pytest.approx(26 / 33)
    # The all-CN trap: high accuracy, chance-level balanced accuracy.
    assert out["balanced_accuracy"] == pytest.approx(0.5)
    assert out["recall_mci_dem"] == pytest.approx(0.0)


def test_jaccard_stability():
    assert metrics.jaccard_stability([[1, 2], [1, 2]])["mean_jaccard"] == pytest.approx(1.0)
    assert metrics.jaccard_stability([[1, 2], [3, 4]])["mean_jaccard"] == pytest.approx(0.0)


# ------------------------------------------------------------ preprocessing --
def test_smote_only_adds_minority_and_reaches_parity():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = np.array([0] * 45 + [1] * 15)
    X_out, y_out = preprocessing.smote_resample(X, y, kind="plain", seed=0)
    assert np.count_nonzero(y_out == 0) == 45          # majority untouched
    assert np.count_nonzero(y_out == 1) == 45          # minority raised to parity
    assert len(X_out) == len(y_out)


def test_smote_neighbour_search_is_scale_invariant():
    """Rescaling one column must not change which neighbours SMOTE picks."""

    rng = np.random.default_rng(1)
    X = np.column_stack([rng.normal(size=40), rng.normal(size=40)])
    y = np.array([0] * 30 + [1] * 10)

    blown_up = X.copy()
    blown_up[:, 0] *= 5000.0                     # calories-vs-fractions mismatch

    _, y_a = preprocessing.smote_resample(X, y, kind="borderline", seed=3)
    Xb, y_b = preprocessing.smote_resample(blown_up, y, kind="borderline", seed=3)
    assert np.count_nonzero(y_a == 1) == np.count_nonzero(y_b == 1)

    # Synthesised points must live on the rescaled axis, not the original one.
    assert np.abs(Xb[:, 0]).max() > 100.0


def test_smote_is_a_no_op_when_minority_is_degenerate():
    X = np.zeros((5, 3))
    y = np.array([0, 0, 0, 0, 1])                       # single minority point
    X_out, y_out = preprocessing.smote_resample(X, y, seed=0)
    assert len(y_out) == 5


def test_fold_preprocessor_statistics_come_from_train_only():
    train = np.array([[1.0], [2.0], [3.0]])
    holdout = np.array([[np.nan]])
    pre = preprocessing.FoldPreprocessor().fit(train, np.array([0, 1, 0]))
    # The imputed value is the TRAIN median (2.0), not anything from holdout.
    assert pre.transform(holdout)[0, 0] == pytest.approx(2.0)


def test_fold_preprocessor_drops_constant_columns():
    train = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    pre = preprocessing.FoldPreprocessor().fit(train, np.array([0, 1, 0]))
    assert pre.transform(train).shape[1] == 1


def test_transform_can_preserve_nan_for_native_tree_splits():
    train = np.array([[1.0], [2.0], [3.0]])
    pre = preprocessing.FoldPreprocessor().fit(train, np.array([0, 1, 0]))
    assert np.isnan(pre.transform(np.array([[np.nan]]), impute=False)[0, 0])


# ---------------------------------------------------------------- thresholds --
def test_youden_threshold_separates_a_clean_split():
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    threshold = metrics.youden_threshold(y, score)
    predicted = (score >= threshold).astype(int)
    assert np.array_equal(predicted, y)


def test_parse_args_survives_jupyter_kernel_argv():
    """Regression: base.ipynb runs this file inside a kernel via runpy, so
    sys.argv still carries '-f .../kernel-xxx.json'."""

    from Binary_Google_SOTA_DualTrack import run as runner

    args = runner.parse_args(["-f", "/root/.../kernel-0dbe8ec8.json"])
    assert args.mode in runner.MODE_SETTINGS
    assert args.track in ("wearable", "mmse_fusion", "both")


def test_env_vars_drive_the_notebook_run(monkeypatch):
    from Binary_Google_SOTA_DualTrack import run as runner

    monkeypatch.setenv("SOTA_MODE", "smoke")
    monkeypatch.setenv("SOTA_TRACK", "wearable")
    monkeypatch.setenv("SOTA_SMOTE", "none")
    args = runner.parse_args(["-f", "/root/kernel.json"])
    assert (args.mode, args.track, args.smote) == ("smoke", "wearable", "none")


def test_soft_voting_weights_match_the_report():
    from Binary_Google_SOTA_DualTrack.models import SOFT_VOTING_WEIGHTS

    assert SOFT_VOTING_WEIGHTS["gbt_leafwise"] == pytest.approx(0.40)   # LightGBM slot
    assert sum(SOFT_VOTING_WEIGHTS.values()) == pytest.approx(1.0)
