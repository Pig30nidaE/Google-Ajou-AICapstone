"""Contract tests for Binary_Google_SensorFM_Nested.

Pure wiring / leakage / geometry checks on tiny synthetic tensors -- NO real
training and NO real data is required (tests that need the repo Data/ or torch
skip themselves when unavailable).  Run with:

    python -m pytest SangHyo/Binary/Binary_Google_SensorFM_Nested/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from sensorfmnested import config as C                                  # noqa: E402
from sensorfmnested.fe_features import build_fe_features, fe_feature_names  # noqa: E402
from sensorfmnested.grids import (                                      # noqa: E402
    DayBank, build_day_bank, build_subject_grids, parse_series,
)
from sensorfmnested.nested_cv import (                                  # noqa: E402
    derive_seed, select_candidate, stratified_subject_folds,
)
from sensorfmnested.probes import LinearProbe, _ecdf_rank               # noqa: E402

DATA_ROOT = EXPERIMENT_ROOT.parents[2] / "Data"


# ------------------------------------------------------------ synthetic data -
def _series(values) -> str:
    return "/".join(str(v) for v in values)


def _make_frames(subjects=("a@x", "b@x"), n_days=3):
    """Tiny synthetic activity/sleep frames in the real CSV schema."""

    rng = np.random.default_rng(0)
    activity_rows, sleep_rows = [], []
    for sid in subjects:
        for day in range(n_days):
            date = pd.Timestamp(2020, 10, 10 + day, 4, 0, 0, tz="Asia/Seoul")
            met = np.round(rng.uniform(0.9, 4.0, C.MINUTES_PER_DAY), 1)
            classes = rng.integers(1, 6, C.ACTIVITY_CLASS_EPOCHS_PER_DAY)
            classes[:6] = 0  # half an hour of non-wear
            activity_rows.append(
                {
                    "EMAIL": sid,
                    "activity_day_start": date.isoformat(),
                    C.INTRADAY_COLUMNS["met_1min"]: _series(met),
                    C.INTRADAY_COLUMNS["activity_class_5min"]: _series(classes),
                }
            )
            bed = date + pd.Timedelta(hours=18, minutes=30)  # 22:30 local
            n_epochs = 60  # 5 hours of sleep
            stages = rng.integers(1, 5, n_epochs)
            hr = rng.integers(45, 70, n_epochs)
            hr[3] = 0  # missing epoch
            rmssd = rng.integers(15, 60, n_epochs)
            sleep_rows.append(
                {
                    "EMAIL": sid,
                    "sleep_bedtime_start": bed.isoformat(),
                    C.INTRADAY_COLUMNS["hypnogram_5min"]: _series(stages),
                    C.INTRADAY_COLUMNS["sleep_hr_5min"]: _series(hr),
                    C.INTRADAY_COLUMNS["sleep_rmssd_5min"]: _series(rmssd),
                }
            )
    return pd.DataFrame(activity_rows), pd.DataFrame(sleep_rows)


def _make_bank(subjects=("a@x", "b@x"), n_days=3) -> DayBank:
    activity, sleep = _make_frames(subjects, n_days)
    grids = build_subject_grids(activity, sleep)
    return build_day_bank(grids, sorted(subjects))


# ----------------------------------------------------------------- parsing ---
def test_parse_series_roundtrip_and_edge_cases():
    assert parse_series("1.2/0.9/1").tolist() == [1.2, 0.9, 1.0]
    assert parse_series(None).size == 0
    assert parse_series("").size == 0
    assert parse_series("nan").size == 0
    assert parse_series("1/2/3", expected=2).tolist() == [1.0, 2.0]


def test_grid_geometry_and_channel_semantics():
    bank = _make_bank()
    assert bank.values.shape[1:] == (C.MINUTES_PER_DAY, C.N_CHANNELS)
    met_i = C.CHANNELS.index("met")
    class_i = C.CHANNELS.index("act_class")
    # Non-wear (class 0) first 30 minutes: met and class both missing there.
    assert not bank.mask[0, :30, met_i].any()
    assert not bank.mask[0, :30, class_i].any()
    assert bank.mask[0, 31:, met_i].all()
    # Sleep placed at 22:30 local == minute 1110 of the 04:00 window.
    stage_channels = [C.CHANNELS.index(f"stage_{s}") for s in
                      ("deep", "light", "rem", "awake")]
    assert bank.mask[0, 1110, stage_channels].all()
    assert not bank.mask[0, 1000, stage_channels].any()
    # One-hot: exactly one stage active per observed sleep minute.
    assert bank.values[0, 1110, stage_channels].sum() == 1.0
    # HR epoch 3 was 0 -> minutes 1125..1129 missing on the HR channel.
    hr_i = C.CHANNELS.index("sleep_hr")
    assert not bank.mask[0, 1125:1130, hr_i].any()
    assert bank.mask[0, 1110:1115, hr_i].all()


def test_sleep_crossing_4am_boundary_spills_into_both_windows():
    activity, sleep = _make_frames(subjects=("a@x",), n_days=2)
    # A sleep starting 03:00 local on day 2: first hour belongs to day 1's
    # window (minutes 1380..1439), remainder to day 2's window (minute 0..).
    bed = pd.Timestamp(2020, 10, 11, 3, 0, 0, tz="Asia/Seoul")
    stages = [1] * 24  # 2 hours
    sleep = pd.concat([sleep, pd.DataFrame([{
        "EMAIL": "a@x",
        "sleep_bedtime_start": bed.isoformat(),
        C.INTRADAY_COLUMNS["hypnogram_5min"]: _series(stages),
        C.INTRADAY_COLUMNS["sleep_hr_5min"]: _series([55] * 24),
        C.INTRADAY_COLUMNS["sleep_rmssd_5min"]: _series([30] * 24),
    }])], ignore_index=True)
    grids = build_subject_grids(activity, sleep)
    grid = grids["a@x"]
    deep_i = C.CHANNELS.index("stage_deep")
    day1 = grid.values[0]  # window 2020-10-10 04:00 -> 10-11 04:00
    day2 = grid.values[1]
    assert grid.mask[0, 1380:1440, deep_i].all() and day1[1380:1440, deep_i].all()
    assert grid.mask[1, 0:60, deep_i].all() and day2[0:60, deep_i].all()


def test_grids_are_subject_local():
    """Rebuilding from a subject subset must reproduce identical arrays."""

    activity, sleep = _make_frames(subjects=("a@x", "b@x"))
    full = build_subject_grids(activity, sleep)
    only_a = build_subject_grids(
        activity[activity["EMAIL"] == "a@x"], sleep[sleep["EMAIL"] == "a@x"]
    )
    np.testing.assert_array_equal(full["a@x"].values, only_a["a@x"].values)
    np.testing.assert_array_equal(full["a@x"].mask, only_a["a@x"].mask)
    np.testing.assert_array_equal(full["a@x"].meta, only_a["a@x"].meta)


def test_fold_channel_stats_use_only_given_subjects():
    bank = _make_bank(subjects=("a@x", "b@x"))
    mean_a, std_a = bank.fold_channel_stats(np.array([0]))
    mean_ab, std_ab = bank.fold_channel_stats(np.array([0, 1]))
    met_i = C.CHANNELS.index("met")
    observed = bank.mask[bank.day_subject == 0][:, :, met_i]
    values = bank.values[bank.day_subject == 0][:, :, met_i]
    expected = values[observed].mean()
    assert abs(mean_a[met_i] - expected) < 1e-4
    assert not np.allclose(mean_a, mean_ab)  # adding a subject changes stats


def test_day_admission_drops_empty_days():
    activity, sleep = _make_frames(subjects=("a@x",), n_days=1)
    # A nearly-empty extra activity day: 20 observed minutes only.
    met = ["0"] * C.MINUTES_PER_DAY
    met[100:120] = ["1.5"] * 20
    classes = ["0"] * C.ACTIVITY_CLASS_EPOCHS_PER_DAY
    activity = pd.concat([activity, pd.DataFrame([{
        "EMAIL": "a@x",
        "activity_day_start": pd.Timestamp(2020, 11, 1, 4, tz="Asia/Seoul").isoformat(),
        C.INTRADAY_COLUMNS["met_1min"]: "/".join(met),
        C.INTRADAY_COLUMNS["activity_class_5min"]: "/".join(classes),
    }])], ignore_index=True)
    grids = build_subject_grids(activity, sleep)
    assert grids["a@x"].n_days == 1  # the junk day was dropped


# ---------------------------------------------------------------- features ---
def test_fe_features_shape_names_and_locality():
    bank = _make_bank()
    fe = build_fe_features(bank)
    assert list(fe.columns) == fe_feature_names()
    assert fe.shape == (2, C.N_CHANNELS * 21 * 2)
    for token in C.FORBIDDEN_FEATURE_TOKENS + C.FORBIDDEN_SUBSTRINGS:
        assert not any(token in c.lower() for c in fe.columns), token
    # Subject-locality: single-subject bank reproduces identical rows.
    bank_a = _make_bank(subjects=("a@x",))
    fe_a = build_fe_features(bank_a)
    np.testing.assert_allclose(
        fe.loc["a@x"].to_numpy(), fe_a.loc["a@x"].to_numpy(), rtol=0, atol=0
    )


def test_fe_cosinor_recovers_known_rhythm():
    """A clean 24h cosine must yield its amplitude/acrophase, IV near zero."""

    from sensorfmnested.fe_features import _day_channel_stats, FE_STATS

    t = np.arange(C.MINUTES_PER_DAY)
    values = 2.0 + 1.5 * np.cos(2 * np.pi * (t - 300) / C.MINUTES_PER_DAY)
    stats = _day_channel_stats(values.astype(np.float64), np.ones_like(t, dtype=bool))
    by_name = dict(zip(FE_STATS, stats))
    assert abs(by_name["cosinor_mesor"] - 2.0) < 1e-6
    assert abs(by_name["cosinor_amplitude"] - 1.5) < 1e-6
    phase = np.arctan2(by_name["cosinor_acro_sin"], by_name["cosinor_acro_cos"])
    assert abs(phase - 2 * np.pi * 300 / C.MINUTES_PER_DAY) < 1e-6


# ------------------------------------------------------------------ folds ----
def test_outer_folds_are_deterministic_and_partition():
    rng = np.random.default_rng(3)
    diag = np.array(["CN"] * 85 + ["MCI"] * 47 + ["Dem"] * 9, dtype=object)
    rng.shuffle(diag)
    folds_a = stratified_subject_folds(diag, 5, derive_seed(C.SEED, 1000, 0))
    folds_b = stratified_subject_folds(diag, 5, derive_seed(C.SEED, 1000, 0))
    for (tr_a, te_a), (tr_b, te_b) in zip(folds_a, folds_b):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(te_a, te_b)


def test_candidate_selection_prefers_simple_within_tolerance():
    candidates = list(C.CANDIDATES)
    scores = {"fe_paper_lr": 0.700, "sensorfm_lr": 0.703, "sensorfm_fe_blend": 0.7045}
    assert select_candidate(scores, candidates) == "fe_paper_lr"
    scores["sensorfm_lr"] = 0.75
    assert select_candidate(scores, candidates) == "sensorfm_lr"


# ------------------------------------------------------------------ probes ---
def test_linear_probe_caps_pca_and_scores_probabilities():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 200)))
    y = (rng.random(30) < 0.4).astype(int)
    probe = LinearProbe(C.ProbeConfig(50, 0.1), seed=0).fit(X, y)
    assert probe.effective_pca_k_ == 29  # min(50, 200, n-1)
    scores = probe.predict_score(X)
    assert scores.shape == (30,) and (scores >= 0).all() and (scores <= 1).all()


def test_ecdf_rank_is_monotone_and_bounded():
    reference = np.array([0.1, 0.2, 0.8])
    ranks = _ecdf_rank(reference, np.array([0.0, 0.15, 0.9]))
    assert (np.diff(ranks) > 0).all()
    assert (ranks > 0).all() and (ranks < 1).all()


# ------------------------------------------------------------- real data -----
@pytest.mark.skipif(not (DATA_ROOT / "1.Training").is_dir(),
                    reason="repo Data/ not present")
def test_real_label_contract():
    from sensorfmnested.data import load_labels

    labels = load_labels(DATA_ROOT, "train")
    assert len(labels) == 141
    assert int(labels["y"].sum()) == 56
