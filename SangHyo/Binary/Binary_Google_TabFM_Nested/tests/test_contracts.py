"""Static + dynamic contracts for Binary_Google_TabFM_Nested.

Run locally:  python -m pytest tests/ -q   (from the experiment folder)
No real TabFM install is needed: adapter tests inject a fake ``tabfm`` module,
and the pipeline stub path is exercised separately.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from tabfmnested import config as C  # noqa: E402
from tabfmnested import modeling as M  # noqa: E402
from tabfmnested.evaluation import pick_threshold_balanced, thresholded_metrics  # noqa: E402
from tabfmnested.features import (  # noqa: E402
    assert_no_forbidden, mmse_features, select_view,
)
from tabfmnested.nested_cv import (  # noqa: E402
    assert_fold_partition, select_candidate, stratified_subject_folds,
)


@pytest.fixture(autouse=True)
def _reset_shared_model(monkeypatch):
    monkeypatch.setattr(M, "_SHARED_MODEL", None)
    monkeypatch.setattr(M, "_SHARED_MODEL_INFO", {})
    monkeypatch.setattr(M, "_STUB_ACTIVE", False)


# ------------------------------------------------------------ TabFM adapter --
class _FakeClassifierFull:
    """Signature accepts the desired kwargs; classes_ deliberately reversed."""

    def __init__(self, model=None, max_num_rows=100, random_state=None):
        self.model = model
        self.max_num_rows = max_num_rows
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.array([1, 0])  # reversed on purpose
        self._mean1 = np.asarray(X)[np.asarray(y) == 1].mean(axis=0)
        return self

    def predict_proba(self, X):
        d = np.linalg.norm(np.asarray(X) - self._mean1, axis=1)
        p1 = 1.0 / (1.0 + d)
        return np.column_stack([p1, 1.0 - p1])  # column 0 = class 1 (reversed)


class _FakeClassifierBare:
    """Signature accepts only the model; desired kwargs must be dropped."""

    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return np.tile([0.4, 0.6], (len(X), 1))


def _install_fake_tabfm(monkeypatch, classifier_class):
    import importlib.machinery

    fake = types.ModuleType("tabfm")
    fake.__spec__ = importlib.machinery.ModuleSpec("tabfm", loader=None)
    fake.TabFMClassifier = classifier_class
    loader = types.SimpleNamespace(load=lambda: "fake-checkpoint")
    fake.tabfm_v1_0_0_pytorch = loader
    monkeypatch.setitem(sys.modules, "tabfm", fake)
    return fake


def _toy_frame(n=30, k=4, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, k)), columns=[f"f{i}" for i in range(k)])
    y = (rng.random(n) < 0.5).astype(int)
    y[:2] = [0, 1]  # both classes guaranteed
    return X, y


def test_tabfm_adapter_passes_supported_kwargs_and_reads_classes(monkeypatch):
    _install_fake_tabfm(monkeypatch, _FakeClassifierFull)
    M.load_shared_model()
    X, y = _toy_frame()
    model = M.TabFMModel(seed=7).fit(X, y)
    assert model.kwargs_accepted_["max_num_rows"] == 256   # default 100 must be raised
    assert model.kwargs_accepted_["random_state"] == 7
    assert model.classifier_.max_num_rows == 256
    scores = model.predict_score(X)
    # classes_ is [1, 0]: the positive column is column 0, found via classes_.
    assert model.positive_column_source_ == "classes_"
    assert np.allclose(scores, model.classifier_.predict_proba(X.to_numpy())[:, 0])


def test_tabfm_adapter_drops_unsupported_kwargs(monkeypatch):
    _install_fake_tabfm(monkeypatch, _FakeClassifierBare)
    M.load_shared_model()
    X, y = _toy_frame()
    model = M.TabFMModel(seed=7).fit(X, y)
    assert model.kwargs_accepted_ == {}
    assert set(model.kwargs_dropped_) == {"max_num_rows", "random_state"}
    assert model.predict_score(X).shape == (len(X),)


def test_tabfm_adapter_imputes_nan_fold_locally(monkeypatch):
    _install_fake_tabfm(monkeypatch, _FakeClassifierBare)
    M.load_shared_model()
    X, y = _toy_frame()
    X.iloc[0, 0] = np.nan

    seen = {}
    original_fit = _FakeClassifierBare.fit

    def spy_fit(self, values, target):
        seen["values"] = np.asarray(values)
        return original_fit(self, values, target)

    monkeypatch.setattr(_FakeClassifierBare, "fit", spy_fit)
    M.TabFMModel(seed=0).fit(X, y)
    assert np.isfinite(seen["values"]).all()
    assert seen["values"][0, 0] == pytest.approx(np.nanmedian(X.to_numpy()[:, 0]))


def test_shared_model_is_loaded_once(monkeypatch):
    fake = _install_fake_tabfm(monkeypatch, _FakeClassifierBare)
    calls = []
    fake.tabfm_v1_0_0_pytorch = types.SimpleNamespace(
        load=lambda: calls.append(1) or "ckpt"
    )
    M.load_shared_model()
    M.load_shared_model()
    assert len(calls) == 1


def test_missing_tabfm_fails_closed(monkeypatch):
    monkeypatch.setitem(sys.modules, "tabfm", None)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(ModuleNotFoundError):
        M.require_tabfm()


# ------------------------------------------------------------- wiring stub ---
def test_wiring_stub_is_labelled_not_google():
    info = M.activate_wiring_stub()
    assert "NOT_GOOGLE" in info["engine"]
    X, y = _toy_frame()
    model = M.TabFMModel(seed=0).fit(X, y)
    scores = model.predict_score(X)
    assert scores.shape == (len(X),) and np.isfinite(scores).all()


def test_stub_refused_outside_smoke():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_bgtf_run_under_test", EXPERIMENT_ROOT / "run.py"
    )
    run_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_module)
    with pytest.raises(RuntimeError, match="smoke"):
        run_module._ensure_tabfm(stub_requested=True, profile_name="default")
    run_module._ensure_tabfm(stub_requested=True, profile_name="smoke")  # no raise


# ------------------------------------------------------------- feature guards
def test_forbidden_columns_rejected():
    for name in ("DIAG_NM", "mmse_diag_like", "wd_n_days_activity", "EMAIL",
                 "activity_non_wear", "wd_cover_sleep"):
        with pytest.raises(AssertionError):
            assert_no_forbidden(["mmse_TOTAL", name])


def test_views_contain_no_forbidden_names():
    for view_name, columns in C.VIEWS.items():
        assert_no_forbidden(columns)
        assert len(columns) == len(set(columns)), view_name


def test_view_selection_fails_closed_on_missing_columns():
    frame = pd.DataFrame({"mmse_TOTAL": [1.0], "wi_met_IS": [0.5]})
    with pytest.raises(KeyError):
        select_view(frame, ("mmse_TOTAL", "definitely_absent"))


# ------------------------------------------------------------------ MMSE ----
def test_mmse_zero_row_becomes_missing_not_lowest_score():
    base = {item: [2.0, 0.0] for item in C.MMSE_ITEMS}
    base["TOTAL"] = [30.0, 0.0]
    base["_sid"] = ["ok", "not_administered"]
    features = mmse_features(pd.DataFrame(base).set_index("_sid"), zero_as_missing=True)
    assert features.loc["ok", "mmse_TOTAL"] == 30.0
    assert np.isnan(features.loc["not_administered", "mmse_TOTAL"])


# ------------------------------------------------------------------ folds ----
def test_subject_folds_partition_and_stratify():
    rng = np.random.default_rng(0)
    diag = np.array(["CN"] * 85 + ["MCI"] * 47 + ["Dem"] * 9, dtype=object)
    rng.shuffle(diag)
    folds = stratified_subject_folds(diag, k=5, seed=1)
    assert_fold_partition(len(diag), folds)
    for _, test_index in folds:
        assert (diag[test_index] == "Dem").sum() >= 1


def test_fold_construction_matches_circadian_nested_run():
    """Fold parity: same derive_seed chain + StratifiedKFold as the reference
    run, so the anchor track is comparable across the two experiments."""

    from tabfmnested.nested_cv import derive_seed
    assert C.SEED == 20260813
    assert derive_seed(C.SEED, 1000, 0) == derive_seed(20260813, 1000, 0)


# -------------------------------------------------------------- selection ----
def test_selection_prefers_anchor_within_tolerance():
    scores = {"lr_mmse_c001": 0.760, "tabfm_mmse": 0.7649, "blend_tabfm_lr": 0.7648}
    assert select_candidate(scores, list(C.CANDIDATES)) == "lr_mmse_c001"
    scores = {"lr_mmse_c001": 0.750, "tabfm_mmse": 0.760, "blend_tabfm_lr": 0.755}
    assert select_candidate(scores, list(C.CANDIDATES)) == "tabfm_mmse"


# -------------------------------------------------------------- thresholds ---
def test_threshold_comes_from_training_side_scores_only():
    y_train = np.array([0, 0, 0, 1, 1])
    train_scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    threshold = pick_threshold_balanced(y_train, train_scores)
    assert 0.3 < threshold < 0.8
    metrics = thresholded_metrics(np.array([0, 1]), np.array([0.2, 0.85]), threshold)
    assert metrics["tp"] == 1 and metrics["tn"] == 1


# ------------------------------------------------- subject-local features ----
def test_features_are_subject_local(monkeypatch):
    from tabfmnested import features as F

    rng = np.random.default_rng(7)

    def _make_raw(subjects: list[str]):
        met = "/".join(str(round(v, 2)) for v in rng.uniform(0.9, 4.0, C.MET_MINUTES_PER_DAY))
        activity_rows, sleep_rows = [], []
        for sid in subjects:
            for day in range(4):
                activity_rows.append(
                    {
                        "EMAIL": sid,
                        "activity_day_start": f"2020-10-{19 + day}T04:00:00+09:00",
                        C.INTRADAY_COLUMNS["met_1min"]: met,
                        C.INTRADAY_COLUMNS["activity_class_5min"]: "...",
                    }
                )
                hyp = "/".join(str(int(v)) for v in rng.integers(1, 5, 60))
                hr = "/".join(str(int(v)) for v in rng.integers(45, 80, 60))
                sleep_rows.append(
                    {
                        "EMAIL": sid,
                        "sleep_bedtime_start": f"2020-10-{19 + day}T21:30:00+09:00",
                        "sleep_bedtime_end": f"2020-10-{20 + day}T05:30:00+09:00",
                        C.INTRADAY_COLUMNS["hypnogram_5min"]: hyp,
                        C.INTRADAY_COLUMNS["sleep_hr_5min"]: hr,
                        C.INTRADAY_COLUMNS["sleep_rmssd_5min"]: hr,
                    }
                )
        mmse_rows = []
        for sid in subjects:
            row = {"SAMPLE_EMAIL": sid, "TOTAL": 28.0}
            row.update({item: 2.0 for item in C.MMSE_ITEMS})
            mmse_rows.append(row)
        return (pd.DataFrame(activity_rows), pd.DataFrame(sleep_rows),
                pd.DataFrame(mmse_rows))

    subjects = [f"s{i}@x" for i in range(6)]
    activity, sleep, mmse = _make_raw(subjects)

    def _fake_read_csv(path):
        text = str(path)
        if "activity" in text:
            return activity[activity["EMAIL"].isin(current)]
        if "sleep" in text:
            return sleep[sleep["EMAIL"].isin(current)]
        return mmse[mmse["SAMPLE_EMAIL"].isin(current)]

    monkeypatch.setattr(F, "read_csv", _fake_read_csv)

    current = subjects
    full = F.build_split_features("unused", "train")
    current = subjects[:3]
    subset = F.build_split_features("unused", "train")

    pd.testing.assert_frame_equal(full.loc[subset.index], subset)


# -------------------------------------------------------- freeze ordering ----
def test_run_py_freezes_before_opening_validation_labels():
    source = (EXPERIMENT_ROOT / "run.py").read_text(encoding="utf-8")
    freeze_at = source.index("VALIDATION_PREDICTIONS_FROZEN.json")
    labels_at = source.index('load_labels(data_root, "val")')
    assert freeze_at < labels_at


# ------------------------------------------------------------- progress -----
def test_progress_heartbeat_and_fold_callback_fire(monkeypatch):
    """A long run must prove liveness *inside* a fold, not only per repeat.

    Regression guard for the 2026-08-14 report that the TabFM run produced no
    output for an hour: logging used to fire once per outer repeat (10% of the
    run), which is indistinguishable from a hang.
    """

    from tabfmnested import nested_cv as N

    monkeypatch.setattr(N, "HEARTBEAT_SECONDS", 0.0)  # log every step

    rng = np.random.default_rng(0)
    n = 60
    diag = np.array(["CN"] * 36 + ["MCI"] * 20 + ["Dem"] * 4, dtype=object)
    rng.shuffle(diag)
    y = np.isin(diag, ("MCI", "Dem")).astype(int)
    features = pd.DataFrame(
        rng.normal(size=(n, len(C.VIEWS["mmse"]))), columns=list(C.VIEWS["mmse"])
    )

    profile = C.Profile(
        name="t", outer_k=3, outer_repeats=1, inner_k=2, inner_repeats=1,
        n_bootstrap=10, candidate_ids=("lr_mmse_c001",),
    )
    anchor = [c for c in C.CANDIDATES if c.candidate_id == "lr_mmse_c001"]

    lines: list[str] = []
    folds: list[dict] = []
    N.run_repeated_nested_cv(
        features, y, diag, anchor, [], profile, seed=1,
        log=lines.append, on_fold=folds.append,
    )

    assert sum("[progress]" in line for line in lines) >= 3, "no intra-fold heartbeat"
    assert sum("[nested-cv] fold" in line for line in lines) == 3, "no per-fold line"
    assert [f["folds_done"] for f in folds] == [1, 2, 3]
    assert all(f["total_folds"] == 3 and "eta_minutes" in f for f in folds)


# --------------------------------------------------------- notebook launch --
def _load_run_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_bgtf_run_argv_test", EXPERIMENT_ROOT / "run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_notebook_kernel_argv_is_ignored(monkeypatch):
    run_module = _load_run_module()

    class _FakeIPython:
        @staticmethod
        def get_ipython():
            return object()

    monkeypatch.setitem(sys.modules, "IPython", _FakeIPython)
    monkeypatch.setattr(
        sys, "argv",
        ["/usr/local/lib/python3.12/dist-packages/colab_kernel_launcher.py",
         "-f", "/root/.local/share/jupyter/runtime/kernel-abc.json"],
    )
    monkeypatch.delenv("BGTF_ARGS", raising=False)
    assert run_module._parse_args().profile == "default"
    monkeypatch.setenv("BGTF_ARGS", "--profile quick")
    assert run_module._parse_args().profile == "quick"


def test_shell_argv_is_still_honored(monkeypatch):
    run_module = _load_run_module()
    monkeypatch.delitem(sys.modules, "IPython", raising=False)
    monkeypatch.setattr(sys, "argv", ["run.py", "--profile", "max"])
    monkeypatch.delenv("BGTF_ARGS", raising=False)
    assert run_module._parse_args().profile == "max"
