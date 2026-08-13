"""Static + dynamic leakage/wiring contracts for Binary_Google_CircadianNested.

Run locally:  python -m pytest tests/ -q   (from the experiment folder)
These tests use synthetic frames shaped like the real CSVs, so they run in
seconds and without the dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from circnested import config as C  # noqa: E402
from circnested.evaluation import pick_threshold_balanced, thresholded_metrics  # noqa: E402
from circnested.features import (  # noqa: E402
    assert_no_forbidden, mmse_features, select_view,
)
from circnested.nested_cv import (  # noqa: E402
    assert_fold_partition, select_candidate, stratified_subject_folds,
)


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
def _mmse_table(rows: dict) -> pd.DataFrame:
    table = pd.DataFrame(rows).set_index("_sid")
    return table


def test_mmse_zero_row_becomes_missing_not_lowest_score():
    base = {item: [2.0, 0.0] for item in C.MMSE_ITEMS}
    base["TOTAL"] = [30.0, 0.0]
    base["_sid"] = ["ok", "not_administered"]
    features = mmse_features(_mmse_table(base), zero_as_missing=True)
    assert features.loc["ok", "mmse_TOTAL"] == 30.0
    assert np.isnan(features.loc["not_administered", "mmse_TOTAL"])
    assert np.isnan(features.loc["not_administered", "mmse_recall"])


def test_mmse_unexpected_coding_fails_closed():
    base = {item: [2.0] for item in C.MMSE_ITEMS}
    base[C.MMSE_ITEMS[0]] = [7.0]
    base["TOTAL"] = [30.0]
    base["_sid"] = ["weird"]
    with pytest.raises(AssertionError):
        mmse_features(_mmse_table(base))


# ------------------------------------------------------------------ folds ----
def test_subject_folds_partition_and_stratify():
    rng = np.random.default_rng(0)
    diag = np.array(["CN"] * 85 + ["MCI"] * 47 + ["Dem"] * 9, dtype=object)
    rng.shuffle(diag)
    folds = stratified_subject_folds(diag, k=5, seed=1)
    assert_fold_partition(len(diag), folds)
    for _, test_index in folds:
        assert (diag[test_index] == "Dem").sum() >= 1


def test_fold_partition_detects_overlap():
    folds = [(np.array([0, 1, 2]), np.array([2, 3]))]
    with pytest.raises(AssertionError):
        assert_fold_partition(4, folds)


# -------------------------------------------------------------- selection ----
def test_selection_prefers_simpler_within_tolerance():
    candidates = [c for c in C.CANDIDATES if c.candidate_id in
                  ("lr_mmse_c001", "obl_fusion")]
    scores = {"lr_mmse_c001": 0.760, "obl_fusion": 0.7649}
    assert select_candidate(scores, candidates) == "lr_mmse_c001"
    scores = {"lr_mmse_c001": 0.750, "obl_fusion": 0.760}
    assert select_candidate(scores, candidates) == "obl_fusion"


def test_candidate_set_pairs_every_fusion_with_its_mmse_twin():
    ids = {c.candidate_id for c in C.CANDIDATES}
    for fusion, twin in (("lr_fusion_c001", "lr_mmse_c001"), ("gbt_fusion", "gbt_mmse"),
                         ("obl_fusion", "obl_mmse"), ("blend_fusion", "blend_mmse")):
        assert fusion in ids and twin in ids


# -------------------------------------------------------------- thresholds ---
def test_threshold_comes_from_training_side_scores_only():
    y_train = np.array([0, 0, 0, 1, 1])
    train_scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    threshold = pick_threshold_balanced(y_train, train_scores)
    assert 0.3 < threshold < 0.8
    metrics = thresholded_metrics(np.array([0, 1]), np.array([0.2, 0.85]), threshold)
    assert metrics["tp"] == 1 and metrics["tn"] == 1


# ------------------------------------------------- subject-local features ----
def test_features_are_subject_local(tmp_path, monkeypatch):
    """Building features for a subset of subjects must reproduce the full-build
    rows bit-identically (proves no cross-subject statistic exists)."""

    from circnested import features as F

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
    """Static order check: in run.py, the freeze write must precede the
    validation label load."""

    source = (EXPERIMENT_ROOT / "run.py").read_text(encoding="utf-8")
    freeze_at = source.index("VALIDATION_PREDICTIONS_FROZEN.json")
    labels_at = source.index('load_labels(data_root, "val")')
    assert freeze_at < labels_at


# --------------------------------------------------------- notebook launch --
def _load_run_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_bgcn_run_under_test", EXPERIMENT_ROOT / "run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_notebook_kernel_argv_is_ignored(monkeypatch):
    """base.ipynb runs this file with runpy inside an IPython kernel, which
    leaves the kernel's own ``-f .../kernel-<uuid>.json`` in sys.argv."""

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
    monkeypatch.delenv("BGCN_ARGS", raising=False)
    assert run_module._parse_args().profile == "default"

    monkeypatch.setenv("BGCN_ARGS", "--profile smoke")
    assert run_module._parse_args().profile == "smoke"


def test_shell_argv_is_still_honored(monkeypatch):
    run_module = _load_run_module()
    monkeypatch.delitem(sys.modules, "IPython", raising=False)
    monkeypatch.setattr(sys, "argv", ["run.py", "--profile", "max"])
    monkeypatch.delenv("BGCN_ARGS", raising=False)
    assert run_module._parse_args().profile == "max"


def test_run_py_has_no_silent_ydf_fallback():
    source = (EXPERIMENT_ROOT / "circnested" / "modeling.py").read_text(encoding="utf-8")
    assert "fallback is forbidden" in source
    assert "HistGradientBoosting" not in source
