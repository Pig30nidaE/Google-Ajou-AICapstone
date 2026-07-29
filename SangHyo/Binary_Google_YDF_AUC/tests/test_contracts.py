"""Fail-closed tests for the Google YDF ROC-AUC experiment."""

from __future__ import annotations

import importlib.util
import inspect
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from SangHyo.Binary_Google_YDF_AUC import data, features, models, selection


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_mmse_source_allowlist_excludes_diagnosis_admin_and_identifiers() -> None:
    selected = set(data.MMSE_ALLOWED_SOURCE_COLUMNS)
    assert selected == {
        data.PERSON_KEY,
        "TOTAL",
        *data.MMSE_ITEMS,
    }
    assert selected.isdisjoint(data.MMSE_FORBIDDEN_SOURCE_COLUMNS)
    assert {"DIAG_NM", "DIAG_SEQ", "DOCTOR_NM", "MMSE_NUM", "MMSE_KIND"}.isdisjoint(
        selected
    )


@pytest.mark.parametrize(
    "selected",
    [
        data.ACTIVITY_ALLOWED_SOURCE_COLUMNS,
        data.SLEEP_ALLOWED_SOURCE_COLUMNS,
    ],
)
def test_wearable_source_allowlists_have_no_label_columns(
    selected: tuple[str, ...],
) -> None:
    upper = {name.upper() for name in selected}
    assert {"DIAG_NM", "DIAG_SEQ", "LABEL", "TARGET", "SAMPLE_EMAIL"}.isdisjoint(
        upper
    )
    assert "EMAIL" in upper


def test_allowlist_reader_passes_exact_usecols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.csv"
    source.write_text("placeholder", encoding="utf-8")
    seen: list[tuple[str, ...]] = []

    def fake_read_csv(
        _path: Path,
        *,
        encoding: str,
        usecols: list[str],
        low_memory: bool,
    ) -> pd.DataFrame:
        assert encoding == "utf-8-sig"
        assert low_memory is False
        seen.append(tuple(usecols))
        return pd.DataFrame({name: [1] for name in usecols})

    monkeypatch.setattr(data.pd, "read_csv", fake_read_csv)
    audit = data.AccessAudit()
    result = data._read_csv_allowlist(  # type: ignore[attr-defined]
        source,
        ("safe_a", "safe_b"),
        audit=audit,
        purpose="unit contract",
    )
    assert seen == [("safe_a", "safe_b")]
    assert list(result.columns) == ["safe_a", "safe_b"]
    assert audit.to_dict()["events"][0]["selected_columns"] == [
        "safe_a",
        "safe_b",
    ]


@pytest.mark.parametrize(
    "name",
    [
        "DIAG_NM",
        "diag_score",
        "doctor_nm",
        "sample_email",
        "subject_id",
        "target",
        "label",
        "w_n_activity_days",
        "w_n_sleep_days",
        "activity_observation_count",
        "activity_non_wear",
        "sleep_coverage",
        "mmse_missing_fraction",
    ],
)
def test_forbidden_direct_leakage_and_collection_proxy_names_fail_closed(
    name: str,
) -> None:
    with pytest.raises(data.LeakageContractError):
        features.assert_feature_names(("mmse__total", name))


def _synthetic_wearable_sources() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    subjects = ["anonymous_a", "anonymous_b", "anonymous_c"]
    activity_rows: list[dict[str, object]] = []
    sleep_rows: list[dict[str, object]] = []
    for subject_index, subject_id in enumerate(subjects):
        for repeat in range(2):
            activity: dict[str, object] = {"EMAIL": subject_id}
            for column_index, column in enumerate(
                data.ACTIVITY_ALLOWED_SOURCE_COLUMNS[1:], start=1
            ):
                activity[column] = float(
                    10 + subject_index + repeat + column_index
                )
            activity_rows.append(activity)

            sleep: dict[str, object] = {"EMAIL": subject_id}
            for column_index, column in enumerate(
                data.SLEEP_ALLOWED_SOURCE_COLUMNS[1:], start=1
            ):
                if column == "sleep_bedtime_start":
                    sleep[column] = f"2025-01-0{repeat + 1}T22:00:00+09:00"
                elif column == "sleep_bedtime_end":
                    sleep[column] = f"2025-01-0{repeat + 2}T06:00:00+09:00"
                elif column == "sleep_midpoint_time":
                    sleep[column] = 14_400.0
                else:
                    sleep[column] = float(
                        20 + subject_index + repeat + column_index
                    )
            sleep_rows.append(sleep)
    return pd.DataFrame(activity_rows), pd.DataFrame(sleep_rows), subjects


def test_built_wearable_bank_excludes_day_count_and_observation_features() -> None:
    activity, sleep, subjects = _synthetic_wearable_sources()
    built = features.build_wearable_features(activity, sleep, subjects)
    assert built.shape == (len(subjects), features.EXPECTED_WEARABLE_FEATURES)
    assert "activity_non_wear" not in data.ACTIVITY_ALLOWED_SOURCE_COLUMNS
    assert not [name for name in built.columns if "non_wear" in str(name)]
    forbidden = re.compile(
        r"(^|[^a-z0-9])(count|coverage|days|missing|observation)"
        r"([^a-z0-9]|$)"
    )
    assert not [
        name for name in map(str.lower, built.columns) if forbidden.search(name)
    ]
    features.assert_feature_names(tuple(map(str, built.columns)))


def test_repeated_subject_folds_are_disjoint_and_cover_each_subject_once() -> None:
    y = np.asarray([0] * 30 + [1] * 20, dtype=np.int64)
    folds = selection.make_repeated_folds(y, repeats=3, seed=812)
    assert len(folds) == 15
    all_indices = set(range(len(y)))
    for repeat in range(3):
        repeat_folds = [fold for fold in folds if fold.repeat == repeat]
        heldout: list[int] = []
        for fold in repeat_folds:
            train = set(map(int, fold.train_indices))
            test = set(map(int, fold.test_indices))
            assert train.isdisjoint(test)
            assert train | test == all_indices
            assert set(np.unique(y[fold.train_indices])) == {0, 1}
            assert set(np.unique(y[fold.test_indices])) == {0, 1}
            heldout.extend(test)
        assert sorted(heldout) == list(range(len(y)))


def test_feature_selection_is_unchanged_by_heldout_rows_and_labels() -> None:
    rng = np.random.default_rng(511)
    y = np.asarray([0] * 30 + [1] * 20, dtype=np.int64)
    X = rng.normal(size=(len(y), 12))
    X[:, 0] = y + rng.normal(scale=0.1, size=len(y))
    fold = selection.make_repeated_folds(y, repeats=1, seed=31)[0]
    kwargs = {
        "view_columns": np.arange(X.shape[1]),
        "top_k": 5,
        "corr_threshold": 0.95,
    }
    before = selection.select_fold_columns(
        X[fold.train_indices],
        y[fold.train_indices],
        **kwargs,
    )

    mutated_X = X.copy()
    mutated_y = y.copy()
    mutated_X[fold.test_indices] = rng.normal(
        loc=1_000.0,
        scale=100.0,
        size=(len(fold.test_indices), X.shape[1]),
    )
    mutated_y[fold.test_indices] = 1 - mutated_y[fold.test_indices]
    after = selection.select_fold_columns(
        mutated_X[fold.train_indices],
        mutated_y[fold.train_indices],
        **kwargs,
    )
    np.testing.assert_array_equal(before, after)


def test_candidate_bank_contains_only_google_ydf_families() -> None:
    bank = selection.candidate_bank()
    assert bank
    assert {candidate.family for candidate in bank} == set(models.YDF_FAMILIES)
    assert all(candidate.family in models.YDF_FAMILIES for candidate in bank)
    source = inspect.getsource(models)
    assert re.search(r"(?m)^\s*(?:from|import)\s+sklearn(?:\.|\s|$)", source) is None
    assert models.ENGINE_NAME == "google_ydf"


def test_missing_ydf_fails_instead_of_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models.importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(ModuleNotFoundError, match="No sklearn.*fallback"):
        models.require_ydf()


def test_sparse_oblique_rejection_is_fail_closed_and_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_calls: list[dict[str, object]] = []

    class RejectingObliqueLearner:
        def __init__(self, **kwargs: object) -> None:
            constructor_calls.append(dict(kwargs))
            raise TypeError("simulated unsupported sparse-oblique runtime")

    fake_ydf = SimpleNamespace(
        GradientBoostedTreesLearner=RejectingObliqueLearner,
        RandomForestLearner=object,
        load_model=lambda _path: None,
    )
    monkeypatch.setattr(models, "require_ydf", lambda: fake_ydf)
    params = next(
        candidate.params
        for candidate in selection.profile_spec("smoke").candidates
        if candidate.family == "sparse_oblique_gbt"
    )
    adapter = models.YDFBinaryModel(
        "sparse_oblique_gbt",
        params,
        seed=7,
        num_threads=1,
    )
    with pytest.raises(models.ObliqueContractError, match="downgrade is forbidden"):
        adapter._build_learner(np.asarray([0, 0, 1, 1]))  # type: ignore[attr-defined]
    assert len(constructor_calls) == 1
    assert constructor_calls[0]["split_axis"] == "SPARSE_OBLIQUE"


def test_reference_ecdf_never_uses_heldout_batch_ranking() -> None:
    ecdf = selection.ReferenceECDF().fit(np.asarray([0.0, 10.0]))
    alone = ecdf.transform(np.asarray([5.0]))[0]
    with_extremes = ecdf.transform(np.asarray([5.0, -1_000.0, 1_000.0]))[0]
    assert alone == pytest.approx(0.5)
    assert with_extremes == pytest.approx(alone)
    restored = selection.ReferenceECDF.from_dict(ecdf.to_dict())
    np.testing.assert_allclose(
        restored.transform(np.asarray([-1.0, 5.0, 11.0])),
        [0.0, 0.5, 1.0],
    )


def test_policy_winners_are_selected_independently_in_each_score_space() -> None:
    y = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    raw = {
        "axis": np.asarray([0.0, 0.1, 0.2, 0.7, 0.8, 0.9]),
        "oblique": np.asarray([0.0, 0.2, 0.8, 0.1, 0.7, 0.9]),
        "rf": np.asarray([0.0, 0.7, 0.8, 0.1, 0.2, 0.9]),
    }
    ecdf = {
        "axis": raw["rf"],
        "oblique": raw["axis"],
        "rf": raw["oblique"],
    }
    _chosen, policies = selection.build_and_select_policies(
        raw,
        ecdf,
        {
            "axis": "axis_gbt",
            "oblique": "sparse_oblique_gbt",
            "rf": "rf",
        },
        y,
        blend_draws=32,
        seed=3,
    )
    by_name = {policy["name"]: policy for policy in policies}
    assert by_name["equal_top2_raw"]["components"] != by_name[
        "equal_top2_ecdf"
    ]["components"]


@pytest.mark.parametrize("profile", ["smoke", "default", "max"])
def test_required_profiles_exist(profile: str) -> None:
    spec = selection.profile_spec(profile)
    assert spec.name == profile
    assert spec.candidates
    assert all(candidate.family in models.YDF_FAMILIES for candidate in spec.candidates)
    assert spec.reportable is (profile != "smoke")


def test_run_help_exposes_required_stages_and_profiles() -> None:
    run_file = EXPERIMENT_ROOT / "run.py"
    if not run_file.is_file():
        pytest.skip("run.py is still being assembled")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    completed = subprocess.run(
        [sys.executable, str(run_file), "--help"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    help_text = (completed.stdout + completed.stderr).lower()
    assert "--stage" in help_text
    for stage in ("inspect", "train", "all"):
        assert re.search(rf"\b{stage}\b", help_text)
    assert "--profile" in help_text
    for profile in ("smoke", "default", "max"):
        assert re.search(rf"\b{profile}\b", help_text)
    for option in (
        "--historical-eval",
        "--num-threads",
        "--skip-install",
    ):
        assert option in help_text


def test_notebook_arguments_and_extended_cli_contract() -> None:
    from SangHyo.Binary_Google_YDF_AUC import run

    assert run.notebook_argv({}) == [
        "--stage",
        "all",
        "--profile",
        "default",
    ]
    supplied = (
        "--stage all --profile max --historical-eval "
        "--num-threads 6 --skip-install"
    )
    argv = run.notebook_argv({"BGYA_ARGS": supplied})
    assert argv == supplied.split()
    parsed = run.build_parser().parse_args(argv)
    assert parsed.stage == "all"
    assert parsed.profile == "max"
    assert parsed.historical_eval is True
    assert parsed.num_threads == 6
    assert parsed.skip_install is True
    assert run.REQUIREMENTS_FILE.name == "requirements_colab.in"
    assert run.REQUIREMENTS_FILE.is_file()


@pytest.mark.skipif(
    importlib.util.find_spec("ydf") is None,
    reason="optional real Google YDF runtime is not installed",
)
def test_real_google_ydf_checkpoint_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(920)
    y = np.asarray([0] * 12 + [1] * 12, dtype=np.int64)
    X = rng.normal(size=(len(y), 4)).astype(np.float32)
    X[:, 0] += 1.5 * y
    params = {
        "num_trees": 20,
        "max_depth": 3,
        "min_examples": 2,
        "shrinkage": 0.08,
        "subsample": 0.8,
        "num_candidate_attributes_ratio": 0.75,
        "l2_regularization": 0.5,
    }
    original = models.YDFBinaryModel(
        "axis_gbt",
        params,
        seed=23,
        num_threads=1,
    ).fit(X, y, ("signal", "feature_1", "feature_2", "feature_3"))
    before = original.predict_score(X)
    checkpoint = original.save(tmp_path / "checkpoint")
    restored = models.YDFBinaryModel.load(checkpoint)
    after = restored.predict_score(X)
    np.testing.assert_allclose(after, before, rtol=0.0, atol=1e-12)
    assert restored.manifest()["engine"] == "google_ydf"
    assert restored.manifest()["fallback_permitted"] is False


@pytest.mark.skipif(
    importlib.util.find_spec("ydf") is None,
    reason="optional real Google YDF runtime is not installed",
)
def test_real_sparse_oblique_checkpoint_keeps_runtime_evidence(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(921)
    y = np.asarray([0] * 12 + [1] * 12, dtype=np.int64)
    X = rng.normal(size=(len(y), 4)).astype(np.float32)
    X[:, 0] += 1.5 * y
    params = next(
        candidate.params
        for candidate in selection.profile_spec("smoke").candidates
        if candidate.family == "sparse_oblique_gbt"
    )
    original = models.YDFBinaryModel(
        "sparse_oblique_gbt",
        params,
        seed=29,
        num_threads=1,
    ).fit(X, y, ("signal", "feature_1", "feature_2", "feature_3"))
    before = original.predict_score(X)
    checkpoint = original.save(tmp_path / "oblique_checkpoint")
    restored = models.YDFBinaryModel.load(checkpoint)
    after = restored.predict_score(X)
    np.testing.assert_allclose(after, before, rtol=0.0, atol=1e-12)
    evidence = restored.manifest()["oblique_contract"]
    assert evidence["runtime_mapping_checked"] is True
    assert evidence["runtime_split_axis"] == "SPARSE_OBLIQUE"
