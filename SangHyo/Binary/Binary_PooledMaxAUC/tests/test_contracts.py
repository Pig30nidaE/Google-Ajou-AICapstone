"""Direct-leakage and wiring contracts.

These tests exist because this experiment deliberately permits *selection*
optimism. That trade is only defensible if the *direct* leakage boundary is
mechanically verified, so the boundary gets tests and the optimism gets
disclosure.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from SangHyo.Binary.Binary_PooledMaxAUC import config as config_module, engine, leakage, models, pipeline
from SangHyo.Binary.Binary_PooledMaxAUC import run

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# leakage guards
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name",
    [
        "DIAG_NM",
        "DIAG_SEQ",
        "DOCTOR_NM",
        "MMSE_NUM",
        "MMSE_KIND",
        "EMAIL",
        "SAMPLE_EMAIL",
        "label",
        "y",
        "diagnosis_code",
        "wear__n_days__mean",
        "wear__coverage__mean",
        "wear__non_wear__mean",
    ],
)
def test_forbidden_feature_names_are_rejected(name):
    with pytest.raises(leakage.LeakageError):
        leakage.assert_no_forbidden_features([name], context="test")


def test_legitimate_feature_names_pass():
    good = [
        "mmse__total",
        "mmse__domain_recall",
        "mmse__item_q13_2",
        "wear__slp_deep_ratio__cv",
        "wear__act_steps__trend_per_week",
        "wear__slp_bedtime_hour__circular_sd",
        "x__recall_x_slp_deep_ratio_cv",
    ]
    leakage.assert_no_forbidden_features(good, context="test")


def test_fold_overlap_is_rejected():
    with pytest.raises(leakage.LeakageError):
        leakage.assert_fold_disjoint(["a", "b"], ["b", "c"], context="test")


def test_screening_must_be_fold_local():
    """The exact V41-style mistake: screening on more rows than the training fold."""

    leakage.assert_screening_is_train_local(100, 100, context="ok")
    with pytest.raises(leakage.LeakageError, match="fold-local"):
        leakage.assert_screening_is_train_local(174, 139, context="global screening")


def test_non_finite_scores_are_rejected():
    with pytest.raises(leakage.LeakageError):
        leakage.assert_finite_scores(np.array([0.1, np.nan]), context="test")


# --------------------------------------------------------------------------- #
# split plan
# --------------------------------------------------------------------------- #
def _synthetic_labels(n_neg=111, n_pos=63, seed=0):
    y = np.array([0] * n_neg + [1] * n_pos)
    rng = np.random.default_rng(seed)
    rng.shuffle(y)
    subjects = [f"s{i:03d}" for i in range(len(y))]
    return y, subjects


def test_split_plan_holds_every_subject_out_exactly_once_per_repeat():
    y, subjects = _synthetic_labels()
    plan = engine.build_split_plan(
        y, subjects, n_splits=5, n_repeats=3, seed=1, min_positive_per_validation_fold=2
    )
    assert plan.n_repeats == 3 and len(plan.records) == 15
    for record in plan.records:
        train = {subjects[i] for i in record.train_indices}
        test = {subjects[i] for i in record.test_indices}
        assert not (train & test)
    for repeat in range(plan.n_repeats):
        held_out = np.concatenate(
            [r.test_indices for r in plan.records if r.repeat == repeat]
        )
        assert sorted(held_out.tolist()) == list(range(len(y)))


def test_split_plan_is_deterministic_and_shared():
    y, subjects = _synthetic_labels()
    first = engine.build_split_plan(y, subjects, n_splits=5, n_repeats=2, seed=7)
    second = engine.build_split_plan(y, subjects, n_splits=5, n_repeats=2, seed=7)
    assert first.plan_hash == second.plan_hash


def test_more_folds_than_minority_is_rejected():
    y = np.array([0] * 40 + [1] * 3)
    subjects = [f"s{i}" for i in range(len(y))]
    with pytest.raises(leakage.LeakageError, match="exceed the minority"):
        engine.build_split_plan(y, subjects, n_splits=5, n_repeats=1, seed=1)


# --------------------------------------------------------------------------- #
# preprocessing / screening / normalization internals
# --------------------------------------------------------------------------- #
def test_preprocessor_statistics_come_only_from_train():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(50, 4))
    X_test = rng.normal(size=(10, 4)) + 100.0  # far outside the training range

    train_state = engine._fit_preprocessor(X_train, winsorize_quantile=0.01, scale=True)
    transformed = engine._apply_preprocessor(X_test, train_state)

    # Fitting on the held-out block instead would re-centre it to ~0; using the
    # training reference must leave it pinned at the top of the training range.
    leaky_state = engine._fit_preprocessor(X_test, winsorize_quantile=0.01, scale=True)
    leaky = engine._apply_preprocessor(X_test, leaky_state)

    assert transformed.mean() > 1.0, "held-out block must stay shifted, not re-centred"
    assert abs(float(leaky.mean())) < 0.5, "sanity: fitting on test would centre it"
    assert not np.allclose(transformed, leaky)


def test_ecdf_uses_training_reference_not_holdout_ranks():
    reference = np.linspace(0.0, 1.0, 101)
    # Two held-out batches with identical values but different companions must
    # map the shared value identically (no transductive normalization).
    a = engine._ecdf_transform(reference, np.array([0.5, 0.9]))
    b = engine._ecdf_transform(reference, np.array([0.5, 0.1, 0.2]))
    assert a[0] == pytest.approx(b[0])


def test_screening_selects_the_informative_column():
    rng = np.random.default_rng(0)
    y = np.array([0] * 40 + [1] * 40)
    noise = rng.normal(size=(80, 5))
    signal = y + rng.normal(scale=0.1, size=80)
    X = np.column_stack([noise, signal])
    chosen = engine._screen_features(X, y, top_k=2, correlation_threshold=0.95)
    assert 5 in chosen.tolist()


def test_screening_prunes_duplicated_columns():
    rng = np.random.default_rng(1)
    y = np.array([0] * 40 + [1] * 40)
    signal = y + rng.normal(scale=0.1, size=80)
    X = np.column_stack([signal, signal * 1.0001, rng.normal(size=80)])
    chosen = engine._screen_features(X, y, top_k=3, correlation_threshold=0.95)
    assert not ({0, 1} <= set(chosen.tolist())), "near-duplicate columns must not both survive"


def test_roc_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=200)
    scores = rng.normal(size=200)
    assert engine.roc_auc(y, scores) == pytest.approx(roc_auc_score(y, scores), abs=1e-9)


# --------------------------------------------------------------------------- #
# candidates
# --------------------------------------------------------------------------- #
def test_missing_library_removes_family_without_substitution():
    usable, skipped = models.available_families(("logreg", "not_a_real_family"))
    assert "logreg" in usable
    assert "not_a_real_family" in skipped


def test_candidate_enumeration_covers_views_and_screening():
    candidates = models.build_candidates(
        ("logreg",),
        ("mmse_core", "all"),
        logreg_c_grid=(0.1, 1.0),
        svm_c_grid=(),
        svm_gamma_grid=(),
        top_k_grid=(15, 40),
    )
    names = {c.name for c in candidates}
    # Narrow MMSE views never screen; wide views get both full and screened variants.
    assert any(c.view == "mmse_core" and c.top_k is None for c in candidates)
    assert any(c.view == "all" and c.top_k == 15 for c in candidates)
    assert not any(c.view == "mmse_core" and c.top_k is not None for c in candidates)
    assert len(names) == len(candidates), "candidate names must be unique"


def test_only_scaled_families_request_scaling():
    assert models.needs_scaling("logreg") and models.needs_scaling("svm_rbf")
    for family in ("lightgbm", "catboost", "xgboost", "randomforest", "ydf_oblique"):
        assert not models.needs_scaling(family)


# --------------------------------------------------------------------------- #
# end-to-end on synthetic data (small, but a real fit)
# --------------------------------------------------------------------------- #
def test_candidate_evaluation_scores_every_subject_once_per_repeat():
    rng = np.random.default_rng(5)
    y, subjects = _synthetic_labels(n_neg=60, n_pos=40, seed=5)
    X = np.column_stack([y + rng.normal(scale=1.0, size=len(y)), rng.normal(size=len(y))])
    names = ("mmse__signal", "wear__noise")
    plan = engine.build_split_plan(y, subjects, n_splits=5, n_repeats=2, seed=1)
    candidate = models.Candidate(name="lr", family="logreg", view="mmse_core", params={"C": 1.0})
    result = engine.evaluate_candidate(
        candidate, X, y, names, plan,
        seed=1, winsorize_quantile=0.01, correlation_threshold=0.95, balanced=True,
    )
    assert result.error is None
    assert result.oof_by_repeat.shape == (2, len(y))
    assert np.isfinite(result.oof_by_repeat).all()
    assert 0.0 <= result.subject_mean_auc <= 1.0
    assert result.subject_mean_auc > 0.6, "an informative feature should be learnable"


def test_ensemble_never_scores_below_its_best_member():
    rng = np.random.default_rng(11)
    y, subjects = _synthetic_labels(n_neg=50, n_pos=30, seed=11)
    plan = engine.build_split_plan(y, subjects, n_splits=4, n_repeats=1, seed=2)
    results = []
    for index, noise in enumerate((0.5, 2.0)):
        X = np.column_stack([y + rng.normal(scale=noise, size=len(y)), rng.normal(size=len(y))])
        candidate = models.Candidate(
            name=f"lr{index}", family="logreg", view="mmse_core", params={"C": 1.0}
        )
        results.append(
            engine.evaluate_candidate(
                candidate, X, y, ("mmse__a", "wear__b"), plan,
                seed=1, winsorize_quantile=0.01, correlation_threshold=0.95, balanced=True,
            )
        )
    best_single = max(r.subject_mean_auc for r in results)
    ensemble = engine.search_ensemble(results, y, n_top=2, n_draws=200, seed=3)
    assert ensemble["enabled"]
    assert ensemble["subject_mean_oof_roc_auc"] >= best_single - 1e-9


# --------------------------------------------------------------------------- #
# wiring / config
# --------------------------------------------------------------------------- #
def test_every_cli_stage_maps_to_a_function():
    parser = run.build_parser()
    action = next(a for a in parser._actions if a.dest == "stage")
    assert set(action.choices) == set(pipeline.STAGES)
    for stage in pipeline.STAGES:
        if stage != "all":
            assert callable(pipeline._STAGE_FUNCTIONS[stage])


def test_entrypoint_main_block_is_minimal():
    tree = ast.parse((PACKAGE_ROOT / "run.py").read_text(encoding="utf-8"))
    blocks = [
        node
        for node in tree.body
        if isinstance(node, ast.If) and "__main__" in ast.dump(node.test)
    ]
    assert len(blocks) == 1 and len(blocks[0].body) == 1


def test_notebook_launch_arguments():
    assert run.notebook_argv({}) == ["--stage", "all"]
    assert run.notebook_argv({"BPM_ARGS": "--stage search --profile fast"}) == [
        "--stage", "search", "--profile", "fast",
    ]
    assert run.strip_jupyter_arguments(["-f", "/x/kernel-1.json", "--stage", "audit"]) == [
        "--stage", "audit",
    ]


def test_config_precedence_cli_beats_environment():
    configured = config_module.load_config(
        None,
        environ={"BPM_PROFILE": "fast", "BPM_CV_SPLITS": "4"},
        cli_overrides={"run.profile": "max"},
    )
    assert configured.run.profile == "max"
    assert configured.cv.n_splits == 4


def test_profile_controls_repeat_count():
    for profile, expected in (("fast", 3), ("default", 10), ("max", 20)):
        configured = config_module.load_config(None, cli_overrides={"run.profile": profile})
        assert configured.resolved_cv().n_repeats == expected


def test_single_split_cohort_is_selectable_for_comparability():
    """`--splits train` must reproduce the 141-subject cohort definition."""

    configured = config_module.load_config(None, cli_overrides={"data.splits": ("train",)})
    assert configured.data.splits == ("train",)


def test_paths_are_not_hard_coded_outside_config():
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path.name in {"config.py", "pipeline.py"} or "tests" in path.parts:
            continue
        assert "/content/drive" not in path.read_text(encoding="utf-8"), path.name
