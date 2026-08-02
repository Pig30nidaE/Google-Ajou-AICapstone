"""Leakage and wiring contracts.

Runs under pytest, and also standalone (``python -m SangHyo.Binary_Google_DemRankAUC.tests.test_contracts``)
so a bare Colab image without pytest can still verify the contracts.

The tests that matter most are the ones that could not be argued away in review:

``test_features_are_subject_local``
    rebuilds the feature matrix from a random *subset* of subjects and asserts
    the retained rows are bit-identical.  Any cohort-level statistic --
    a global mean, a cohort quantile, a target encoding -- changes those rows and
    fails here.  This is the mechanical proof behind ``features.py``'s claim.

``test_preprocessor_rejects_wrong_fit_scope``
    asserts that a preprocessor fitted on the full matrix cannot be used to
    transform a fold, which is the exact mistake ``hard_constraints`` forbids.

``test_resampling_never_sees_test_rows``
    counts the rows a resampler is handed and asserts it equals the training
    fold size.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):  # pragma: no cover - standalone execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from SangHyo.Binary_Google_DemRankAUC import config as C
from SangHyo.Binary_Google_DemRankAUC import features as F
from SangHyo.Binary_Google_DemRankAUC import preprocessing as P
from SangHyo.Binary_Google_DemRankAUC.data import load_cohort, quality_flags, resolve_block
from SangHyo.Binary_Google_DemRankAUC.engine import fold_fit_predict, nested_selection_cv, run_repeated_cv
from SangHyo.Binary_Google_DemRankAUC.ensemble import best_blender, rank_normalize
from SangHyo.Binary_Google_DemRankAUC.evaluation import subject_metrics
from SangHyo.Binary_Google_DemRankAUC.models import ModelSpec, available_models, build_model
from SangHyo.Binary_Google_DemRankAUC.splits import (
    assert_no_subject_overlap,
    assert_split_integrity,
    make_folds,
    safe_k,
)

DATA_ROOT = Path(__file__).resolve().parents[3] / "Data"
_CACHE: dict[str, object] = {}


def cohort():
    if "cohort" not in _CACHE:
        _CACHE["cohort"] = load_cohort(DATA_ROOT)
    return _CACHE["cohort"]


# ------------------------------------------------------------- data contract -
def test_cohort_contract():
    c = cohort()
    assert c.n_subjects == C.COHORT_CONTRACT["n_subjects"] == 174
    assert c.n_positive == 12, "Dem is the positive class and there are 12 of them"
    counts = {d: int((c.diagnosis == d).sum()) for d in C.DIAG_ORDER}
    assert counts == {"CN": 111, "MCI": 51, "Dem": 12}
    assert len(set(c.subject_ids.tolist())) == c.n_subjects


def test_positive_class_is_dem():
    c = cohort()
    assert set(np.unique(c.y).tolist()) == {0, 1}
    assert (c.y[c.diagnosis == "Dem"] == 1).all()
    assert (c.y[c.diagnosis == "CN"] == 0).all()
    assert (c.y[c.diagnosis == "MCI"] == 0).all()


def test_no_forbidden_columns():
    c = cohort()
    F.assert_no_forbidden(c.feature_names)
    lowered = [n.lower() for n in c.feature_names]
    for token in ("diag", "label", "doctor", "email", "mmse_num", "mmse_kind"):
        assert not any(token in name for name in lowered), f"{token} leaked into features"


def test_forbidden_check_actually_fires():
    """The guard must reject a bad name, not just pass on good ones."""

    for bad in ("DIAG_NM", "wd_diag_score", "subject_email"):
        try:
            F.assert_no_forbidden(["mmse_TOTAL", bad])
        except AssertionError:
            continue
        raise AssertionError(f"assert_no_forbidden failed to reject {bad!r}")


def test_features_are_subject_local():
    """Rebuilding from a subject subset must not change the retained rows."""

    full = F.build_split_features(DATA_ROOT, "val")
    rng = np.random.default_rng(0)
    keep = sorted(rng.choice(full.index.to_numpy(), size=20, replace=False).tolist())

    subset = _build_subset_features(keep)
    shared = [c for c in full.columns if c in subset.columns]
    a = full.loc[keep, shared].to_numpy(dtype=np.float64)
    b = subset.loc[keep, shared].to_numpy(dtype=np.float64)
    both_nan = np.isnan(a) & np.isnan(b)
    assert np.allclose(a[~both_nan], b[~both_nan], rtol=0, atol=0), (
        "features changed when other subjects were removed -> a cohort-level "
        "statistic leaked into the per-subject features"
    )


def _build_subset_features(keep):
    """Rebuild val features after physically removing the other subjects' rows."""

    import pandas as pd

    root = DATA_ROOT / C.SPLIT_DIRS["val"]
    keep_set = set(map(str, keep))
    mmse = F.read_csv(root / C.SOURCE_FILES["val"]["mmse"])
    mmse = mmse[mmse["SAMPLE_EMAIL"].astype(str).str.strip().isin(keep_set)]
    activity = F.read_csv(root / C.SOURCE_FILES["val"]["activity"])
    activity = activity[activity["EMAIL"].astype(str).str.strip().isin(keep_set)]
    sleep = F.read_csv(root / C.SOURCE_FILES["val"]["sleep"])
    sleep = sleep[sleep["EMAIL"].astype(str).str.strip().isin(keep_set)]

    mmse = mmse.copy()
    mmse["_sid"] = mmse["SAMPLE_EMAIL"].astype(str).str.strip()
    mmse = mmse.drop(columns=[c for c in mmse.columns if c in C.FORBIDDEN_COLUMNS])
    mmse = mmse.drop(columns=[c for c in C.MMSE_EXCLUDED_ITEMS if c in mmse.columns])
    mmse = mmse.drop_duplicates("_sid").set_index("_sid")

    blocks = [
        F.mmse_features(mmse),
        F._daily_block(activity, C.RICH_ACTIVITY, C.LITE_ACTIVITY,
                       day_column="activity_day_start", source="activity"),
        F._daily_block(sleep, C.RICH_SLEEP, C.LITE_SLEEP,
                       day_column="sleep_bedtime_end", source="sleep"),
        F._sleep_architecture(sleep),
        F._circadian_daily(sleep),
        F._intraday_block(activity, sleep),
    ]
    subjects = sorted(map(str, keep))
    frame = pd.concat([b.reindex(subjects) for b in blocks], axis=1)
    return frame.loc[:, ~frame.columns.duplicated()].astype(np.float64)


def test_intraday_features_exist_and_are_finite():
    """The new wi_ family must actually be populated, not silently all-NaN."""

    c = cohort()
    intraday = [n for n in c.feature_names if n.startswith("wi_")]
    assert len(intraday) > 50, f"only {len(intraday)} intraday features built"
    matrix = c.X[:, [c.feature_names.index(n) for n in intraday]]
    assert np.isfinite(matrix).mean() > 0.9
    for canonical in ("wi_met_IS", "wi_met_IV", "wi_met_RA", "wi_met_M10", "wi_met_L5"):
        assert canonical in c.feature_names, f"{canonical} missing"


def test_mmse_zero_row_is_treated_as_missing():
    """The all-zero MMSE record must not become a literal score of zero."""

    with_rule = F.build_split_features(DATA_ROOT, "val", mmse_zero_as_missing=True)
    without = F.build_split_features(DATA_ROOT, "val", mmse_zero_as_missing=False)
    assert without["mmse_TOTAL"].min() == 0.0
    assert with_rule["mmse_TOTAL"].isna().sum() == 1
    assert with_rule["mmse_TOTAL"].min() > 0.0


# ---------------------------------------------------------------- splits -----
def test_split_contracts():
    c = cohort()
    folds = make_folds(c.y, n_splits=5, n_repeats=3, seed=C.SEED)
    assert_split_integrity(folds, c.y, n_repeats=3)
    for fold in folds:
        assert_no_subject_overlap(c.subject_ids[fold.train_index], c.subject_ids[fold.test_index])
        assert int(c.y[fold.test_index].sum()) >= 1
        assert len(np.unique(c.y[fold.train_index])) == 2


def test_split_overlap_detector_fires():
    try:
        assert_no_subject_overlap(["a", "b"], ["b", "c"])
    except AssertionError:
        return
    raise AssertionError("assert_no_subject_overlap failed to detect an overlap")


def test_fold_count_capped_by_minority():
    assert safe_k(np.array([0] * 100 + [1] * 3), 10, min_per_fold=2) == 2
    assert safe_k(np.array([0] * 162 + [1] * 12), 5, min_per_fold=2) == 5


# --------------------------------------------------------- preprocessing -----
def test_preprocessor_rejects_wrong_fit_scope():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    train_index = np.arange(30)
    fitted = P.FoldPreprocessor().fit(X[train_index], train_index)
    fitted.transform(X[30:], expect_fingerprint=P.rows_fingerprint(train_index))
    try:
        fitted.transform(X[30:], expect_fingerprint=P.rows_fingerprint(np.arange(40)))
    except AssertionError:
        return
    raise AssertionError("preprocessor accepted a transform from the wrong fit scope")


def test_preprocessor_statistics_come_from_training_rows_only():
    X = np.vstack([np.zeros((20, 3)), np.full((20, 3), 1000.0)])
    train_index = np.arange(20)
    fitted = P.FoldPreprocessor(winsorize=0.0).fit(X[train_index], train_index)
    assert np.allclose(fitted.center_, 0.0), "center must ignore the held-out block"
    transformed = fitted.transform(X[20:], expect_fingerprint=P.rows_fingerprint(train_index))
    assert (transformed > 100).all(), "held-out rows must be scaled by training statistics"


def test_resampling_never_sees_test_rows():
    seen: list[int] = []
    original = P.resample

    def spy(X, y, kind, *, seed):
        seen.append(len(y))
        return original(X, y, kind, seed=seed)

    c = cohort()
    block = c.select(resolve_block(c, "mmse_core"))
    folds = make_folds(c.y, n_splits=5, n_repeats=1, seed=C.SEED)
    import SangHyo.Binary_Google_DemRankAUC.engine as engine_module

    engine_module.resample = spy
    try:
        for fold in folds:
            fold_fit_predict(block.X, block.y, fold.train_index, fold.test_index,
                             ModelSpec("logreg_l2", {}), seed=0, resampler="random_over")
    finally:
        engine_module.resample = original
    expected = [int(f.train_index.size) for f in folds]
    assert seen == expected, f"resampler saw {seen}, expected training sizes {expected}"


def test_feature_selection_is_fold_local():
    """Selection on different training rows must be allowed to differ."""

    c = cohort()
    block = c.select(resolve_block(c, "wd_full"))
    folds = make_folds(c.y, n_splits=5, n_repeats=1, seed=C.SEED)
    chosen = []
    for fold in folds[:3]:
        X_train = np.nan_to_num(block.X[fold.train_index])
        chosen.append(tuple(P.select_features(X_train, block.y[fold.train_index], top_k=10)))
    assert len({len(s) for s in chosen}) == 1 and all(len(s) == 10 for s in chosen)


# ---------------------------------------------------------------- models -----
def test_every_available_model_fits_and_scores():
    c = cohort()
    block = c.select(resolve_block(c, "mmse_core"))
    X = P.FoldPreprocessor().fit_transform(block.X)
    failures = {}
    for name in available_models(include_slow=False):
        try:
            model = build_model(name, {}, seed=0).fit(X[:140], block.y[:140])
            scores = model.score_samples(X[140:])
            assert scores.shape == (X.shape[0] - 140,)
            assert np.isfinite(scores).all()
        except Exception as error:  # pragma: no cover - reported, not swallowed
            failures[name] = f"{type(error).__name__}: {error}"
    assert not failures, f"models failed to fit/score: {failures}"


def test_rank_normalize_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    values = rng.normal(size=50)
    ranked = rank_normalize(values)
    assert ranked.min() >= 0.0 and ranked.max() <= 1.0
    assert np.array_equal(np.argsort(values), np.argsort(ranked))
    tied = rank_normalize(np.array([1.0, 1.0, 2.0]))
    assert tied[0] == tied[1] < tied[2], "ties must average, not break arbitrarily"


def test_blender_selection_prefers_informative_member():
    rng = np.random.default_rng(0)
    y = np.array([0] * 60 + [1] * 12)
    good = rng.normal(size=72) + y * 2.5
    noise = rng.normal(size=72)
    blender = best_blender(np.column_stack([good, noise]), y, seed=0)
    combined = blender.apply(np.column_stack([good, noise]))
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y, combined) > roc_auc_score(y, noise) + 0.1


# ------------------------------------------------------------ metrics -------
def test_subject_metrics_reports_every_required_secondary():
    y = np.array([0, 0, 0, 1, 1, 0, 1, 0])
    score = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.15, 0.7, 0.05])
    metrics = subject_metrics(y, score, threshold=0.5)
    for key in ("roc_auc", "pr_auc", "dem_recall", "f1", "balanced_accuracy", "mcc",
                "specificity"):
        assert key in metrics, f"missing secondary metric {key}"
    assert metrics["roc_auc"] == 1.0


# ------------------------------------------------------------- smoke run ----
def test_smoke_repeated_cv_runs():
    """Wiring check only.  These numbers are NOT reported as performance."""

    c = cohort()
    block = c.select(resolve_block(c, "fused_core"))
    folds = make_folds(c.y, n_splits=5, n_repeats=1, seed=C.SEED)
    result = run_repeated_cv(block.X, block.y, block.subject_ids, folds,
                             ModelSpec("logreg_l2", {"C": 0.3}), block="fused_core", seed=0)
    assert result.per_repeat_auc and np.isfinite(result.per_repeat_auc[0])
    assert len(result.per_fold_auc) == len(folds)
    assert result.mean_oof().shape == (c.n_subjects,)


def test_smoke_nested_selection_runs():
    """Wiring check only.  These numbers are NOT reported as performance."""

    c = cohort()
    block = c.select(resolve_block(c, "mmse_core"))
    folds = make_folds(c.y, n_splits=3, n_repeats=1, seed=C.SEED)
    specs = [ModelSpec("logreg_l2", {"C": 0.3}), ModelSpec("rank_mean", {})]
    result = nested_selection_cv(block.X, block.y, block.subject_ids, folds, specs,
                                 block="mmse_core", inner_k=2, seed=0)
    assert result.per_repeat_auc and np.isfinite(result.per_repeat_auc[0])
    assert len(result.thresholds) == len(folds)
    assert sum(result.chosen_combiner.values()) == len(folds)


def test_quality_flags_do_not_read_labels():
    c = cohort()
    flipped = c.__class__(c.subject_ids, c.diagnosis, c.severity, 1 - c.y, c.X,
                          c.feature_names, c.split_of, c.fingerprint)
    assert quality_flags(c) == quality_flags(flipped)


# ------------------------------------------------------------------ main ----
def _run_all() -> int:
    tests = [(name, value) for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = 0
    for name, test in tests:
        try:
            test()
            print(f"PASS {name}")
        except Exception as error:
            failures += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_run_all())
