from __future__ import annotations

import inspect
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from Codex_Dementia_ROCAUC.config import make_config
from Codex_Dementia_ROCAUC.data import _KNOWN_FILES, _target_from_diagnosis
from Codex_Dementia_ROCAUC.ensemble import fit_blend_policy
from Codex_Dementia_ROCAUC.features import aggregate_wearable_sequences
from Codex_Dementia_ROCAUC.leakage import LeakageError, assert_no_forbidden_features
from Codex_Dementia_ROCAUC.models.base import model_specs
from Codex_Dementia_ROCAUC.models.tabular import select_spec_columns
from Codex_Dementia_ROCAUC.run import TRAINING_ACKNOWLEDGEMENT
from Codex_Dementia_ROCAUC.splits import build_repeated_group_plan
from Codex_Dementia_ROCAUC.validation import validate_code_without_fit


class ZeroFitContracts(unittest.TestCase):
    """No test in this class is allowed to instantiate or fit an estimator."""

    def test_target_is_only_dementia_positive(self):
        config = make_config().data
        diagnosis = np.asarray(["CN", "MCI", "Dem", "CN", "Dem"])
        observed = _target_from_diagnosis(diagnosis, config)
        np.testing.assert_array_equal(observed, [0, 0, 1, 0, 1])

    def test_known_training_contract(self):
        config = make_config()
        self.assertEqual(config.data.expected_train_subjects, 141)
        self.assertEqual(
            config.data.expected_train_diagnosis_counts,
            {"CN": 85, "MCI": 47, "Dem": 9},
        )
        self.assertEqual(config.cv.outer_folds, 3)

    def test_raw_file_fingerprints_cannot_open_label_or_mmse_files(self):
        for layout in _KNOWN_FILES.values():
            self.assertEqual(
                set(layout),
                {"directory", "activity", "sleep"},
            )

    def test_required_model_families_are_registered(self):
        names = {spec.name for spec in model_specs()}
        self.assertTrue(
            {
                "catboost",
                "lightgbm",
                "xgboost",
                "extra_trees",
                "random_forest",
                "hist_gradient_boosting",
                "elastic_logreg",
                "rbf_svm",
                "balanced_random_forest",
                "easy_ensemble",
                "mlp",
                "tabnet",
                "tabnet_pretrained",
                "tsmixer",
            }
            <= names
        )

    def test_anchor_is_name_fixed_not_full_label_selected(self):
        spec = next(
            spec for spec in model_specs() if spec.name == "univariate_logreg"
        )
        names = (
            "wearable__activity__scalar__steps__mean",
            "wearable__activity__scalar__low__std",
            "wearable__sleep__scalar__duration__mean",
        )
        values = np.arange(15, dtype=float).reshape(5, 3)
        selected, selected_names = select_spec_columns(values, names, spec)
        self.assertEqual(selected.shape, (5, 1))
        self.assertEqual(
            selected_names, ("wearable__activity__scalar__low__std",)
        )

    def test_blend_requires_strict_repeat_majority_for_base_promotion(self):
        y = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        anchor = np.asarray([0.2, 0.7, 0.3, 0.6, 0.4, 0.5])
        candidate_repeats = np.asarray(
            [
                [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
                [0.4, 0.5, 0.6, 0.1, 0.2, 0.3],
            ]
        )
        candidate = candidate_repeats.mean(axis=0)
        policy = fit_blend_policy(
            y,
            {"anchor": anchor, "candidate": candidate},
            anchor="anchor",
            max_members=1,
            minimum_auc_gain=0.0025,
            weight_trials=0,
            seed=3,
            repeat_oof_by_model={
                "anchor": np.stack([anchor, anchor]),
                "candidate": candidate_repeats,
            },
        )
        self.assertEqual(policy.base_model, "anchor")
        comparison = next(
            record
            for record in policy.selection_trace
            if record.get("action") == "compare_best_individual_to_anchor"
        )
        self.assertEqual(comparison["repeat_wins"], 1)
        self.assertEqual(comparison["required_repeat_wins"], 2)

    def test_forbidden_feature_assertion_fails_closed(self):
        with self.assertRaises(LeakageError):
            assert_no_forbidden_features(("activity_steps", "DIAG_NM"))
        with self.assertRaises(LeakageError):
            assert_no_forbidden_features(("sleep_duration", "subject_id"))
        with self.assertRaises(LeakageError):
            assert_no_forbidden_features(("sleep_duration", "feature__subject_hash"))
        with self.assertRaises(LeakageError):
            assert_no_forbidden_features(("activity_steps", "diag_nm"))
        with self.assertRaises(LeakageError):
            assert_no_forbidden_features(("sleep_duration", "feature__mmse_kind"))

    def test_repeated_group_plan_has_subject_disjoint_folds(self):
        groups = np.asarray([f"s{i:02d}" for i in range(18)])
        y = np.asarray([0] * 12 + [1] * 6)
        plan = build_repeated_group_plan(
            y,
            groups,
            n_splits=3,
            n_repeats=2,
            seed=7,
            minimum_positive_validation=1,
            layer="unit",
        )
        self.assertEqual(len(plan.records), 6)
        for record in plan.records:
            self.assertFalse(
                set(groups[record.train_indices])
                & set(groups[record.validation_indices])
            )
            self.assertGreaterEqual(int(y[record.validation_indices].sum()), 1)

    def test_feature_aggregation_signature_has_no_target(self):
        parameters = inspect.signature(aggregate_wearable_sequences).parameters
        self.assertNotIn("y", parameters)
        self.assertNotIn("target", parameters)

    def test_training_cli_requires_literal_acknowledgement(self):
        self.assertEqual(
            TRAINING_ACKNOWLEDGEMENT,
            "I_UNDERSTAND_THIS_RUNS_TRAINING",
        )

    def test_direct_entrypoint_can_import_repository_sibling_package(self):
        package_root = Path(__file__).resolve().parents[1]
        run_path = package_root / "run.py"
        probe = (
            "import runpy; "
            f"runpy.run_path({str(run_path)!r}, run_name='direct_import_probe'); "
            "from SangHyo.Binary_Google_ROCAUC_Champion.data import AccessAudit; "
            "assert AccessAudit is not None"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=package_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_validate_code_report_declares_zero_fits(self):
        package_root = Path(__file__).resolve().parents[1]
        report = validate_code_without_fit(package_root)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["training_calls_executed"], 0)


if __name__ == "__main__":
    unittest.main()
