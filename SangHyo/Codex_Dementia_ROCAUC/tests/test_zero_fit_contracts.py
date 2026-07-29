from __future__ import annotations

import inspect
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import warnings

import numpy as np

from Codex_Dementia_ROCAUC import run as run_module
from Codex_Dementia_ROCAUC.config import make_config
from Codex_Dementia_ROCAUC.data import _KNOWN_FILES, _target_from_diagnosis
from Codex_Dementia_ROCAUC.ensemble import fit_blend_policy
from Codex_Dementia_ROCAUC.features import aggregate_wearable_sequences
from Codex_Dementia_ROCAUC.leakage import LeakageError, assert_no_forbidden_features
from Codex_Dementia_ROCAUC.models.base import model_specs
from Codex_Dementia_ROCAUC.models.tabnet import TabNetBinaryEstimator
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

    def test_wrong_training_ack_fails_before_dependency_install(self):
        with mock.patch.object(run_module, "_ensure_dependencies") as ensure:
            with self.assertRaises(SystemExit) as raised:
                run_module.main(
                    [
                        "train",
                        "--execute-training",
                        "NOT_THE_ACKNOWLEDGEMENT",
                    ],
                    namespace={},
                )
        self.assertEqual(raised.exception.code, 2)
        ensure.assert_not_called()

    def test_jupyter_kernel_arguments_are_removed_narrowly(self):
        kernel = "/root/.local/share/jupyter/runtime/kernel-unit-test.json"
        self.assertEqual(
            run_module._without_jupyter_kernel_args(["-f", kernel]),
            [],
        )
        self.assertEqual(
            run_module._without_jupyter_kernel_args([kernel]),
            [],
        )
        explicit = ["train", "--profile", "standard"]
        self.assertEqual(
            run_module._without_jupyter_kernel_args(explicit),
            explicit,
        )

    def test_original_base_dispatches_unique_standard_training_without_fit(self):
        project_root = Path(__file__).resolve().parents[3]
        captured = []
        fake_train = types.ModuleType(
            f"{run_module.__package__}.train"
        )
        fake_train.run_experiment = captured.append
        kernel_args = [
            "--HistoryManager.hist_file=:memory:",
            "-f",
            "/root/.local/share/jupyter/runtime/kernel-unit-test.json",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "Data"
            (data_root / "1.Training").mkdir(parents=True)
            (data_root / "2.Validation").mkdir()
            namespace = {
                "PROJECT_ROOT": project_root,
                "DATA_ROOT": data_root,
                "USER_ROOT": project_root / "SangHyo",
                "RUN_PATH": Path(__file__).resolve().parents[1] / "run.py",
            }
            drive = Path(temporary) / "MyDrive"
            drive.mkdir()
            results_root = drive / "Codex_Dementia_ROCAUC_result"
            with (
                mock.patch.object(
                    run_module,
                    "DEFAULT_RESULTS_ROOT",
                    results_root,
                ),
                mock.patch.object(
                    run_module,
                    "_ensure_dependencies",
                ) as ensure,
                mock.patch.dict(
                    sys.modules,
                    {f"{run_module.__package__}.train": fake_train},
                ),
            ):
                exit_code = run_module.main(
                    kernel_args,
                    namespace=namespace,
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(captured), 1)
        config = captured[0]
        self.assertEqual(config.profile, "standard")
        self.assertEqual(config.data.resolved_root(), data_root.resolve())
        self.assertEqual(config.neural.device, "auto")
        self.assertEqual(config.runtime.n_jobs, 4)
        output = config.runtime.resolved_output()
        self.assertEqual(output.parent, results_root.resolve())
        self.assertTrue(output.name.endswith("_utc"))
        ensure.assert_called_once_with(
            include_models=True,
            skip_install=False,
            enforce_requirements=True,
        )

    def test_base_dependency_check_uses_experiment_colab_requirements(self):
        with (
            mock.patch.object(
                run_module.importlib.util,
                "find_spec",
                return_value=object(),
            ),
            mock.patch.object(
                run_module.importlib,
                "import_module",
                return_value=object(),
            ),
            mock.patch.object(
                run_module.subprocess,
                "run",
            ) as pip_run,
        ):
            run_module._ensure_dependencies(
                include_models=True,
                skip_install=False,
                enforce_requirements=True,
            )

        command = pip_run.call_args.args[0]
        self.assertEqual(command[:4], [
            sys.executable,
            "-m",
            "pip",
            "install",
        ])
        self.assertEqual(
            command[-2:],
            ["-r", str(run_module.REQUIREMENTS_COLAB)],
        )
        self.assertTrue(run_module.REQUIREMENTS_COLAB.is_file())
        self.assertTrue(pip_run.call_args.kwargs["check"])

    def test_tabnet_uses_epoch_scheduler_with_no_metric_argument(self):
        class FakeAdamW:
            pass

        class FakeStepLR:
            pass

        class FakeTabNetClassifier:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_torch = types.ModuleType("torch")
        fake_torch.optim = types.SimpleNamespace(
            AdamW=FakeAdamW,
            lr_scheduler=types.SimpleNamespace(StepLR=FakeStepLR),
        )
        fake_package = types.ModuleType("pytorch_tabnet")
        fake_package.__path__ = []
        fake_tab_model = types.ModuleType("pytorch_tabnet.tab_model")
        fake_tab_model.TabNetClassifier = FakeTabNetClassifier
        with mock.patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "pytorch_tabnet": fake_package,
                "pytorch_tabnet.tab_model": fake_tab_model,
            },
        ):
            model = TabNetBinaryEstimator(
                device_name="cpu"
            )._new_classifier()

        self.assertIs(model.kwargs["scheduler_fn"], FakeStepLR)
        self.assertEqual(
            model.kwargs["scheduler_params"],
            {"step_size": 20, "gamma": 0.5},
        )

    def test_real_tabnet_scheduler_callback_contract_without_fit(self):
        try:
            import torch
            from pytorch_tabnet.callbacks import LRSchedulerCallback
        except ImportError as error:
            self.skipTest(f"optional TabNet runtime unavailable: {error}")

        model = TabNetBinaryEstimator(device_name="cpu")._new_classifier()
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.AdamW([parameter], lr=0.002)
        callback = LRSchedulerCallback(
            scheduler_fn=model.scheduler_fn,
            scheduler_params=dict(model.scheduler_params),
            optimizer=optimizer,
            early_stopping_metric="valid_auc",
            is_batch_level=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            callback.on_epoch_end(0, {"valid_auc": 0.5})
        self.assertFalse(callback.is_metric_related)

    def test_direct_entrypoint_can_import_repository_sibling_package(self):
        package_root = Path(__file__).resolve().parents[1]
        run_path = package_root / "run.py"
        probe = (
            "import runpy, sys\n"
            f"run_path = {str(run_path)!r}\n"
            "sys.argv = [run_path, '--help']\n"
            "try:\n"
            "    runpy.run_path(run_path, run_name='__main__')\n"
            "except SystemExit as exc:\n"
            "    assert exc.code in (None, 0)\n"
            "from SangHyo.Binary_Google_ROCAUC_Champion.data import AccessAudit\n"
            "assert AccessAudit is not None\n"
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-c", probe],
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
