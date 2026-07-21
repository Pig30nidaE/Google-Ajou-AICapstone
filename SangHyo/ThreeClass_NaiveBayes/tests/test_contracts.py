from __future__ import annotations

import ast
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import CLASS_NAMES  # noqa: E402
from evaluation import (  # noqa: E402
    class_prior_probabilities,
    normalize_probabilities,
)
from modeling import (  # noqa: E402
    build_pipeline,
    fit_grid_search,
    predict_probabilities,
    selected_feature_names,
    validate_split_count,
)
from train import assert_disjoint_subjects, prepare_output_dir  # noqa: E402
from run_base import resolve_base_settings  # noqa: E402


class StaticContracts(unittest.TestCase):
    def test_all_python_files_parse(self) -> None:
        for path in sorted(ROOT.rglob("*.py")):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_new_code_cannot_resolve_cognitive_source(self) -> None:
        forbidden_path = "SourceData/3.CognitiveFunction"
        for path in sorted(ROOT.glob("*.py")):
            self.assertNotIn(forbidden_path, path.read_text(encoding="utf-8"))

    def test_no_machine_specific_path_is_hardcoded(self) -> None:
        for path in sorted(ROOT.glob("*.py")):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("/Users/", source)
            if path.name != "run_base.py":
                self.assertNotIn("/content/drive", source)

    def test_pipeline_ends_in_gaussian_naive_bayes(self) -> None:
        pipeline = build_pipeline()
        self.assertEqual(
            list(pipeline.named_steps),
            ["imputer", "scaler", "variance", "selector", "model"],
        )
        self.assertIsInstance(pipeline.named_steps["model"], GaussianNB)

    def test_validation_predictions_freeze_before_label_access(self) -> None:
        source = (ROOT / "train.py").read_text(encoding="utf-8")
        frozen = source.index('output / "VALIDATION_PREDICTIONS_FROZEN.json"')
        labels = source.index("validation_y = load_aligned_labels")
        self.assertLess(frozen, labels)


class ModelingContracts(unittest.TestCase):
    @staticmethod
    def synthetic_data() -> tuple[pd.DataFrame, np.ndarray]:
        rng = np.random.default_rng(20260721)
        labels = np.repeat(np.arange(3, dtype=np.int64), 12)
        values = rng.normal(size=(len(labels), 20))
        values[:, :5] += labels[:, None] * 0.8
        values[0, 7] = np.nan
        values[13, 11] = np.nan
        frame = pd.DataFrame(
            values,
            columns=[f"activity__synthetic_{index}" for index in range(values.shape[1])],
        )
        return frame, labels

    def test_fast_search_handles_missing_values_and_probabilities(self) -> None:
        frame, labels = self.synthetic_data()
        search = fit_grid_search(
            frame,
            labels,
            inner_folds=3,
            seed=71,
            fast=True,
            n_jobs=1,
        )
        probabilities = predict_probabilities(search.best_estimator_, frame)
        self.assertEqual(probabilities.shape, (len(frame), len(CLASS_NAMES)))
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
        selected = selected_feature_names(
            search.best_estimator_,
            list(frame.columns),
        )
        self.assertGreater(len(selected), 0)
        self.assertLessEqual(len(selected), frame.shape[1])

    def test_split_count_rejects_too_many_folds(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "smallest class count"):
            validate_split_count(labels, 3, context="test")

    def test_probability_normalization_preserves_exact_zero(self) -> None:
        values = normalize_probabilities(np.asarray([[0.0, 2.0, 0.0]]))
        np.testing.assert_array_equal(values, np.asarray([[0.0, 1.0, 0.0]]))

    def test_class_prior_uses_only_fit_labels(self) -> None:
        fit_labels = np.asarray([0, 0, 0, 1, 1, 2], dtype=np.int64)
        probabilities = class_prior_probabilities(fit_labels, 2)
        np.testing.assert_allclose(
            probabilities,
            np.asarray([[0.5, 1.0 / 3.0, 1.0 / 6.0]] * 2),
        )

    def test_training_validation_subjects_must_be_disjoint(self) -> None:
        with self.assertRaisesRegex(AssertionError, "overlap"):
            assert_disjoint_subjects(
                np.asarray(["Person@Example.com"]),
                np.asarray([" person@example.com "]),
            )

    def test_output_directory_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new-output"
            self.assertEqual(prepare_output_dir(output), output.resolve())
            (output / "existing.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_output_dir(output)

    def test_base_launcher_uses_notebook_data_root(self) -> None:
        project_root = ROOT.parents[1]
        settings = resolve_base_settings(
            {
                "PROJECT_ROOT": project_root,
                "DATA_ROOT": project_root / "Data",
                "NAIVE_BAYES_RUN_MODE": "smoke",
                "NAIVE_BAYES_RESULTS_ROOT": Path(tempfile.gettempdir()),
            }
        )
        self.assertEqual(settings["mode"], "smoke")
        self.assertEqual(settings["training_root"], project_root / "Data/1.Training")
        self.assertEqual(settings["validation_root"], project_root / "Data/2.Validation")


if __name__ == "__main__":
    unittest.main()
