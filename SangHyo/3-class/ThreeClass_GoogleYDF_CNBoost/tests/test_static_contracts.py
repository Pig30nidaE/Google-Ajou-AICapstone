from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class StaticContracts(unittest.TestCase):
    def test_all_python_files_compile(self) -> None:
        for path in sorted(ROOT.glob("*.py")):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_mmse_source_cannot_be_resolved(self) -> None:
        source = (ROOT / "feature_engineering.py").read_text(encoding="utf-8")
        self.assertNotIn("SourceData/3.CognitiveFunction", source)
        tree = ast.parse(source)
        split_files = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SplitFiles"
        )
        fields = {
            node.target.id
            for node in split_files.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        self.assertEqual(fields, {"root", "activity", "sleep", "labels"})
        self.assertIn('"mmse_source_opened": False', source)
        self.assertIn('"mmse_values_used": False', source)

    def test_validation_prediction_is_frozen_before_label_access(self) -> None:
        source = (ROOT / "train.py").read_text(encoding="utf-8")
        frozen = source.index('output / "VALIDATION_PREDICTIONS_FROZEN.json"')
        first_label_access = source.index(
            "discover_split_files(config.validation_root, require_labels=True)"
        )
        self.assertLess(frozen, first_label_access)
        prefix = source[:first_label_access]
        self.assertNotIn("load_consistent_labels(validation", prefix)

    def test_fixed_blend_has_no_class_scale_search(self) -> None:
        source = (ROOT / "evaluation.py").read_text(encoding="utf-8")
        self.assertIn("FIXED_BLEND_RECIPES", source)
        self.assertIn('"class_probability_scaling_used": False', source)
        self.assertNotIn("optimize_class_scale", source)

    def test_google_candidates_and_binary_adapter_are_present(self) -> None:
        source = (ROOT / "models.py").read_text(encoding="utf-8")
        for name in (
            "ydf_multiclass",
            "ydf_hierarchical",
            "ydf_random_forest",
            "ydf_ovr",
            "tabnet",
        ):
            self.assertIn(f'"{name}"', source)
        self.assertIn("if raw.ndim == 1", source)
        self.assertIn("label_classes=list(target_names)", source)
        learner_call = source[
            source.index("learner = ydf.GradientBoostedTreesLearner"):
            source.index("model = learner.train")
        ]
        self.assertNotIn("verbose=", learner_call)
        self.assertIn("learner = ydf.RandomForestLearner", source)
        self.assertIn("winner_take_all=False", source)

    def test_cn_reference_is_fold_local_and_stage_safe(self) -> None:
        from preprocessing import FoldFeatureSelector

        rng = np.random.default_rng(17)
        frame = pd.DataFrame(
            {
                "activity__event7__rest__median": rng.normal(size=18),
                "activity__event14__rest__iqr": rng.normal(size=18),
                "sleep__event7__deep__median": rng.normal(size=18),
                "sleep__event28__restless__iqr": rng.normal(size=18),
            }
        )
        labels = np.asarray([0] * 8 + [1] * 6 + [2] * 4)
        selector = FoldFeatureSelector(
            max_features=4,
            min_features_per_modality=1,
            correlation_threshold=0.999,
        )
        values = selector.fit_transform(frame, labels, task="multiclass")
        self.assertEqual(values.shape, (18, 4))
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(selector.manifest()["cn_reference_absolute_deviation"])

        stage = FoldFeatureSelector(
            max_features=2,
            min_features_per_modality=1,
            correlation_threshold=0.999,
        )
        stage.fit(frame.iloc[8:].reset_index(drop=True), labels[8:] - 1, task="mci_vs_dem")
        self.assertFalse(stage.manifest()["cn_reference_absolute_deviation"])


if __name__ == "__main__":
    unittest.main()
