"""Static and privacy contract tests for PerformanceLab.

These tests use only the Python standard library. They deliberately avoid opening
any benchmark label file. Run from the repository root with:

    python -m unittest discover -s ThreeClass_PerformanceLab/tests -v
"""

from __future__ import annotations

import csv
import json
import os
import re
import unittest
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LAB_ROOT.parent


class StaticContractTests(unittest.TestCase):
    def test_locked_config(self) -> None:
        path = LAB_ROOT / "config" / "locked_discovery_v1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["config_version"], "performance_lab_preregistered_v1")
        self.assertFalse(payload["data"]["include_mmse"])
        self.assertFalse(payload["data"]["include_absolute_dates"])
        self.assertEqual(payload["data"]["training_subjects_expected"], 141)
        self.assertEqual(
            payload["target"]["prediction_index"],
            "last_activity_day_end_timestamp_plus09",
        )
        self.assertEqual(payload["data"]["class_counts_expected"], {"CN": 85, "MCI": 47, "DEM": 9})
        self.assertEqual(payload["cv"]["outer_folds"], 3)
        self.assertEqual(len(payload["cv"]["outer_seeds"]), 5)
        self.assertEqual(payload["cv"]["model_seeds"], [17011, 27011])
        self.assertEqual(payload["models"]["event_tcn28_v1"]["epochs"], 120)
        self.assertFalse(payload["models"]["event_tcn28_v1"]["early_stopping"])
        self.assertEqual(
            payload["models"]["mask_tcn_35d_legacy_v1"]["daily_value_features"], 49
        )
        self.assertEqual(
            payload["models"]["mask_tcn_35d_legacy_v1"]["input_channels"], 147
        )
        self.assertFalse(
            payload["models"]["coverage_only_v1"]["selection_eligible"]
        )
        self.assertFalse(payload["selection"]["adaptive_ensemble_weights"])
        self.assertEqual(payload["selection"]["final_candidate_frequency_unit"], "15_outer_folds")
        self.assertEqual(payload["go_gate"]["zero_recall_repeats_max_per_mci_or_dem"], 1)
        self.assertEqual(payload["go_gate"]["primary_minus_coverage_positive_repeats_min"], 4)
        self.assertTrue(
            payload["go_gate"][
                "incremental_vs_elastic_gates_apply_when_final_is_not_elastic"
            ]
        )
        self.assertTrue(payload["go_gate"]["config_identity_must_match_checkpoints"])

    def test_no_benchmark_label_path_in_new_sources(self) -> None:
        forbidden = [
            "2." + "Validation/" + "LabelingData",
            "val_" + "label.csv",
            "ALLOW_" + "MMSE_FEATURES",
        ]
        suffixes = {".py", ".md", ".json", ".txt", ".ipynb"}
        for path in LAB_ROOT.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                self.assertNotIn(token, text, msg=f"Forbidden token in {path.relative_to(LAB_ROOT)}")

    def test_discovery_notebook_is_physically_training_only(self) -> None:
        path = LAB_ROOT / "01_train_only_discovery_colab.ipynb"
        if not path.exists():
            self.skipTest("Notebook has not been generated yet")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        source = "\n".join(
            "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list)
            else str(cell.get("source", ""))
            for cell in notebook["cells"]
        )
        for token in ["2.Validation", "val_label", "validation_label", "/Users/pig30nidae"]:
            self.assertNotIn(token, source)
        for required in ["SUBJECT_HASH_KEY", "FAST_MODE", "A100", "performance_lab_core"]:
            self.assertIn(required, source)
        self.assertLess(
            source.index('os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"'),
            source.index("import torch"),
        )
        for required in [
            "nested_result=nested_result",
            "decision=decision",
            "privacy_audit_pre_freeze.json",
            "RUN_INVALID_PRIVACY.json",
            "TRAINING_COMPLETE.json",
        ]:
            self.assertIn(required, source)

        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            self.assertIsNone(cell.get("execution_count"))
            self.assertEqual(cell.get("outputs"), [])
            cell_source = cell.get("source", [])
            cell_source = "".join(cell_source) if isinstance(cell_source, list) else str(cell_source)
            compile(cell_source, f"notebook_cell_{index}", "exec")

    def test_freeze_and_inference_identity_fail_closed(self) -> None:
        source = (LAB_ROOT / "performance_lab_core.py").read_text(encoding="utf-8")
        for required in [
            'nested_evidence.get("input_hash") != current_input_hash',
            'nested_evidence.get("code_hash") != current_code_hash',
            "expected_nested_run_hash = stable_json_hash",
            "recomputed_decision = select_and_assess",
            "Raw ensembles are not accepted",
            "frozen_config is mandatory",
        ]:
            self.assertIn(required, source)

    def test_requirements_leave_torch_to_colab_runtime(self) -> None:
        text = (LAB_ROOT / "requirements_colab.txt").read_text(encoding="utf-8").lower()
        self.assertNotRegex(text, r"(?m)^\s*torch(?:==|>=|<=)")
        self.assertNotIn("tabpfn", text)
        self.assertNotIn("tabicl", text)

    def test_local_eda_outputs_are_aggregate(self) -> None:
        artifact_dir = LAB_ROOT / "artifacts" / "local_eda"
        expected = {
            "data_audit.json",
            "class_feature_summary.csv",
            "EDA_REPORT_KO.md",
            "feature_contract_smoke.json",
        }
        self.assertEqual({path.name for path in artifact_dir.iterdir() if path.is_file()}, expected)
        audit = json.loads((artifact_dir / "data_audit.json").read_text(encoding="utf-8"))
        privacy = audit["privacy_contract"]
        self.assertTrue(privacy["artifacts_are_aggregate_only"])
        self.assertFalse(privacy["identifier_values_persisted"])
        self.assertFalse(audit["validation_isolation"]["validation_label_files_opened"])
        self.assertFalse(audit["mmse_exclusion"]["training_mmse_file_opened"])
        self.assertEqual(audit["training_labels"]["subject_count"], 141)
        self.assertEqual(audit["training_labels"]["class_counts"], {"CN": 85, "MCI": 47, "DEM": 9})

        with (artifact_dir / "class_feature_summary.csv").open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            rows = list(reader)
        forbidden_columns = {"EMAIL", "SAMPLE_EMAIL", "subject_id", "subject_hash", "hash"}
        self.assertFalse(fieldnames & forbidden_columns)
        self.assertEqual(len(rows), 1293)
        self.assertEqual(len({row["feature"] for row in rows}), 431)

        smoke = json.loads(
            (artifact_dir / "feature_contract_smoke.json").read_text(encoding="utf-8")
        )
        self.assertEqual(smoke["subjects"], 141)
        self.assertTrue(smoke["contract_checks"]["feature_bundle_validated"])
        self.assertFalse(smoke["contract_checks"]["validation_source_or_labels_opened"])
        self.assertFalse(smoke["contract_checks"]["raw_or_hashed_subject_ids_persisted"])

    def test_aggregate_artifacts_do_not_contain_source_identifiers(self) -> None:
        """Read source ID columns in memory and report only a match count on failure."""

        if os.environ.get("PERFORMANCE_LAB_RUN_SOURCE_PRIVACY_TEST") != "1":
            self.skipTest("Opt-in source-ID privacy scan; aggregate regex checks run by default")

        source_specs = [
            (PROJECT_ROOT / "Data/1.Training/SourceData/1.Gait/train_activity.csv", "EMAIL"),
            (PROJECT_ROOT / "Data/1.Training/SourceData/2.Sleep/train_sleep.csv", "EMAIL"),
            (PROJECT_ROOT / "Data/2.Validation/SourceData/1.Gait/val_activity.csv", "EMAIL"),
            (PROJECT_ROOT / "Data/2.Validation/SourceData/2.Sleep/val_sleep.csv", "EMAIL"),
        ]
        identifiers: set[str] = set()
        for path, column in source_specs:
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    value = (row.get(column) or "").strip()
                    if value:
                        identifiers.add(value)

        payload_paths = [
            LAB_ROOT / "README_KO.md",
            LAB_ROOT / "EXPERIMENT_DESIGN_KO.md",
            LAB_ROOT / "EDA_REPORT_KO.md",
            LAB_ROOT / "artifacts/local_eda/data_audit.json",
            LAB_ROOT / "artifacts/local_eda/class_feature_summary.csv",
            LAB_ROOT / "artifacts/local_eda/EDA_REPORT_KO.md",
            LAB_ROOT / "artifacts/local_eda/feature_contract_smoke.json",
        ]
        notebook = LAB_ROOT / "01_train_only_discovery_colab.ipynb"
        if notebook.exists():
            payload_paths.append(notebook)
        payload = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in payload_paths)
        match_count = sum(identifier in payload for identifier in identifiers)
        self.assertEqual(match_count, 0, msg=f"Aggregate artifact identifier match count: {match_count}")

    def test_no_email_like_values_in_aggregate_csv(self) -> None:
        text = (LAB_ROOT / "artifacts/local_eda/class_feature_summary.csv").read_text(
            encoding="utf-8"
        )
        self.assertIsNone(re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text))


if __name__ == "__main__":
    unittest.main()
