"""Static safety contracts for the binary wearable experiment.

The suite intentionally imports none of the project modules, so it can run before
Colab-only ML dependencies are installed.  It checks source structure with the
standard-library AST and text only; it never opens source data or labels.
"""

from __future__ import annotations

import ast
import math
import re
import unittest
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODULES = (
    "run.py",
    "data.py",
    "preprocessing.py",
    "models.py",
    "train.py",
    "eda.py",
)


def _source(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _tree(name: str) -> ast.Module:
    path = ROOT / name
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [name for item in node.elts for name in _target_names(item)]
    return []


def _literal(node: ast.AST) -> object:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def _assignments(tree: ast.AST) -> Iterable[tuple[str, object]]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = _literal(node.value) if node.value is not None else None
            for target in targets:
                for name in _target_names(target):
                    yield name, value


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [
            item
            for key, child in value.items()
            for item in (*_strings(key), *_strings(child))
        ]
    if isinstance(value, (tuple, list, set, frozenset)):
        return [item for child in value for item in _strings(child)]
    return []


def _called_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


class StaticContractTests(unittest.TestCase):
    def test_expected_modules_exist_and_parse(self) -> None:
        missing = [name for name in EXPECTED_MODULES if not (ROOT / name).is_file()]
        self.assertFalse(missing, msg=f"Missing experiment modules: {missing}")
        for name in EXPECTED_MODULES:
            _tree(name)

    def test_mmse_is_explicitly_blocked_and_never_configured_on(self) -> None:
        trees = [_tree("data.py"), _tree("preprocessing.py")]
        blocked_values: list[str] = []
        opt_in_names = {"include_mmse", "use_mmse", "allow_mmse", "read_mmse"}

        for tree in trees:
            for name, value in _assignments(tree):
                lowered = name.lower()
                if any(token in lowered for token in ("forbidden", "blocked", "excluded", "deny")):
                    blocked_values.extend(item.lower() for item in _strings(value))
                if lowered in opt_in_names:
                    self.assertIsNot(value, True, msg=f"{name} must not enable MMSE")

            for node in ast.walk(tree):
                # An explicit deny-list token is allowed; an MMSE path is not.
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    value = node.value.lower()
                    data_suffixes = (".csv", ".tsv", ".parquet", ".xlsx", ".xls")
                    self.assertFalse(
                        "mmse" in value and value.rstrip().endswith(data_suffixes),
                        msg=f"MMSE source path must never be embedded: {node.value}",
                    )
                if isinstance(node, ast.Call):
                    for keyword in node.keywords:
                        if keyword.arg and keyword.arg.lower() in opt_in_names:
                            self.assertIsNot(
                                _literal(keyword.value),
                                True,
                                msg=f"Call enables forbidden option {keyword.arg}",
                            )

        self.assertTrue(
            any("mmse" in value for value in blocked_values),
            msg="MMSE must appear in an explicit forbidden/blocked/excluded/deny list",
        )

        combined = (_source("data.py") + _source("preprocessing.py")).lower()
        self.assertIn("activity", combined)
        self.assertIn("sleep", combined)
        data_source = _source("data.py")
        self.assertNotIn('root.glob("LabelingData/*', data_source)
        self.assertIn('LabelingData/1.Gait/*label.csv', data_source)
        self.assertIn('LabelingData/2.Sleep/*label.csv', data_source)
        self.assertNotIn('3.CognitiveFunction/*', data_source)

    def test_binary_class_names_are_cn_and_combined_mci_dem(self) -> None:
        candidates: list[list[str]] = []
        for module in ("data.py", "train.py"):
            for name, value in _assignments(_tree(module)):
                if "class" in name.lower() and "name" in name.lower():
                    strings = _strings(value)
                    if strings:
                        candidates.append(strings)

        def canonical(label: str) -> str:
            return re.sub(r"[^A-Z]", "", label.upper())

        valid = []
        for values in candidates:
            canonical_values = {canonical(value) for value in values}
            if (
                len(values) == 2
                and "CN" in canonical_values
                and canonical_values & {"MCIDEM", "IMPAIRED"}
            ):
                valid.append(values)
        self.assertTrue(
            valid,
            msg="Declare exactly two classes: CN and IMPAIRED/MCI_DEM (MCI + DEM)",
        )

        data_source = _source("data.py").upper()
        self.assertIn("MCI", data_source)
        self.assertIn("DEM", data_source)

    def test_run_file_is_the_single_python_entrypoint(self) -> None:
        tree = _tree("run.py")
        functions = {
            node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertIn("main", functions)

        has_main_guard = False
        main_called_under_guard = False
        for node in tree.body:
            if not isinstance(node, ast.If):
                continue
            test_text = ast.unparse(node.test)
            if "__name__" in test_text and "__main__" in test_text:
                has_main_guard = True
                main_called_under_guard = any(
                    isinstance(child, ast.Call) and _called_name(child) == "main"
                    for statement in node.body
                    for child in ast.walk(statement)
                )
        self.assertTrue(has_main_guard, msg="run.py needs an if __name__ == '__main__' guard")
        self.assertTrue(main_called_under_guard, msg="The main guard must call main()")

        readme = _source("README_KO.md")
        self.assertIn('USER_FOLDER = "SangHyo"', readme)
        self.assertIn('RUN_FILE = "Binary_Wearable_GoogleModels/run.py"', readme)
        self.assertIn("BINARY_RUN_MODE", readme)
        self.assertIn("smoke", readme.lower())

    def test_validation_predictions_are_frozen_before_evaluation(self) -> None:
        source = _source("train.py")
        tree = _tree("train.py")
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertIn("freeze_validation_predictions", functions)
        self.assertIn("validation_predictions_label_free_hashed.csv", source)
        self.assertIn("VALIDATION_PREDICTIONS_FROZEN.json", source)
        self.assertIn("prediction_sha256", source)
        self.assertIn("file_sha256(prediction_path)", source)
        self.assertIn("evaluation_read_probabilities_from_frozen_csv", source)

        scopes_with_safe_order: list[str] = []
        any_freeze_call = False
        any_evaluation_call = False
        for scope_name, scope in functions.items():
            calls = [node for node in ast.walk(scope) if isinstance(node, ast.Call)]
            freeze_lines = [
                node.lineno
                for node in calls
                if _called_name(node) == "freeze_validation_predictions"
            ]
            evaluation_lines = [
                node.lineno
                for node in calls
                if "validation" in _called_name(node).lower()
                and any(
                    token in _called_name(node).lower() for token in ("evaluate", "score")
                )
            ]
            any_freeze_call = any_freeze_call or bool(freeze_lines)
            any_evaluation_call = any_evaluation_call or bool(evaluation_lines)
            if freeze_lines and evaluation_lines and min(freeze_lines) < min(evaluation_lines):
                scopes_with_safe_order.append(scope_name)

        self.assertTrue(any_freeze_call, msg="freeze_validation_predictions() must actually be called")
        self.assertTrue(any_evaluation_call, msg="Validation evaluation must be an explicit call")
        self.assertTrue(
            scopes_with_safe_order,
            msg="One orchestration function must freeze label-free predictions before evaluation",
        )

    def test_subject_level_repeated_nested_cv_contract(self) -> None:
        train_tree = _tree("train.py")
        train_source = _source("train.py").lower()
        data_source = _source("data.py").lower()
        assignments = {name.lower(): value for name, value in _assignments(train_tree)}

        self.assertEqual(assignments.get("cv_unit"), "subject")
        self.assertIn("inner", train_source)
        self.assertIn("outer", train_source)
        self.assertIn("repeat", train_source)
        self.assertRegex(train_source, r"stratified(?:group)?kfold")

        repeat_values = [
            value
            for name, value in assignments.items()
            if "outer" in name and "repeat" in name and isinstance(value, int)
        ]
        self.assertTrue(repeat_values and max(repeat_values) >= 2)

        # Aggregation and uniqueness checks jointly make the CV input one row per person.
        combined = data_source + train_source
        self.assertIn("groupby", data_source)
        self.assertIn("duplicated", combined)
        self.assertTrue(any(token in combined for token in ("subject", "participant", "email")))
        self.assertNotIn("timeseriessplit", train_source)

        risky_split_inputs: list[tuple[int, str]] = []
        for node in ast.walk(train_tree):
            if not isinstance(node, ast.Call) or _called_name(node) != "split" or not node.args:
                continue
            first_arg = ast.unparse(node.args[0]).lower()
            if any(token in first_arg for token in ("daily", "day_rows", "events", "raw_rows")):
                risky_split_inputs.append((node.lineno, first_arg))
        self.assertFalse(
            risky_split_inputs,
            msg=f"A CV splitter appears to receive daily/raw rows: {risky_split_inputs}",
        )

    def test_accuracy_target_is_point_nine_but_not_claimed_as_guaranteed(self) -> None:
        assignments = {name: value for name, value in _assignments(_tree("train.py"))}
        self.assertIn("TARGET_ACCURACY", assignments)
        self.assertTrue(math.isclose(float(assignments["TARGET_ACCURACY"]), 0.90))

        readme = _source("README_KO.md")
        self.assertIn("0.90", readme)
        self.assertIn("30/33", readme)
        self.assertIn("보장", readme)

    def test_all_three_google_related_model_families_are_present(self) -> None:
        source = _source("models.py").lower()
        assignments = {name.lower(): value for name, value in _assignments(_tree("models.py"))}
        model_values = [
            item.lower()
            for name, value in assignments.items()
            if "model" in name and "name" in name
            for item in _strings(value)
        ]
        joined = " ".join(model_values)

        self.assertTrue("ydf" in joined or "yggdrasil" in joined)
        self.assertIn("tabnet", joined)
        self.assertIn("transformer", joined)
        self.assertRegex(source, r"ydf|yggdrasil")
        self.assertIn("tabnet", source)
        self.assertIn("transformer", source)

        readme = _source("README_KO.md")
        self.assertIn("Yandex Research", readme)
        self.assertIn("FT-Transformer 자체는 Google 모델이 아니", readme)

    def test_gitignore_keeps_runtime_artifacts_out_but_requirements_in(self) -> None:
        patterns = {
            line.strip()
            for line in _source(".gitignore").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertTrue({"outputs/", "cache/", "checkpoints/"}.issubset(patterns))
        self.assertIn("!requirements_colab.txt", patterns)


if __name__ == "__main__":
    unittest.main()
