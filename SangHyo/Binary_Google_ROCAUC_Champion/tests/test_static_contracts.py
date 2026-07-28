"""Source-only contracts for the ROC-AUC champion experiment.

These tests intentionally never import the experiment package.  Importing the
training modules could initialize PyTorch, TabPFN, or data-loading code; instead
we parse source files with the standard-library :mod:`ast` module.  The file can
be run with either ``python tests/test_static_contracts.py`` or a pytest/unittest
test runner.
"""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _source(filename: str) -> str:
    return (PACKAGE_ROOT / filename).read_text(encoding="utf-8")


def _tree(filename: str) -> ast.Module:
    return ast.parse(_source(filename), filename=str(PACKAGE_ROOT / filename))


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _function(
    tree: ast.Module,
    name: str,
    *,
    class_name: str | None = None,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    scope: list[ast.stmt] = tree.body
    if class_name is not None:
        classes = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        if len(classes) != 1:
            raise AssertionError(f"Expected one class {class_name!r}")
        scope = classes[0].body
    matches = [
        node
        for node in scope
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1:
        owner = f"{class_name}." if class_name else ""
        raise AssertionError(f"Expected one function {owner}{name}")
    return matches[0]


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected one class {name!r}")
    return matches[0]


def _assignment(tree: ast.Module, name: str) -> ast.AST:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                return node.value
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            if node.value is None:
                raise AssertionError(f"Assignment {name!r} has no value")
            return node.value
    raise AssertionError(f"Top-level assignment {name!r} not found")


def _calls(node: ast.AST, suffix: str) -> list[ast.Call]:
    return [
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Call)
        and _dotted_name(candidate.func).endswith(suffix)
    ]


def _dict_string_keys(node: ast.AST) -> set[str]:
    keys: set[str] = set()
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Dict):
            continue
        for key in candidate.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
    return keys


def _literal_frozenset(node: ast.AST) -> set[str]:
    if not (
        isinstance(node, ast.Call)
        and _dotted_name(node.func) == "frozenset"
        and len(node.args) == 1
    ):
        raise AssertionError("Expected frozenset(<literal>)")
    value = ast.literal_eval(node.args[0])
    return set(value)


class StaticContracts(unittest.TestCase):
    """Fail closed when a safety-critical source structure changes."""

    def test_wearable_cannot_open_mmse_or_cognitive_source(self) -> None:
        tree = _tree("data.py")
        loader = _function(tree, "load_wearable_sequence")
        loader_names = {
            _dotted_name(node.func)
            for node in ast.walk(loader)
            if isinstance(node, ast.Call)
        }
        self.assertNotIn("load_mmse_allowed", loader_names)
        self.assertNotIn("_read_csv_allowlist", loader_names)

        mmse_loader = _function(tree, "load_mmse_allowed")
        guards = []
        for node in ast.walk(mmse_loader):
            if not isinstance(node, ast.If):
                continue
            rendered = ast.unparse(node.test)
            raises_contract = any(
                isinstance(child, ast.Raise)
                and child.exc is not None
                and _dotted_name(
                    child.exc.func if isinstance(child.exc, ast.Call) else child.exc
                ).endswith("LeakageContractError")
                for child in ast.walk(node)
            )
            if (
                "resolved_track" in rendered
                and "mmse" in rendered
                and isinstance(node.test, ast.Compare)
                and any(isinstance(op, ast.NotEq) for op in node.test.ops)
                and raises_contract
            ):
                guards.append(node)
        self.assertEqual(
            len(guards),
            1,
            "load_mmse_allowed must reject every normalized non-MMSE track",
        )
        reads = _calls(mmse_loader, "_read_csv_allowlist")
        self.assertEqual(len(reads), 1)
        self.assertLess(
            guards[0].lineno,
            reads[0].lineno,
            "Track guard must run before the CognitiveFunction file read",
        )

        train_tree = _tree("train.py")
        for helper_name in (
            "_prepare_training_dataset",
            "_prepare_validation_dataset",
        ):
            helper = _function(train_tree, helper_name)
            rendered = ast.unparse(helper)
            self.assertIn("track == 'mmse'", rendered)
            self.assertIn("load_mmse_allowed", rendered)
            self.assertIn("else None", rendered)

    def test_mmse_csv_read_uses_explicit_allowlist(self) -> None:
        tree = _tree("data.py")
        reader = _function(tree, "_read_csv_allowlist")
        read_calls = _calls(reader, "pd.read_csv")
        self.assertEqual(len(read_calls), 1)
        usecols = [
            keyword.value
            for keyword in read_calls[0].keywords
            if keyword.arg == "usecols"
        ]
        self.assertEqual(len(usecols), 1)
        self.assertEqual(ast.unparse(usecols[0]), "list(selected)")

        mmse_loader = _function(tree, "load_mmse_allowed")
        calls = _calls(mmse_loader, "_read_csv_allowlist")
        self.assertEqual(len(calls), 1)
        positional_names = {
            argument.id
            for argument in calls[0].args
            if isinstance(argument, ast.Name)
        }
        self.assertIn("MMSE_ALLOWED_SOURCE_COLUMNS", positional_names)

        domains = ast.literal_eval(_assignment(tree, "MMSE_DOMAINS"))
        person_key = ast.literal_eval(_assignment(tree, "PERSON_KEY"))
        allowed = {
            person_key,
            "TOTAL",
            *(
                item
                for domain_items in domains.values()
                for item in domain_items
            ),
        }
        forbidden = _literal_frozenset(
            _assignment(tree, "MMSE_FORBIDDEN_SOURCE_COLUMNS")
        )
        self.assertTrue(
            {
                "DIAG_NM",
                "DIAG_SEQ",
                "DOCTOR_NM",
                "MMSE_NUM",
                "MMSE_KIND",
                "EMAIL",
            }.issubset(forbidden)
        )
        self.assertTrue(allowed.isdisjoint(forbidden))

        allowed_assignment = ast.unparse(
            _assignment(tree, "MMSE_ALLOWED_SOURCE_COLUMNS")
        )
        self.assertIn("PERSON_KEY", allowed_assignment)
        self.assertIn("'TOTAL'", allowed_assignment)
        self.assertIn("*MMSE_ITEMS", allowed_assignment)

    def test_feature_names_fail_closed_and_wearable_rejects_mmse(self) -> None:
        tree = _tree("features.py")
        forbidden = _literal_frozenset(
            _assignment(tree, "_FORBIDDEN_EXACT_TOKENS")
        )
        required = {
            "id",
            "identifier",
            "sample",
            "email",
            "diag",
            "diagnosis",
            "label",
            "target",
            "period",
            "sequence",
            "length",
            "coverage",
            "count",
            "counts",
            "mask",
            "padding",
            "missing",
            "missingness",
            "nonwear",
            "order",
            "index",
        }
        self.assertTrue(required.issubset(forbidden))

        contract = _function(tree, "assert_feature_contract")
        rendered = ast.unparse(contract)
        for required_guard in (
            "non_wear",
            "n_days",
            "resolved_track == 'wearable'",
            "lowered.startswith('mmse__')",
            "LeakageContractError",
        ):
            self.assertIn(required_guard, rendered)

        builder = _function(tree, "build_champion_dataset")
        builder_source = ast.unparse(builder)
        self.assertIn("elif mmse is not None", builder_source)
        self.assertIn(
            "MMSE data was supplied to the wearable track",
            builder_source,
        )
        self.assertGreaterEqual(
            len(_calls(builder, "assert_feature_contract")),
            1,
        )

    def test_wearable_source_schema_and_calendar_proxies_are_locked(self) -> None:
        tree = _tree("features.py")
        count = ast.literal_eval(
            _assignment(tree, "EXPECTED_WEARABLE_SOURCE_FEATURE_COUNT")
        )
        digest = ast.literal_eval(
            _assignment(tree, "EXPECTED_WEARABLE_SOURCE_SCHEMA_SHA256")
        )
        self.assertEqual(count, 113)
        self.assertEqual(len(digest), 64)

        time_tokens = _literal_frozenset(
            _assignment(tree, "_FORBIDDEN_WEARABLE_TIME_TOKENS")
        )
        self.assertTrue(
            {
                "date",
                "datetime",
                "timestamp",
                "day",
                "weekday",
                "calendar",
                "elapsed",
                "start",
                "end",
                "time",
            }.issubset(time_tokens)
        )
        source_contract = _function(tree, "assert_wearable_source_contract")
        rendered = ast.unparse(source_contract)
        self.assertIn("EXPECTED_WEARABLE_SOURCE_SCHEMA_SHA256", rendered)
        self.assertIn("_FORBIDDEN_WEARABLE_SUBSTRINGS", rendered)
        self.assertIn("_FORBIDDEN_WEARABLE_TIME_TOKENS", rendered)
        self.assertIn("LeakageContractError", rendered)

        summary = _function(tree, "summarize_wearable_sequences")
        self.assertEqual(
            len(_calls(summary, "assert_wearable_source_contract")),
            1,
            "Schema must be checked before every summary construction",
        )
        post_init = _function(
            tree, "__post_init__", class_name="ChampionDataset"
        )
        self.assertEqual(
            len(_calls(post_init, "assert_wearable_source_contract")),
            1,
            "Sequence Transformer inputs must use the same locked schema",
        )

    def test_nested_cv_protocol_is_fixed_to_5_by_5_or_10_and_4_by_2(self) -> None:
        evaluation_tree = _tree("evaluation.py")
        config_class = _class(evaluation_tree, "NestedCVConfig")
        defaults: dict[str, object] = {}
        for node in config_class.body:
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.value is not None
            ):
                try:
                    defaults[node.target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
        self.assertEqual(defaults["outer_splits"], 5)
        self.assertEqual(defaults["outer_repeats"], 5)
        self.assertEqual(defaults["inner_splits"], 4)
        self.assertEqual(defaults["inner_repeats"], 2)

        validate = _function(
            evaluation_tree, "validate", class_name="NestedCVConfig"
        )
        rendered = ast.unparse(validate)
        compact = "".join(rendered.split())
        self.assertIn("self.outer_splits!=5", compact)
        self.assertIn("self.outer_repeatsnotin{5,10}", compact)
        self.assertIn("self.inner_splits!=4", compact)
        self.assertIn("self.inner_repeats!=2", compact)

        train_tree = _tree("train.py")
        nested = _function(train_tree, "nested", class_name="RunConfig")
        nested_source = ast.unparse(nested)
        self.assertIn("profile not in {'default', 'max'}", nested_source)
        self.assertIn(
            "outer_repeats=5 if profile == 'default' else 10",
            nested_source,
        )

    def test_no_smote_or_oversampling_implementation(self) -> None:
        forbidden_symbols: list[str] = []
        forbidden_imports: list[str] = []
        for path in sorted(PACKAGE_ROOT.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        lowered = alias.name.lower()
                        if "imblearn" in lowered or "smote" in lowered:
                            forbidden_imports.append(f"{path.name}:{alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = (node.module or "").lower()
                    names = [alias.name.lower() for alias in node.names]
                    if (
                        "imblearn" in module
                        or "smote" in module
                        or any("smote" in name for name in names)
                    ):
                        forbidden_imports.append(
                            f"{path.name}:{node.module}:{','.join(names)}"
                        )
                elif isinstance(node, ast.Name):
                    lowered = node.id.lower()
                    if lowered.startswith("smote") or lowered in {
                        "borderlinesmote",
                        "smotenc",
                    }:
                        forbidden_symbols.append(f"{path.name}:{node.id}")
                elif isinstance(node, ast.Attribute):
                    lowered = node.attr.lower()
                    if lowered.startswith("smote") or lowered in {
                        "borderlinesmote",
                        "smotenc",
                    }:
                        forbidden_symbols.append(f"{path.name}:{node.attr}")
            self.assertNotIn(
                "fit_resample",
                {
                    node.attr
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Attribute)
                },
                f"{path.name} contains an oversampler-style fit_resample call",
            )
        requirements = _source("requirements_colab.txt").lower()
        self.assertNotIn("imbalanced-learn", requirements)
        self.assertNotIn("imblearn", requirements)
        self.assertEqual(forbidden_imports, [])
        self.assertEqual(forbidden_symbols, [])

    def test_both_validation_tracks_are_finalized_and_verified_before_labels(
        self,
    ) -> None:
        tree = _tree("train.py")
        track_order = tuple(ast.literal_eval(_assignment(tree, "TRACK_ORDER")))
        self.assertEqual(track_order, ("mmse", "wearable"))

        finalize_method = _function(
            tree, "finalize", class_name="TwoTrackValidationFreeze"
        )
        finalize_source = ast.unparse(finalize_method)
        self.assertIn("set(self.records) != set(TRACK_ORDER)", finalize_source)
        internal_verify = _calls(finalize_method, "self.verify")
        manifest_writes = _calls(finalize_method, "write_json")
        self.assertEqual(len(internal_verify), 1)
        self.assertEqual(len(manifest_writes), 1)
        self.assertLess(internal_verify[0].lineno, manifest_writes[0].lineno)

        run = _function(tree, "run_experiment")
        finalize_calls = _calls(run, "freeze.finalize")
        verify_calls = _calls(run, "freeze.verify")
        label_calls = [
            call
            for call in _calls(run, "load_diagnoses")
            if any(
                isinstance(argument, ast.Constant)
                and argument.value == "val"
                for argument in call.args
            )
        ]
        self.assertEqual(len(finalize_calls), 1)
        self.assertEqual(len(label_calls), 1)
        label_line = label_calls[0].lineno
        self.assertLess(finalize_calls[0].lineno, label_line)
        self.assertTrue(
            any(
                finalize_calls[0].lineno < call.lineno < label_line
                for call in verify_calls
            ),
            "A frozen-file hash verification must occur after finalize and "
            "before the first Validation label read",
        )
        record_calls = _calls(run, "freeze.record")
        self.assertEqual(len(record_calls), 1)
        self.assertLess(record_calls[0].lineno, finalize_calls[0].lineno)

        all_val_label_calls: list[str] = []
        for path in sorted(PACKAGE_ROOT.glob("*.py")):
            parsed = ast.parse(
                path.read_text(encoding="utf-8"), filename=str(path)
            )
            for call in _calls(parsed, "load_diagnoses"):
                if any(
                    isinstance(argument, ast.Constant)
                    and argument.value == "val"
                    for argument in call.args
                ):
                    all_val_label_calls.append(f"{path.name}:{call.lineno}")
        self.assertEqual(all_val_label_calls, [f"train.py:{label_line}"])

    def test_prediction_outputs_hash_subject_ids_and_never_name_raw_id_columns(
        self,
    ) -> None:
        tree = _tree("train.py")
        hash_function = _function(tree, "subject_hash")
        hash_calls = {_dotted_name(call.func) for call in ast.walk(hash_function)
                      if isinstance(call, ast.Call)}
        self.assertIn("hashlib.sha256", hash_calls)
        self.assertTrue(
            any(name.endswith("hexdigest") for name in hash_calls),
            "Subject hashes must use the complete SHA-256 hex digest",
        )

        forbidden_output_keys = {
            "subject_id",
            "subject_ids",
            "patient_id",
            "person_id",
            "SAMPLE_EMAIL",
            "EMAIL",
        }
        for function_name in (
            "_write_nested_artifacts",
            "_write_validation_prediction",
        ):
            function = _function(tree, function_name)
            keys = _dict_string_keys(function)
            self.assertIn("subject_sha256", keys)
            self.assertTrue(keys.isdisjoint(forbidden_output_keys))
            self.assertGreaterEqual(len(_calls(function, "subject_hash")), 1)

        validation_writer = _function(tree, "_write_validation_prediction")
        validation_keys = _dict_string_keys(validation_writer)
        self.assertIn("contains_raw_identifier", validation_keys)
        false_values = [
            value
            for node in ast.walk(validation_writer)
            if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant)
            and key.value == "contains_raw_identifier"
        ]
        self.assertEqual(len(false_values), 1)
        self.assertIs(ast.literal_eval(false_values[0]), False)

        evaluation_tree = _tree("evaluation.py")
        nested = _function(evaluation_tree, "run_repeated_nested_cv")
        nested_keys = _dict_string_keys(nested)
        self.assertIn("outer_test_subject_hashes", nested_keys)
        self.assertGreaterEqual(len(_calls(nested, "hashlib.sha256")), 1)
        self.assertTrue(nested_keys.isdisjoint(forbidden_output_keys))

    def test_tabpfn_checkpoint_is_explicit_v2_6_not_moving_default(self) -> None:
        tree = _tree("models.py")
        factory = _function(tree, "_make_estimator")
        versioned = _calls(
            factory, "TabPFNClassifier.create_default_for_version"
        )
        self.assertEqual(len(versioned), 1)
        self.assertGreaterEqual(len(versioned[0].args), 1)
        self.assertEqual(
            _dotted_name(versioned[0].args[0]),
            "ModelVersion.V2_6",
        )

        calls = [_dotted_name(call.func) for call in ast.walk(tree)
                 if isinstance(call, ast.Call)]
        self.assertNotIn("TabPFNClassifier", calls)
        self.assertNotIn("TabPFNClassifier.create_default", calls)
        self.assertEqual(
            [
                name
                for name in calls
                if name.endswith("create_default_for_version")
            ],
            ["TabPFNClassifier.create_default_for_version"],
        )

        manifest = _function(
            tree, "manifest", class_name="FoldLocalTableModel"
        )
        manifest_source = ast.unparse(manifest)
        self.assertIn("'model_version': 'v2.6'", manifest_source)
        self.assertIn("'moving_default_used': False", manifest_source)
        self.assertNotIn("'moving_default_used': True", manifest_source)

    def test_launcher_has_no_smoke_training_mode(self) -> None:
        tree = _tree("run.py")
        parser = _function(tree, "_parser")
        mode_calls = []
        for call in _calls(parser, "add_argument"):
            if (
                call.args
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value == "--mode"
            ):
                mode_calls.append(call)
        self.assertEqual(len(mode_calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in mode_calls[0].keywords}
        self.assertEqual(ast.literal_eval(keywords["choices"]), ("full",))
        self.assertEqual(ast.literal_eval(keywords["default"]), "full")

        smoke_comparisons: list[str] = []
        for path in sorted(PACKAGE_ROOT.glob("*.py")):
            parsed = ast.parse(
                path.read_text(encoding="utf-8"), filename=str(path)
            )
            for node in ast.walk(parsed):
                if isinstance(node, ast.Compare) and any(
                    isinstance(comparator, ast.Constant)
                    and str(comparator.value).lower() == "smoke"
                    for comparator in node.comparators
                ):
                    smoke_comparisons.append(
                        f"{path.name}:{node.lineno}:{ast.unparse(node)}"
                    )
        self.assertEqual(smoke_comparisons, [])

        train_tree = _tree("train.py")
        nested = _function(train_tree, "nested", class_name="RunConfig")
        nested_source = ast.unparse(nested)
        self.assertIn("{'default', 'max'}", nested_source)
        self.assertNotIn("'smoke'", nested_source.lower())

    def test_metrics_keep_repeat_and_subject_mean_estimands_separate(self) -> None:
        metrics_tree = _tree("metrics.py")
        summary = _function(metrics_tree, "summarize_repeated_oof")
        keys = _dict_string_keys(summary)
        self.assertTrue(
            {
                "repeat_level_roc_auc",
                "subject_mean_repeated_oof",
                "separation_warning",
                "estimand",
                "bootstrap_ci_attached",
                "score_aggregation",
            }.issubset(keys)
        )
        repeat_bootstrap_flags = [
            value
            for node in ast.walk(summary)
            if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant)
            and key.value == "bootstrap_ci_attached"
        ]
        self.assertEqual(len(repeat_bootstrap_flags), 1)
        self.assertIs(ast.literal_eval(repeat_bootstrap_flags[0]), False)

        summary_source = ast.unparse(summary)
        self.assertIn("subject_mean_scores = np.mean(matrix, axis=0)", summary_source)
        self.assertIn(
            "evaluate_binary_scores(target, subject_mean_scores",
            summary_source,
        )
        self.assertIn(
            "they are not a confidence interval for the mean",
            _source("metrics.py"),
        )

        evaluation_tree = _tree("evaluation.py")
        result_summary = _function(
            evaluation_tree, "summary", class_name="NestedCVResult"
        )
        result_keys = _dict_string_keys(result_summary)
        self.assertIn("estimand_warning", result_keys)
        result_source = ast.unparse(result_summary)
        self.assertIn("summarize_repeated_oof", result_source)
        self.assertIn("np.mean(self.repeat_scores, axis=0)", result_source)
        self.assertIn("paired_bootstrap_auc_difference", result_source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
