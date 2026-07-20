from pathlib import Path
import ast
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_python_files_parse():
    for path in ROOT.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_training_entrypoint_contains_all_required_models():
    models = (ROOT / "models.py").read_text(encoding="utf-8")
    assert "fit_transformer" in models
    assert "fit_tabnet" in models
    assert "GradientBoostedTreesLearner" in models


def test_ydf_016_learner_does_not_receive_unsupported_verbose_argument():
    tree = ast.parse((ROOT / "models.py").read_text(encoding="utf-8"))
    learner_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "GradientBoostedTreesLearner"
    ]
    assert len(learner_calls) == 1
    assert "verbose" not in {keyword.arg for keyword in learner_calls[0].keywords}


def test_colab_pins_the_tested_ydf_version():
    requirements = (ROOT / "requirements_colab.txt").read_text(encoding="utf-8").splitlines()
    assert "ydf==0.16.1" in requirements


def test_direct_diagnosis_is_explicitly_blocked():
    features = (ROOT / "feature_engineering.py").read_text(encoding="utf-8")
    assert '"diag_nm"' in features
    assert "assert_no_forbidden_features" in features
    assert "direct_diagnosis_columns_used" in features


def test_subject_level_nested_cv_and_three_metrics_are_present():
    training = (ROOT / "train.py").read_text(encoding="utf-8")
    assert "StratifiedKFold" in training
    assert "macro_f1" in training
    assert "roc_auc_ovr_macro" in training
    assert "accuracy" in training
    assert "validation_predictions_label_free_hashed.csv" in training


def test_notebook_points_to_this_experiment():
    notebook_text = (ROOT.parent / "base.ipynb").read_text(encoding="utf-8")
    assert "ThreeClass_TransformerTabNet_Google" in notebook_text
    assert "drive.mount" in notebook_text
    notebook = json.loads(notebook_text)
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"base.ipynb:cell-{index}")
