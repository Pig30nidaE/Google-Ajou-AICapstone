import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR / "src"))
DATA_ROOT = EXP_DIR.parents[2] / "Data"

from cheon.features import PAPER_FEATURES  # noqa: E402
from cheon.pipeline import Experiment  # noqa: E402


def synthetic(n_subj: int = 24, seed: int = 0):
    rng = np.random.default_rng(seed)
    groups, y = [], []
    for s in range(n_subj):
        lab = int(s % 3 == 0)
        for _ in range(int(rng.integers(6, 12))):
            groups.append(f"subj{s:03d}")
            y.append(lab)
    groups, y = np.array(groups), np.array(y)
    X = rng.normal(size=(len(y), len(PAPER_FEATURES))) + y[:, None] * 0.6
    return X, y, groups


def make_experiment(tmp_path: Path, **over) -> Experiment:
    cfg = {"experiment": "subject_independent", "output_dir": "results/test", "seeds": [0], "n_splits": 3,
           "inner_splits": 3, "split_unit": "subject", "shuffle": True, "n_threads": 2, "threshold_rule": "model_default",
           "base_models": {"enabled": False, "repeats": 0},
           "feature_selection": {"scope": "train_fold", "max_k": 4, "k_fixed": 3},
           "final_model": {"params": "paper_table2", "also_evaluate_all_features": True, "also_evaluate_paper_k": 3},
           "hyperparameters": {"grid": "paper_anchored_2x2x2x2", "also_report_fixed_paper_params": True}}
    cfg.update(over)
    ex = Experiment(cfg, yaml.dump(cfg), exp_dir=tmp_path, repo_root=tmp_path, data_root=tmp_path, smoke=False)
    X, y, groups = synthetic()
    ex.X, ex.y, ex.groups, ex.record_id = X, y, groups, np.arange(len(y))
    ex.max_k = 4
    ex.data_fp = "synthetic"
    return ex


@pytest.fixture
def real_data_available():
    return DATA_ROOT.exists()
