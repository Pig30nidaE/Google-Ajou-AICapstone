"""Model factories. LightGBM default = library defaults (A12); paper final = Table 2 (p.167)."""
from __future__ import annotations

import warnings

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

LGBM_DEFAULT: dict = {}  # LightGBM library defaults: num_leaves=31, min_child_samples=20, n_estimators=100, learning_rate=0.1
# Table 2 (p.167). `min_data_in_leaf` is the LightGBM core name; the scikit-learn wrapper alias is `min_child_samples`.
PAPER_TABLE2: dict = {"min_child_samples": 41, "num_leaves": 330, "n_estimators": 1000, "learning_rate": 0.08}
LGBM_LIBRARY_DEFAULT_VALUES = {"min_child_samples": 20, "num_leaves": 31, "n_estimators": 100, "learning_rate": 0.1}

# N (operational, paper-anchored, A20): every paper-tuned parameter takes its library default or its Table-2 value.
def paper_anchored_grid() -> list[dict]:
    grid = []
    for mcs in (LGBM_LIBRARY_DEFAULT_VALUES["min_child_samples"], PAPER_TABLE2["min_child_samples"]):
        for nl in (LGBM_LIBRARY_DEFAULT_VALUES["num_leaves"], PAPER_TABLE2["num_leaves"]):
            for ne in (LGBM_LIBRARY_DEFAULT_VALUES["n_estimators"], PAPER_TABLE2["n_estimators"]):
                for lr in (LGBM_LIBRARY_DEFAULT_VALUES["learning_rate"], PAPER_TABLE2["learning_rate"]):
                    grid.append({"min_child_samples": mcs, "num_leaves": nl, "n_estimators": ne, "learning_rate": lr})
    return grid


def params_key(p: dict) -> str:
    return "|".join(f"{k}={p[k]}" for k in sorted(p))


def make_lgbm(params: dict, seed: int, n_threads: int) -> LGBMClassifier:
    return LGBMClassifier(random_state=seed, n_jobs=n_threads, verbose=-1, **params)


def make_base_models(seed: int, n_threads: int) -> dict:
    """The seven candidate algorithms of §3.2(2), library defaults (A12), no scaling (A13)."""
    return {
        "LightGBM": make_lgbm(LGBM_DEFAULT, seed, n_threads),
        "Random forest": RandomForestClassifier(random_state=seed, n_jobs=n_threads),
        "Decision tree": DecisionTreeClassifier(random_state=seed),
        "K-Nearest Neighbor": KNeighborsClassifier(n_jobs=n_threads),
        "Multi-Layer Perceptron": MLPClassifier(random_state=seed),
        "Support vector machine": SVC(random_state=seed),
        "Logistic regression": LogisticRegression(),
    }


def score_and_predict(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Continuous score for ROC/PR (proba of class 1 or decision_function) and the model's own hard prediction."""
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[:, 1]
    else:
        score = model.decision_function(X)
    pred = model.predict(X)
    return np.asarray(score, dtype=float), np.asarray(pred).astype(int)
