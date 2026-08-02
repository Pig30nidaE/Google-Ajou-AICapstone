"""Hyperparameter search spaces, sampled by the inner-fold tuner.

Two things worth noting:

1. **Feature selection is part of the search space.**  ``top_k`` (how many
   features survive the fold-internal univariate ranking) and ``corr_threshold``
   (redundancy pruning) are sampled alongside the model hyperparameters, so the
   tuner optimises the *pipeline*, not just the estimator.  With ~110 candidate
   features and only ~110 training subjects per outer fold, how aggressively you
   prune matters more than any single tree parameter.

2. **Every draw is scored by inner-fold ROC-AUC only.**  The outer fold and the
   33-subject validation set are never consulted, so the reported number stays
   an honest estimate of out-of-sample ranking quality.

Random search (not Optuna) is deliberate: no extra dependency, deterministic
under a seed, trivially parallel, and adequate for a space this small.  A
two-stage successive-halving screen (see ``engine.tune_kind``) is what buys the
extra candidates, not a smarter proposal distribution.
"""

from __future__ import annotations

import numpy as np

# Shared pipeline knobs, sampled for every learner kind.
SELECTION_SPACE = {
    "top_k": [10, 15, 20, 30, 45, 70, 0],       # 0 == keep everything that survives pruning
    "corr_threshold": [0.90, 0.95, 0.99, 1.01],  # 1.01 == no correlation pruning
}

MODEL_SPACES = {
    "ydf_gbt": {
        "num_trees": [150, 250, 400, 600, 900],
        "max_depth": [2, 3, 4, 5, 6],
        "min_examples": [3, 5, 8, 12, 20],
        "shrinkage": [0.01, 0.02, 0.03, 0.05, 0.08, 0.12],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "num_candidate_attributes_ratio": [0.3, 0.5, 0.7, 0.9, 1.0],
        "l2_regularization": [0.0, 0.5, 1.0, 3.0, 10.0],
    },
    "ydf_rf": {
        "num_trees": [300, 500, 800, 1200],
        "max_depth": [4, 6, 8, 12, 16],
        "min_examples": [2, 3, 5, 8, 12],
        "num_candidate_attributes_ratio": [0.2, 0.3, 0.5, 0.7, 1.0],
    },
    "logreg": {
        "C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
        "penalty": ["l2", "l1", "elasticnet"],
        "l1_ratio": [0.15, 0.5, 0.85],
    },
    "svm": {
        "C": [0.1, 0.3, 1.0, 3.0, 10.0],
        "gamma": ["scale", "auto", 0.005, 0.01, 0.05],
    },
}

# Oblique variants reuse the base tree space plus YDF's sparse-oblique knobs.
OBLIQUE_EXTRA = {
    "sparse_oblique_normalization": ["NONE", "STANDARD_DEVIATION", "MIN_MAX"],
    "sparse_oblique_num_projections_exponent": [0.5, 1.0, 1.5],
    "sparse_oblique_projection_density_factor": [1.0, 2.0, 3.0, 5.0],
}
MODEL_SPACES["ydf_gbt_oblique"] = {**MODEL_SPACES["ydf_gbt"], **OBLIQUE_EXTRA}
MODEL_SPACES["ydf_rf_oblique"] = {**MODEL_SPACES["ydf_rf"], **OBLIQUE_EXTRA}


def _sample_value(value, rng: np.random.Generator):
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return float(value)
    return int(value)


def sample_params(kind: str, rng: np.random.Generator) -> dict:
    """One random pipeline configuration (selection knobs + model knobs)."""

    space = {**SELECTION_SPACE, **MODEL_SPACES[kind]}
    out: dict = {}
    for key, values in space.items():
        out[key] = _sample_value(values[int(rng.integers(len(values)))], rng)
    # l1_ratio only applies to elasticnet; drop it otherwise so keys stay canonical
    if kind == "logreg" and out.get("penalty") != "elasticnet":
        out.pop("l1_ratio", None)
    return out


def default_params(kind: str) -> dict:
    """Middle-of-the-space defaults, used for smoke runs and as a search anchor."""

    space = {**SELECTION_SPACE, **MODEL_SPACES[kind]}
    out = {key: values[len(values) // 2] for key, values in space.items()}
    if kind == "logreg":
        out["penalty"] = "l2"
        out.pop("l1_ratio", None)
    return out


def params_key(kind: str, params: dict) -> str:
    """Stable key so duplicate random draws are only evaluated once."""

    return kind + "|" + ",".join(f"{k}={params[k]}" for k in sorted(params))


def space_size(kind: str) -> int:
    space = {**SELECTION_SPACE, **MODEL_SPACES[kind]}
    total = 1
    for values in space.values():
        total *= len(values)
    return total


__all__ = ["MODEL_SPACES", "OBLIQUE_EXTRA", "SELECTION_SPACE", "default_params",
           "params_key", "sample_params", "space_size"]
