"""Random-search hyperparameter spaces for the Google YDF models.

Selection is done by repeated nested cross-validation on the 141 training
subjects only (see train.run_tuning_experiment).  The 33-subject validation set
is never used to pick hyperparameters, so it stays an honest held-out test.

Random search (rather than Optuna) is deliberate: it needs no extra dependency,
is deterministic under a seed, and is more than adequate for this small,
low-dimensional space where prior experiments already show a ~0.71 AUROC
ceiling.  The point of tuning is to find the best-within-ceiling configuration,
not to break the data-imposed limit.
"""

from __future__ import annotations

import numpy as np

# Search spaces centred on the current defaults, kept small so each YDF fit is
# fast (num_trees bounded) and the whole search fits comfortably under 20 hours.
GBT_SPACE = {
    "num_trees": [200, 300, 400, 500, 600],
    "max_depth": [3, 4, 5, 6],
    "min_examples": [5, 8, 10, 15],
    "shrinkage": [0.02, 0.03, 0.05, 0.08, 0.10],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "num_candidate_attributes_ratio": [0.6, 0.8, 1.0],
    "l2_regularization": [0.0, 0.5, 1.0, 2.0],
}
RF_SPACE = {
    "num_trees": [300, 500, 800],
    "max_depth": [4, 5, 6, 8],
    "min_examples": [3, 5, 8],
    "num_candidate_attributes_ratio": [0.4, 0.6, 0.8],
}


def _sample_space(space: dict, rng: np.random.Generator) -> dict:
    out = {}
    for key, values in space.items():
        choice = values[int(rng.integers(len(values)))]
        out[key] = float(choice) if isinstance(choice, float) else int(choice)
    return out


def sample_config(rng: np.random.Generator) -> dict:
    """One random {gbt: {...}, rf: {...}} hyperparameter configuration."""

    return {"gbt": _sample_space(GBT_SPACE, rng), "rf": _sample_space(RF_SPACE, rng)}


def config_key(config: dict) -> str:
    """Stable string key to skip duplicate random draws."""

    parts = []
    for model in ("gbt", "rf"):
        parts.append(model + ":" + ",".join(f"{k}={config[model][k]}" for k in sorted(config[model])))
    return "|".join(parts)


__all__ = ["GBT_SPACE", "RF_SPACE", "config_key", "sample_config"]
