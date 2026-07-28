"""Fixed, deliberately small configurations -- searching would cost more than it buys.

``Binary_Google_MaxAUC_Tuned`` measured what an aggressive search does to this
dataset: the selected ``top_k`` never converged across folds
(``[70, 10, 70, 10, 70, 15, 0, 10, 70, 30, 20, 45, 20, 10, 45]``), inner-fold AUC
ran +0.053 above outer-fold AUC on average, and the tuned 151-feature pipeline
finished *below* an untuned 39-feature one.  That was with 56 positives; here
there are 12, so the same search would be strictly noisier.

The response is to spend the compute budget on **repeats** (which shrink the
variance of the estimate) instead of on **search** (which here mostly fits
noise).  Each learner therefore gets one pre-specified configuration, and only
``top_k`` -- kept to a handful of features -- varies, chosen inside the fold.
``SEARCH_SPACE`` exists for an explicit opt-in comparison, not for the default.
"""

from __future__ import annotations

# Feature budget per learner. Small on purpose: with ~9 training positives,
# anything past a handful of columns is fitting individual patients.
DEFAULT_TOP_K = 5
DEFAULT_CORR_THRESHOLD = 0.90

DEFAULTS = {
    "univariate": {"C": 1.0, "top_k": 0, "corr_threshold": 1.01},   # ranks all, keeps 1
    "logreg": {"C": 0.3, "top_k": DEFAULT_TOP_K, "corr_threshold": DEFAULT_CORR_THRESHOLD},
    "ydf_gbt": {
        "num_trees": 120, "max_depth": 2, "min_examples": 5, "shrinkage": 0.05,
        "subsample": 0.9, "num_candidate_attributes_ratio": 0.8, "l2_regularization": 5.0,
        "top_k": DEFAULT_TOP_K, "corr_threshold": DEFAULT_CORR_THRESHOLD,
    },
    "ydf_rf": {
        "num_trees": 400, "max_depth": 3, "min_examples": 5,
        "num_candidate_attributes_ratio": 0.5,
        "top_k": DEFAULT_TOP_K, "corr_threshold": DEFAULT_CORR_THRESHOLD,
    },
    "ydf_gbt_oblique": {
        "num_trees": 120, "max_depth": 2, "min_examples": 5, "shrinkage": 0.05,
        "subsample": 0.9, "num_candidate_attributes_ratio": 0.8, "l2_regularization": 5.0,
        "sparse_oblique_normalization": "STANDARD_DEVIATION",
        "sparse_oblique_num_projections_exponent": 1.0,
        "top_k": DEFAULT_TOP_K, "corr_threshold": DEFAULT_CORR_THRESHOLD,
    },
}

# Opt-in only (DEMSCREEN_SEARCH=1): a minimal grid over the two knobs that
# actually matter at this sample size.
SEARCH_SPACE = {
    "top_k": [1, 3, 5, 8],
    "C": [0.1, 0.3, 1.0],
}


def default_params(kind: str, *, smote: bool = False) -> dict:
    params = dict(DEFAULTS[kind])
    params["smote"] = bool(smote)
    return params


def search_configs(kind: str, *, smote: bool = False) -> list[dict]:
    """Small explicit grid; only ``top_k`` (and ``C`` for linear models) moves."""

    base = default_params(kind, smote=smote)
    configs = []
    top_ks = [0] if kind == "univariate" else SEARCH_SPACE["top_k"]
    c_values = SEARCH_SPACE["C"] if kind in {"logreg", "univariate"} else [base.get("C", 1.0)]
    for top_k in top_ks:
        for c in c_values:
            config = dict(base)
            config["top_k"] = top_k
            config["C"] = c
            configs.append(config)
    return configs


__all__ = ["DEFAULTS", "DEFAULT_TOP_K", "SEARCH_SPACE", "default_params", "search_configs"]
