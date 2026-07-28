"""Hyperparameter handling: small in-fold grids, plus an optional Optuna phase.

Two very different things live here, and conflating them is exactly the mistake
``Binary_Google_MaxAUC_Tuned`` measured (10.6 hours of search produced nested
0.7172 against a non-nested 0.8017 -- an optimism of +0.084):

``grid(...)`` + :func:`select_spec_inner`
    A *small* pre-specified grid, searched **inside each outer fold** on inner
    OOF data.  This is the only tuning whose result may enter the headline,
    because the outer-test subjects never influenced it.

:func:`optuna_refine`
    A wide search whose objective is repeated-CV OOF AUC over the whole cohort.
    That objective sees every subject, so its winning score is **not** an
    unbiased estimate and never becomes the headline.  It is run to answer a
    diagnostic question -- how much does a big search inflate the apparent score
    at this sample size? -- and the run reports that gap explicitly.

Grid sizes are deliberately tiny.  At 12 positives an inner fold has 2-3
positives, so the inner AUC used to pick between configurations is itself made
of a handful of comparisons; offering it 50 options mostly lets it fit noise.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .ensemble import safe_auc
from .models import ModelSpec, available_models


def _probe_optuna() -> bool:
    import importlib.util

    return importlib.util.find_spec("optuna") is not None


HAS_OPTUNA = _probe_optuna()


# ------------------------------------------------------------- in-fold grid --
_GRIDS: dict[str, list[dict]] = {
    "logreg_en": [
        {"C": 0.05, "l1_ratio": 0.5},
        {"C": 0.3, "l1_ratio": 0.5},
        {"C": 1.0, "l1_ratio": 0.2},
    ],
    "logreg_l2": [{"C": 0.05}, {"C": 0.3}, {"C": 3.0}],
    "svm_rbf": [{"C": 1.0, "gamma": "scale"}, {"C": 10.0, "gamma": "scale"}],
    "svm_linear": [{"C": 0.05}, {"C": 1.0}],
    "random_forest": [
        {"n_estimators": 500, "max_depth": 3},
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 3},
    ],
    "extra_trees": [
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 2},
        {"n_estimators": 500, "max_depth": 4, "min_samples_leaf": 1},
    ],
    "hist_gb": [
        {"max_depth": 2, "learning_rate": 0.05, "max_iter": 200},
        {"max_depth": 3, "learning_rate": 0.03, "max_iter": 300, "l2_regularization": 5.0},
    ],
    "mlp": [{"hidden": 16, "alpha": 0.1}, {"hidden": 32, "alpha": 0.01}],
    "lightgbm": [
        {"num_leaves": 4, "max_depth": 2, "n_estimators": 250, "learning_rate": 0.05},
        {"num_leaves": 7, "max_depth": 3, "n_estimators": 400, "learning_rate": 0.03},
    ],
    "xgboost": [
        {"max_depth": 2, "n_estimators": 250, "learning_rate": 0.05},
        {"max_depth": 3, "n_estimators": 400, "learning_rate": 0.03},
    ],
    "catboost": [
        {"depth": 2, "iterations": 300, "learning_rate": 0.05},
        {"depth": 4, "iterations": 500, "learning_rate": 0.03},
    ],
    "balanced_rf": [{"n_estimators": 500, "max_depth": 3}, {"n_estimators": 500, "max_depth": None}],
    "easy_ensemble": [{"n_estimators": 20}],
    "ydf_gbt": [
        {"num_trees": 150, "max_depth": 3, "shrinkage": 0.05, "l2_regularization": 5.0},
        {"num_trees": 300, "max_depth": 2, "shrinkage": 0.03, "l2_regularization": 10.0},
    ],
    "ydf_rf": [{"num_trees": 300, "max_depth": 4}, {"num_trees": 500, "max_depth": 6}],
    "ydf_gbt_oblique": [
        {"num_trees": 150, "max_depth": 3, "shrinkage": 0.05, "l2_regularization": 5.0},
        {"num_trees": 300, "max_depth": 2, "shrinkage": 0.03, "l2_regularization": 10.0},
    ],
    "tabnet": [
        {"n_d": 8, "n_a": 8, "n_steps": 3, "gamma": 1.3, "lambda_sparse": 1e-3,
         "virtual_batch_size": 16, "lr": 2e-2, "pretrain": True},
    ],
    "rank_mean": [{"margin": 0.05}, {"margin": 0.10}],
    "univariate": [{"C": 1.0}],
    "bag_logreg_l2": [{"n_members": 15, "negative_ratio": 3.0, "C": 0.3}],
    "bag_hist_gb": [{"n_members": 15, "negative_ratio": 3.0, "max_depth": 2}],
}


def grid(model: str, *, level: str = "small") -> list[dict]:
    """Configurations to try for one model inside a fold."""

    configurations = _GRIDS.get(model, [{}])
    if level == "single":
        return [dict(configurations[0])]
    return [dict(c) for c in configurations]


def candidates(models: Sequence[str], *, level: str = "small") -> list[list[ModelSpec]]:
    """Nested-loop candidate list: one inner list of variants per model."""

    out: list[list[ModelSpec]] = []
    for model in models:
        out.append([ModelSpec(name=model, params=params) for params in grid(model, level=level)])
    return out


def default_specs(models: Sequence[str]) -> list[ModelSpec]:
    return [ModelSpec(name=model, params=grid(model, level="single")[0]) for model in models]


def select_spec_inner(variants: Sequence[ModelSpec], score_fn: Callable[[ModelSpec], np.ndarray],
                      y: np.ndarray) -> tuple[ModelSpec, np.ndarray, float]:
    """Choose the variant with the best inner-OOF AUC.

    ``score_fn`` must return inner OOF scores computed on the outer-training
    block only; the caller (``engine.nested_selection_cv``) guarantees that.
    """

    best_spec, best_scores, best_auc = None, None, -1.0
    errors: list[str] = []
    for spec in variants:
        try:
            scores = score_fn(spec)
        except Exception as error:
            errors.append(f"{spec.name}: {type(error).__name__}: {error}")
            continue
        auc = safe_auc(y, scores)
        if auc > best_auc:
            best_spec, best_scores, best_auc = spec, scores, auc
    if best_spec is None:
        raise RuntimeError("; ".join(errors) or "no usable variant")
    return best_spec, best_scores, float(best_auc)


# ---------------------------------------------------------------- Optuna -----
_SEARCH_SPACES: dict[str, Callable] = {}


def _space_logreg_en(trial):
    return {
        "C": trial.suggest_float("C", 1e-3, 30.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
    }


def _space_hist_gb(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 4),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.1, 30.0, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 15),
    }


def _space_random_forest(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
        "max_depth": trial.suggest_categorical("max_depth", [2, 3, 4, 6, None]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3]),
    }


def _space_extra_trees(trial):
    return _space_random_forest(trial)


def _space_svm_rbf(trial):
    return {
        "C": trial.suggest_float("C", 1e-2, 100.0, log=True),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }


def _space_lightgbm(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 3, 15),
        "max_depth": trial.suggest_int("max_depth", 2, 4),
        "min_child_samples": trial.suggest_int("min_child_samples", 2, 15),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
    }


def _space_xgboost(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 4),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 8.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
    }


def _space_catboost(trial):
    return {
        "iterations": trial.suggest_int("iterations", 200, 800, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 2, 5),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
    }


def _space_ydf(trial):
    return {
        "num_trees": trial.suggest_int("num_trees", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "shrinkage": trial.suggest_float("shrinkage", 0.01, 0.15, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.5, 30.0, log=True),
        "min_examples": trial.suggest_int("min_examples", 2, 12),
    }


def _space_ydf_rf(trial):
    return {
        "num_trees": trial.suggest_int("num_trees", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_examples": trial.suggest_int("min_examples", 2, 12),
    }


def _space_tabnet(trial):
    n_d = trial.suggest_int("n_d", 4, 24, step=4)
    return {
        "n_d": n_d,
        "n_a": trial.suggest_int("n_a", 4, 24, step=4),
        "n_steps": trial.suggest_int("n_steps", 2, 5),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-2, log=True),
        "virtual_batch_size": trial.suggest_categorical("virtual_batch_size", [8, 16, 32]),
        "lr": trial.suggest_float("lr", 5e-3, 5e-2, log=True),
        "patience": trial.suggest_int("patience", 10, 30),
        "pretrain": trial.suggest_categorical("pretrain", [True, False]),
    }


_SEARCH_SPACES.update(
    {
        "logreg_en": _space_logreg_en,
        "hist_gb": _space_hist_gb,
        "random_forest": _space_random_forest,
        "extra_trees": _space_extra_trees,
        "svm_rbf": _space_svm_rbf,
        "lightgbm": _space_lightgbm,
        "xgboost": _space_xgboost,
        "catboost": _space_catboost,
        "ydf_gbt": _space_ydf,
        "ydf_gbt_oblique": _space_ydf,
        "ydf_rf": _space_ydf_rf,
        "tabnet": _space_tabnet,
    }
)


def has_search_space(model: str) -> bool:
    return model in _SEARCH_SPACES


def optuna_refine(model: str, objective_fn: Callable[[dict], float], *, n_trials: int,
                  seed: int = 0, log: Callable[[str], None] = lambda _m: None) -> dict:
    """Bayesian refinement of one model.

    ``objective_fn`` receives a parameter dict and returns repeated-CV OOF AUC.
    Because that objective is computed on every subject, the best value it finds
    is optimistic by construction; the caller re-evaluates the winner inside the
    nested loop and reports both numbers side by side.

    Falls back to a seeded random search over the same spaces when Optuna is not
    installed, so the phase still runs on a bare Colab image.
    """

    if not has_search_space(model):
        return {"model": model, "skipped": "no search space defined"}
    if n_trials <= 0:
        return {"model": model, "skipped": "n_trials <= 0"}

    history: list[dict] = []
    if HAS_OPTUNA:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        def _objective(trial):
            params = _SEARCH_SPACES[model](trial)
            value = float(objective_fn(params))
            history.append({"trial": trial.number, "value": value, "params": dict(params)})
            return value

        study.optimize(_objective, n_trials=int(n_trials), catch=(Exception,))
        best = study.best_trial
        log(f"optuna[{model}] best={best.value:.4f} params={best.params}")
        return {
            "model": model,
            "backend": "optuna-tpe",
            "n_trials": int(n_trials),
            "best_value_non_nested": float(best.value),
            "best_params": dict(history[int(np.argmax([h['value'] for h in history]))]["params"])
            if history else dict(best.params),
            "history": history,
        }

    # Seeded random search fallback over the same distributions.
    rng = np.random.default_rng(seed)
    sampler = _RandomTrial(rng)
    best_value, best_params = -1.0, {}
    for index in range(int(n_trials)):
        sampler.reset()
        params = _SEARCH_SPACES[model](sampler)
        try:
            value = float(objective_fn(params))
        except Exception:
            continue
        history.append({"trial": index, "value": value, "params": dict(params)})
        if value > best_value:
            best_value, best_params = value, dict(params)
    return {
        "model": model,
        "backend": "random-search (optuna not installed)",
        "n_trials": int(n_trials),
        "best_value_non_nested": float(best_value),
        "best_params": best_params,
        "history": history,
    }


class _RandomTrial:
    """Minimal Optuna-trial shim so the same spaces drive the fallback search."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def reset(self) -> None:
        return None

    def suggest_float(self, _name: str, low: float, high: float, *, log: bool = False,
                      step: float | None = None) -> float:
        if log:
            return float(np.exp(self.rng.uniform(np.log(low), np.log(high))))
        return float(self.rng.uniform(low, high))

    def suggest_int(self, _name: str, low: int, high: int, *, step: int = 1) -> int:
        values = np.arange(low, high + 1, step)
        return int(self.rng.choice(values))

    def suggest_categorical(self, _name: str, choices: Sequence):
        return choices[int(self.rng.integers(0, len(choices)))]


def default_model_pool(requested: Sequence[str] | None) -> tuple[str, ...]:
    """Resolve the requested model list against what is installed."""

    installed = set(available_models())
    if requested is None:
        return tuple(sorted(installed))
    resolved = [name for name in requested if name in installed]
    if not resolved:
        raise ValueError(
            f"None of the requested models are available: {list(requested)}; "
            f"installed = {sorted(installed)}"
        )
    return tuple(resolved)


__all__ = [
    "HAS_OPTUNA", "candidates", "default_model_pool", "default_specs", "grid",
    "has_search_space", "optuna_refine", "select_spec_inner",
]
