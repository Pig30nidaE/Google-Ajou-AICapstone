"""Google-first model candidates for small subject-level tabular data.

Primary candidates use Google's Yggdrasil Decision Forests (YDF).  A TabNet
candidate is retained because TabNet originated at Google Research and can add
neural diversity, but inner OOF selection may assign it zero ensemble weight.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import joblib
import numpy as np
import pandas as pd

from feature_engineering import CLASS_NAMES
from preprocessing import FoldFeatureSelector


CANDIDATE_NAMES = (
    "ydf_multiclass",
    "ydf_hierarchical",
    "ydf_random_forest",
    "ydf_ovr",
    "tabnet",
)
GOOGLE_MODEL_EVIDENCE = {
    "ydf_multiclass": "Google Yggdrasil Decision Forests GradientBoostedTreesLearner",
    "ydf_hierarchical": "Google YDF binary cascade (CN vs impaired, then MCI vs DEM)",
    "ydf_random_forest": "Google YDF probability-voting RandomForestLearner",
    "ydf_ovr": "Google YDF one-vs-rest probability ensemble",
    "tabnet": "TabNet, published by Google Research",
}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def balanced_class_weights(y: np.ndarray, power: float) -> np.ndarray:
    target = np.asarray(y, dtype=np.int64)
    counts = np.bincount(target, minlength=int(target.max()) + 1).astype(float)
    if np.any(counts == 0):
        raise ValueError(f"All fitted classes must be present; counts={counts.tolist()}")
    raw = len(target) / (len(counts) * counts)
    weights = np.power(raw, float(np.clip(power, 0.0, 1.25)))
    return weights / weights.mean()


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    values = np.clip(values, 1e-9, 1.0)
    return values / values.sum(axis=1, keepdims=True)


class YDFClassifierAdapter:
    def __init__(self, model: Any, feature_names: list[str], target_names: tuple[str, ...]):
        self.model = model
        self.feature_names = list(feature_names)
        self.target_names = tuple(target_names)

    def _frame(self, X: np.ndarray) -> pd.DataFrame:
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"YDF input shape mismatch: {values.shape}; "
                f"expected (*, {len(self.feature_names)})"
            )
        return pd.DataFrame(values, columns=self.feature_names)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.model.predict(self._frame(X)), dtype=np.float64)
        model_classes = tuple(str(value) for value in self.model.label_classes())
        if raw.ndim == 1:
            if len(model_classes) != 2:
                raise ValueError(
                    f"One-dimensional YDF output with {len(model_classes)} classes"
                )
            by_model_order = np.column_stack([1.0 - raw, raw])
        elif raw.ndim == 2 and raw.shape[1] == len(model_classes):
            by_model_order = raw
        else:
            raise ValueError(
                f"Unexpected YDF output {raw.shape} for classes {model_classes}"
            )
        missing = sorted(set(self.target_names) - set(model_classes))
        if missing:
            raise ValueError(f"YDF model is missing expected class(es): {missing}")
        order = [model_classes.index(name) for name in self.target_names]
        return normalize_probabilities(by_model_order[:, order])

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        self.model.save(str(output / "model"))
        (output / "adapter.json").write_text(
            json.dumps(
                {
                    "feature_names": self.feature_names,
                    "target_names": list(self.target_names),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "YDFClassifierAdapter":
        import ydf

        root = Path(path)
        metadata = json.loads((root / "adapter.json").read_text(encoding="utf-8"))
        model = ydf.load_model(str(root / "model"))
        return cls(
            model,
            feature_names=list(metadata["feature_names"]),
            target_names=tuple(metadata["target_names"]),
        )


class TabNetAdapter:
    def __init__(self, model: Any):
        self.model = model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return normalize_probabilities(self.model.predict_proba(np.asarray(X, dtype=np.float32)))

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(output))

    @classmethod
    def load(cls, path: str | Path) -> "TabNetAdapter":
        from pytorch_tabnet.tab_model import TabNetClassifier

        model = TabNetClassifier()
        model.load_model(str(path))
        return cls(model)


@dataclass
class FittedCandidate:
    name: str
    selectors: dict[str, FoldFeatureSelector]
    models: dict[str, Any]
    params: dict[str, Any]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.name in {"ydf_multiclass", "ydf_random_forest", "tabnet"}:
            transformed = self.selectors["main"].transform(X)
            return self.models["main"].predict_proba(transformed)
        if self.name == "ydf_hierarchical":
            cn_values = self.selectors["cn"].transform(X)
            p_cn_stage = self.models["cn"].predict_proba(cn_values)
            p_cn = p_cn_stage[:, 0]
            p_impaired = p_cn_stage[:, 1]
            stage_values = self.selectors["stage"].transform(X)
            p_stage = self.models["stage"].predict_proba(stage_values)
            return normalize_probabilities(
                np.column_stack(
                    [p_cn, p_impaired * p_stage[:, 0], p_impaired * p_stage[:, 1]]
                )
            )
        if self.name == "ydf_ovr":
            columns = []
            for class_id, class_name in enumerate(CLASS_NAMES):
                key = f"class_{class_id}"
                transformed = self.selectors[key].transform(X)
                p_binary = self.models[key].predict_proba(transformed)
                # Target order was forced to OTHER, then the named class.
                columns.append(p_binary[:, 1])
            return normalize_probabilities(np.column_stack(columns))
        raise ValueError(f"Unknown fitted candidate: {self.name}")

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "google_model": GOOGLE_MODEL_EVIDENCE[self.name],
            "params": self.params,
            "selectors": {key: selector.manifest() for key, selector in self.selectors.items()},
        }

    def save(self, path: str | Path) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        for key, selector in self.selectors.items():
            joblib.dump(selector, root / f"selector_{key}.joblib")
        for key, model in self.models.items():
            if isinstance(model, YDFClassifierAdapter):
                model.save(root / f"ydf_{key}")
            elif isinstance(model, TabNetAdapter):
                model.save(root / f"tabnet_{key}")
            else:
                raise TypeError(f"Unsupported checkpoint model: {type(model)}")
        (root / "candidate_manifest.json").write_text(
            json.dumps(self.manifest(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "FittedCandidate":
        root = Path(path)
        manifest = json.loads((root / "candidate_manifest.json").read_text(encoding="utf-8"))
        name = str(manifest["name"])
        selectors: dict[str, FoldFeatureSelector] = {}
        models: dict[str, Any] = {}
        for selector_path in sorted(root.glob("selector_*.joblib")):
            key = selector_path.stem.removeprefix("selector_")
            selectors[key] = joblib.load(selector_path)
        for model_path in sorted(root.glob("ydf_*")):
            if model_path.is_dir():
                key = model_path.name.removeprefix("ydf_")
                models[key] = YDFClassifierAdapter.load(model_path)
        for model_path in sorted(root.glob("tabnet_*.zip")):
            key = model_path.stem.removeprefix("tabnet_")
            models[key] = TabNetAdapter.load(model_path)
        return cls(name=name, selectors=selectors, models=models, params=manifest["params"])


def _selector(params: dict[str, Any]) -> FoldFeatureSelector:
    return FoldFeatureSelector(
        max_features=int(params["max_features"]),
        max_missing_fraction=float(params.get("max_missing_fraction", 0.35)),
        correlation_threshold=float(params.get("correlation_threshold", 0.975)),
        cn_focus=float(params.get("cn_focus", 0.45)),
        min_features_per_modality=int(params.get("min_features_per_modality", 12)),
    )


def _fit_ydf_classifier(
    X: np.ndarray,
    y: np.ndarray,
    target_names: tuple[str, ...],
    params: dict[str, Any],
    seed: int,
    selected_feature_names: list[str] | None = None,
) -> YDFClassifierAdapter:
    import ydf

    set_global_seed(seed)
    target = np.asarray(y, dtype=np.int64)
    feature_names = (
        list(selected_feature_names)
        if selected_feature_names is not None
        else [f"f_{index:04d}" for index in range(X.shape[1])]
    )
    if len(feature_names) != X.shape[1]:
        raise ValueError("Selected feature names do not match the YDF input width")
    frame = pd.DataFrame(np.asarray(X, dtype=np.float32), columns=feature_names)
    frame["label"] = [target_names[int(value)] for value in target]
    weights = balanced_class_weights(target, float(params.get("class_weight_power", 0.5)))
    class_weights = {target_names[index]: float(weights[index]) for index in range(len(weights))}
    loss = (
        "BINOMIAL_LOG_LIKELIHOOD"
        if len(target_names) == 2
        else "MULTINOMIAL_LOG_LIKELIHOOD"
    )
    learner = ydf.GradientBoostedTreesLearner(
        label="label",
        label_classes=list(target_names),
        class_weights=class_weights,
        loss=loss,
        num_trees=int(params["num_trees"]),
        max_depth=int(params["max_depth"]),
        min_examples=int(params["min_examples"]),
        shrinkage=float(params["shrinkage"]),
        subsample=float(params["subsample"]),
        num_candidate_attributes_ratio=float(params["num_candidate_attributes_ratio"]),
        use_hessian_gain=bool(params["use_hessian_gain"]),
        l2_regularization=float(params["l2_regularization"]),
        validation_ratio=0.0,
        random_seed=int(seed),
        num_threads=int(params.get("num_threads", 32)),
    )
    model = learner.train(frame)
    return YDFClassifierAdapter(model, feature_names, target_names)


def _fit_tabnet(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    seed: int,
) -> TabNetAdapter:
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier

    set_global_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("The full TabNet candidate requires a CUDA runtime")
    model = TabNetClassifier(
        n_d=int(params["n_d"]),
        n_a=int(params["n_d"]),
        n_steps=int(params["n_steps"]),
        gamma=float(params["gamma"]),
        lambda_sparse=float(params["lambda_sparse"]),
        mask_type=str(params["mask_type"]),
        optimizer_fn=torch.optim.AdamW,
        optimizer_params={
            "lr": float(params["lr"]),
            "weight_decay": float(params["weight_decay"]),
        },
        scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
        scheduler_params={
            "T_max": int(params["epochs"]),
            "eta_min": float(params["lr"]) * 0.02,
        },
        seed=int(seed),
        verbose=0,
        device_name="cuda",
    )
    weights = balanced_class_weights(y, float(params.get("class_weight_power", 0.5)))
    model.fit(
        X_train=np.asarray(X, dtype=np.float32),
        y_train=np.asarray(y, dtype=np.int64),
        max_epochs=int(params["epochs"]),
        patience=0,
        batch_size=min(int(params["batch_size"]), max(16, len(y))),
        virtual_batch_size=min(int(params["virtual_batch_size"]), max(8, len(y))),
        num_workers=0,
        drop_last=False,
        weights={index: float(weights[index]) for index in range(3)},
    )
    return TabNetAdapter(model)


def _fit_ydf_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    seed: int,
    selected_feature_names: list[str],
) -> YDFClassifierAdapter:
    import ydf

    set_global_seed(seed)
    target = np.asarray(y, dtype=np.int64)
    feature_names = list(selected_feature_names)
    if len(feature_names) != X.shape[1]:
        raise ValueError("Selected feature names do not match the Random Forest input width")
    frame = pd.DataFrame(np.asarray(X, dtype=np.float32), columns=feature_names)
    frame["label"] = [CLASS_NAMES[int(value)] for value in target]
    weights = balanced_class_weights(target, float(params.get("class_weight_power", 0.5)))
    learner = ydf.RandomForestLearner(
        label="label",
        label_classes=list(CLASS_NAMES),
        class_weights={CLASS_NAMES[index]: float(weights[index]) for index in range(3)},
        num_trees=int(params["num_trees"]),
        max_depth=int(params["max_depth"]),
        min_examples=int(params["min_examples"]),
        num_candidate_attributes_ratio=float(params["num_candidate_attributes_ratio"]),
        bootstrap_training_dataset=True,
        bootstrap_size_ratio=float(params["bootstrap_size_ratio"]),
        sampling_with_replacement=True,
        winner_take_all=False,
        random_seed=int(seed),
        num_threads=int(params.get("num_threads", 32)),
    )
    model = learner.train(frame)
    return YDFClassifierAdapter(model, feature_names, CLASS_NAMES)


def fit_candidate(
    candidate_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict[str, Any],
    seed: int,
) -> FittedCandidate:
    """Fit one complete candidate, including all fold-local selectors."""

    target = np.asarray(y, dtype=np.int64)
    if candidate_name == "ydf_multiclass":
        selector = _selector(params)
        values = selector.fit_transform(X, target, task="multiclass")
        model = _fit_ydf_classifier(
            values,
            target,
            CLASS_NAMES,
            params,
            seed,
            selector.selected_feature_names,
        )
        return FittedCandidate(candidate_name, {"main": selector}, {"main": model}, params)

    if candidate_name == "ydf_hierarchical":
        cn_target = (target != 0).astype(np.int64)
        cn_selector = _selector(params)
        cn_values = cn_selector.fit_transform(X, cn_target, task="cn_vs_impaired")
        cn_model = _fit_ydf_classifier(
            cn_values,
            cn_target,
            ("CN", "IMPAIRED"),
            params,
            seed,
            cn_selector.selected_feature_names,
        )
        impaired = target != 0
        stage_target = (target[impaired] == 2).astype(np.int64)
        stage_selector = _selector(params)
        stage_values = stage_selector.fit_transform(
            X.loc[impaired].reset_index(drop=True),
            stage_target,
            task="mci_vs_dem",
        )
        stage_model = _fit_ydf_classifier(
            stage_values,
            stage_target,
            ("MCI", "DEM"),
            params,
            seed + 10_007,
            stage_selector.selected_feature_names,
        )
        return FittedCandidate(
            candidate_name,
            {"cn": cn_selector, "stage": stage_selector},
            {"cn": cn_model, "stage": stage_model},
            params,
        )

    if candidate_name == "ydf_random_forest":
        selector = _selector(params)
        values = selector.fit_transform(X, target, task="multiclass")
        model = _fit_ydf_random_forest(
            values,
            target,
            params,
            seed,
            selector.selected_feature_names,
        )
        return FittedCandidate(candidate_name, {"main": selector}, {"main": model}, params)

    if candidate_name == "ydf_ovr":
        selectors: dict[str, FoldFeatureSelector] = {}
        models: dict[str, YDFClassifierAdapter] = {}
        for class_id, class_name in enumerate(CLASS_NAMES):
            key = f"class_{class_id}"
            binary_target = (target == class_id).astype(np.int64)
            selector = _selector(params)
            values = selector.fit_transform(
                X,
                target,
                task="one_vs_rest",
                positive_class=class_id,
            )
            model = _fit_ydf_classifier(
                values,
                binary_target,
                ("OTHER", class_name),
                params,
                seed + class_id * 10_007,
                selector.selected_feature_names,
            )
            selectors[key] = selector
            models[key] = model
        return FittedCandidate(candidate_name, selectors, models, params)

    if candidate_name == "tabnet":
        selector = _selector(params)
        values = selector.fit_transform(X, target, task="multiclass")
        model = _fit_tabnet(values, target, params, seed)
        return FittedCandidate(candidate_name, {"main": selector}, {"main": model}, params)

    raise ValueError(f"Unknown candidate: {candidate_name}")


def suggest_parameters(candidate_name: str, trial: Any, *, fast: bool) -> dict[str, Any]:
    if candidate_name == "ydf_hierarchical":
        feature_choices = [24, 32, 48, 64]
    elif candidate_name == "ydf_ovr":
        feature_choices = [32, 48, 64]
    elif candidate_name == "ydf_random_forest":
        feature_choices = [48, 64, 96, 128]
    else:
        feature_choices = [48, 64, 96]
    common = {
        "max_features": trial.suggest_categorical("max_features", feature_choices),
        "max_missing_fraction": 0.35,
        "correlation_threshold": trial.suggest_categorical(
            "correlation_threshold", [0.95, 0.975, 0.99]
        ),
        "cn_focus": trial.suggest_float("cn_focus", 0.25, 0.75),
        "min_features_per_modality": 12,
        "class_weight_power": trial.suggest_float("class_weight_power", 0.0, 1.0),
    }
    if candidate_name == "ydf_random_forest":
        return {
            **common,
            "num_trees": 60
            if fast
            else trial.suggest_categorical("num_trees", [600, 1000, 1600, 2400]),
            "max_depth": trial.suggest_categorical("max_depth", [4, 6, 8, 12]),
            "min_examples": trial.suggest_int("min_examples", 2, 10),
            "num_candidate_attributes_ratio": trial.suggest_categorical(
                "num_candidate_attributes_ratio", [0.25, 0.35, 0.50, 0.75]
            ),
            "bootstrap_size_ratio": trial.suggest_categorical(
                "bootstrap_size_ratio", [0.65, 0.80, 1.0]
            ),
            "num_threads": 32,
        }
    if candidate_name.startswith("ydf_"):
        return {
            **common,
            "num_trees": 40
            if fast
            else trial.suggest_categorical("num_trees", [300, 600, 1000, 1600]),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_examples": trial.suggest_int("min_examples", 2, 14),
            "shrinkage": trial.suggest_categorical(
                "shrinkage", [0.02, 0.03, 0.05, 0.08, 0.10]
            ),
            "subsample": trial.suggest_categorical("subsample", [0.70, 0.85, 1.0]),
            "num_candidate_attributes_ratio": trial.suggest_categorical(
                "num_candidate_attributes_ratio", [0.35, 0.50, 0.75, 1.0]
            ),
            "use_hessian_gain": trial.suggest_categorical(
                "use_hessian_gain", [False, True]
            ),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1e-3, 20.0, log=True
            ),
            "num_threads": 32,
        }
    if candidate_name == "tabnet":
        n_d = trial.suggest_categorical("n_d", [16, 24, 32, 48, 64])
        return {
            **common,
            "n_d": n_d,
            "n_steps": trial.suggest_int("n_steps", 3, 7),
            "gamma": trial.suggest_float("gamma", 1.0, 1.8),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-7, 1e-3, log=True),
            "mask_type": trial.suggest_categorical("mask_type", ["entmax", "sparsemax"]),
            "lr": trial.suggest_float("lr", 3e-4, 3e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
            "epochs": 12
            if fast
            else trial.suggest_categorical("epochs", [200, 350, 500]),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "virtual_batch_size": trial.suggest_categorical(
                "virtual_batch_size", [16, 32, 64]
            ),
        }
    raise ValueError(f"Unknown candidate: {candidate_name}")
