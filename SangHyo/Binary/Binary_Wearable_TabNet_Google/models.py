"""Binary TabNet primary model and Google YDF diversity model."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd


CLASS_NAMES = ("CN", "MCI_DEM")
MODEL_NAMES = ("tabnet", "ydf")
GOOGLE_MODEL_EVIDENCE = {
    "tabnet": {
        "origin": "Google Research",
        "paper": "TabNet: Attentive Interpretable Tabular Learning",
    },
    "ydf": {
        "origin": "Google",
        "library": "Yggdrasil Decision Forests",
    },
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


def balanced_class_weights(y: np.ndarray, power: float = 0.35) -> np.ndarray:
    target = np.asarray(y, dtype=np.int64)
    counts = np.bincount(target, minlength=2).astype(float)
    if len(counts) != 2 or np.any(counts == 0):
        raise ValueError(f"Both binary classes are required; counts={counts.tolist()}")
    raw = len(target) / (2.0 * counts)
    values = np.power(raw, float(np.clip(power, 0.0, 1.0)))
    return values / values.mean()


def normalize_probabilities(values: np.ndarray) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] != 2:
        raise ValueError(f"Expected [N,2] probabilities, received {probability.shape}")
    probability = np.clip(probability, 1e-8, 1.0)
    return probability / probability.sum(axis=1, keepdims=True)


class TabNetAdapter:
    def __init__(self, model: Any):
        self.model = model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.model.predict_proba(np.asarray(X, dtype=np.float32))
        return normalize_probabilities(raw)

    def save(self, base_path: str | Path) -> Path:
        path = Path(base_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        saved = Path(self.model.save_model(str(path)))
        if not saved.is_file() or saved.suffix != ".zip":
            raise FileNotFoundError(f"TabNet checkpoint was not created: {saved}")
        return saved

    @classmethod
    def load(cls, path: str | Path, *, device_name: str = "cpu") -> "TabNetAdapter":
        from pytorch_tabnet.tab_model import TabNetClassifier

        model = TabNetClassifier(device_name=device_name, verbose=0)
        model.load_model(str(path))
        return cls(model)


class YDFAdapter:
    def __init__(self, model: Any, feature_names: list[str]):
        self.model = model
        self.feature_names = list(feature_names)

    def _frame(self, X: np.ndarray) -> pd.DataFrame:
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("YDF feature shape differs from the fitted schema")
        return pd.DataFrame(values, columns=self.feature_names)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.model.predict(self._frame(X)), dtype=np.float64)
        model_classes = tuple(str(value) for value in self.model.label_classes())
        if raw.ndim == 1:
            if len(model_classes) != 2:
                raise ValueError("Unexpected one-dimensional multiclass YDF output")
            raw = np.column_stack([1.0 - raw, raw])
        if raw.shape != (len(X), len(model_classes)):
            raise ValueError(
                f"Unexpected YDF output {raw.shape} for classes {model_classes}"
            )
        if set(model_classes) != set(CLASS_NAMES):
            raise ValueError(f"Unexpected YDF labels: {model_classes}")
        order = [model_classes.index(name) for name in CLASS_NAMES]
        return normalize_probabilities(raw[:, order])

    def save(self, path: str | Path) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=False)
        self.model.save(str(root / "model"))
        (root / "adapter.json").write_text(
            json.dumps(
                {"feature_names": self.feature_names, "class_names": list(CLASS_NAMES)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "YDFAdapter":
        import ydf

        root = Path(path)
        metadata = json.loads((root / "adapter.json").read_text(encoding="utf-8"))
        if tuple(metadata["class_names"]) != CLASS_NAMES:
            raise ValueError("Stored YDF class contract differs from this experiment")
        return cls(ydf.load_model(str(root / "model")), metadata["feature_names"])


@dataclass
class PlattCalibrator:
    """Monotonic OOF-fitted probability calibration with identity fallback."""

    model: Any | None = None
    reason: str = "not fitted"

    def fit(self, probability: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        p = np.clip(np.asarray(probability, dtype=np.float64), 1e-5, 1 - 1e-5)
        target = np.asarray(y, dtype=np.int64)
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        candidate = LogisticRegression(C=1.0, solver="lbfgs", random_state=0)
        candidate.fit(logits, target)
        slope = float(candidate.coef_[0, 0])
        if not np.isfinite(slope) or slope <= 0.0:
            self.model = None
            self.reason = f"identity fallback because fitted slope={slope:.6g}"
        else:
            self.model = candidate
            self.reason = "monotonic logistic calibration fitted on inner OOF only"
        return self

    def transform(self, probability: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probability, dtype=np.float64), 1e-5, 1 - 1e-5)
        if self.model is None:
            return p
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        return np.clip(self.model.predict_proba(logits)[:, 1], 1e-7, 1 - 1e-7)


def fit_tabnet(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    *,
    seed: int,
    device_name: str,
) -> TabNetAdapter:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch

    set_global_seed(seed)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Full TabNet run requires a CUDA runtime")
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
        device_name=device_name,
    )
    weights = balanced_class_weights(y, power=float(params["class_weight_power"]))
    weight_map = {class_id: float(weights[class_id]) for class_id in range(2)}
    if set(weight_map) != {0, 1}:
        raise AssertionError("TabNet binary class weights must be keyed by 0 and 1")
    batch_size = min(int(params["batch_size"]), max(16, len(y)))
    virtual_batch = min(int(params["virtual_batch_size"]), batch_size)
    model.fit(
        X_train=np.asarray(X, dtype=np.float32),
        y_train=np.asarray(y, dtype=np.int64),
        max_epochs=int(params["epochs"]),
        patience=0,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch,
        num_workers=0,
        drop_last=False,
        weights=weight_map,
    )
    return TabNetAdapter(model)


def fit_ydf(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    *,
    seed: int,
) -> YDFAdapter:
    import ydf

    set_global_seed(seed)
    names = [f"f_{index:04d}" for index in range(X.shape[1])]
    frame = pd.DataFrame(np.asarray(X, dtype=np.float32), columns=names)
    frame["label"] = [CLASS_NAMES[int(value)] for value in y]
    weights = balanced_class_weights(y, power=float(params["class_weight_power"]))
    learner = ydf.GradientBoostedTreesLearner(
        label="label",
        label_classes=list(CLASS_NAMES),
        class_weights={CLASS_NAMES[i]: float(weights[i]) for i in range(2)},
        loss="BINOMIAL_LOG_LIKELIHOOD",
        num_trees=int(params["num_trees"]),
        max_depth=int(params["max_depth"]),
        min_examples=int(params["min_examples"]),
        shrinkage=float(params["shrinkage"]),
        subsample=float(params["subsample"]),
        num_candidate_attributes_ratio=float(params["num_candidate_attributes_ratio"]),
        l2_regularization=float(params["l2_regularization"]),
        validation_ratio=0.0,
        random_seed=int(seed),
        num_threads=int(params.get("num_threads", 16)),
    )
    return YDFAdapter(learner.train(frame), names)


def suggest_parameters(model_name: str, trial: Any, *, smoke: bool) -> dict[str, Any]:
    max_features = trial.suggest_categorical("max_features", [32] if smoke else [32, 48, 64, 96])
    common = {
        "max_features": max_features,
        "class_weight_power": trial.suggest_float("class_weight_power", 0.0, 0.55),
    }
    if model_name == "tabnet":
        return {
            **common,
            "n_d": trial.suggest_categorical("n_d", [16] if smoke else [16, 24, 32]),
            "n_steps": trial.suggest_int("n_steps", 3, 3 if smoke else 6),
            "gamma": trial.suggest_float("gamma", 1.0, 1.5),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 5e-4, log=True),
            "mask_type": trial.suggest_categorical("mask_type", ["entmax"] if smoke else ["entmax", "sparsemax"]),
            "lr": trial.suggest_float("lr", 5e-4, 1.5e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 3e-3, log=True),
            "epochs": 3 if smoke else trial.suggest_categorical("epochs", [200, 300, 450]),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "virtual_batch_size": trial.suggest_categorical("virtual_batch_size", [16, 32]),
        }
    if model_name == "ydf":
        return {
            **common,
            "num_trees": 30 if smoke else trial.suggest_categorical("num_trees", [300, 600, 1000]),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "min_examples": trial.suggest_int("min_examples", 3, 14),
            "shrinkage": trial.suggest_categorical("shrinkage", [0.03, 0.05, 0.08]),
            "subsample": trial.suggest_categorical("subsample", [0.75, 0.9, 1.0]),
            "num_candidate_attributes_ratio": trial.suggest_categorical(
                "num_candidate_attributes_ratio", [0.35, 0.6, 1.0]
            ),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            "num_threads": min(16, os.cpu_count() or 1),
        }
    raise ValueError(f"Unknown model: {model_name}")


__all__ = [
    "CLASS_NAMES",
    "GOOGLE_MODEL_EVIDENCE",
    "MODEL_NAMES",
    "PlattCalibrator",
    "TabNetAdapter",
    "YDFAdapter",
    "fit_tabnet",
    "fit_ydf",
    "normalize_probabilities",
    "set_global_seed",
    "suggest_parameters",
]
