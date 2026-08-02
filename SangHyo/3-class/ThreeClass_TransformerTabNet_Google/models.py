"""FT-Transformer, TabNet, and Google's YDF model adapters.

The three adapters expose the same ``predict_proba`` and ``save`` interface so
that nested cross-validation can blend them without model-specific shortcuts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random
from typing import Any

import numpy as np


CLASS_NAMES = ("CN", "MCI", "DEM")
MODEL_NAMES = ("transformer", "tabnet", "ydf")


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


def balanced_class_weights(y: np.ndarray, power: float = 0.5) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=3).astype(float)
    if np.any(counts == 0):
        raise ValueError(f"Every training fold must contain all classes; counts={counts}")
    raw = len(y) / (3.0 * counts)
    weights = np.power(raw, power)
    return weights / weights.mean()


@dataclass(frozen=True)
class TransformerConfig:
    n_features: int
    d_token: int = 48
    n_heads: int = 8
    n_layers: int = 3
    ff_multiplier: float = 2.0
    dropout: float = 0.15
    n_classes: int = 3


def _build_ft_transformer(config: TransformerConfig):
    import torch
    from torch import nn

    class NumericalFeatureTokenizer(nn.Module):
        def __init__(self, n_features: int, d_token: int):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(n_features, d_token))
            self.bias = nn.Parameter(torch.empty(n_features, d_token))
            self.feature_embedding = nn.Parameter(torch.empty(n_features, d_token))
            nn.init.xavier_uniform_(self.weight)
            nn.init.zeros_(self.bias)
            nn.init.normal_(self.feature_embedding, std=0.02)

        def forward(self, x):
            return (
                x.unsqueeze(-1) * self.weight.unsqueeze(0)
                + self.bias.unsqueeze(0)
                + self.feature_embedding.unsqueeze(0)
            )

    class FTTransformer(nn.Module):
        def __init__(self, cfg: TransformerConfig):
            super().__init__()
            self.tokenizer = NumericalFeatureTokenizer(cfg.n_features, cfg.d_token)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_token))
            nn.init.normal_(self.cls_token, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_token,
                nhead=cfg.n_heads,
                dim_feedforward=int(cfg.d_token * cfg.ff_multiplier),
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(cfg.d_token),
                nn.Linear(cfg.d_token, cfg.d_token),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_token, cfg.n_classes),
            )

        def forward(self, x):
            tokens = self.tokenizer(x)
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            encoded = self.encoder(torch.cat([cls, tokens], dim=1))
            return self.head(encoded[:, 0])

    return FTTransformer(config)


class TransformerAdapter:
    def __init__(self, model: Any, config: TransformerConfig, device: str):
        self.model = model
        self.config = config
        self.device = device

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        self.model.eval()
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(X), 512):
                batch = torch.as_tensor(X[start : start + 512], dtype=torch.float32, device=device)
                chunks.append(torch.softmax(self.model(batch), dim=1).cpu().numpy())
        return np.concatenate(chunks, axis=0).astype(np.float64)

    def save(self, path: str | Path) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        export_model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(
            {"config": asdict(self.config), "state_dict": export_model.cpu().state_dict()},
            path,
        )


class TabNetAdapter:
    def __init__(self, model: Any):
        self.model = model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(X), dtype=np.float64)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))


class YDFAdapter:
    def __init__(self, model: Any, feature_names: list[str]):
        self.model = model
        self.feature_names = list(feature_names)

    def _frame(self, X: np.ndarray):
        import pandas as pd

        return pd.DataFrame(X, columns=self.feature_names)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(self.model.predict(self._frame(X)), dtype=np.float64)
        model_classes = list(self.model.label_classes())
        if probabilities.ndim != 2 or probabilities.shape[1] != 3:
            raise ValueError(f"Unexpected YDF prediction shape: {probabilities.shape}")
        order = [model_classes.index(name) for name in CLASS_NAMES]
        return probabilities[:, order]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        (path / "adapter_metadata.json").write_text(
            json.dumps({"feature_names": self.feature_names, "class_names": CLASS_NAMES}, indent=2),
            encoding="utf-8",
        )


def fit_transformer(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    seed: int,
) -> TransformerAdapter:
    import torch
    from torch import nn
    from torch.optim.swa_utils import AveragedModel
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    set_global_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("FT-Transformer full run requires the requested CUDA GPU runtime")
    device = torch.device("cuda")
    d_token = int(params["d_token"])
    n_heads = int(params["n_heads"])
    if d_token % n_heads:
        raise ValueError("d_token must be divisible by n_heads")
    config = TransformerConfig(
        n_features=X.shape[1],
        d_token=d_token,
        n_heads=n_heads,
        n_layers=int(params["n_layers"]),
        ff_multiplier=float(params["ff_multiplier"]),
        dropout=float(params["dropout"]),
    )
    model = _build_ft_transformer(config).to(device)
    class_weight = balanced_class_weights(y, power=float(params.get("class_weight_power", 0.5)))
    sample_weight = class_weight[y]
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weight, dtype=torch.double),
        num_samples=max(len(y), int(params.get("samples_per_epoch", len(y)))),
        replacement=True,
        generator=generator,
    )
    dataset = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(int(params["batch_size"]), max(8, len(y))),
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    epochs = int(params["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=float(params["lr"]) * 0.02
    )
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(params.get("label_smoothing", 0.03)),
    )
    averaged = AveragedModel(model)
    average_from = max(1, int(epochs * 0.60))
    noise_std = float(params.get("noise_std", 0.01))
    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if noise_std > 0:
                xb = xb + torch.randn_like(xb) * noise_std
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise FloatingPointError("FT-Transformer produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        if epoch >= average_from:
            averaged.update_parameters(model)
    averaged.eval()
    return TransformerAdapter(averaged, config=config, device="cuda")


def fit_tabnet(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    seed: int,
) -> TabNetAdapter:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch

    set_global_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("TabNet full run requires the requested CUDA GPU runtime")
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
        scheduler_params={"T_max": int(params["epochs"]), "eta_min": float(params["lr"]) * 0.02},
        seed=seed,
        verbose=0,
        device_name="cuda",
    )
    class_weight = balanced_class_weights(
        y, power=float(params.get("class_weight_power", 0.5))
    )
    model.fit(
        X_train=X.astype(np.float32),
        y_train=y.astype(np.int64),
        max_epochs=int(params["epochs"]),
        patience=0,
        batch_size=min(int(params["batch_size"]), max(16, len(y))),
        virtual_batch_size=min(int(params["virtual_batch_size"]), max(8, len(y))),
        num_workers=0,
        drop_last=False,
        weights={class_id: float(class_weight[class_id]) for class_id in range(3)},
    )
    return TabNetAdapter(model)


def fit_ydf(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    seed: int,
) -> YDFAdapter:
    import pandas as pd
    import ydf

    set_global_seed(seed)
    feature_names = [f"f_{i:04d}" for i in range(X.shape[1])]
    frame = pd.DataFrame(X, columns=feature_names)
    frame["label"] = [CLASS_NAMES[int(v)] for v in y]
    weights = balanced_class_weights(y, power=float(params.get("class_weight_power", 0.5)))
    class_weights = {CLASS_NAMES[i]: float(weights[i]) for i in range(3)}
    learner = ydf.GradientBoostedTreesLearner(
        label="label",
        label_classes=list(CLASS_NAMES),
        class_weights=class_weights,
        loss="MULTINOMIAL_LOG_LIKELIHOOD",
        num_trees=int(params["num_trees"]),
        max_depth=int(params["max_depth"]),
        min_examples=int(params["min_examples"]),
        shrinkage=float(params["shrinkage"]),
        subsample=float(params["subsample"]),
        use_hessian_gain=bool(params["use_hessian_gain"]),
        l2_regularization=float(params["l2_regularization"]),
        validation_ratio=0.0,
        random_seed=seed,
        num_threads=int(params.get("num_threads", 32)),
    )
    model = learner.train(frame)
    return YDFAdapter(model, feature_names=feature_names)


def fit_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    seed: int,
):
    if model_name == "transformer":
        return fit_transformer(X, y, params, seed)
    if model_name == "tabnet":
        return fit_tabnet(X, y, params, seed)
    if model_name == "ydf":
        return fit_ydf(X, y, params, seed)
    raise ValueError(f"Unknown model: {model_name}")


def suggest_parameters(model_name: str, trial: Any, fast: bool = False) -> dict[str, Any]:
    """Define a broad but bounded search space for the A100/High-RAM run."""
    max_features = trial.suggest_categorical("max_features", [64, 96, 128, 192])
    if model_name == "transformer":
        return {
            "max_features": max_features,
            "d_token": trial.suggest_categorical("d_token", [32, 48, 64]),
            "n_heads": trial.suggest_categorical("n_heads", [4, 8]),
            "n_layers": trial.suggest_int("n_layers", 2, 5),
            "ff_multiplier": trial.suggest_categorical("ff_multiplier", [1.5, 2.0, 3.0]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.35),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 3e-2, log=True),
            "epochs": 12 if fast else trial.suggest_categorical("epochs", [180, 280, 420]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.10),
            "noise_std": trial.suggest_float("noise_std", 0.0, 0.04),
            "class_weight_power": trial.suggest_float("class_weight_power", 0.0, 0.75),
        }
    if model_name == "tabnet":
        n_d = trial.suggest_categorical("n_d", [16, 24, 32, 48, 64])
        return {
            "max_features": max_features,
            "n_d": n_d,
            "n_steps": trial.suggest_int("n_steps", 3, 7),
            "gamma": trial.suggest_float("gamma", 1.0, 1.8),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
            "mask_type": trial.suggest_categorical("mask_type", ["entmax", "sparsemax"]),
            "lr": trial.suggest_float("lr", 3e-4, 3e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
            "epochs": 12 if fast else trial.suggest_categorical("epochs", [200, 350, 500]),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "virtual_batch_size": trial.suggest_categorical("virtual_batch_size", [16, 32, 64]),
            "class_weight_power": trial.suggest_float("class_weight_power", 0.0, 0.75),
        }
    if model_name == "ydf":
        return {
            "max_features": max_features,
            "num_trees": 40 if fast else trial.suggest_categorical("num_trees", [300, 600, 1000, 1600]),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_examples": trial.suggest_int("min_examples", 2, 14),
            "shrinkage": trial.suggest_categorical("shrinkage", [0.02, 0.03, 0.05, 0.08, 0.10]),
            "subsample": trial.suggest_categorical("subsample", [0.70, 0.85, 1.0]),
            "use_hessian_gain": trial.suggest_categorical("use_hessian_gain", [False, True]),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            "class_weight_power": trial.suggest_float("class_weight_power", 0.0, 0.75),
            "num_threads": 32,
        }
    raise ValueError(f"Unknown model: {model_name}")
