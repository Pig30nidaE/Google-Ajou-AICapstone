"""Google-first model adapters for CN versus impaired classification.

YDF Gradient Boosted Trees and Random Forest are Google Yggdrasil models.
TabNet was published by Google Research (the maintained PyTorch package is a
third-party implementation).  The neural Transformer here is a compact
numerical-token adaptation of Google's original Transformer architecture; it
is not presented as FT-Transformer or TabTransformer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd


CLASS_NAMES = ("CN", "MCI_DEM")
MODEL_NAMES = ("ydf_gbt", "ydf_rf", "tabnet", "transformer")
GOOGLE_MODEL_EVIDENCE = {
    "ydf_gbt": {
        "name": "Yggdrasil Decision Forests Gradient Boosted Trees",
        "origin": "Google",
        "url": "https://github.com/google/yggdrasil-decision-forests",
    },
    "ydf_rf": {
        "name": "Yggdrasil Decision Forests Random Forest",
        "origin": "Google",
        "url": "https://github.com/google/yggdrasil-decision-forests",
    },
    "tabnet": {
        "name": "TabNet",
        "origin": "Google Research paper; third-party PyTorch implementation",
        "url": "https://research.google/pubs/tabnet-attentive-interpretable-tabular-learning/",
    },
    "transformer": {
        "name": "Numerical-token Transformer",
        "origin": "Custom tabular adaptation of the Google Transformer architecture",
        "url": "https://research.google/pubs/attention-is-all-you-need/",
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


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Expected binary probability matrix; got {values.shape}")
    values = np.clip(values, 1e-8, 1.0)
    return values / values.sum(axis=1, keepdims=True)


def balanced_class_weights(y: np.ndarray, power: float = 0.5) -> np.ndarray:
    target = np.asarray(y, dtype=np.int64)
    counts = np.bincount(target, minlength=2).astype(float)
    if np.any(counts == 0):
        raise ValueError(f"Both binary classes are required; counts={counts.tolist()}")
    weights = np.power(len(target) / (2.0 * counts), float(np.clip(power, 0.0, 1.0)))
    return weights / weights.mean()


class YDFAdapter:
    def __init__(self, model: Any, feature_names: list[str]):
        self.model = model
        self.feature_names = list(feature_names)

    def _frame(self, X: np.ndarray) -> pd.DataFrame:
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("YDF input shape differs from its fitted feature manifest")
        return pd.DataFrame(values, columns=self.feature_names)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.model.predict(self._frame(X)), dtype=np.float64)
        model_classes = tuple(str(value) for value in self.model.label_classes())
        if set(model_classes) != set(CLASS_NAMES):
            raise ValueError(f"Unexpected YDF label classes: {model_classes}")
        if raw.ndim == 1:
            # Binary YDF emits the probability of label_classes()[1].
            ordered = np.column_stack([1.0 - raw, raw])
        elif raw.ndim == 2 and raw.shape[1] == 2:
            ordered = raw
        else:
            raise ValueError(f"Unexpected YDF prediction shape: {raw.shape}")
        order = [model_classes.index(name) for name in CLASS_NAMES]
        return normalize_probabilities(ordered[:, order])

    def save(self, path: str | Path) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        self.model.save(str(root / "model"))
        (root / "adapter.json").write_text(
            json.dumps(
                {"feature_names": self.feature_names, "class_names": list(CLASS_NAMES)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


class TabNetAdapter:
    def __init__(self, model: Any):
        self.model = model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return normalize_probabilities(
            self.model.predict_proba(np.asarray(X, dtype=np.float32))
        )

    def save(self, path: str | Path) -> None:
        root = Path(path)
        root.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(root))


@dataclass(frozen=True)
class TransformerConfig:
    n_features: int
    d_token: int = 32
    n_heads: int = 4
    n_layers: int = 2
    ff_multiplier: float = 2.0
    dropout: float = 0.15
    n_classes: int = 2


def _build_transformer(config: TransformerConfig):
    import torch
    from torch import nn

    class NumericalTokenizer(nn.Module):
        def __init__(self, n_features: int, d_token: int):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(n_features, d_token))
            self.bias = nn.Parameter(torch.zeros(n_features, d_token))
            self.feature_embedding = nn.Parameter(torch.empty(n_features, d_token))
            nn.init.xavier_uniform_(self.weight)
            nn.init.normal_(self.feature_embedding, std=0.02)

        def forward(self, x):
            return (
                x.unsqueeze(-1) * self.weight.unsqueeze(0)
                + self.bias.unsqueeze(0)
                + self.feature_embedding.unsqueeze(0)
            )

    class WearableTransformer(nn.Module):
        def __init__(self, cfg: TransformerConfig):
            super().__init__()
            self.tokenizer = NumericalTokenizer(cfg.n_features, cfg.d_token)
            self.cls_token = nn.Parameter(torch.empty(1, 1, cfg.d_token))
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
            return self.head(self.encoder(torch.cat([cls, tokens], dim=1))[:, 0])

    return WearableTransformer(config)


class TransformerAdapter:
    def __init__(self, model: Any, config: TransformerConfig, device: str):
        self.model = model
        self.config = config
        self.device = device

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        device = torch.device(self.device if self.device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model.to(device).eval()
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(X), 512):
                batch = torch.as_tensor(
                    X[start : start + 512], dtype=torch.float32, device=device
                )
                chunks.append(torch.softmax(self.model(batch), dim=1).cpu().numpy())
        return normalize_probabilities(np.concatenate(chunks, axis=0))

    def save(self, path: str | Path) -> None:
        import torch

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.model.to("cpu")
        torch.save(
            {"config": asdict(self.config), "state_dict": self.model.state_dict()}, output
        )


def default_parameters(model_name: str, *, fast: bool) -> dict[str, Any]:
    """Conservative fixed settings; model selection happens on inner OOF data."""

    if model_name == "ydf_gbt":
        return {
            "num_trees": 60 if fast else 700,
            "max_depth": 3,
            "min_examples": 7,
            "shrinkage": 0.04,
            "subsample": 0.85,
            "num_candidate_attributes_ratio": 0.5,
            "use_hessian_gain": True,
            "l2_regularization": 1.0,
            "class_weight_power": 0.35,
        }
    if model_name == "ydf_rf":
        return {
            "num_trees": 80 if fast else 1000,
            "max_depth": 8,
            "min_examples": 5,
            "num_candidate_attributes_ratio": 0.5,
            "bootstrap_size_ratio": 0.8,
            "class_weight_power": 0.25,
        }
    if model_name == "tabnet":
        return {
            "n_d": 24,
            "n_steps": 4,
            "gamma": 1.3,
            "lambda_sparse": 1e-5,
            "mask_type": "entmax",
            "lr": 0.006,
            "weight_decay": 1e-4,
            "epochs": 8 if fast else 180,
            "batch_size": 64,
            "virtual_batch_size": 16,
            "class_weight_power": 0.35,
        }
    if model_name == "transformer":
        return {
            "d_token": 32,
            "n_heads": 4,
            "n_layers": 2,
            "ff_multiplier": 2.0,
            "dropout": 0.18,
            "lr": 8e-4,
            "weight_decay": 3e-4,
            "epochs": 8 if fast else 160,
            "batch_size": 64,
            "label_smoothing": 0.03,
            "noise_std": 0.01,
            "class_weight_power": 0.35,
        }
    raise ValueError(f"Unknown model: {model_name}")


def _fit_ydf(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    params: dict[str, Any],
    seed: int,
) -> YDFAdapter:
    import ydf

    values = np.asarray(X, dtype=np.float32)
    if values.shape[1] != len(feature_names):
        raise ValueError("YDF feature names do not match transformed width")
    frame = pd.DataFrame(values, columns=feature_names)
    frame["label"] = [CLASS_NAMES[int(value)] for value in y]
    weights = balanced_class_weights(y, float(params["class_weight_power"]))
    common = {
        "label": "label",
        "label_classes": list(CLASS_NAMES),
        "class_weights": {CLASS_NAMES[index]: float(weights[index]) for index in range(2)},
        "num_trees": int(params["num_trees"]),
        "max_depth": int(params["max_depth"]),
        "min_examples": int(params["min_examples"]),
        "num_candidate_attributes_ratio": float(params["num_candidate_attributes_ratio"]),
        "random_seed": int(seed),
        "num_threads": min(32, max(1, os.cpu_count() or 1)),
    }
    if model_name == "ydf_gbt":
        learner = ydf.GradientBoostedTreesLearner(
            **common,
            loss="BINOMIAL_LOG_LIKELIHOOD",
            shrinkage=float(params["shrinkage"]),
            subsample=float(params["subsample"]),
            use_hessian_gain=bool(params["use_hessian_gain"]),
            l2_regularization=float(params["l2_regularization"]),
            validation_ratio=0.0,
        )
    elif model_name == "ydf_rf":
        learner = ydf.RandomForestLearner(
            **common,
            bootstrap_training_dataset=True,
            bootstrap_size_ratio=float(params["bootstrap_size_ratio"]),
            sampling_with_replacement=True,
            winner_take_all=False,
        )
    else:
        raise ValueError(f"Not a YDF model: {model_name}")
    return YDFAdapter(learner.train(frame), feature_names)


def _fit_tabnet(
    X: np.ndarray, y: np.ndarray, params: dict[str, Any], seed: int
) -> TabNetAdapter:
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier

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
        seed=int(seed),
        verbose=0,
        device_name="auto",
    )
    weights = balanced_class_weights(y, float(params["class_weight_power"]))
    model.fit(
        X_train=np.asarray(X, dtype=np.float32),
        y_train=np.asarray(y, dtype=np.int64),
        max_epochs=int(params["epochs"]),
        patience=0,
        batch_size=min(int(params["batch_size"]), max(16, len(y))),
        virtual_batch_size=min(int(params["virtual_batch_size"]), max(8, len(y))),
        num_workers=0,
        drop_last=False,
        weights={index: float(weights[index]) for index in range(2)},
    )
    return TabNetAdapter(model)


def _fit_transformer(
    X: np.ndarray, y: np.ndarray, params: dict[str, Any], seed: int
) -> TransformerAdapter:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TransformerConfig(
        n_features=X.shape[1],
        d_token=int(params["d_token"]),
        n_heads=int(params["n_heads"]),
        n_layers=int(params["n_layers"]),
        ff_multiplier=float(params["ff_multiplier"]),
        dropout=float(params["dropout"]),
    )
    model = _build_transformer(config).to(device)
    class_weights = balanced_class_weights(y, float(params["class_weight_power"]))
    sample_weights = class_weights[np.asarray(y, dtype=np.int64)]
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(y),
        replacement=True,
        generator=generator,
    )
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.long),
        ),
        batch_size=min(int(params["batch_size"]), max(8, len(y))),
        sampler=sampler,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(params["epochs"]))
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(params["label_smoothing"]))
    for _ in range(int(params["epochs"])):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if float(params["noise_std"]) > 0:
                xb = xb + torch.randn_like(xb) * float(params["noise_std"])
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise FloatingPointError("Transformer produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
    return TransformerAdapter(model, config, str(device))


def fit_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    seed: int,
    fast: bool,
):
    set_global_seed(seed)
    params = default_parameters(model_name, fast=fast)
    if model_name in {"ydf_gbt", "ydf_rf"}:
        fitted = _fit_ydf(model_name, X, y, feature_names, params, seed)
    elif model_name == "tabnet":
        fitted = _fit_tabnet(X, y, params, seed)
    elif model_name == "transformer":
        fitted = _fit_transformer(X, y, params, seed)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return fitted, params
