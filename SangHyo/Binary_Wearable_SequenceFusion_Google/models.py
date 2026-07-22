"""Model implementations for the wearable sequence-fusion experiment.

The two tree models are provided by Google's Yggdrasil Decision Forests
(YDF).  ``sequence_transformer`` is a compact temporal adaptation of the
Transformer architecture introduced by Google Research.  The
``conv_bilstm`` branch is retained as a leakage-safe successor to the model in
``SangHyo/previous/Experiment2.py``.
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
MODEL_NAMES = ("ydf_gbt", "ydf_rf", "conv_bilstm", "sequence_transformer")
ENSEMBLE_WEIGHTS = {
    "ydf_gbt": 0.20,
    "ydf_rf": 0.10,
    "conv_bilstm": 0.40,
    "sequence_transformer": 0.30,
}
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
    "sequence_transformer": {
        "name": "Temporal sequence Transformer",
        "origin": "Custom wearable adaptation of Google's Transformer architecture",
        "url": "https://research.google/pubs/attention-is-all-you-need/",
    },
    "conv_bilstm": {
        "name": "Regularized Conv1D + bidirectional LSTM",
        "origin": "Reference branch based on SangHyo/previous/Experiment2.py",
        "url": None,
    },
}


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds without forcing a slow CUDA algorithm set."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
    except ImportError:
        pass


def _class_weights(y: np.ndarray, power: float = 0.25) -> np.ndarray:
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=2).astype(float)
    if np.any(counts == 0):
        raise ValueError(f"Both classes are required, observed {counts.tolist()}")
    weights = np.power(len(y) / (2.0 * counts), power)
    return weights / weights.mean()


class YDFBinaryModel:
    """Thin adapter preserving the canonical CN/MCI_DEM probability order."""

    def __init__(self, model: Any, feature_names: list[str], model_name: str):
        self.model = model
        self.feature_names = list(feature_names)
        self.model_name = str(model_name)

    def _frame(self, X: np.ndarray) -> pd.DataFrame:
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"YDF feature mismatch: {values.shape} vs {len(self.feature_names)}"
            )
        return pd.DataFrame(values, columns=self.feature_names)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.model.predict(self._frame(X)), dtype=np.float64)
        classes = [str(item) for item in self.model.label_classes()]
        if set(classes) != set(CLASS_NAMES):
            raise ValueError(f"Unexpected YDF classes: {classes}")
        if raw.ndim == 1:
            raw = np.column_stack([1.0 - raw, raw])
        if raw.ndim != 2 or raw.shape[1] != 2:
            raise ValueError(f"Unexpected YDF prediction shape: {raw.shape}")
        raw = raw[:, [classes.index(name) for name in CLASS_NAMES]]
        raw = np.clip(raw, 1e-7, 1.0)
        return raw / raw.sum(axis=1, keepdims=True)

    def save(self, path: str | Path) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        self.model.save(str(root / "model"))
        (root / "adapter.json").write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "class_names": list(CLASS_NAMES),
                    "feature_names": self.feature_names,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def fit_ydf(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    seed: int,
    fast: bool = False,
) -> YDFBinaryModel:
    """Fit one shallow, regularized Google YDF model at subject level."""

    import ydf

    if model_name not in {"ydf_gbt", "ydf_rf"}:
        raise ValueError(f"Not a YDF model: {model_name}")
    values = np.asarray(X, dtype=np.float32)
    target = np.asarray(y, dtype=np.int64)
    frame = pd.DataFrame(values, columns=feature_names)
    frame["label"] = [CLASS_NAMES[int(item)] for item in target]
    class_weights = _class_weights(target, power=0.25)
    common = dict(
        label="label",
        label_classes=list(CLASS_NAMES),
        class_weights={
            CLASS_NAMES[index]: float(class_weights[index]) for index in range(2)
        },
        max_depth=3 if model_name == "ydf_gbt" else 8,
        min_examples=7 if model_name == "ydf_gbt" else 5,
        num_candidate_attributes_ratio=0.45,
        random_seed=int(seed),
        num_threads=min(32, max(1, os.cpu_count() or 1)),
        # Bound each native C++ training call so the six-hour experiment
        # deadline cannot be consumed by one forest fit.
        maximum_training_duration_seconds=30.0 if fast else 300.0,
    )
    if model_name == "ydf_gbt":
        learner = ydf.GradientBoostedTreesLearner(
            **common,
            num_trees=40 if fast else 650,
            loss="BINOMIAL_LOG_LIKELIHOOD",
            shrinkage=0.035,
            subsample=0.80,
            l2_regularization=2.0,
            validation_ratio=0.0,
            adapt_subsample_for_maximum_training_duration=True,
        )
    else:
        learner = ydf.RandomForestLearner(
            **common,
            num_trees=60 if fast else 900,
            bootstrap_training_dataset=True,
            bootstrap_size_ratio=0.80,
            adapt_bootstrap_size_ratio_for_maximum_training_duration=True,
        )
    return YDFBinaryModel(learner.train(frame), feature_names, model_name)


@dataclass(frozen=True)
class NeuralConfig:
    model_name: str
    n_features: int
    sequence_length: int = 28
    hidden_size: int = 64
    dropout: float = 0.25


def _build_torch_model(config: NeuralConfig):
    import torch
    from torch import nn

    class ConvBiLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = config.hidden_size
            self.input_norm = nn.LayerNorm(config.n_features)
            self.conv1 = nn.Conv1d(config.n_features, width, kernel_size=5, padding=2)
            self.norm1 = nn.BatchNorm1d(width)
            self.conv2 = nn.Conv1d(
                width, width, kernel_size=3, padding=2, dilation=2, bias=False
            )
            self.norm2 = nn.BatchNorm1d(width)
            self.dropout = nn.Dropout(config.dropout)
            self.rnn = nn.LSTM(
                input_size=width,
                hidden_size=width // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.attention = nn.Sequential(
                nn.Linear(width, width // 2), nn.Tanh(), nn.Linear(width // 2, 1)
            )
            self.head = nn.Sequential(
                nn.LayerNorm(width * 2),
                nn.Linear(width * 2, width // 2),
                nn.GELU(),
                nn.Dropout(config.dropout + 0.05),
                nn.Linear(width // 2, 1),
            )

        def forward(self, x):
            x = self.input_norm(x).transpose(1, 2)
            residual = torch.nn.functional.gelu(self.norm1(self.conv1(x)))
            x = self.dropout(
                torch.nn.functional.gelu(self.norm2(self.conv2(residual))) + residual
            )
            x, _ = self.rnn(x.transpose(1, 2))
            weights = torch.softmax(self.attention(x), dim=1)
            attentive = (weights * x).sum(dim=1)
            pooled = x.mean(dim=1)
            return self.head(torch.cat([attentive, pooled], dim=1)).squeeze(1)

    class SequenceTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = config.hidden_size
            self.input_norm = nn.LayerNorm(config.n_features)
            self.projection = nn.Linear(config.n_features, width)
            self.position = nn.Parameter(
                torch.empty(1, config.sequence_length, width)
            )
            nn.init.normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=4,
                dim_feedforward=width * 2,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Sequential(
                nn.LayerNorm(width * 2),
                nn.Linear(width * 2, width),
                nn.GELU(),
                nn.Dropout(config.dropout + 0.05),
                nn.Linear(width, 1),
            )

        def forward(self, x):
            x = self.projection(self.input_norm(x)) + self.position[:, : x.shape[1]]
            x = self.encoder(x)
            return self.head(torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=1)).squeeze(1)

    if config.model_name == "conv_bilstm":
        return ConvBiLSTM()
    if config.model_name == "sequence_transformer":
        return SequenceTransformer()
    raise ValueError(f"Unknown neural model: {config.model_name}")


class NeuralSequenceModel:
    def __init__(self, model: Any, config: NeuralConfig, device: str):
        self.model = model
        self.config = config
        self.device = device

    def predict_proba(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        import torch

        device = torch.device(
            self.device if self.device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.model.to(device).eval()
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(X), batch_size):
                batch = torch.as_tensor(
                    X[start : start + batch_size], dtype=torch.float32, device=device
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    logits = self.model(batch)
                outputs.append(torch.sigmoid(logits.float()).cpu().numpy())
        impaired = np.clip(np.concatenate(outputs), 1e-7, 1.0 - 1e-7)
        return np.column_stack([1.0 - impaired, impaired])

    def save(self, path: str | Path) -> None:
        import torch

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.model.to("cpu")
        torch.save(
            {"config": asdict(self.config), "state_dict": self.model.state_dict()},
            output,
        )


def aggregate_view_probabilities(
    probabilities: np.ndarray, view_subject_indices: np.ndarray, n_subjects: int
) -> np.ndarray:
    """Average the same fixed number of views for each subject."""

    impaired = np.asarray(probabilities, dtype=np.float64)
    if impaired.ndim == 2:
        impaired = impaired[:, 1]
    mapping = np.asarray(view_subject_indices, dtype=np.int64)
    result = np.zeros(n_subjects, dtype=np.float64)
    counts = np.zeros(n_subjects, dtype=np.int64)
    np.add.at(result, mapping, impaired)
    np.add.at(counts, mapping, 1)
    if np.any(counts == 0) or len(set(counts.tolist())) != 1:
        raise AssertionError(f"Unequal/missing subject views: {counts.tolist()}")
    return result / counts


def _subject_auc(
    y_subject: np.ndarray,
    p_view: np.ndarray,
    view_subject_indices: np.ndarray,
) -> float:
    from sklearn.metrics import roc_auc_score

    probability = aggregate_view_probabilities(
        p_view, view_subject_indices, len(y_subject)
    )
    return float(roc_auc_score(y_subject, probability))


def _train_torch_epochs(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    patience: int,
) -> tuple[Any, int, list[dict[str, float]]]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=2e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=learning_rate * 0.08
    )
    weights = _class_weights(np.asarray(y, dtype=np.int64), power=0.25)
    positive_weight = float(weights[1] / weights[0])
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32, device=device)
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
        ),
        batch_size=min(batch_size, len(X)),
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    use_amp = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # PyTorch builds before torch.amp unification
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_auc = -np.inf
    best_epoch = epochs
    best_state: dict[str, Any] | None = None
    wait = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                logits = model(batch_x)
                # Light label smoothing stabilizes the tiny subject cohort.
                smooth_y = batch_y * 0.96 + 0.02
                loss = criterion(logits, smooth_y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        row = {"epoch": float(epoch), "loss": float(np.mean(losses))}
        if validation is not None:
            X_valid, y_valid_subject, valid_view_subject, _ = validation
            wrapper = NeuralSequenceModel(model, NeuralConfig("temporary", X.shape[-1]), device.type)
            p_valid = wrapper.predict_proba(X_valid)[:, 1]
            auc = _subject_auc(y_valid_subject, p_valid, valid_view_subject)
            row["validation_subject_auc"] = auc
            if auc > best_auc + 1e-4:
                best_auc = auc
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                wait = 0
            else:
                wait += 1
            if epoch >= 12 and wait >= patience:
                history.append(row)
                break
        history.append(row)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, int(best_epoch), history


def select_neural_epoch(
    model_name: str,
    X_train: np.ndarray,
    y_train_view: np.ndarray,
    X_early: np.ndarray,
    y_early_subject: np.ndarray,
    early_view_subject: np.ndarray,
    *,
    seed: int,
    fast: bool = False,
) -> tuple[int, list[dict[str, float]]]:
    """Choose an epoch using only an internal subject holdout."""

    set_global_seed(seed)
    config = NeuralConfig(
        model_name=model_name,
        n_features=X_train.shape[-1],
        sequence_length=X_train.shape[1],
    )
    model = _build_torch_model(config)
    _, best_epoch, history = _train_torch_epochs(
        model,
        X_train,
        y_train_view,
        seed=seed,
        epochs=3 if fast else 120,
        batch_size=32,
        learning_rate=4e-4 if model_name == "conv_bilstm" else 3e-4,
        validation=(X_early, y_early_subject, early_view_subject, np.empty(0)),
        patience=2 if fast else 16,
    )
    return best_epoch, history


def fit_neural_fixed_epochs(
    model_name: str,
    X: np.ndarray,
    y_view: np.ndarray,
    *,
    epochs: int,
    seed: int,
) -> tuple[NeuralSequenceModel, list[dict[str, float]]]:
    """Refit on the complete fold after a train-only epoch was selected."""

    import torch

    set_global_seed(seed)
    config = NeuralConfig(
        model_name=model_name,
        n_features=X.shape[-1],
        sequence_length=X.shape[1],
    )
    model = _build_torch_model(config)
    model, _, history = _train_torch_epochs(
        model,
        X,
        y_view,
        seed=seed,
        epochs=max(1, int(epochs)),
        batch_size=32,
        learning_rate=4e-4 if model_name == "conv_bilstm" else 3e-4,
        validation=None,
        patience=max(1, int(epochs)),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return NeuralSequenceModel(model, config, device), history


def blend_probabilities(probabilities: dict[str, np.ndarray]) -> np.ndarray:
    missing = sorted(set(ENSEMBLE_WEIGHTS) - set(probabilities))
    if missing:
        raise KeyError(f"Missing ensemble probabilities: {missing}")
    blended = sum(
        ENSEMBLE_WEIGHTS[name] * np.asarray(probabilities[name], dtype=np.float64)
        for name in ENSEMBLE_WEIGHTS
    )
    return np.clip(blended, 1e-7, 1.0 - 1e-7)
