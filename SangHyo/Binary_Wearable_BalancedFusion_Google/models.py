"""Small-data model branches used by the balanced fusion experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
import pandas as pd


CLASS_NAMES = ("CN", "MCI_DEM")
MODEL_NAMES = (
    "elastic_net",
    "ydf_subject",
    "ydf_daily",
    "tabnet",
    "transformer",
)
GOOGLE_MODELS = {
    "ydf_subject": "Google Yggdrasil Decision Forests",
    "ydf_daily": "Google Yggdrasil Decision Forests",
    "tabnet": "Google Research TabNet",
    "transformer": "compact temporal adaptation of Google's Transformer",
}


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
    except ImportError:
        pass


def balanced_class_weights(y: np.ndarray) -> np.ndarray:
    target = np.asarray(y, dtype=np.int64)
    counts = np.bincount(target, minlength=2).astype(float)
    if np.any(counts == 0):
        raise ValueError(f"Both classes are required; counts={counts.tolist()}")
    values = len(target) / (2.0 * counts)
    return values / values.mean()


def _positive_from_ydf(model: Any, frame: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict(frame), dtype=np.float64)
    classes = [str(value) for value in model.label_classes()]
    if set(classes) != set(CLASS_NAMES):
        raise ValueError(f"Unexpected YDF classes: {classes}")
    if raw.ndim == 1:
        raw = np.column_stack([1.0 - raw, raw])
    if raw.shape != (len(frame), 2):
        raise ValueError(f"Unexpected YDF probability shape: {raw.shape}")
    ordered = raw[:, [classes.index(name) for name in CLASS_NAMES]]
    ordered = np.clip(ordered, 1e-7, 1.0)
    ordered /= ordered.sum(axis=1, keepdims=True)
    return ordered[:, 1]


class ElasticNetAdapter:
    def __init__(self, model: Any):
        self.model = model

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        if not np.array_equal(np.asarray(self.model.classes_), np.asarray([0, 1])):
            raise ValueError(
                f"Unexpected Elastic-Net class order: {self.model.classes_}"
            )
        return np.asarray(
            self.model.predict_proba(np.asarray(X, dtype=np.float32))[:, 1],
            dtype=np.float64,
        )

    def save(self, path: str | Path) -> None:
        import joblib

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, output)

    @classmethod
    def load(cls, path: str | Path) -> "ElasticNetAdapter":
        import joblib

        return cls(joblib.load(path))


def fit_elastic_net(X: np.ndarray, y: np.ndarray, *, seed: int) -> ElasticNetAdapter:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=0.20,
        C=0.20,
        class_weight="balanced",
        solver="saga",
        max_iter=5000,
        tol=1e-5,
        random_state=int(seed),
    )
    model.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64))
    return ElasticNetAdapter(model)


class YDFSubjectAdapter:
    def __init__(self, model: Any, feature_names: Sequence[str]):
        self.model = model
        self.feature_names = list(map(str, feature_names))

    def _frame(self, X: np.ndarray) -> pd.DataFrame:
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("YDF subject feature schema mismatch")
        return pd.DataFrame(values, columns=self.feature_names)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        return _positive_from_ydf(self.model, self._frame(X))

    def save(self, path: str | Path) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=False)
        self.model.save(str(root / "model"))
        (root / "adapter.json").write_text(
            json.dumps(
                {
                    "kind": "subject",
                    "class_names": list(CLASS_NAMES),
                    "feature_names": self.feature_names,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "YDFSubjectAdapter":
        import ydf

        root = Path(path)
        metadata = json.loads((root / "adapter.json").read_text(encoding="utf-8"))
        if metadata["kind"] != "subject" or tuple(metadata["class_names"]) != CLASS_NAMES:
            raise ValueError("Invalid YDF subject checkpoint metadata")
        return cls(ydf.load_model(str(root / "model")), metadata["feature_names"])


def fit_ydf_subject(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    seed: int,
    fast: bool,
) -> YDFSubjectAdapter:
    import ydf

    values = np.asarray(X, dtype=np.float32)
    target = np.asarray(y, dtype=np.int64)
    names = [f"subject_{index:03d}" for index in range(values.shape[1])]
    if len(feature_names) != len(names):
        raise ValueError("Subject feature name count differs from matrix width")
    frame = pd.DataFrame(values, columns=names)
    frame["label"] = [CLASS_NAMES[int(value)] for value in target]
    weights = balanced_class_weights(target)
    learner = ydf.GradientBoostedTreesLearner(
        label="label",
        label_classes=list(CLASS_NAMES),
        class_weights={CLASS_NAMES[index]: float(weights[index]) for index in range(2)},
        loss="BINOMIAL_LOG_LIKELIHOOD",
        num_trees=30 if fast else 320,
        max_depth=2,
        min_examples=max(6, int(round(len(target) * 0.07))),
        shrinkage=0.035,
        subsample=0.85,
        num_candidate_attributes_ratio=0.70,
        l2_regularization=4.0,
        validation_ratio=0.15,
        early_stopping="LOSS_INCREASE",
        early_stopping_num_trees_look_ahead=25,
        random_seed=int(seed),
        num_threads=min(24, max(1, os.cpu_count() or 1)),
        maximum_training_duration_seconds=20.0 if fast else 75.0,
    )
    return YDFSubjectAdapter(learner.train(frame), names)


class YDFDailyAdapter:
    """Daily-row YDF with equal subject contribution and subject aggregation."""

    def __init__(self, model: Any, feature_names: Sequence[str], view_length: int):
        self.model = model
        self.feature_names = list(map(str, feature_names))
        self.view_length = int(view_length)

    def _frame(self, temporal: np.ndarray) -> pd.DataFrame:
        values = np.asarray(temporal, dtype=np.float32)
        if values.ndim != 3 or values.shape[1] != self.view_length:
            raise ValueError("Daily YDF requires fixed [N,T,F] views")
        flat = values.reshape(-1, values.shape[2])
        if flat.shape[1] != len(self.feature_names):
            raise ValueError("Daily YDF feature schema mismatch")
        return pd.DataFrame(flat, columns=self.feature_names)

    def predict_score(self, temporal: np.ndarray) -> np.ndarray:
        values = np.asarray(temporal, dtype=np.float32)
        daily = _positive_from_ydf(self.model, self._frame(values)).reshape(
            len(values), self.view_length
        )
        ordered = np.sort(daily, axis=1)
        trim = max(1, int(round(self.view_length * 0.10)))
        trimmed_mean = ordered[:, trim:-trim].mean(axis=1)
        # Median is robust; the upper quartile retains intermittent risk events.
        return np.clip(
            0.50 * np.median(daily, axis=1)
            + 0.30 * trimmed_mean
            + 0.20 * np.quantile(daily, 0.75, axis=1),
            1e-7,
            1 - 1e-7,
        )

    def save(self, path: str | Path) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=False)
        self.model.save(str(root / "model"))
        (root / "adapter.json").write_text(
            json.dumps(
                {
                    "kind": "daily",
                    "class_names": list(CLASS_NAMES),
                    "feature_names": self.feature_names,
                    "view_length": self.view_length,
                    "aggregation": "0.50 median + 0.30 trimmed mean + 0.20 q75",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "YDFDailyAdapter":
        import ydf

        root = Path(path)
        metadata = json.loads((root / "adapter.json").read_text(encoding="utf-8"))
        if metadata["kind"] != "daily" or tuple(metadata["class_names"]) != CLASS_NAMES:
            raise ValueError("Invalid YDF daily checkpoint metadata")
        return cls(
            ydf.load_model(str(root / "model")),
            metadata["feature_names"],
            metadata["view_length"],
        )


def fit_ydf_daily(
    temporal: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    fast: bool,
) -> YDFDailyAdapter:
    import ydf

    values = np.asarray(temporal, dtype=np.float32)
    target = np.asarray(y, dtype=np.int64)
    if values.ndim != 3 or len(values) != len(target):
        raise ValueError("Daily YDF inputs are not aligned")
    n_subjects, view_length, width = values.shape
    names = [f"daily_{index:03d}" for index in range(width)]
    frame = pd.DataFrame(values.reshape(-1, width), columns=names)
    repeated = np.repeat(target, view_length)
    frame["label"] = [CLASS_NAMES[int(value)] for value in repeated]
    subject_class_weights = balanced_class_weights(target)
    subject_weights = subject_class_weights[target]
    # Every subject contributes exactly the same total mass regardless of rows.
    frame["sample_weight"] = np.repeat(subject_weights / view_length, view_length)
    if not np.allclose(
        frame["sample_weight"].to_numpy().reshape(n_subjects, view_length).sum(axis=1),
        subject_weights,
    ):
        raise AssertionError("Daily row weights do not preserve equal subject mass")
    learner = ydf.GradientBoostedTreesLearner(
        label="label",
        label_classes=list(CLASS_NAMES),
        weights="sample_weight",
        loss="BINOMIAL_LOG_LIKELIHOOD",
        num_trees=30 if fast else 420,
        max_depth=3,
        min_examples=max(24, view_length),
        shrinkage=0.035,
        subsample=0.80,
        num_candidate_attributes_ratio=0.55,
        l2_regularization=5.0,
        # A random daily-row validation would place observations from the same
        # subject on both sides.  Model complexity is therefore fixed here and
        # judged by the surrounding subject-level inner OOF instead.
        validation_ratio=0.0,
        random_seed=int(seed),
        num_threads=min(24, max(1, os.cpu_count() or 1)),
        maximum_training_duration_seconds=20.0 if fast else 90.0,
    )
    return YDFDailyAdapter(learner.train(frame), names, view_length)


class TabNetAdapter:
    def __init__(self, model: Any):
        self.model = model

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        if not np.array_equal(np.asarray(self.model.classes_), np.asarray([0, 1])):
            raise ValueError(f"Unexpected TabNet class order: {self.model.classes_}")
        probability = np.asarray(
            self.model.predict_proba(np.asarray(X, dtype=np.float32)),
            dtype=np.float64,
        )
        if probability.shape != (len(X), 2):
            raise ValueError(f"Unexpected TabNet output: {probability.shape}")
        return np.clip(probability[:, 1], 1e-7, 1 - 1e-7)

    def save(self, base_path: str | Path) -> Path:
        path = Path(base_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stored = Path(self.model.save_model(str(path)))
        if not stored.is_file() or stored.suffix != ".zip":
            raise FileNotFoundError(f"TabNet checkpoint not created: {stored}")
        return stored

    @classmethod
    def load(cls, path: str | Path) -> "TabNetAdapter":
        from pytorch_tabnet.tab_model import TabNetClassifier

        model = TabNetClassifier(device_name="cpu", verbose=0)
        model.load_model(str(path))
        return cls(model)


def _new_tabnet(seed: int, device_name: str):
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch

    return TabNetClassifier(
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.25,
        lambda_sparse=1e-5,
        mask_type="entmax",
        optimizer_fn=torch.optim.AdamW,
        optimizer_params={"lr": 0.004, "weight_decay": 8e-4},
        scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
        scheduler_params={"T_max": 160, "eta_min": 8e-5},
        seed=int(seed),
        verbose=0,
        device_name=device_name,
    )


def fit_tabnet(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    fast: bool,
    device_name: str,
) -> TabNetAdapter:
    from sklearn.model_selection import StratifiedShuffleSplit

    set_global_seed(seed)
    values = np.asarray(X, dtype=np.float32)
    target = np.asarray(y, dtype=np.int64)
    if device_name == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Full TabNet fitting requires CUDA")
    weights = balanced_class_weights(target)
    weight_map = {index: float(weights[index]) for index in range(2)}
    max_epochs = 3 if fast else 160
    if fast or min(np.bincount(target)) < 6:
        chosen_epochs = max_epochs
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=0.18, random_state=int(seed)
        )
        fit_index, early_index = next(splitter.split(values, target))
        probe = _new_tabnet(seed, device_name)
        probe.fit(
            X_train=values[fit_index],
            y_train=target[fit_index],
            eval_set=[(values[early_index], target[early_index])],
            eval_name=["subject_holdout"],
            eval_metric=["auc"],
            max_epochs=max_epochs,
            patience=18,
            batch_size=min(64, len(fit_index)),
            virtual_batch_size=min(16, len(fit_index)),
            num_workers=0,
            drop_last=False,
            weights=weight_map,
        )
        best_epoch = getattr(probe, "best_epoch", max_epochs - 1)
        chosen_epochs = int(np.clip(int(best_epoch) + 1, 20, max_epochs))
        del probe
    final = _new_tabnet(seed + 1, device_name)
    final.fit(
        X_train=values,
        y_train=target,
        max_epochs=chosen_epochs,
        patience=0,
        batch_size=min(64, len(values)),
        virtual_batch_size=min(16, len(values)),
        num_workers=0,
        drop_last=False,
        weights=weight_map,
    )
    return TabNetAdapter(final)


@dataclass(frozen=True)
class TransformerConfig:
    n_features: int
    sequence_length: int
    width: int = 48
    heads: int = 4
    layers: int = 2
    dropout: float = 0.30


def _build_transformer(config: TransformerConfig):
    from torch import nn
    import torch

    class CompactTemporalTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_norm = nn.LayerNorm(config.n_features)
            self.projection = nn.Linear(config.n_features, config.width)
            self.position = nn.Parameter(
                torch.empty(1, config.sequence_length, config.width)
            )
            nn.init.normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=config.width,
                nhead=config.heads,
                dim_feedforward=config.width * 2,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
            self.head = nn.Sequential(
                nn.LayerNorm(config.width * 2),
                nn.Linear(config.width * 2, config.width // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.width // 2, 1),
            )

        def forward(self, x):
            hidden = self.projection(self.input_norm(x))
            hidden = hidden + self.position[:, : hidden.shape[1]]
            hidden = self.encoder(hidden)
            pooled = torch.cat([hidden.mean(dim=1), hidden.amax(dim=1)], dim=1)
            return self.head(pooled).squeeze(1)

    return CompactTemporalTransformer()


class TransformerAdapter:
    def __init__(self, model: Any, config: TransformerConfig, device: str):
        self.model = model
        self.config = config
        self.device = device

    def predict_score(self, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
        import torch

        requested = self.device if self.device == "cuda" and torch.cuda.is_available() else "cpu"
        device = torch.device(requested)
        self.model.to(device).eval()
        rows: list[np.ndarray] = []
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
                rows.append(torch.sigmoid(logits.float()).cpu().numpy())
        return np.clip(np.concatenate(rows), 1e-7, 1 - 1e-7)

    def save(self, path: str | Path) -> None:
        import torch

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.model.to("cpu")
        torch.save(
            {"config": asdict(self.config), "state_dict": self.model.state_dict()},
            output,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TransformerAdapter":
        import torch

        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch versions before the weights_only argument
            payload = torch.load(path, map_location="cpu")
        config = TransformerConfig(**payload["config"])
        model = _build_transformer(config)
        model.load_state_dict(payload["state_dict"])
        return cls(model, config, "cpu")


def _train_transformer_epochs(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    epochs: int,
    validation: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[Any, int]:
    import torch
    from sklearn.metrics import roc_auc_score
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    set_global_seed(seed)
    config = TransformerConfig(n_features=X.shape[2], sequence_length=X.shape[1])
    model = _build_transformer(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=8e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=2e-5
    )
    weights = balanced_class_weights(y)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(weights[1] / weights[0]), device=device)
    )
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
        ),
        batch_size=min(32, len(X)),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    best_epoch = epochs
    best_auc = -np.inf
    best_state = None
    wait = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(batch_x)
                loss = criterion(logits, batch_y * 0.96 + 0.02)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        if validation is not None:
            X_valid, y_valid = validation
            probe = TransformerAdapter(model, config, device.type)
            probability = probe.predict_score(X_valid)
            auc = float(roc_auc_score(y_valid, probability))
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
            if epoch >= 15 and wait >= 16:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, int(best_epoch)


def fit_transformer(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    fast: bool,
) -> TransformerAdapter:
    import torch
    from sklearn.model_selection import StratifiedShuffleSplit

    values = np.asarray(X, dtype=np.float32)
    target = np.asarray(y, dtype=np.int64)
    max_epochs = 3 if fast else 110
    if fast or min(np.bincount(target)) < 6:
        chosen_epochs = max_epochs
    else:
        split = StratifiedShuffleSplit(
            n_splits=1, test_size=0.18, random_state=int(seed)
        )
        fit_index, early_index = next(split.split(values, target))
        _, chosen_epochs = _train_transformer_epochs(
            values[fit_index],
            target[fit_index],
            seed=seed,
            epochs=max_epochs,
            validation=(values[early_index], target[early_index]),
        )
        chosen_epochs = int(np.clip(chosen_epochs, 15, max_epochs))
    model, _ = _train_transformer_epochs(
        values,
        target,
        seed=seed + 1,
        epochs=chosen_epochs,
        validation=None,
    )
    config = TransformerConfig(
        n_features=values.shape[2], sequence_length=values.shape[1]
    )
    return TransformerAdapter(
        model,
        config,
        "cuda" if torch.cuda.is_available() else "cpu",
    )


def fit_model(
    model_name: str,
    *,
    subject_X: np.ndarray,
    temporal_X: np.ndarray,
    y: np.ndarray,
    subject_feature_names: Sequence[str],
    seed: int,
    fast: bool,
    device_name: str,
) -> Any:
    if model_name == "elastic_net":
        return fit_elastic_net(subject_X, y, seed=seed)
    if model_name == "ydf_subject":
        return fit_ydf_subject(
            subject_X, y, subject_feature_names, seed=seed, fast=fast
        )
    if model_name == "ydf_daily":
        return fit_ydf_daily(temporal_X, y, seed=seed, fast=fast)
    if model_name == "tabnet":
        return fit_tabnet(
            subject_X,
            y,
            seed=seed,
            fast=fast,
            device_name=device_name,
        )
    if model_name == "transformer":
        return fit_transformer(temporal_X, y, seed=seed, fast=fast)
    raise ValueError(f"Unknown model: {model_name}")


def predict_model(
    model_name: str,
    model: Any,
    *,
    subject_X: np.ndarray,
    temporal_X: np.ndarray,
) -> np.ndarray:
    if model_name in {"elastic_net", "ydf_subject", "tabnet"}:
        return np.asarray(model.predict_score(subject_X), dtype=np.float64)
    if model_name in {"ydf_daily", "transformer"}:
        return np.asarray(model.predict_score(temporal_X), dtype=np.float64)
    raise ValueError(f"Unknown model: {model_name}")


__all__ = [
    "CLASS_NAMES",
    "GOOGLE_MODELS",
    "MODEL_NAMES",
    "ElasticNetAdapter",
    "TabNetAdapter",
    "TransformerAdapter",
    "YDFDailyAdapter",
    "YDFSubjectAdapter",
    "balanced_class_weights",
    "fit_model",
    "predict_model",
    "set_global_seed",
]
