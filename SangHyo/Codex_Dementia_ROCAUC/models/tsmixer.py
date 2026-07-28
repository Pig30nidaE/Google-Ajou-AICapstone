"""Compact TSMixer classifier for real subject-level daily sequences.

The branch is applicable only when every subject in a fit/predict set has at
least ``sequence_length`` aligned days.  It never pads short histories and does
not expose sequence length, availability masks, subject IDs, or absolute dates
to the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import copy
import random
from typing import Any, Mapping, Sequence

import numpy as np


TSMIXER_SEQUENCE_LENGTH_CHOICES = (21, 28, 35)


@dataclass
class SequencePreprocessor:
    sequence_length: int = 28
    min_channel_coverage: float = 0.20
    max_channels: int = 96

    def fit(
        self,
        sequences: Sequence[np.ndarray],
        feature_names: Sequence[str],
    ) -> "SequencePreprocessor":
        values = [np.asarray(sequence, dtype=np.float64) for sequence in sequences]
        if not values:
            raise ValueError("No training sequences")
        if min(sequence.shape[0] for sequence in values) < int(self.sequence_length):
            raise ValueError(
                "TSMixer disabled: a fold-training subject has fewer than "
                f"{self.sequence_length} aligned days"
            )
        width = values[0].shape[1]
        if any(sequence.ndim != 2 or sequence.shape[1] != width for sequence in values):
            raise ValueError("Inconsistent sequence channel schema")
        recent = np.concatenate(
            [sequence[-int(self.sequence_length) :] for sequence in values], axis=0
        )
        recent[~np.isfinite(recent)] = np.nan
        coverage = np.mean(np.isfinite(recent), axis=0)
        medians = np.nanmedian(recent, axis=0)
        q25, q75 = np.nanquantile(recent, [0.25, 0.75], axis=0)
        scales = q75 - q25
        fallback = np.nanstd(recent, axis=0)
        bad = ~np.isfinite(scales) | (scales < 1e-8)
        scales[bad] = fallback[bad]
        scales[~np.isfinite(scales) | (scales < 1e-8)] = 1.0
        medians[~np.isfinite(medians)] = 0.0
        variance = np.nanvar((recent - medians) / scales, axis=0)
        eligible = np.flatnonzero(
            (coverage >= float(self.min_channel_coverage))
            & np.isfinite(variance)
            & (variance > 1e-8)
        )
        if len(eligible) == 0:
            raise ValueError("No sequence channel passes fold-local coverage/variance checks")
        order = eligible[np.argsort(-variance[eligible], kind="stable")]
        selected = order[: min(int(self.max_channels), len(order))]
        self.selected_indices_ = np.asarray(selected, dtype=np.int64)
        self.selected_feature_names_ = tuple(
            str(feature_names[index]) for index in self.selected_indices_
        )
        self.medians_ = medians[self.selected_indices_]
        self.scales_ = scales[self.selected_indices_]
        return self

    def transform(self, sequences: Sequence[np.ndarray]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for sequence in sequences:
            values = np.asarray(sequence, dtype=np.float64)
            if values.shape[0] < int(self.sequence_length):
                raise ValueError(
                    "TSMixer prediction subject has fewer than the fitted sequence length"
                )
            recent = values[-int(self.sequence_length) :, self.selected_indices_]
            recent = np.where(np.isfinite(recent), recent, self.medians_[None, :])
            recent = (recent - self.medians_[None, :]) / self.scales_[None, :]
            rows.append(np.clip(recent, -12.0, 12.0).astype(np.float32))
        return np.stack(rows, axis=0)


def sequence_applicability(
    sequences: Sequence[np.ndarray], *, sequence_length: int
) -> tuple[bool, str]:
    if not sequences:
        return False, "no sequences"
    lengths = [int(np.asarray(sequence).shape[0]) for sequence in sequences]
    if min(lengths) < int(sequence_length):
        return (
            False,
            f"minimum aligned days {min(lengths)} < required {sequence_length}",
        )
    return (
        True,
        f"aligned daily sequences retained (min/median/max="
        f"{min(lengths)}/{int(np.median(lengths))}/{max(lengths)})",
    )


def _set_seed(seed: int, deterministic: bool) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)


def _build_network(
    *,
    sequence_length: int,
    n_channels: int,
    hidden_size: int,
    n_blocks: int,
    dropout: float,
):
    import torch
    from torch import nn

    class MixerBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            temporal_hidden = max(sequence_length, sequence_length * 2)
            self.temporal_norm = nn.LayerNorm(sequence_length)
            self.temporal = nn.Sequential(
                nn.Linear(sequence_length, temporal_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(temporal_hidden, sequence_length),
            )
            self.channel_norm = nn.LayerNorm(hidden_size)
            self.channel = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
            )

        def forward(self, x):
            temporal = self.temporal_norm(x.transpose(1, 2))
            x = x + self.temporal(temporal).transpose(1, 2)
            x = x + self.channel(self.channel_norm(x))
            return x

    class TSMixerNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_norm = nn.LayerNorm(n_channels)
            self.projection = nn.Linear(n_channels, hidden_size)
            self.blocks = nn.ModuleList([MixerBlock() for _ in range(n_blocks)])
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size * 3),
                nn.Linear(hidden_size * 3, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, x):
            x = self.projection(self.input_norm(x))
            for block in self.blocks:
                x = block(x)
            pooled = torch.cat([x.mean(dim=1), x.amax(dim=1), x[:, -1]], dim=1)
            return self.head(pooled).squeeze(1)

    return TSMixerNetwork()


class TSMixerBinaryEstimator:
    def __init__(
        self,
        *,
        feature_names: tuple[str, ...],
        sequence_length: int = 28,
        hidden_size: int = 64,
        n_blocks: int = 2,
        dropout: float = 0.30,
        learning_rate: float = 7e-4,
        weight_decay: float = 0.01,
        focal_gamma: float = 1.0,
        min_channel_coverage: float = 0.20,
        max_channels: int = 96,
        max_epochs: int = 160,
        patience: int = 25,
        batch_size: int = 32,
        validation_fraction: float = 0.20,
        device: str = "auto",
        mixed_precision: bool = True,
        deterministic: bool = True,
        seed: int = 0,
    ) -> None:
        self.feature_names = feature_names
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.focal_gamma = focal_gamma
        self.min_channel_coverage = min_channel_coverage
        self.max_channels = max_channels
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.device = device
        self.mixed_precision = mixed_precision
        self.deterministic = deterministic
        self.seed = seed

    def _resolved_device(self):
        import torch

        if self.device in {"cpu", "cuda"}:
            if self.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _loss(self, logits, target, pos_weight):
        import torch
        import torch.nn.functional as functional

        base = functional.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
            pos_weight=pos_weight,
        )
        probability = torch.sigmoid(logits)
        pt = torch.where(target > 0.5, probability, 1.0 - probability)
        return (torch.pow(1.0 - pt, float(self.focal_gamma)) * base).mean()

    def _train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int,
        validation: tuple[np.ndarray, np.ndarray] | None,
    ):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from ..metrics import safe_roc_auc

        device = self._resolved_device()
        model = _build_network(
            sequence_length=int(self.sequence_length),
            n_channels=X.shape[2],
            hidden_size=int(self.hidden_size),
            n_blocks=int(self.n_blocks),
            dropout=float(self.dropout),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        dataset = TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
        )
        generator = torch.Generator().manual_seed(int(self.seed))
        loader = DataLoader(
            dataset,
            batch_size=min(int(self.batch_size), len(dataset)),
            shuffle=True,
            generator=generator,
            num_workers=0,
            drop_last=False,
        )
        counts = np.bincount(y.astype(np.int64), minlength=2)
        positive_weight = torch.tensor(
            [float(counts[0] / max(1, counts[1]))],
            dtype=torch.float32,
            device=device,
        )
        amp_enabled = bool(self.mixed_precision and device.type == "cuda")
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled,
        )
        best_state = None
        best_auc = -np.inf
        best_epoch = 0
        stale = 0
        for epoch in range(int(epochs)):
            model.train()
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                amp_context = (
                    torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                    )
                    if amp_enabled
                    else nullcontext()
                )
                with amp_context:
                    loss = self._loss(model(batch_x), batch_y, positive_weight)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
            if validation is None:
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                continue
            validation_x, validation_y = validation
            score = self._predict_network(model, validation_x, device)
            auc = safe_roc_auc(validation_y, score)
            if auc > best_auc + 1e-6:
                best_auc = auc
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            if stale >= int(self.patience):
                break
        if best_state is None:
            best_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        return model, best_epoch + 1

    @staticmethod
    def _predict_network(model, X: np.ndarray, device) -> np.ndarray:
        import torch

        model.eval()
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(X), 256):
                batch = torch.as_tensor(
                    X[start : start + 256], dtype=torch.float32, device=device
                )
                outputs.append(torch.sigmoid(model(batch)).cpu().numpy())
        return np.clip(np.concatenate(outputs), 1e-7, 1.0 - 1e-7)

    def fit(self, sequences: Sequence[np.ndarray], y: Sequence[int]):
        from sklearn.model_selection import StratifiedShuffleSplit

        target = np.asarray(y, dtype=np.int64)
        raw_sequences = tuple(
            np.asarray(sequence, dtype=np.float64) for sequence in sequences
        )
        if len(raw_sequences) != len(target):
            raise ValueError("TSMixer sequence/target cardinality mismatch")
        _set_seed(int(self.seed), bool(self.deterministic))
        counts = np.bincount(target, minlength=2)
        if int(counts.min()) < 3:
            raise ValueError(
                f"TSMixer internal validation requires >=3 per class; {counts.tolist()}"
            )
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=float(self.validation_fraction),
            random_state=int(self.seed),
        )
        train_index, validation_index = next(
            splitter.split(np.zeros(len(target), dtype=np.int8), target)
        )
        # Epoch selection has its own training-only sequence preprocessor.  The
        # internal early-stopping subjects therefore do not determine channel
        # coverage, medians, scales, or variance selection.
        early_preprocessor = SequencePreprocessor(
            sequence_length=int(self.sequence_length),
            min_channel_coverage=float(self.min_channel_coverage),
            max_channels=int(self.max_channels),
        ).fit(
            [raw_sequences[index] for index in train_index],
            self.feature_names,
        )
        early_train = early_preprocessor.transform(
            [raw_sequences[index] for index in train_index]
        )
        early_validation = early_preprocessor.transform(
            [raw_sequences[index] for index in validation_index]
        )
        _, selected_epochs = self._train(
            early_train,
            target[train_index],
            epochs=int(self.max_epochs),
            validation=(early_validation, target[validation_index]),
        )
        # Once the epoch count is fixed, refit preprocessing and the network on
        # every subject in the enclosing fold-training scope.
        self.preprocessor_ = SequencePreprocessor(
            sequence_length=int(self.sequence_length),
            min_channel_coverage=float(self.min_channel_coverage),
            max_channels=int(self.max_channels),
        ).fit(raw_sequences, self.feature_names)
        values = self.preprocessor_.transform(raw_sequences)
        _set_seed(int(self.seed), bool(self.deterministic))
        self.model_, _ = self._train(
            values,
            target,
            epochs=selected_epochs,
            validation=None,
        )
        self.selected_epochs_ = selected_epochs
        self.classes_ = np.asarray([0, 1], dtype=np.int64)
        return self

    def predict_proba(self, sequences: Sequence[np.ndarray]) -> np.ndarray:
        values = self.preprocessor_.transform(sequences)
        positive = self._predict_network(
            self.model_,
            values,
            self._resolved_device(),
        )
        return np.column_stack([1.0 - positive, positive])


def build_tsmixer_estimator(
    params: Mapping[str, Any],
    *,
    feature_names: tuple[str, ...],
    data_config,
    neural_config,
    seed: int,
) -> TSMixerBinaryEstimator:
    return TSMixerBinaryEstimator(
        feature_names=feature_names,
        sequence_length=int(params.get("sequence_length", 28)),
        hidden_size=int(params.get("hidden_size", 64)),
        n_blocks=int(params.get("n_blocks", 2)),
        dropout=float(params.get("dropout", 0.30)),
        learning_rate=float(params.get("learning_rate", 7e-4)),
        weight_decay=float(params.get("weight_decay", 0.01)),
        focal_gamma=float(params.get("focal_gamma", 1.0)),
        min_channel_coverage=float(data_config.sequence_min_channel_coverage),
        max_channels=int(data_config.max_sequence_channels),
        max_epochs=int(neural_config.max_epochs),
        patience=int(neural_config.patience),
        batch_size=int(neural_config.batch_size),
        validation_fraction=float(neural_config.validation_fraction),
        device=str(neural_config.device),
        mixed_precision=bool(neural_config.mixed_precision),
        deterministic=bool(neural_config.deterministic),
        seed=int(seed),
    )


def suggest_tsmixer_params(trial, fixed: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(fixed)
    params["sequence_length"] = trial.suggest_categorical(
        "sequence_length", list(TSMIXER_SEQUENCE_LENGTH_CHOICES)
    )
    params["hidden_size"] = trial.suggest_categorical(
        "hidden_size", [32, 48, 64, 96]
    )
    params["n_blocks"] = trial.suggest_int("n_blocks", 1, 4)
    params["dropout"] = trial.suggest_float("dropout", 0.15, 0.50)
    params["learning_rate"] = trial.suggest_float(
        "learning_rate", 1e-4, 3e-3, log=True
    )
    params["weight_decay"] = trial.suggest_float(
        "weight_decay", 1e-5, 0.1, log=True
    )
    params["focal_gamma"] = trial.suggest_float("focal_gamma", 0.0, 2.5)
    return params


__all__ = [
    "SequencePreprocessor",
    "TSMIXER_SEQUENCE_LENGTH_CHOICES",
    "TSMixerBinaryEstimator",
    "build_tsmixer_estimator",
    "sequence_applicability",
    "suggest_tsmixer_params",
]
