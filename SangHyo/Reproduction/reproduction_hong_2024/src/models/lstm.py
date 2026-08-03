"""The paper's LSTM: one LSTM layer, one dense layer, sigmoid output.

Reported configuration (§4.2): LSTM 128 units -> Dense 64 units -> Dense 1
sigmoid, Adam at learning rate 0.001, threshold 0.5, and early stopping (stated
in the limitations section).  Everything else -- batch size, epoch budget,
dropout, patience, the validation source for early stopping -- is unreported and
lives in the config, documented as A-10 through A-14.

The paper does not report its framework.  This package uses PyTorch to match the
rest of the repository, so framework-level defaults remain an explicit source of
non-equivalence.  This is a reported-method reconstruction, not a bit-exact one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LSTMConfig:
    lstm_units: int = 128
    dense_units: int = 64
    learning_rate: float = 0.001
    dropout: float = 0.0
    recurrent_dropout: float = 0.0
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10
    early_stopping: bool = True
    class_weight: bool = False
    threshold: float = 0.5
    seed: int = 42

    def describe(self) -> dict[str, Any]:
        return dict(self.__dict__)


class SequenceLSTM:
    """A thin, deterministic wrapper so the engine never touches torch directly."""

    def __init__(self, config: LSTMConfig, *, n_features: int, sequence_length: int,
                 device: str = "cpu") -> None:
        self.config = config
        self.n_features = int(n_features)
        self.sequence_length = int(sequence_length)
        self.device = device
        self.model: Any = None
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.best_epoch: int | None = None
        self.meta: dict[str, Any] = {}

    # -- construction ---------------------------------------------------------
    def _build(self) -> Any:
        import torch
        from torch import nn

        cfg = self.config
        torch.manual_seed(cfg.seed)

        class Net(nn.Module):
            def __init__(self, n_features: int) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_features,
                    hidden_size=cfg.lstm_units,
                    num_layers=1,
                    batch_first=True,
                )
                self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
                self.dense = nn.Linear(cfg.lstm_units, cfg.dense_units)
                self.activation = nn.ReLU()
                self.out = nn.Linear(cfg.dense_units, 1)

            def forward(self, x):  # noqa: ANN001 - torch tensors
                # The paper's diagram takes the final hidden state, not a pooled
                # sequence, so the last timestep is what reaches the dense layer.
                _, (hidden, _) = self.lstm(x)
                h = self.dropout(hidden[-1])
                h = self.activation(self.dense(h))
                return self.out(h).squeeze(-1)

        return Net(self.n_features).to(self.device)

    # -- training -------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "SequenceLSTM":
        """Train on ``(X, y)``, early-stopping on *validation* when supplied.

        ``validation`` must never be the outer test set.  ``audit/leakage.py``
        checks that, and the engine only ever passes an inner/held-out-train
        slice here.
        """
        cfg = self.config
        if cfg.early_stopping and validation is None:
            raise ValueError(
                "early_stopping=True requires an explicit train-side validation set; "
                "silently running every epoch would not reproduce the reported method"
            )

        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(cfg.seed)
        self.model = self._build()

        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32))
        y_t = torch.as_tensor(np.asarray(y, dtype=np.float32))
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=cfg.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(cfg.seed),
            drop_last=False,
        )

        if cfg.class_weight:
            counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
            pos_weight = torch.as_tensor(
                [counts[0] / max(counts[1], 1)], dtype=torch.float32, device=self.device
            )
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()
        optimiser = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

        best_loss, best_state, bad_epochs = float("inf"), None, 0
        for epoch in range(cfg.max_epochs):
            self.model.train()
            running, n_seen = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimiser.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimiser.step()
                running += float(loss) * len(xb)
                n_seen += len(xb)
            train_loss = running / max(n_seen, 1)
            self.history["train_loss"].append(train_loss)

            if validation is None:
                self.history["val_loss"].append(float("nan"))
                continue

            val_loss = self._evaluate_loss(validation, criterion)
            self.history["val_loss"].append(val_loss)
            if val_loss < best_loss - 1e-5:
                best_loss, bad_epochs = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                self.best_epoch = epoch
            else:
                bad_epochs += 1
                if cfg.early_stopping and bad_epochs >= cfg.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.meta = {
            "n_epochs_run": len(self.history["train_loss"]),
            "best_epoch": self.best_epoch,
            "best_val_loss": None if best_loss == float("inf") else best_loss,
            "early_stopped": bool(
                validation is not None
                and cfg.early_stopping
                and len(self.history["train_loss"]) < cfg.max_epochs
            ),
            "used_validation": validation is not None,
            "early_stopping_requested": bool(cfg.early_stopping),
            "early_stopping_applied": bool(cfg.early_stopping and validation is not None),
        }
        return self

    def _evaluate_loss(self, validation: tuple[np.ndarray, np.ndarray], criterion: Any) -> float:
        import torch

        Xv, yv = validation
        if len(Xv) == 0:
            return float("nan")
        self.model.eval()
        with torch.no_grad():
            xb = torch.as_tensor(np.asarray(Xv, dtype=np.float32)).to(self.device)
            yb = torch.as_tensor(np.asarray(yv, dtype=np.float32)).to(self.device)
            return float(criterion(self.model(xb), yb))

    # -- inference ------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("model must be fitted before predict_proba")
        if len(X) == 0:
            return np.empty(0, dtype=np.float64)
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X), 512):
                batch = torch.as_tensor(
                    np.asarray(X[start : start + 512], dtype=np.float32)
                ).to(self.device)
                outputs.append(torch.sigmoid(self.model(batch)).cpu().numpy())
        return np.concatenate(outputs).astype(np.float64)

    # -- persistence ----------------------------------------------------------
    def save(self, path: str) -> str:
        import torch

        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": self.config.describe(),
                "n_features": self.n_features,
                "sequence_length": self.sequence_length,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str, *, device: str = "cpu") -> "SequenceLSTM":
        import torch

        blob = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            LSTMConfig(**blob["config"]),
            n_features=blob["n_features"],
            sequence_length=blob["sequence_length"],
            device=device,
        )
        model.model = model._build()
        model.model.load_state_dict(blob["state_dict"])
        return model


@dataclass
class LSTMSearchSpace:
    """The paper's grid (§4.2), for the inner CV of experiment C only.

    Learning rate is given as the range 0.001-0.01; a range is not a grid, so a
    small explicit list stands in for it (assumption A-15).
    """

    lstm_units: tuple[int, ...] = (64, 128, 256)
    dense_units: tuple[int, ...] = (32, 64, 128)
    learning_rate: tuple[float, ...] = (0.001, 0.005, 0.01)
    dropout: tuple[float, ...] = (0.0,)
    batch_size: tuple[int, ...] = (64,)
    extra: dict[str, Any] = field(default_factory=dict)

    def candidates(self, *, limit: int | None = None, seed: int = 42) -> list[dict[str, Any]]:
        from itertools import product

        grid = [
            {
                "lstm_units": lstm,
                "dense_units": dense,
                "learning_rate": lr,
                "dropout": dropout,
                "batch_size": batch,
            }
            for lstm, dense, lr, dropout, batch in product(
                self.lstm_units, self.dense_units, self.learning_rate,
                self.dropout, self.batch_size,
            )
        ]
        if limit is not None and len(grid) > limit:
            rng = np.random.default_rng(seed)
            picked = rng.choice(len(grid), size=limit, replace=False)
            grid = [grid[i] for i in sorted(picked)]
        return grid
