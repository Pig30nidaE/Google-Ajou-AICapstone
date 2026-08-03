"""LSTM, Bi-LSTM and 1D-CNN.

**Neither paper reports any architecture detail for these three models**
(``unresolved_questions.md`` Q1): no layer count, hidden size, kernel size, filter
count, dropout, optimizer, learning rate, epoch budget or batch size.  Section
3.3.2 explains what an LSTM *is* and stops there.

Everything below is therefore ``assumption_variant_minimal_architecture``, sized
conservatively for 141 training subjects.  If these numbers do not reproduce the
paper's 1D-CNN AUC of 0.810, that is a consequence of the missing report, not a
failed reproduction -- ``summary()`` carries that flag into every result file.

torch is imported lazily so that ``--dry-run``, ``--audit-only`` and the static
tests all work in an environment without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ARCHITECTURE_SOURCE = "assumption_variant_minimal_architecture"

DEFAULTS: dict[str, dict[str, Any]] = {
    "lstm": {"hidden_size": 64, "num_layers": 1, "dropout": 0.2, "bidirectional": False},
    "bilstm": {"hidden_size": 64, "num_layers": 1, "dropout": 0.2, "bidirectional": True},
    "cnn1d": {"filters": (64, 64), "kernel_size": 3, "pool_size": 2, "dropout": 0.2},
}

TRAINING_DEFAULTS = {
    "loss": "bce",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "batch_size": 16,
    "max_epochs": 100,
    "early_stopping_patience": 10,
    "early_stopping_metric": "auc",   # matches the reported primary metric
    "validation_fraction": 0.2,
    "pos_weight": None,   # the paper applies no imbalance correction
    "grad_clip": 1.0,
}


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC that returns 0.5 rather than raising on a single-class slice."""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, np.asarray(scores, dtype=np.float64)))


def _stratified_monitor_split(
    y: np.ndarray, *, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split training rows into (monitor, fit) indices, preserving class balance.

    Returns empty monitor indices when either side would end up single-class; the
    caller then falls back to monitoring the training loss.
    """
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    monitor: list[int] = []
    for label in (0, 1):
        members = np.flatnonzero(y == label)
        rng.shuffle(members)
        take = int(round(len(members) * float(fraction)))
        monitor.extend(members[:take].tolist())

    monitor_idx = np.asarray(sorted(monitor), dtype=int)
    fit_idx = np.setdiff1d(np.arange(len(y)), monitor_idx)
    both_sides_have_two_classes = (
        len(monitor_idx) >= 2
        and len(np.unique(y[monitor_idx])) == 2
        and len(fit_idx) >= 2
        and len(np.unique(y[fit_idx])) == 2
    )
    if not both_sides_have_two_classes:
        return np.array([], dtype=int), np.arange(len(y))
    return monitor_idx, fit_idx


class ModelDependencyError(ImportError):
    """Raised when torch is unavailable but a sequence model was requested."""


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _require_torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ModelDependencyError(
            "PyTorch is required for lstm/bilstm/cnn1d. Install it via "
            "requirements_colab.txt, or restrict config.models to the tree learners."
        ) from error
    return torch


def _build_module(name: str, n_features: int, params: dict[str, Any]):
    """Construct the nn.Module.  Defined inside so torch stays a lazy import."""
    torch = _require_torch()
    nn = torch.nn

    class RecurrentClassifier(nn.Module):
        """LSTM / Bi-LSTM with padding-aware pooling."""

        def __init__(self) -> None:
            super().__init__()
            self.bidirectional = bool(params["bidirectional"])
            self.rnn = nn.LSTM(
                input_size=n_features,
                hidden_size=int(params["hidden_size"]),
                num_layers=int(params["num_layers"]),
                batch_first=True,
                bidirectional=self.bidirectional,
                dropout=float(params["dropout"]) if int(params["num_layers"]) > 1 else 0.0,
            )
            out_dim = int(params["hidden_size"]) * (2 if self.bidirectional else 1)
            self.dropout = nn.Dropout(float(params["dropout"]))
            self.head = nn.Linear(out_dim, 1)

        def forward(self, x, lengths=None):
            if lengths is None:
                output, _ = self.rnn(x)
                pooled = output[:, -1, :]
            else:
                lengths_cpu = lengths.detach().cpu().clamp(min=1)
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths_cpu, batch_first=True, enforce_sorted=False
                )
                packed_out, _ = self.rnn(packed)
                output, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_out, batch_first=True, total_length=x.shape[1]
                )
                # Mean over valid timesteps only -- padding must not enter the pool.
                mask = (
                    torch.arange(x.shape[1], device=x.device)[None, :]
                    < lengths.to(x.device)[:, None]
                ).unsqueeze(-1).float()
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            return self.head(self.dropout(pooled)).squeeze(-1)

    class Conv1dClassifier(nn.Module):
        """Two Conv1d blocks then masked global average pooling."""

        def __init__(self) -> None:
            super().__init__()
            filters = tuple(int(f) for f in params["filters"])
            kernel = int(params["kernel_size"])
            layers: list[Any] = []
            in_channels = n_features
            for out_channels in filters:
                layers += [
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2),
                    nn.ReLU(),
                    nn.MaxPool1d(int(params["pool_size"])),
                ]
                in_channels = out_channels
            self.features = nn.Sequential(*layers)
            self.dropout = nn.Dropout(float(params["dropout"]))
            self.head = nn.Linear(in_channels, 1)
            self.n_pools = len(filters)
            self.pool_size = int(params["pool_size"])

        def forward(self, x, lengths=None):
            # (B, T, F) -> (B, F, T) for Conv1d.
            z = self.features(x.transpose(1, 2))
            if lengths is None:
                pooled = z.mean(dim=2)
            else:
                scaled = lengths.to(z.device).float()
                for _ in range(self.n_pools):
                    scaled = torch.floor(scaled / self.pool_size)
                scaled = scaled.clamp(min=1.0)
                mask = (
                    torch.arange(z.shape[2], device=z.device)[None, :] < scaled[:, None]
                ).unsqueeze(1).float()
                pooled = (z * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)
            return self.head(self.dropout(pooled)).squeeze(-1)

    if name in ("lstm", "bilstm"):
        return RecurrentClassifier()
    if name == "cnn1d":
        return Conv1dClassifier()
    raise ValueError(f"unknown sequence model: {name!r}")


@dataclass
class SequenceModel:
    """Uniform wrapper matching :class:`~..models.tabular.TabularModel`."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"
    module: Any = None
    n_features: int | None = None
    history: dict[str, Any] = field(default_factory=dict)

    # -- construction ---------------------------------------------------------
    def build(self, n_features: int) -> "SequenceModel":
        torch = _require_torch()
        self.n_features = int(n_features)
        self.module = _build_module(self.name, self.n_features, self.params)
        self.module.to(torch.device(self.device))
        return self

    # -- training -------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        lengths: np.ndarray | None = None,
        seed: int = 42,
    ) -> "SequenceModel":
        torch = _require_torch()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"{self.name} expects (N, T, F); got shape {X.shape}")
        y = np.asarray(y, dtype=np.float32)
        if self.module is None:
            self.build(X.shape[2])
        if X.shape[2] != self.n_features:
            raise ValueError(
                f"feature count changed after build: expected {self.n_features}, got {X.shape[2]}"
            )

        cfg = {**TRAINING_DEFAULTS, **self.training}
        device = torch.device(self.device)
        generator = torch.Generator().manual_seed(int(seed))

        lengths_arr = (
            np.full(len(X), X.shape[1], dtype=np.int64)
            if lengths is None
            else np.asarray(lengths, dtype=np.int64)
        )

        # Early stopping needs a held-out slice, carved from the training rows only.
        # It is stratified: an unstratified draw of ~28 subjects from a 40%-positive
        # pool swings the monitored class balance enough to make the signal noise.
        val_idx, tr_idx = _stratified_monitor_split(
            y, fraction=float(cfg["validation_fraction"]), seed=seed
        )

        tensors = {
            "X": torch.from_numpy(X).to(device),
            "y": torch.from_numpy(y).to(device),
            "len": torch.from_numpy(lengths_arr),
        }

        pos_weight = cfg.get("pos_weight")
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=None if pos_weight is None
            else torch.tensor(float(pos_weight), device=device)
        )
        optimizer = torch.optim.Adam(self.module.parameters(), lr=float(cfg["learning_rate"]))

        # The project's primary metric is ROC-AUC, and BCE loss on ~28 monitored
        # subjects rises as soon as the net grows confident even while its ranking
        # keeps improving -- which stopped training at epoch 0 and restored an
        # essentially untrained model.  Monitor what we actually report.
        metric = str(cfg.get("early_stopping_metric", "auc")).lower()
        if metric not in ("auc", "loss"):
            raise ValueError(f"early_stopping_metric must be auc/loss, got {metric!r}")
        higher_is_better = metric == "auc"

        best_state = {k: v.detach().clone() for k, v in self.module.state_dict().items()}
        best_score = -np.inf if higher_is_better else np.inf
        best_tiebreak = np.inf   # monitor loss, used only when the metric ties
        best_epoch = 0
        patience_left = int(cfg["early_stopping_patience"])
        losses: list[float] = []
        monitored: list[float] = []
        batch_size = int(cfg["batch_size"])

        for epoch in range(int(cfg["max_epochs"])):
            self.module.train()
            perm = torch.randperm(len(tr_idx), generator=generator).numpy()
            shuffled = tr_idx[perm]
            epoch_loss, seen = 0.0, 0
            for start in range(0, len(shuffled), batch_size):
                batch = shuffled[start:start + batch_size]
                if len(batch) < 2:
                    continue
                optimizer.zero_grad()
                logits = self.module(tensors["X"][batch], tensors["len"][batch])
                loss = criterion(logits, tensors["y"][batch])
                loss.backward()
                if cfg.get("grad_clip"):
                    torch.nn.utils.clip_grad_norm_(
                        self.module.parameters(), float(cfg["grad_clip"])
                    )
                optimizer.step()
                epoch_loss += float(loss.item()) * len(batch)
                seen += len(batch)
            losses.append(epoch_loss / max(seen, 1))

            if len(val_idx):
                self.module.eval()
                with torch.no_grad():
                    logits = self.module(tensors["X"][val_idx], tensors["len"][val_idx])
                    current_loss = float(criterion(logits, tensors["y"][val_idx]).item())
                    current = (
                        _safe_auc(y[val_idx], torch.sigmoid(logits).cpu().numpy())
                        if higher_is_better else current_loss
                    )
            else:
                # No usable monitor split: fall back to training loss, which always
                # improves, so this degenerates to "train for max_epochs".
                current_loss = losses[-1]
                current = -losses[-1] if higher_is_better else losses[-1]
            monitored.append(current)

            if higher_is_better:
                # AUC saturates: on a small monitor split it can hit 1.0 at epoch 0
                # and never "improve" again, which would restore untrained weights.
                # Break ties on the monitor loss so training keeps progressing.
                improved = current > best_score + 1e-5 or (
                    abs(current - best_score) <= 1e-5 and current_loss < best_tiebreak - 1e-5
                )
            else:
                improved = current < best_score - 1e-5

            if improved:
                best_score = current
                best_tiebreak = current_loss
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in self.module.state_dict().items()}
                patience_left = int(cfg["early_stopping_patience"])
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        self.module.load_state_dict(best_state)
        # Restoring epoch 0 means we report a barely-trained network. That is a
        # training failure, not a finding, so make it impossible to miss.
        train_loss_improved = bool(losses and (losses[0] - min(losses)) > 0.05)
        self.history = {
            "epochs_run": len(losses),
            "train_loss": losses,
            "monitor_metric": metric,
            "monitor_score": monitored,
            "best_monitor_score": float(best_score),
            "best_epoch": int(best_epoch),
            "early_stopped": len(losses) < int(cfg["max_epochs"]),
            "monitor_split_size": int(len(val_idx)),
            "monitor_split_positives": int(y[val_idx].sum()) if len(val_idx) else 0,
            "degenerate_training": bool(best_epoch == 0 and train_loss_improved),
        }
        return self

    # -- inference ------------------------------------------------------------
    def predict_proba(self, X: np.ndarray, *, lengths: np.ndarray | None = None) -> np.ndarray:
        torch = _require_torch()
        if self.module is None:
            raise RuntimeError(f"{self.name} has not been built/fitted")
        X = np.asarray(X, dtype=np.float32)
        device = torch.device(self.device)
        lengths_arr = (
            np.full(len(X), X.shape[1], dtype=np.int64)
            if lengths is None
            else np.asarray(lengths, dtype=np.int64)
        )
        self.module.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(X), 256):
                chunk = slice(start, start + 256)
                logits = self.module(
                    torch.from_numpy(X[chunk]).to(device),
                    torch.from_numpy(lengths_arr[chunk]),
                )
                outputs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(outputs).astype(np.float64) if outputs else np.array([])

    # -- persistence ----------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        torch = _require_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "name": self.name,
                "params": self.params,
                "training": self.training,
                "n_features": self.n_features,
                "state_dict": self.module.state_dict(),
                "history": self.history,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "SequenceModel":
        torch = _require_torch()
        blob = torch.load(Path(path), map_location=device, weights_only=False)
        model = cls(
            name=blob["name"], params=blob["params"], training=blob["training"],
            device=device, history=blob.get("history", {}),
        )
        model.build(int(blob["n_features"]))
        model.module.load_state_dict(blob["state_dict"])
        model.module.eval()
        return model

    # -- reporting ------------------------------------------------------------
    def architecture_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "name": self.name,
            "architecture_source": ARCHITECTURE_SOURCE,
            "reported_in_paper": False,
            "note": (
                "Neither the thesis nor the journal article reports any architecture "
                "detail for this model; every value here is our assumption."
            ),
            "params": dict(self.params),
            "training": {**TRAINING_DEFAULTS, **self.training},
            "n_features": self.n_features,
        }
        if self.module is not None:
            summary["n_parameters"] = int(
                sum(p.numel() for p in self.module.parameters())
            )
            summary["module_repr"] = repr(self.module)
        if self.history:
            summary["training_diagnostics"] = {
                key: self.history[key]
                for key in (
                    "epochs_run", "best_epoch", "monitor_metric", "best_monitor_score",
                    "early_stopped", "monitor_split_size", "monitor_split_positives",
                    "degenerate_training",
                )
                if key in self.history
            }
            if self.history.get("degenerate_training"):
                summary["training_diagnostics"]["warning"] = (
                    "Early stopping restored the epoch-0 weights while the training "
                    "loss was still improving: this model is essentially untrained "
                    "and its metrics are not a performance result."
                )
        return summary

    def summary(self) -> dict[str, Any]:
        return self.architecture_summary()


def build_sequence_model(
    name: str,
    *,
    device: str = "cpu",
    overrides: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
) -> SequenceModel:
    if name not in DEFAULTS:
        raise ValueError(f"unknown sequence model {name!r}; expected one of {tuple(DEFAULTS)}")
    params = dict(DEFAULTS[name])
    params.update(overrides or {})
    if name == "lstm":
        params["bidirectional"] = False
    elif name == "bilstm":
        params["bidirectional"] = True
    return SequenceModel(
        name=name, params=params, training=dict(training or {}), device=device
    )
