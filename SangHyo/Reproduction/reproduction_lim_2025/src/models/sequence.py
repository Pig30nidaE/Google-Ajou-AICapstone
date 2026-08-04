"""LSTM, Bi-LSTM and 1D-CNN.

**Neither paper reports the implemented architecture for these three models**
(``unresolved_questions.md`` Q1): no layer count, hidden size, kernel size, filter
count, dropout, optimizer, learning rate, epoch budget or batch size.  Section
3.3.2 provides only generic structural descriptions.

Everything below is therefore ``assumption_variant_minimal_architecture``, sized
conservatively for 141 training subjects.  The only structural clues that the
papers do provide are followed: the Bi-LSTM concatenates the final forward and
backward states, and the 1D-CNN sends pooled feature maps through Flatten and a
fully-connected classifier.  Filter counts, kernel sizes and all training
settings remain assumptions, and ``summary()`` carries that flag into every
result file.

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
    "lstm": {
        "hidden_size": 64, "num_layers": 1, "dropout": 0.2,
        "bidirectional": False, "readout": "last_hidden",
    },
    "bilstm": {
        "hidden_size": 64, "num_layers": 1, "dropout": 0.2,
        "bidirectional": True, "readout": "last_hidden_concat",
    },
    "cnn1d": {
        # Section 3.3.2 describes exactly three stages -- "1) Conv1D 계층,
        # 2) Pooling 계층, 3) Fully Connected 계층" -- i.e. a single conv block,
        # not a stack.  conv_padding='valid' matches the Keras Conv1D default,
        # and the papers name Keras layers throughout.
        "filters": (64,), "kernel_size": 3, "pool_size": 2,
        "conv_padding": "valid", "dropout": 0.2, "readout": "flatten",
    },
}

TRAINING_DEFAULTS = {
    "loss": "bce",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    # Keras `model.fit` defaults to 32, and Adam to lr=1e-3.  The papers name
    # Keras layers but report no training settings, so the framework defaults are
    # the least-invented choice available.
    "batch_size": 32,
    "max_epochs": 100,
    "early_stopping": True,
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


def _build_module(
    name: str, n_features: int, n_timesteps: int, params: dict[str, Any]
):
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

        def forward(self, x, lengths=None, padding_side="post"):
            if lengths is None:
                _, (hidden, _) = self.rnn(x)
            else:
                lengths_cpu = lengths.detach().cpu().clamp(min=1)
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths_cpu, batch_first=True, enforce_sorted=False
                )
                _, (hidden, _) = self.rnn(packed)
            if self.bidirectional:
                # Last layer's forward/backward states.  This is the closest
                # executable interpretation of the papers' "concatenate" clue.
                pooled = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                pooled = hidden[-1]
            return self.head(self.dropout(pooled)).squeeze(-1)

    class Conv1dClassifier(nn.Module):
        """Two Conv1d blocks, Flatten, then a fully-connected classifier."""

        def __init__(self) -> None:
            super().__init__()
            filters = tuple(int(f) for f in params["filters"])
            kernel = int(params["kernel_size"])
            pool_size = int(params["pool_size"])
            if not filters or any(value <= 0 for value in filters):
                raise ValueError("cnn1d filters must be a non-empty sequence of positives")
            if kernel < 1:
                raise ValueError("cnn1d kernel_size must be a positive integer")
            conv_padding = str(params.get("conv_padding", "same")).lower()
            if conv_padding not in ("same", "valid"):
                raise ValueError(
                    f"cnn1d conv_padding must be same/valid, got {conv_padding!r}"
                )
            if conv_padding == "same" and kernel % 2 == 0:
                raise ValueError(
                    "cnn1d kernel_size must be odd when conv_padding='same' so the "
                    "configured sequence length is preserved"
                )
            if pool_size < 1:
                raise ValueError("cnn1d pool_size must be a positive integer")

            # Keras Conv1D defaults to padding='valid'; the papers name Keras
            # layers (Conv1D / MaxPooling1D), so 'valid' is the faithful default
            # for the paper-literal arm.  'same' stays available for the
            # extension arms, where preserving T keeps short folds usable.
            pad = kernel // 2 if conv_padding == "same" else 0

            layers: list[Any] = []
            in_channels = n_features
            timesteps = int(n_timesteps)
            for out_channels in filters:
                layers += [
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding=pad),
                    nn.ReLU(),
                    nn.MaxPool1d(pool_size),
                ]
                in_channels = out_channels
                timesteps = timesteps + 2 * pad - (kernel - 1)   # conv
                if timesteps < 1:
                    raise ValueError(
                        f"cnn1d sequence length {n_timesteps} is too short for "
                        f"{len(filters)} conv layers of kernel {kernel} "
                        f"with conv_padding={conv_padding!r}"
                    )
                timesteps //= pool_size                          # pool
                if timesteps < 1:
                    raise ValueError(
                        f"cnn1d sequence length {n_timesteps} is too short for "
                        f"{len(filters)} pool layers of size {pool_size}"
                    )
            self.features = nn.Sequential(*layers)
            self.dropout = nn.Dropout(float(params["dropout"]))
            self.n_pools = len(filters)
            self.pool_size = pool_size
            self.conv_padding = conv_padding
            self.output_timesteps = timesteps
            self.head = nn.Linear(in_channels * timesteps, 1)

        def forward(self, x, lengths=None, padding_side="post"):
            # (B, T, F) -> (B, F, T) for Conv1d.
            # The input builder has already restored padding to exact zeros.
            # The papers do not report a Conv mask, and approximating post-conv
            # support from lengths is wrong after repeated convolution/pooling
            # for odd lengths.  Preserve the padded tensor verbatim and let the
            # reported Flatten readout see the fixed-length feature map.
            z = self.features(x.transpose(1, 2))
            flattened = z.flatten(start_dim=1)
            return self.head(self.dropout(flattened)).squeeze(-1)

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
    n_timesteps: int | None = None
    padding_side: str = "post"
    history: dict[str, Any] = field(default_factory=dict)

    # -- construction ---------------------------------------------------------
    def build(self, n_features: int, n_timesteps: int) -> "SequenceModel":
        torch = _require_torch()
        self.n_features = int(n_features)
        self.n_timesteps = int(n_timesteps)
        self.module = _build_module(
            self.name, self.n_features, self.n_timesteps, self.params
        )
        self.module.to(torch.device(self.device))
        return self

    # -- training -------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        lengths: np.ndarray | None = None,
        padding_side: str = "post",
        seed: int = 42,
    ) -> "SequenceModel":
        torch = _require_torch()
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"{self.name} expects (N, T, F); got shape {X.shape}")
        y = np.asarray(y, dtype=np.float32)
        if y.shape != (len(X),):
            raise ValueError(
                f"labels must have shape ({len(X)},), got {y.shape}"
            )
        if padding_side not in ("pre", "post"):
            raise ValueError(f"padding_side must be pre/post, got {padding_side!r}")
        self.padding_side = padding_side
        if self.module is None:
            self.build(X.shape[2], X.shape[1])
        if X.shape[2] != self.n_features:
            raise ValueError(
                f"feature count changed after build: expected {self.n_features}, got {X.shape[2]}"
            )
        if X.shape[1] != self.n_timesteps:
            raise ValueError(
                f"sequence length changed after build: expected {self.n_timesteps}, "
                f"got {X.shape[1]}"
            )

        cfg = {**TRAINING_DEFAULTS, **self.training}
        device = torch.device(self.device)
        generator = torch.Generator().manual_seed(int(seed))
        max_epochs = int(cfg["max_epochs"])
        batch_size = int(cfg["batch_size"])
        if max_epochs < 1:
            raise ValueError("max_epochs must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        lengths_arr = (
            np.full(len(X), X.shape[1], dtype=np.int64)
            if lengths is None
            else np.asarray(lengths, dtype=np.int64)
        )
        if lengths_arr.shape != (len(X),):
            raise ValueError(
                f"lengths must have shape ({len(X)},), got {lengths_arr.shape}"
            )
        if (lengths_arr < 1).any() or (lengths_arr > X.shape[1]).any():
            raise ValueError("every sequence length must be in [1, T]")

        early_stopping_enabled = bool(cfg.get("early_stopping", True))
        if early_stopping_enabled:
            # This is a reconstruction assumption, not a reported paper step.
            # It is stratified so the tiny monitor slice retains both classes.
            val_idx, tr_idx = _stratified_monitor_split(
                y, fraction=float(cfg["validation_fraction"]), seed=seed
            )
        else:
            # Experiment A uses all 141 official Training subjects.  The papers
            # report no internal holdout or neural-network early stopping.
            val_idx = np.array([], dtype=int)
            tr_idx = np.arange(len(y), dtype=int)

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

        # Extension arms monitor the reported ranking metric.  BCE on ~28 subjects
        # can rise while ranking improves; experiment A disables this unreported
        # monitor entirely and retains the final fixed epoch.
        metric = str(cfg.get("early_stopping_metric", "auc")).lower()
        if metric not in ("auc", "loss"):
            raise ValueError(f"early_stopping_metric must be auc/loss, got {metric!r}")
        higher_is_better = metric == "auc"

        initial_state = {
            k: v.detach().clone() for k, v in self.module.state_dict().items()
        }
        best_state = {k: v.detach().clone() for k, v in initial_state.items()}
        best_score = -np.inf if higher_is_better else np.inf
        best_tiebreak = np.inf   # monitor loss, used only when the metric ties
        best_epoch = -1
        patience_left = int(cfg["early_stopping_patience"])
        losses: list[float] = []
        monitored: list[float] = []
        optimizer_steps = 0

        for epoch in range(max_epochs):
            self.module.train()
            perm = torch.randperm(len(tr_idx), generator=generator).numpy()
            shuffled = tr_idx[perm]
            epoch_loss, seen = 0.0, 0
            for start in range(0, len(shuffled), batch_size):
                batch = shuffled[start:start + batch_size]
                batch_device = torch.as_tensor(
                    batch, dtype=torch.long, device=device
                )
                batch_cpu = torch.as_tensor(batch, dtype=torch.long)
                optimizer.zero_grad()
                logits = self.module(
                    tensors["X"][batch_device], tensors["len"][batch_cpu],
                    padding_side=self.padding_side,
                )
                loss = criterion(logits, tensors["y"][batch_device])
                loss.backward()
                if cfg.get("grad_clip"):
                    torch.nn.utils.clip_grad_norm_(
                        self.module.parameters(), float(cfg["grad_clip"])
                    )
                optimizer.step()
                optimizer_steps += 1
                epoch_loss += float(loss.item()) * len(batch)
                seen += len(batch)
            losses.append(epoch_loss / max(seen, 1))

            if len(val_idx):
                self.module.eval()
                with torch.no_grad():
                    val_device = torch.as_tensor(
                        val_idx, dtype=torch.long, device=device
                    )
                    val_cpu = torch.as_tensor(val_idx, dtype=torch.long)
                    logits = self.module(
                        tensors["X"][val_device], tensors["len"][val_cpu],
                        padding_side=self.padding_side,
                    )
                    current_loss = float(
                        criterion(logits, tensors["y"][val_device]).item()
                    )
                    current = (
                        _safe_auc(y[val_idx], torch.sigmoid(logits).cpu().numpy())
                        if higher_is_better else current_loss
                    )
            else:
                # With early stopping disabled, the full official Training split is
                # fitted for exactly max_epochs and the final epoch is retained.
                current_loss = losses[-1]
                current = -losses[-1] if higher_is_better else losses[-1]
            monitored.append(current)

            if not early_stopping_enabled:
                improved = True
            elif higher_is_better:
                # AUC can saturate on a small monitor split after the first trained
                # epoch. Break ties on loss so later trained states remain eligible.
                improved = current > best_score + 1e-5 or (
                    abs(current - best_score) <= 1e-5 and current_loss < best_tiebreak - 1e-5
                )
            elif not higher_is_better:
                improved = current < best_score - 1e-5

            if improved:
                best_score = current
                best_tiebreak = current_loss
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in self.module.state_dict().items()}
                patience_left = int(cfg["early_stopping_patience"])
            elif early_stopping_enabled:
                patience_left -= 1
                if patience_left <= 0:
                    break

        self.module.load_state_dict(best_state)
        parameter_delta_sq = 0.0
        for name, value in best_state.items():
            delta = value.detach().float().cpu() - initial_state[name].detach().float().cpu()
            parameter_delta_sq += float((delta * delta).sum().item())
        parameter_delta_l2 = float(np.sqrt(parameter_delta_sq))
        finite_training = bool(losses and np.isfinite(np.asarray(losses)).all())
        # epoch 0 is the first *trained* epoch, not the random initialisation.
        # Degeneracy therefore means no optimiser update / no parameter change /
        # non-finite optimisation, not merely best_epoch == 0.
        degenerate_training = bool(
            optimizer_steps == 0
            or not np.isfinite(parameter_delta_l2)
            or parameter_delta_l2 <= 1e-12
            or not finite_training
        )
        self.history = {
            "epochs_run": len(losses),
            "train_loss": losses,
            "monitor_metric": metric,
            "monitor_score": monitored,
            "best_monitor_score": float(best_score),
            "best_epoch": int(best_epoch),
            "early_stopping_enabled": early_stopping_enabled,
            "early_stopped": bool(
                early_stopping_enabled and len(losses) < max_epochs
            ),
            "optimizer_fit_size": int(len(tr_idx)),
            "optimizer_fit_positives": int(y[tr_idx].sum()),
            "optimizer_steps": int(optimizer_steps),
            "parameter_delta_l2": parameter_delta_l2,
            "monitor_split_size": int(len(val_idx)),
            "monitor_split_positives": int(y[val_idx].sum()) if len(val_idx) else 0,
            "degenerate_training": degenerate_training,
        }
        return self

    # -- inference ------------------------------------------------------------
    def predict_proba(
        self,
        X: np.ndarray,
        *,
        lengths: np.ndarray | None = None,
        padding_side: str | None = None,
    ) -> np.ndarray:
        torch = _require_torch()
        if self.module is None:
            raise RuntimeError(f"{self.name} has not been built/fitted")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"{self.name} expects (N, T, F); got shape {X.shape}")
        if X.shape[2] != self.n_features:
            raise ValueError(
                f"feature count changed after build: expected {self.n_features}, "
                f"got {X.shape[2]}"
            )
        if X.shape[1] != self.n_timesteps:
            raise ValueError(
                f"sequence length changed after build: expected {self.n_timesteps}, "
                f"got {X.shape[1]}"
            )
        device = torch.device(self.device)
        side = self.padding_side if padding_side is None else padding_side
        if side not in ("pre", "post"):
            raise ValueError(f"padding_side must be pre/post, got {side!r}")
        lengths_arr = (
            np.full(len(X), X.shape[1], dtype=np.int64)
            if lengths is None
            else np.asarray(lengths, dtype=np.int64)
        )
        if lengths_arr.shape != (len(X),):
            raise ValueError(
                f"lengths must have shape ({len(X)},), got {lengths_arr.shape}"
            )
        if (lengths_arr < 1).any() or (lengths_arr > X.shape[1]).any():
            raise ValueError("every sequence length must be in [1, T]")
        self.module.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(X), 256):
                chunk = slice(start, start + 256)
                logits = self.module(
                    torch.from_numpy(X[chunk]).to(device),
                    torch.from_numpy(lengths_arr[chunk]),
                    padding_side=side,
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
                "n_timesteps": self.n_timesteps,
                "padding_side": self.padding_side,
                "architecture_version": 2,
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
        if int(blob.get("architecture_version", 1)) != 2:
            raise ValueError(
                "legacy sequence checkpoint uses the pre-fix pooling architecture; "
                "rerun the experiment to create a version-2 checkpoint"
            )
        model = cls(
            name=blob["name"], params=blob["params"], training=blob["training"],
            device=device, history=blob.get("history", {}),
            padding_side=blob.get("padding_side", "post"),
        )
        model.build(int(blob["n_features"]), int(blob["n_timesteps"]))
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
            "n_timesteps": self.n_timesteps,
            "padding_side": self.padding_side,
            "architecture_version": 2,
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
                    "early_stopping_enabled", "early_stopped", "optimizer_fit_size",
                    "optimizer_fit_positives", "optimizer_steps", "parameter_delta_l2",
                    "monitor_split_size",
                    "monitor_split_positives", "degenerate_training",
                )
                if key in self.history
            }
            if self.history.get("degenerate_training"):
                summary["training_diagnostics"]["warning"] = (
                    "No finite optimiser-driven parameter change was retained. This "
                    "model is effectively untrained and its metrics are not a "
                    "performance result."
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
