"""Base learners with a uniform, fold-safe ``fit(train_idx)/predict_proba(idx)``.

Every learner owns the whole :class:`SubjectData` and selects rows by global
subject index.  All preprocessing (imputation, winsorization, scaling, feature
selection, sequence normalization) is fit on the given ``train_idx`` only, so
the same instance can be reused inside nested cross-validation without leaking
validation information.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

from .datalib import SubjectData, make_windows


def _safe_f_classif(x: np.ndarray, y: np.ndarray):
    """ANOVA F-scores with constant columns scored 0 (never selected)."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores, pvalues = f_classif(x, y)
    return np.nan_to_num(scores, nan=0.0), np.nan_to_num(pvalues, nan=1.0)

try:  # torch is Colab-supplied; the tabular path must still run without it.
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only on CPU-without-torch hosts
    TORCH_AVAILABLE = False


def cuda_available() -> bool:
    """True only when torch is importable and a CUDA device is present."""

    if not TORCH_AVAILABLE:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover
        return False


class _FoldTabularPrep:
    """Median impute -> 1/99 winsorize -> standardize -> SelectKBest, train-fit."""

    def __init__(self, max_features: int) -> None:
        self.max_features = max_features

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_FoldTabularPrep":
        self.median_ = np.nanmedian(x, axis=0)
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)
        filled = self._impute(x)
        self.low_ = np.percentile(filled, 1, axis=0)
        self.high_ = np.percentile(filled, 99, axis=0)
        clipped = np.clip(filled, self.low_, self.high_)
        self.mean_ = clipped.mean(axis=0)
        self.std_ = clipped.std(axis=0)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        scaled = (clipped - self.mean_) / self.std_
        k = min(self.max_features, scaled.shape[1])
        self.selector_ = SelectKBest(_safe_f_classif, k=k).fit(scaled, y)
        return self

    def _impute(self, x: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(x), x, self.median_)
        return filled

    def transform(self, x: np.ndarray) -> np.ndarray:
        clipped = np.clip(self._impute(x), self.low_, self.high_)
        scaled = (clipped - self.mean_) / self.std_
        return self.selector_.transform(scaled)


def _make_estimator(kind: str):
    if kind == "gbt":
        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=12,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=0,
        )
    if kind == "logreg":
        return LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=4,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=0,
        )
    raise ValueError(f"Unknown tabular estimator: {kind}")


class TabularLearner:
    """A single fold-safe tabular estimator over per-subject summaries."""

    def __init__(self, data: SubjectData, kind: str, max_features: int = 30) -> None:
        self.data = data
        self.kind = kind
        self.max_features = max_features

    def fit(self, train_idx: np.ndarray) -> "TabularLearner":
        x = self.data.tabular[train_idx]
        y = self.data.y[train_idx]
        self.prep_ = _FoldTabularPrep(self.max_features).fit(x, y)
        self.model_ = _make_estimator(self.kind).fit(self.prep_.transform(x), y)
        return self

    def predict_proba(self, idx: np.ndarray) -> np.ndarray:
        return self.predict_proba_matrix(self.data.tabular[idx])

    def predict_proba_matrix(self, tabular: np.ndarray) -> np.ndarray:
        """Score an external summary matrix (e.g. the validation split)."""

        x = self.prep_.transform(np.asarray(tabular, dtype=np.float64))
        return self.model_.predict_proba(x)[:, 1]


if TORCH_AVAILABLE:

    class _ConvBiLSTM(nn.Module):
        """Small Conv1D -> BiLSTM -> Dense, matching the report's winning recipe."""

        def __init__(self, n_channels: int) -> None:
            super().__init__()
            self.conv = nn.Conv1d(n_channels, 32, kernel_size=3, padding=1)
            self.act = nn.ReLU()
            self.dropout1 = nn.Dropout(0.5)
            self.lstm = nn.LSTM(
                32, 8, batch_first=True, bidirectional=True
            )
            self.dropout2 = nn.Dropout(0.5)
            self.head = nn.Sequential(
                nn.Linear(16, 8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 1)
            )

        def forward(self, x):  # x: (batch, days, channels)
            h = self.conv(x.transpose(1, 2))           # (batch, 32, days)
            h = self.dropout1(self.act(h)).transpose(1, 2)  # (batch, days, 32)
            out, _ = self.lstm(h)                      # (batch, days, 16)
            pooled = out.mean(dim=1)                   # temporal average pool
            return self.head(self.dropout2(pooled)).squeeze(-1)


class SequenceLearner:
    """Conv1D+BiLSTM over 7-day windows; subject prob = mean of its windows.

    Fold-local channel median/IQR normalization is fit on the training windows.
    Class imbalance is handled with a positive-class weight in the loss.  If
    torch is unavailable the learner reports it is disabled and the pipeline
    proceeds with the tabular ensemble only.
    """

    available = TORCH_AVAILABLE

    def __init__(
        self,
        data: SubjectData,
        *,
        epochs: int = 60,
        batch_size: int = 64,
        lr: float = 3e-3,
        weight_decay: float = 5e-3,
        seed: int = 0,
    ) -> None:
        self.data = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self._windows, self._subject_of_window = make_windows(data.sequences)

    def _normalize_fit(self, windows: np.ndarray) -> None:
        flat = windows.reshape(-1, windows.shape[-1]).astype(np.float64)
        median = np.nanmedian(flat, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        filled = np.where(np.isfinite(flat), flat, median)
        q25, q75 = np.percentile(filled, [25, 75], axis=0)
        iqr = np.where((q75 - q25) < 1e-8, 1.0, q75 - q25)
        self._median = median
        self._iqr = iqr

    def _normalize_apply(self, windows: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(windows), windows, self._median)
        scaled = (filled - self._median) / self._iqr
        return np.clip(scaled, -5.0, 5.0).astype(np.float32)

    def _windows_for(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        wanted = np.zeros(self.data.n_subjects, dtype=bool)
        wanted[idx] = True
        mask = wanted[self._subject_of_window]
        return self._windows[mask], self._subject_of_window[mask]

    def fit(self, train_idx: np.ndarray) -> "SequenceLearner":
        if not TORCH_AVAILABLE:
            return self
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        windows, subject_of_window = self._windows_for(train_idx)
        self._normalize_fit(windows)
        x = torch.tensor(self._normalize_apply(windows), device=device)
        y = torch.tensor(
            self.data.y[subject_of_window].astype(np.float32), device=device
        )
        n_pos = float((y == 1).sum().item())
        n_neg = float((y == 0).sum().item())
        pos_weight = torch.tensor(
            [n_neg / n_pos if n_pos > 0 else 1.0], device=device
        )
        model = _ConvBiLSTM(x.shape[-1]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        model.train()
        n = x.shape[0]
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        for _ in range(self.epochs):
            order = torch.randperm(n, generator=generator).to(device)
            for start in range(0, n, self.batch_size):
                batch = order[start : start + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(model(x[batch]), y[batch])
                loss.backward()
                optimizer.step()
        self._model = model
        self._device = device
        return self

    def _score_windows(self, windows: np.ndarray, group: np.ndarray, n_out: int) -> np.ndarray:
        x = torch.tensor(self._normalize_apply(windows), device=self._device)
        self._model.eval()
        with torch.no_grad():
            window_probs = torch.sigmoid(self._model(x)).cpu().numpy()
        sums = np.zeros(n_out)
        counts = np.zeros(n_out)
        for prob, row in zip(window_probs, group):
            sums[row] += prob
            counts[row] += 1
        out = np.full(n_out, np.nan)
        nonzero = counts > 0
        out[nonzero] = sums[nonzero] / counts[nonzero]
        return out

    def predict_proba(self, idx: np.ndarray) -> np.ndarray:
        if not TORCH_AVAILABLE:
            return np.full(len(idx), np.nan)
        windows, subject_of_window = self._windows_for(idx)
        position = {int(subject): row for row, subject in enumerate(idx)}
        group = np.asarray([position[int(s)] for s in subject_of_window])
        return self._score_windows(windows, group, len(idx))

    def predict_proba_sequences(self, sequences: Sequence[np.ndarray]) -> np.ndarray:
        """Score an external list of subject sequences (the validation split)."""

        if not TORCH_AVAILABLE:
            return np.full(len(sequences), np.nan)
        windows, subject_of_window = make_windows(list(sequences))
        return self._score_windows(windows, subject_of_window, len(sequences))


__all__ = ["SequenceLearner", "TabularLearner", "TORCH_AVAILABLE", "cuda_available"]
