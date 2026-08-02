"""Feature scaling, fitted on training sequences only.

A scaler is the easiest place in a sequence pipeline to leak, because the natural
thing to write -- fit on the stacked ``(n, L, F)`` array of everything -- silently
uses test statistics.  ``SequenceScaler`` therefore records the fingerprint of the
data it was fitted on, and ``audit/leakage.py`` refuses to continue if that
fingerprint does not match the training set it was supposed to see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..sequences.builder import SequenceSet


def fingerprint(sequences: SequenceSet) -> str:
    """A cheap, stable id for "which sequences were these"."""
    ids = sequences.provenance.get("sequence_id")
    if ids is None or not len(ids):
        return "empty"
    import hashlib

    digest = hashlib.sha256("\n".join(sorted(map(str, ids))).encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass
class SequenceScaler:
    """Per-feature standardisation across the (sequence, timestep) axes.

    Statistics are computed over all timesteps of the training sequences, so a
    feature has one mean and one scale regardless of its position in the window.
    """

    method: str = "standard"
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    fitted_on: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def fit(self, train: SequenceSet) -> "SequenceScaler":
        if self.method == "none":
            self.fitted_on = fingerprint(train)
            return self
        if not len(train):
            raise ValueError("cannot fit a scaler on zero training sequences")

        flat = train.X.reshape(-1, train.X.shape[-1]).astype(np.float64)
        if self.method == "standard":
            self.mean_ = flat.mean(axis=0)
            scale = flat.std(axis=0)
        elif self.method == "minmax":
            self.mean_ = flat.min(axis=0)
            scale = flat.max(axis=0) - flat.min(axis=0)
        else:
            raise ValueError("method must be 'standard', 'minmax' or 'none'")

        # A constant feature within this training fold (the rarer one-hot bins do
        # this) would otherwise divide by zero.
        scale = np.where(scale <= 1e-12, 1.0, scale)
        self.scale_ = scale
        self.fitted_on = fingerprint(train)
        self.meta = {
            "n_train_sequences": len(train),
            "n_features": int(train.X.shape[-1]),
            "n_degenerate_features": int((scale == 1.0).sum()),
            "split_name": train.split_name,
        }
        return self

    def transform(self, sequences: SequenceSet) -> SequenceSet:
        if self.method == "none":
            return sequences
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("scaler must be fitted before transform")
        if not len(sequences):
            return sequences
        scaled = (sequences.X.astype(np.float64) - self.mean_) / self.scale_
        return SequenceSet(
            X=scaled.astype(np.float32),
            y=sequences.y,
            provenance=sequences.provenance,
            feature_columns=sequences.feature_columns,
            sequence_length=sequences.sequence_length,
            split_name=sequences.split_name,
        )

    def fit_transform_pair(
        self, train: SequenceSet, *others: SequenceSet
    ) -> tuple[SequenceSet, ...]:
        """Fit on *train* and transform train plus every other split.

        This is the only sanctioned way to scale in this package: the fit sees one
        argument, and it is the training one.
        """
        self.fit(train)
        return (self.transform(train), *(self.transform(other) for other in others))

    def describe(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "fitted_on_fingerprint": self.fitted_on,
            **self.meta,
        }


def flatten(sequences: SequenceSet) -> np.ndarray:
    """``(n, L, F) -> (n, L*F)`` for the non-temporal baselines."""
    return sequences.X.reshape(len(sequences), -1)


def mean_over_time(sequences: SequenceSet) -> np.ndarray:
    """``(n, L, F) -> (n, F)`` by averaging each feature across the window."""
    return sequences.X.mean(axis=1)


def last_day(sequences: SequenceSet) -> np.ndarray:
    """``(n, L, F) -> (n, F)`` keeping only the final day of the window."""
    return sequences.X[:, -1, :]


def summary_statistics(sequences: SequenceSet) -> np.ndarray:
    """``(n, L, F) -> (n, 4F)``: mean, std, min and max of each feature."""
    X = sequences.X
    return np.concatenate(
        [X.mean(axis=1), X.std(axis=1), X.min(axis=1), X.max(axis=1)], axis=1
    )


BASELINE_REPRESENTATIONS = {
    "flatten": flatten,
    "mean": mean_over_time,
    "last_day": last_day,
    "summary": summary_statistics,
}


def represent(sequences: SequenceSet, kind: str) -> np.ndarray:
    """Reduce sequences to a 2-D matrix for a non-temporal model.

    The paper never states which of these it used for SVM / LR / RF / XGBoost, so
    the choice is config-driven and recorded as assumption A-09 rather than
    asserted to be the paper's.
    """
    if kind not in BASELINE_REPRESENTATIONS:
        raise ValueError(
            f"representation must be one of {sorted(BASELINE_REPRESENTATIONS)}, got {kind!r}"
        )
    return BASELINE_REPRESENTATIONS[kind](sequences)


def representation_feature_names(
    feature_columns: tuple[str, ...], sequence_length: int, kind: str
) -> list[str]:
    if kind == "flatten":
        return [f"{name}__t{t}" for t in range(sequence_length) for name in feature_columns]
    if kind in ("mean", "last_day"):
        return list(feature_columns)
    if kind == "summary":
        return [
            f"{name}__{stat}"
            for stat in ("mean", "std", "min", "max")
            for name in feature_columns
        ]
    raise ValueError(f"unknown representation {kind!r}")
