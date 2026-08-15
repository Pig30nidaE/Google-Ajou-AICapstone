"""Downstream heads: fold-local (impute -> scale -> PCA -> logistic) probes.

Mirrors SensorFM M.3.4: embeddings reduced to <=50 principal components, then
a logistic head; identically applied to the engineered-feature baseline
(M.3.6).  Every stage fits inside the current training fold only.  The blend
candidate rank-averages the two families' scores through the training fold's
ECDF (same construction as Binary_Google_TabFM_Nested's blend).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    BLEND_EMB_CONFIG,
    BLEND_FE_CONFIG,
    LR_PARAMS,
    ProbeConfig,
)


class LinearProbe:
    """One (impute, scale, PCA-K, LR-C) pipeline with a probability score."""

    def __init__(self, config: ProbeConfig, seed: int) -> None:
        self.config = config
        self.seed = int(seed)
        self.pipeline: Pipeline | None = None
        self.effective_pca_k_: int | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LinearProbe":
        n_samples, n_features = X.shape
        # PCA rank cannot exceed min(n_samples, n_features); cap fold-locally
        # and record it (113-subject folds support K=50, inner folds are close).
        k = int(min(self.config.pca_k, n_features, max(1, n_samples - 1)))
        self.effective_pca_k_ = k
        self.pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=k, svd_solver="full", random_state=self.seed)),
                ("lr", LogisticRegression(C=self.config.lr_c, random_state=self.seed,
                                          **LR_PARAMS)),
            ]
        )
        self.pipeline.fit(X.to_numpy(dtype=np.float64), np.asarray(y, dtype=int))
        return self

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("predict_score before fit")
        probabilities = self.pipeline.predict_proba(X.to_numpy(dtype=np.float64))
        classes = list(self.pipeline.named_steps["lr"].classes_)
        return probabilities[:, classes.index(1)]


def _ecdf_rank(reference: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Map ``scores`` to (0, 1) by the ECDF of ``reference`` (training scores)."""

    reference = np.sort(np.asarray(reference, dtype=np.float64))
    if reference.size == 0:
        return np.full_like(np.asarray(scores, dtype=np.float64), 0.5)
    positions = np.searchsorted(reference, np.asarray(scores, dtype=np.float64),
                                side="right")
    return (positions + 0.5) / (reference.size + 1.0)


class BlendProbe:
    """Rank-average of the FE and embedding probes with fixed member configs."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.fe_probe = LinearProbe(BLEND_FE_CONFIG, seed)
        self.emb_probe = LinearProbe(BLEND_EMB_CONFIG, seed + 1)
        self._train_scores: dict[str, np.ndarray] | None = None

    def fit(self, X_fe: pd.DataFrame, X_emb: pd.DataFrame, y: np.ndarray) -> "BlendProbe":
        self.fe_probe.fit(X_fe, y)
        self.emb_probe.fit(X_emb, y)
        self._train_scores = {
            "fe": self.fe_probe.predict_score(X_fe),
            "emb": self.emb_probe.predict_score(X_emb),
        }
        return self

    def predict_score(self, X_fe: pd.DataFrame, X_emb: pd.DataFrame) -> np.ndarray:
        if self._train_scores is None:
            raise RuntimeError("predict_score before fit")
        fe_rank = _ecdf_rank(self._train_scores["fe"], self.fe_probe.predict_score(X_fe))
        emb_rank = _ecdf_rank(self._train_scores["emb"], self.emb_probe.predict_score(X_emb))
        return 0.5 * (fe_rank + emb_rank)


__all__ = ["BlendProbe", "LinearProbe", "_ecdf_rank"]
