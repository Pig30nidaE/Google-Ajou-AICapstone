"""Google Yggdrasil Decision Forests (YDF) learners.

This experiment uses Google's YDF — Gradient Boosted Trees and Random Forest —
as the base models.  YDF is chosen over TabNet/Transformer for this problem
because the dataset is small (141 subjects) and tabular, where decision forests
are the strongest and most stable choice; the prior TabNet experiment on this
same data collapsed to balanced accuracy 0.43.  YDF handles NaN natively and
needs no scaling, so the fold-local step is just class-balanced training.

If ``ydf`` is not importable (e.g. a quick local CPU check without the package),
each learner transparently falls back to scikit-learn HistGradientBoosting so
the pipeline still runs; on Colab with ``ydf`` installed the real Google model
is used.  ``nested_cv_report.json`` records which engine ran.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import SubjectData

CLASS_NAMES = ("CN", "MCI_DEM")

try:
    import ydf  # noqa: F401
    YDF_AVAILABLE = True
except Exception:  # pragma: no cover
    YDF_AVAILABLE = False


def _balanced_class_weights(y: np.ndarray) -> dict:
    n = len(y)
    counts = {c: int(np.sum(y == c)) for c in (0, 1)}
    return {CLASS_NAMES[c]: (n / (2.0 * counts[c]) if counts[c] else 1.0) for c in (0, 1)}


class YDFLearner:
    """A single YDF learner (kind='gbt' or 'rf'); sklearn fallback if no ydf."""

    def __init__(self, data: SubjectData, kind: str, *, seed: int = 0) -> None:
        self.data = data
        self.kind = kind
        self.seed = seed
        self.engine_ = "ydf" if YDF_AVAILABLE else "sklearn_fallback"

    def _frame(self, idx: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(self.data.X[idx], columns=list(self.data.feature_names))

    def fit(self, train_idx: np.ndarray) -> "YDFLearner":
        y = self.data.y[train_idx]
        if YDF_AVAILABLE:
            import ydf
            frame = self._frame(train_idx).copy()
            frame["label"] = [CLASS_NAMES[int(v)] for v in y]
            weights = _balanced_class_weights(y)
            common = dict(
                label="label", label_classes=list(CLASS_NAMES),
                class_weights=weights, random_seed=self.seed,
            )
            if self.kind == "gbt":
                learner = ydf.GradientBoostedTreesLearner(
                    loss="BINOMIAL_LOG_LIKELIHOOD", num_trees=300, max_depth=4,
                    min_examples=8, shrinkage=0.05, subsample=0.8,
                    num_candidate_attributes_ratio=0.8, l2_regularization=1.0,
                    validation_ratio=0.0, **common,
                )
            elif self.kind == "rf":
                learner = ydf.RandomForestLearner(
                    num_trees=500, max_depth=6, min_examples=5,
                    num_candidate_attributes_ratio=0.6, **common,
                )
            else:
                raise ValueError(self.kind)
            self.model_ = learner.train(frame)
            classes = tuple(str(c) for c in self.model_.label_classes())
            self._pos = classes.index("MCI_DEM")
            self._binary_1d = len(classes) == 2
        else:
            # NaN-tolerant sklearn fallback (HistGradientBoosting handles NaN).
            depth = 4 if self.kind == "gbt" else 6
            self.model_ = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_depth=depth,
                min_samples_leaf=8, l2_regularization=1.0,
                class_weight="balanced", random_state=self.seed,
            ).fit(self.data.X[train_idx], y)
        return self

    def predict_proba(self, idx: np.ndarray) -> np.ndarray:
        return self.predict_proba_matrix(self.data.X[idx])

    def predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if YDF_AVAILABLE:
            frame = pd.DataFrame(X, columns=list(self.data.feature_names))
            raw = np.asarray(self.model_.predict(frame), dtype=np.float64)
            if raw.ndim == 1:
                # YDF binary returns P(positive class); positive is index 1.
                return raw if self._pos == 1 else 1.0 - raw
            return raw[:, self._pos]
        return self.model_.predict_proba(X)[:, 1]


__all__ = ["YDFLearner", "YDF_AVAILABLE", "CLASS_NAMES"]
