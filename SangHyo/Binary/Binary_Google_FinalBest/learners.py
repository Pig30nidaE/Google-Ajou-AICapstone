"""Google-model learners for the final best experiment.

Primary: Google Yggdrasil Decision Forests (Gradient Boosted Trees + Random
Forest) — the strongest, most stable choice for this small (141-subject) tabular
problem.  Optional: Google TabNet, included only when a GPU + pytorch-tabnet are
available; the quality gate in the engine drops it automatically if it does not
beat chance (as it did on this data before), so it can never hurt performance.

All preprocessing is fit on the given ``train_idx`` only.  YDF ingests NaN
natively; TabNet gets fold-local median-impute + standardize.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import SubjectData

CLASS_NAMES = ("CN", "MCI_DEM")

# Default YDF hyperparameters (used when no tuned params are supplied).
DEFAULT_GBT_PARAMS = dict(
    num_trees=400, max_depth=4, min_examples=8, shrinkage=0.04, subsample=0.85,
    num_candidate_attributes_ratio=0.9, l2_regularization=1.0,
)
DEFAULT_RF_PARAMS = dict(
    num_trees=600, max_depth=6, min_examples=5, num_candidate_attributes_ratio=0.6,
)

try:
    import ydf  # noqa: F401
    YDF_AVAILABLE = True
except Exception:  # pragma: no cover
    YDF_AVAILABLE = False


def tabnet_available() -> bool:
    try:
        import torch  # noqa: F401
        from pytorch_tabnet.tab_model import TabNetClassifier  # noqa: F401
        import torch as _t
        return bool(_t.cuda.is_available())
    except Exception:
        return False


def _balanced_class_weights(y: np.ndarray) -> dict:
    n = len(y)
    counts = {c: int(np.sum(y == c)) for c in (0, 1)}
    return {CLASS_NAMES[c]: (n / (2.0 * counts[c]) if counts[c] else 1.0) for c in (0, 1)}


class YDFLearner:
    """Google YDF learner (kind='gbt' or 'rf'); sklearn fallback if ydf missing."""

    def __init__(self, data: SubjectData, kind: str, *, seed: int = 0,
                 params: dict | None = None) -> None:
        self.data = data
        self.kind = kind
        self.seed = seed
        self.params = dict(params) if params else {}
        self.engine_ = "ydf" if YDF_AVAILABLE else "sklearn_fallback"

    def _merged_params(self) -> dict:
        base = dict(DEFAULT_GBT_PARAMS if self.kind == "gbt" else DEFAULT_RF_PARAMS)
        base.update(self.params)
        return base

    def fit(self, train_idx: np.ndarray) -> "YDFLearner":
        y = self.data.y[train_idx]
        p = self._merged_params()
        if YDF_AVAILABLE:
            import ydf
            frame = pd.DataFrame(self.data.X[train_idx], columns=list(self.data.feature_names))
            frame["label"] = [CLASS_NAMES[int(v)] for v in y]
            common = dict(label="label", label_classes=list(CLASS_NAMES),
                          class_weights=_balanced_class_weights(y), random_seed=self.seed,
                          num_threads=int(os.cpu_count() or 4))
            if self.kind == "gbt":
                learner = ydf.GradientBoostedTreesLearner(
                    loss="BINOMIAL_LOG_LIKELIHOOD", validation_ratio=0.0,
                    num_trees=int(p["num_trees"]), max_depth=int(p["max_depth"]),
                    min_examples=int(p["min_examples"]), shrinkage=float(p["shrinkage"]),
                    subsample=float(p["subsample"]),
                    num_candidate_attributes_ratio=float(p["num_candidate_attributes_ratio"]),
                    l2_regularization=float(p["l2_regularization"]), **common)
            elif self.kind == "rf":
                learner = ydf.RandomForestLearner(
                    num_trees=int(p["num_trees"]), max_depth=int(p["max_depth"]),
                    min_examples=int(p["min_examples"]),
                    num_candidate_attributes_ratio=float(p["num_candidate_attributes_ratio"]),
                    **common)
            else:
                raise ValueError(self.kind)
            self.model_ = learner.train(frame)
            classes = tuple(str(c) for c in self.model_.label_classes())
            self._pos = classes.index("MCI_DEM")
        else:
            # NaN-tolerant sklearn fallback (local wiring only) mapped from YDF HP.
            self.model_ = HistGradientBoostingClassifier(
                max_iter=int(p["num_trees"]), learning_rate=float(p.get("shrinkage", 0.05)),
                max_depth=int(p["max_depth"]), min_samples_leaf=int(p["min_examples"]),
                l2_regularization=float(p.get("l2_regularization", 1.0)),
                class_weight="balanced", random_state=self.seed
            ).fit(self.data.X[train_idx], y)
        return self

    def predict_proba(self, idx: np.ndarray) -> np.ndarray:
        return self.predict_proba_matrix(self.data.X[idx])

    def predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if YDF_AVAILABLE:
            raw = np.asarray(self.model_.predict(pd.DataFrame(X, columns=list(self.data.feature_names))),
                             dtype=np.float64)
            if raw.ndim == 1:
                return raw if self._pos == 1 else 1.0 - raw
            return raw[:, self._pos]
        return self.model_.predict_proba(X)[:, 1]

    def save(self, path) -> None:
        """Persist the fitted model so it can be reloaded without retraining."""

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {"type": "ydf_learner", "kind": self.kind, "engine": self.engine_,
                "feature_names": list(self.data.feature_names)}
        if YDF_AVAILABLE:
            # YDF's save() does internal temp-file work that can fail on the
            # mounted Google Drive (FUSE) filesystem.  Save to a local temp dir
            # first, then copy the finished model tree to the destination.
            import shutil
            import tempfile

            dest = path / "ydf_model"
            with tempfile.TemporaryDirectory() as tmp:
                local = Path(tmp) / "ydf_model"
                self.model_.save(str(local))
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(local, dest)
            meta["pos_index"] = int(self._pos)
        else:
            joblib.dump(self.model_, path / "sklearn.joblib")
        (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


class TabNetLearner:
    """Optional Google TabNet learner (GPU); fold-local impute + standardize."""

    available = staticmethod(tabnet_available)

    def __init__(self, data: SubjectData, *, seed: int = 0, epochs: int = 120) -> None:
        self.data = data
        self.seed = seed
        self.epochs = epochs

    def _prep_fit(self, X):
        self._median = np.nanmedian(X, axis=0)
        self._median = np.where(np.isfinite(self._median), self._median, 0.0)
        filled = np.where(np.isfinite(X), X, self._median)
        self._mu = filled.mean(0)
        self._sd = np.where(filled.std(0) < 1e-8, 1.0, filled.std(0))

    def _prep(self, X):
        filled = np.where(np.isfinite(X), X, self._median)
        return ((filled - self._mu) / self._sd).astype(np.float32)

    def fit(self, train_idx: np.ndarray) -> "TabNetLearner":
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier
        np.random.seed(self.seed); torch.manual_seed(self.seed)
        X = self.data.X[train_idx]; y = self.data.y[train_idx].astype(np.int64)
        self._prep_fit(X)
        counts = np.bincount(y, minlength=2)
        w = {0: len(y) / (2 * max(1, counts[0])), 1: len(y) / (2 * max(1, counts[1]))}
        self.model_ = TabNetClassifier(
            n_d=8, n_a=8, n_steps=3, gamma=1.3, lambda_sparse=1e-3,
            optimizer_fn=torch.optim.AdamW, optimizer_params={"lr": 2e-2, "weight_decay": 1e-3},
            scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
            scheduler_params={"T_max": self.epochs, "eta_min": 2e-4},
            seed=self.seed, verbose=0, device_name="cuda")
        self.model_.fit(self._prep(X), y, max_epochs=self.epochs, patience=self.epochs,
                        batch_size=min(64, len(y)), virtual_batch_size=min(32, len(y)),
                        weights=w)
        return self

    def predict_proba(self, idx: np.ndarray) -> np.ndarray:
        return self.predict_proba_matrix(self.data.X[idx])

    def predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(self._prep(np.asarray(X, float)))[:, 1]

    def save(self, path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model_.save_model(str(path / "tabnet"))  # writes tabnet.zip
        joblib.dump({"median": self._median, "mu": self._mu, "sd": self._sd},
                    path / "prep.joblib")
        (path / "meta.json").write_text(
            json.dumps({"type": "tabnet_learner", "feature_names": list(self.data.feature_names)},
                       ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["CLASS_NAMES", "TabNetLearner", "YDFLearner", "YDF_AVAILABLE", "tabnet_available"]
