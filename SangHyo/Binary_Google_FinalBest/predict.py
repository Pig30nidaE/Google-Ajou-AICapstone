"""Load a saved Binary_Google_FinalBest deployment and predict without retraining.

The training run writes a ``deployment/`` bundle (fitted YDF models + Platt
calibrator + thresholds + feature list).  This module reloads that bundle and
scores any split, reproducing the reported validation predictions exactly
(YDF is deterministic under the saved seed, and the same fitted models used for
the validation freeze are the ones saved).

Usage (reproduce validation, optionally scoring accuracy)::

    python -m SangHyo.Binary_Google_FinalBest.predict \
        --deployment-dir /content/drive/MyDrive/Binary_Google_FinalBest_result/<RUN>/deployment \
        --data-root /content/drive/Shareddrives/GoogleAI_contest/Data \
        --split val --evaluate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .engine import binary_metrics
from .features import load_split, load_validation_labels_checked


class _LoadedYDF:
    def __init__(self, engine, model, feature_names, pos_index=None):
        self.engine = engine
        self.model = model
        self.feature_names = list(feature_names)
        self.pos_index = pos_index

    def predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.engine == "ydf":
            raw = np.asarray(self.model.predict(pd.DataFrame(X, columns=self.feature_names)),
                             dtype=np.float64)
            if raw.ndim == 1:
                return raw if self.pos_index == 1 else 1.0 - raw
            return raw[:, self.pos_index]
        return self.model.predict_proba(X)[:, 1]


class _LoadedTabNet:
    def __init__(self, model, prep, feature_names):
        self.model = model
        self.prep = prep
        self.feature_names = list(feature_names)

    def predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        filled = np.where(np.isfinite(X), X, self.prep["median"])
        scaled = ((filled - self.prep["mu"]) / self.prep["sd"]).astype(np.float32)
        return self.model.predict_proba(scaled)[:, 1]


def _load_learner(model_dir: Path):
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    if meta["type"] == "ydf_learner":
        if meta["engine"] == "ydf":
            import ydf
            model = ydf.load_model(str(model_dir / "ydf_model"))
            return _LoadedYDF("ydf", model, meta["feature_names"], meta.get("pos_index"))
        return _LoadedYDF("sklearn_fallback", joblib.load(model_dir / "sklearn.joblib"),
                          meta["feature_names"])
    if meta["type"] == "tabnet_learner":
        from pytorch_tabnet.tab_model import TabNetClassifier
        model = TabNetClassifier()
        model.load_model(str(model_dir / "tabnet.zip"))
        return _LoadedTabNet(model, joblib.load(model_dir / "prep.joblib"), meta["feature_names"])
    raise ValueError(f"Unknown model type: {meta['type']}")


class Deployment:
    """A reloaded final model: ensemble + calibrator + recommended threshold."""

    def __init__(self, deployment_dir: str | Path) -> None:
        dep = Path(deployment_dir)
        self.config = json.loads((dep / "deployment.json").read_text(encoding="utf-8"))
        self.feature_names = self.config["feature_names"]
        self.final_weights = self.config["final_weights"]
        self.thresholds = self.config["thresholds_from_training_oof"]
        self.recommended = self.config["recommended_threshold"]
        self.calibrator = joblib.load(dep / "calibrator.joblib")
        self.learners = {name: _load_learner(dep / f"model_{name}") for name in self.final_weights}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Raw ensemble impaired-probability (the space thresholds live in)."""

        total = sum(self.final_weights.values())
        combined = np.zeros(len(X))
        for name, weight in self.final_weights.items():
            combined += weight * self.learners[name].predict_proba_matrix(X)
        return combined / total

    def predict_proba_calibrated(self, X: np.ndarray) -> np.ndarray:
        return self.calibrator.transform(self.predict_proba(X))

    def predict(self, X: np.ndarray, threshold: str | float | None = None) -> np.ndarray:
        prob = self.predict_proba(X)
        if threshold is None:
            threshold = self.recommended
        t = self.thresholds[threshold] if isinstance(threshold, str) else float(threshold)
        return (prob >= t).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment-dir", required=True)
    parser.add_argument("--data-root", required=True, help="repo Data root or a split root")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--evaluate", action="store_true",
                        help="also open labels and score (val labels are safe to open post-hoc)")
    args = parser.parse_args()

    deployment = Deployment(args.deployment_dir)
    root = Path(args.data_root)
    split_root = root / ("2.Validation" if args.split == "val" else "1.Training")
    if not split_root.is_dir():
        split_root = root  # caller passed the split root directly
    data = load_split(split_root, require_labels=False, split=args.split,
                      feature_subset=deployment.feature_names)

    prob = deployment.predict_proba(data.X)
    pred = deployment.predict(data.X)
    frame = pd.DataFrame({
        "subject_index": np.arange(data.n_subjects),
        "prob_impaired": prob,
        "prob_impaired_calibrated": deployment.predict_proba_calibrated(data.X),
        "prediction": pred,
    })
    out = args.output_csv or "predictions.csv"
    frame.to_csv(out, index=False)
    print(f"Wrote {len(frame)} predictions to {out} "
          f"(threshold rule = {deployment.recommended} = "
          f"{deployment.thresholds[deployment.recommended]:.3f})")

    if args.evaluate and args.split == "val":
        y = load_validation_labels_checked(split_root, data.subject_ids)
        metrics = binary_metrics(y, pred, prob)
        print("\nReproduced validation metrics at recommended threshold:")
        for key in ("accuracy", "balanced_accuracy", "impaired_recall", "cn_specificity", "roc_auc"):
            print(f"  {key:18s}: {metrics[key]:.4f}")
        print(f"  confusion         : {metrics['confusion']}")


if __name__ == "__main__":
    main()
