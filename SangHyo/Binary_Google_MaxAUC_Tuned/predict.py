"""Reproduce predictions from a saved deployment bundle, without retraining.

    USER_FOLDER = "SangHyo"
    RUN_FILE    = "Binary_Google_MaxAUC_Tuned/predict.py"

Auto-detects the newest ``deployment/`` under the Drive results root; set
``PREDICT_DEPLOYMENT_DIR`` to pin a specific run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from SangHyo.Binary_Google_MaxAUC_Tuned.engine import binary_metrics, blend_scores, safe_auc
from SangHyo.Binary_Google_MaxAUC_Tuned.features import (
    load_split,
    load_validation_labels_checked,
)

DEFAULT_RESULTS_ROOT = Path("/content/drive/MyDrive/Binary_Google_MaxAUC_Tuned_result")


class _LoadedYDF:
    def __init__(self, directory: Path, meta: dict) -> None:
        import ydf

        self.model = ydf.load_model(str(directory / "ydf_model"))
        self.pos = int(meta.get("pos_index", 1))
        self.columns = [f"f{i}" for i in range(int(meta["n_features"]))]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.model.predict(pd.DataFrame(np.asarray(X, float), columns=self.columns)),
                         dtype=np.float64)
        if raw.ndim == 1:
            return raw if self.pos == 1 else 1.0 - raw
        return raw[:, self.pos]


class _LoadedSklearn:
    def __init__(self, directory: Path, meta: dict) -> None:
        bundle = joblib.load(directory / "sklearn.joblib")
        self.kind = meta.get("kind", "")
        if isinstance(bundle, dict):
            self.median, self.mean, self.std = bundle["median"], bundle["mean"], bundle["std"]
            self.model = bundle["model"]
        else:  # YDF sklearn fallback stores the bare estimator
            self.median = self.mean = self.std = None
            self.model = bundle

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.median is not None:
            X = (np.where(np.isfinite(X), X, self.median) - self.mean) / self.std
        return self.model.predict_proba(X)[:, 1]


def _load_learner(directory: Path):
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    if meta.get("type") == "ydf" and meta.get("engine") == "ydf" and (directory / "ydf_model").exists():
        return _LoadedYDF(directory, meta)
    return _LoadedSklearn(directory, meta)


class Deployment:
    def __init__(self, deployment_dir) -> None:
        dep = Path(deployment_dir)
        if not (dep / "deployment.json").is_file():
            raise FileNotFoundError(f"No deployment.json in {dep}")
        self.dir = dep
        self.config = json.loads((dep / "deployment.json").read_text(encoding="utf-8"))
        self.feature_names = self.config["feature_names"]
        self.eligible = list(self.config["eligible"])
        self.weights = np.array([self.config["weights"][k] for k in self.eligible], dtype=float)
        self.thresholds = self.config["thresholds_from_nested_oof"]
        self.recommended = self.config["recommended_threshold"]
        self.item_max = {k: float(v) for k, v in self.config["item_max"].items()}
        self.include_wearable = self.config.get("include_wearable", True)
        self.drop_suspect = self.config.get("drop_suspect", False)
        self.calibrator = joblib.load(dep / "calibrator.joblib")
        self.models = {k: _load_learner(dep / f"model_{k}") for k in self.eligible}
        self.cols = {k: np.load(dep / f"cols_{k}.npy") for k in self.eligible}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        matrix = np.column_stack([
            self.models[k].predict_proba(X[:, self.cols[k]]) for k in self.eligible
        ])
        return 1.0 / (1.0 + np.exp(-blend_scores(matrix, self.weights)))

    def predict(self, X: np.ndarray, threshold=None) -> np.ndarray:
        prob = self.predict_proba(X)
        rule = threshold or self.recommended
        t = self.thresholds[rule] if isinstance(rule, str) else float(rule)
        return (prob >= t).astype(int)


def _resolve_deployment(explicit):
    for value in (explicit, os.environ.get("PREDICT_DEPLOYMENT_DIR")):
        if value:
            path = Path(value).expanduser()
            if (path / "deployment.json").is_file():
                return path
            raise FileNotFoundError(f"deployment.json not found under {path}")
    root = Path(os.environ.get("PREDICT_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    if root.is_dir():
        found = sorted(p for p in root.glob("*/deployment") if (p / "deployment.json").is_file())
        if found:
            print(f"[predict] auto-detected deployment: {found[-1]}")
            return found[-1]
    raise FileNotFoundError("No deployment bundle found. Run run.py once, or set "
                            "PREDICT_DEPLOYMENT_DIR.")


def _resolve_data_root(explicit, namespace):
    candidates = []
    for value in (explicit, os.environ.get("PREDICT_DATA_ROOT"),
                  os.environ.get("SANGHYO_DATA_ROOT"), namespace.get("DATA_ROOT")):
        if value:
            candidates.append(Path(os.fspath(value)).expanduser())
    if namespace.get("PROJECT_ROOT"):
        candidates.append(Path(os.fspath(namespace["PROJECT_ROOT"])) / "Data")
    candidates += [REPOSITORY_ROOT / "Data",
                   Path("/content/drive/Shareddrives/GoogleAI_contest/Data"),
                   Path("/content/drive/MyDrive/GoogleAI_contest/Data")]
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "1.Training").is_dir() and (resolved / "2.Validation").is_dir():
            return resolved
    for candidate in candidates:
        if candidate.expanduser().exists():
            return candidate.expanduser().resolve()
    raise FileNotFoundError("Could not resolve a data root; set --data-root or PREDICT_DATA_ROOT.")


def main(namespace=None):
    namespace = globals() if namespace is None else namespace
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment-dir", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default=None, choices=(None, "train", "val"))
    parser.add_argument("--output-csv", default=None)
    args, _unknown = parser.parse_known_args()

    dep = Deployment(_resolve_deployment(args.deployment_dir))
    data_root = _resolve_data_root(args.data_root, namespace)
    split = args.split or os.environ.get("PREDICT_SPLIT", "val")
    split_root = data_root / ("2.Validation" if split == "val" else "1.Training")

    data = load_split(split_root, require_labels=False, split=split, item_max=dep.item_max,
                      include_wearable=dep.include_wearable, drop_suspect=dep.drop_suspect)
    if list(data.feature_names) != list(dep.feature_names):
        raise AssertionError("Feature schema mismatch between deployment and data")

    prob = dep.predict_proba(data.X)
    pred = dep.predict(data.X)
    frame = pd.DataFrame({"subject_index": np.arange(data.n_subjects),
                          "prob_impaired": prob, "prediction": pred})
    out = (args.output_csv or os.environ.get("PREDICT_OUTPUT_CSV")
           or str(dep.dir.parent / f"reproduced_predictions_{split}.csv"))
    frame.to_csv(out, index=False)
    print(f"[predict] deployment {dep.dir}")
    print(f"[predict] models     {', '.join(dep.eligible)}")
    print(f"[predict] {split}: {data.n_subjects} subjects -> {out}")
    print(f"[predict] threshold {dep.recommended} = {dep.thresholds[dep.recommended]:.3f}")

    if split == "val" and os.environ.get("PREDICT_EVALUATE", "1") in {"1", "true"}:
        y = load_validation_labels_checked(split_root, data.subject_ids)
        metrics = binary_metrics(y, pred, prob)
        confusion = metrics["confusion"]
        print("\n[predict] reproduced validation metrics:")
        print(f"    ROC-AUC           : {safe_auc(y, prob):.4f}")
        print(f"    accuracy          : {metrics['accuracy']:.4f}")
        print(f"    balanced_accuracy : {metrics['balanced_accuracy']:.4f}")
        print(f"    confusion         : CN {confusion['tn']}/{confusion['tn'] + confusion['fp']}, "
              f"impaired {confusion['tp']}/{confusion['tp'] + confusion['fn']}")


if __name__ == "__main__":
    main()
