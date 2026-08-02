"""Load a saved Binary_MMSE_MaxAUC deployment and predict without retraining.

Run via base.ipynb (RUN_FILE = "Binary_MMSE_MaxAUC/predict.py"), CLI, or import.
Auto-detects the latest deployment/ under the Drive results root; set
PREDICT_DEPLOYMENT_DIR to target a specific run.
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

from SangHyo.Binary.Binary_MMSE_MaxAUC.engine import binary_metrics
from SangHyo.Binary.Binary_MMSE_MaxAUC.features import load_split, load_validation_labels_checked

DEFAULT_RESULTS_ROOT = Path("/content/drive/MyDrive/Binary_MMSE_MaxAUC_result")


class Deployment:
    def __init__(self, deployment_dir):
        dep = Path(deployment_dir)
        if not (dep / "deployment.json").is_file():
            raise FileNotFoundError(f"No deployment.json in {dep}")
        self.dir = dep
        self.config = json.loads((dep / "deployment.json").read_text(encoding="utf-8"))
        self.feature_names = self.config["feature_names"]
        self.final_weights = self.config["final_weights"]
        self.thresholds = self.config["thresholds_from_training_oof"]
        self.recommended = self.config["recommended_threshold"]
        self.include_wearable = self.config.get("include_wearable", False)
        self.item_max = {k: float(v) for k, v in self.config["item_max"].items()}
        self.calibrator = joblib.load(dep / "calibrator.joblib")
        self.models = {n: joblib.load(dep / f"model_{n}.joblib") for n in self.final_weights}

    def _score(self, bundle, X):
        Xi = np.where(np.isfinite(X), X, bundle["median"])
        return bundle["model"].predict_proba((Xi - bundle["mean"]) / bundle["std"])[:, 1]

    def predict_proba(self, X):
        total = sum(self.final_weights.values())
        combined = np.zeros(len(X))
        for name, weight in self.final_weights.items():
            combined += weight * self._score(self.models[name], np.asarray(X, float))
        return combined / total

    def predict(self, X, threshold=None):
        prob = self.predict_proba(X)
        rule = threshold or self.recommended
        t = self.thresholds[rule] if isinstance(rule, str) else float(rule)
        return (prob >= t).astype(int)


def _resolve_deployment(explicit):
    for value in (explicit, os.environ.get("PREDICT_DEPLOYMENT_DIR")):
        if value:
            p = Path(value).expanduser()
            if (p / "deployment.json").is_file():
                return p
            raise FileNotFoundError(f"deployment.json not found under {p}")
    root = Path(os.environ.get("PREDICT_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    if root.is_dir():
        cands = sorted(p for p in root.glob("*/deployment") if (p / "deployment.json").is_file())
        if cands:
            print(f"[predict] auto-detected deployment: {cands[-1]}")
            return cands[-1]
    raise FileNotFoundError("No deployment bundle found. Run run.py once, or set PREDICT_DEPLOYMENT_DIR.")


def _resolve_data_root(explicit, namespace):
    candidates = []
    for v in (explicit, os.environ.get("PREDICT_DATA_ROOT"), os.environ.get("SANGHYO_DATA_ROOT"),
              namespace.get("DATA_ROOT")):
        if v:
            candidates.append(Path(os.fspath(v)).expanduser())
    if namespace.get("PROJECT_ROOT"):
        candidates.append(Path(os.fspath(namespace["PROJECT_ROOT"])) / "Data")
    candidates += [REPOSITORY_ROOT / "Data", Path("/content/drive/Shareddrives/GoogleAI_contest/Data"),
                   Path("/content/drive/MyDrive/GoogleAI_contest/Data")]
    for c in candidates:
        r = c.expanduser().resolve()
        if (r / "1.Training").is_dir() and (r / "2.Validation").is_dir():
            return r
    for c in candidates:
        if c.expanduser().exists():
            return c.expanduser().resolve()
    raise FileNotFoundError("Could not resolve a data root; set --data-root or PREDICT_DATA_ROOT.")


def main(namespace=None):
    namespace = globals() if namespace is None else namespace
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment-dir", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default=None, choices=(None, "train", "val"))
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--evaluate", action="store_true")
    args, _unknown = parser.parse_known_args()

    dep = Deployment(_resolve_deployment(args.deployment_dir))
    data_root = _resolve_data_root(args.data_root, namespace)
    split = args.split or os.environ.get("PREDICT_SPLIT", "val")
    split_root = Path(data_root) / ("2.Validation" if split == "val" else "1.Training")
    if not split_root.is_dir():
        split_root = Path(data_root)

    data = load_split(split_root, require_labels=False, split=split,
                      include_wearable=dep.include_wearable, item_max=dep.item_max)
    if list(data.feature_names) != list(dep.feature_names):
        raise AssertionError("Feature schema mismatch between deployment and data")
    prob = dep.predict_proba(data.X)
    frame = pd.DataFrame({"subject_index": np.arange(data.n_subjects), "prob_impaired": prob,
                          "prediction": dep.predict(data.X)})
    out = args.output_csv or os.environ.get("PREDICT_OUTPUT_CSV") or str(dep.dir.parent / f"reproduced_predictions_{split}.csv")
    frame.to_csv(out, index=False)
    print(f"[predict] deployment {dep.dir}\n[predict] {split}: {data.n_subjects} subjects -> {out}"
          f"\n[predict] threshold {dep.recommended} = {dep.thresholds[dep.recommended]:.3f}")

    if (args.evaluate or os.environ.get("PREDICT_EVALUATE", "1") in {"1", "true"}) and split == "val":
        y = load_validation_labels_checked(split_root, data.subject_ids)
        m = binary_metrics(y, dep.predict(data.X), prob)
        c = m["confusion"]
        print("\n[predict] reproduced validation metrics:")
        print(f"    ROC-AUC           : {m['roc_auc']:.4f}")
        print(f"    accuracy          : {m['accuracy']:.4f}")
        print(f"    balanced_accuracy : {m['balanced_accuracy']:.4f}")
        print(f"    confusion         : CN {c['tn']}/{c['tn']+c['fp']}, impaired {c['tp']}/{c['tp']+c['fn']}")


if __name__ == "__main__":
    main()
