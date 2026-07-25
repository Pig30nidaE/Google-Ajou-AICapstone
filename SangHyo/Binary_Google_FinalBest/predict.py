"""Load a saved Binary_Google_FinalBest deployment and predict without retraining.

The training run writes a ``deployment/`` bundle (fitted YDF models + Platt
calibrator + thresholds + feature list).  This module reloads that bundle and
scores any split, reproducing the reported validation predictions exactly
(YDF is deterministic under the saved seed, and the same fitted models used for
the validation freeze are the ones saved).

Three ways to run it:

1. **base.ipynb (Cell 2)** — set only::

       USER_FOLDER = "SangHyo"
       RUN_FILE = "Binary_Google_FinalBest/predict.py"

   It auto-detects the most recent ``deployment/`` under the Drive results root
   and uses the notebook's ``DATA_ROOT``.  To target a specific run, set
   ``os.environ["PREDICT_DEPLOYMENT_DIR"]`` (and optionally ``PREDICT_SPLIT``)
   in a cell before running.

2. **CLI** ::

       python -m SangHyo.Binary_Google_FinalBest.predict \
           --deployment-dir <RUN>/deployment --data-root <Data> --split val --evaluate

3. **import** ::

       from SangHyo.Binary_Google_FinalBest.predict import Deployment
       dep = Deployment("<RUN>/deployment"); dep.predict(X)
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

# Absolute imports (not relative) so this file also works when base.ipynb runs
# it standalone via runpy.run_path(run_name="__main__").
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from SangHyo.Binary_Google_FinalBest.engine import binary_metrics
from SangHyo.Binary_Google_FinalBest.features import load_split, load_validation_labels_checked

DEFAULT_RESULTS_ROOT = Path("/content/drive/MyDrive/Binary_Google_FinalBest_result")


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


def _import_ydf():
    """Import ydf, installing the pinned version on demand (fresh Colab session)."""

    try:
        import ydf
        return ydf
    except Exception:
        import subprocess
        print("[predict] ydf가 없어 설치합니다 (ydf==0.16.1)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                        "ydf==0.16.1"], check=True)
        import ydf
        return ydf


def _load_learner(model_dir: Path):
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    if meta["type"] == "ydf_learner":
        if meta["engine"] == "ydf":
            ydf = _import_ydf()
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
        if not (dep / "deployment.json").is_file():
            raise FileNotFoundError(f"No deployment.json in {dep}")
        self.dir = dep
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


# --------------------------------------------------------------------------- #
# Runner: resolves config from CLI args, env vars, base.ipynb globals, or auto.
# --------------------------------------------------------------------------- #

def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _autodetect_deployment() -> Path | None:
    roots = []
    env_root = os.environ.get("PREDICT_RESULTS_ROOT")
    if env_root:
        roots.append(Path(env_root))
    roots.append(DEFAULT_RESULTS_ROOT)
    for root in roots:
        if root.is_dir():
            candidates = sorted(p for p in root.glob("*/deployment") if (p / "deployment.json").is_file())
            if candidates:
                return candidates[-1]  # newest UTC-timestamped run
    return None


def _resolve_deployment(explicit: str | None) -> Path:
    for value in (explicit, os.environ.get("PREDICT_DEPLOYMENT_DIR")):
        if value:
            path = Path(value).expanduser()
            if (path / "deployment.json").is_file():
                return path
            raise FileNotFoundError(f"deployment.json not found under {path}")
    auto = _autodetect_deployment()
    if auto is not None:
        print(f"[predict] auto-detected deployment: {auto}")
        return auto
    raise FileNotFoundError(
        "No deployment bundle found. Run Binary_Google_FinalBest/run.py once to "
        "create one, or set PREDICT_DEPLOYMENT_DIR / --deployment-dir."
    )


def _resolve_data_root(explicit: str | None, namespace: dict) -> Path:
    """Resolve the Data root the same way run.py does (robust candidate list)."""

    candidates: list[Path] = []
    for value in (explicit, os.environ.get("PREDICT_DATA_ROOT"),
                  os.environ.get("SANGHYO_DATA_ROOT"), namespace.get("DATA_ROOT")):
        if value:
            candidates.append(Path(os.fspath(value)).expanduser())
    project = namespace.get("PROJECT_ROOT")
    if project:
        candidates.append(Path(os.fspath(project)) / "Data")
    candidates.extend([
        REPOSITORY_ROOT / "Data",
        Path("/content/drive/Shareddrives/GoogleAI_contest/Data"),
        Path("/content/drive/MyDrive/GoogleAI_contest/Data"),
    ])
    # Prefer a candidate that actually contains the split directories.
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "1.Training").is_dir() and (resolved / "2.Validation").is_dir():
            return resolved
    # Fall back to any existing candidate (e.g. a split root passed directly).
    for candidate in candidates:
        if candidate.expanduser().exists():
            return candidate.expanduser().resolve()
    raise FileNotFoundError(
        "Could not resolve a data root; set --data-root or PREDICT_DATA_ROOT. "
        f"Checked: {', '.join(str(c) for c in candidates)}"
    )


def main(namespace: dict | None = None) -> None:
    namespace = globals() if namespace is None else namespace
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment-dir", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default=None, choices=(None, "train", "val"))
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--no-evaluate", action="store_true")
    # base.ipynb runs via runpy inside ipykernel, whose "-f kernel.json" stays in
    # sys.argv; ignore unknown notebook-owned arguments.
    args, _unknown = parser.parse_known_args()

    deployment_dir = _resolve_deployment(args.deployment_dir)
    data_root = _resolve_data_root(args.data_root, namespace)
    split = args.split or os.environ.get("PREDICT_SPLIT", "val")
    if args.no_evaluate:
        evaluate = False
    elif args.evaluate:
        evaluate = True
    else:
        evaluate = _env_bool("PREDICT_EVALUATE", split == "val")

    deployment = Deployment(deployment_dir)
    split_root = Path(data_root) / ("2.Validation" if split == "val" else "1.Training")
    if not split_root.is_dir():
        split_root = Path(data_root)
    data = load_split(split_root, require_labels=False, split=split,
                      feature_subset=deployment.feature_names)

    prob = deployment.predict_proba(data.X)
    pred = deployment.predict(data.X)
    frame = pd.DataFrame({
        "subject_index": np.arange(data.n_subjects),
        "prob_impaired": prob,
        "prob_impaired_calibrated": deployment.predict_proba_calibrated(data.X),
        "prediction": pred,
    })
    out = args.output_csv or os.environ.get("PREDICT_OUTPUT_CSV")
    if out is None:
        out = str(deployment.dir.parent / f"reproduced_predictions_{split}.csv")
    frame.to_csv(out, index=False)
    rule = deployment.recommended
    print(f"[predict] deployment : {deployment.dir}")
    print(f"[predict] data split : {split} ({data.n_subjects} subjects)")
    print(f"[predict] threshold  : {rule} = {deployment.thresholds[rule]:.3f}")
    print(f"[predict] wrote {len(frame)} predictions -> {out}")

    if evaluate and split == "val":
        y = load_validation_labels_checked(split_root, data.subject_ids)
        metrics = binary_metrics(y, pred, prob)
        c = metrics["confusion"]
        print("\n[predict] reproduced validation metrics at recommended threshold:")
        for key in ("accuracy", "balanced_accuracy", "impaired_recall", "cn_specificity", "roc_auc"):
            print(f"    {key:18s}: {metrics[key]:.4f}")
        print(f"    confusion         : CN {c['tn']}/{c['tn']+c['fp']}, "
              f"impaired {c['tp']}/{c['tp']+c['fn']}  => {c['tn']+c['tp']}/{len(y)}")


if __name__ == "__main__":
    main()
