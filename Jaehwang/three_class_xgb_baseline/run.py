"""Fixed-domain single XGBoost, subject-disjoint five-fold OOF evaluation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

from data import DOMAIN_FEATURES, inspect_input, load_dataset, derive_wearable_features

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parents[1] / "Taehyun/data/processed/tabular/patient_level_circadian_v3.csv"
MODEL_PARAMS = {
    "objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss",
    "n_estimators": 100, "max_depth": 2, "learning_rate": 0.04,
    "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 1.0,
    "reg_alpha": 0.1, "reg_lambda": 5.0, "tree_method": "hist",
    "device": "cpu", "max_bin": 256, "random_state": 42, "n_jobs": 1,
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def make_splits(y, groups):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

    unique = len(np.unique(groups)) == len(groups)
    splitter = (StratifiedKFold if unique else StratifiedGroupKFold)(n_splits=5, shuffle=True, random_state=42)
    iterator = splitter.split(np.zeros(len(y)), y) if unique else splitter.split(np.zeros(len(y)), y, groups)
    splits = list(iterator)
    visits = np.zeros(len(y), dtype=int)
    for train, test in splits:
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Subject overlap between train and test")
        if set(y[train]) != {0, 1, 2} or set(y[test]) != {0, 1, 2}:
            raise ValueError("Every fold must contain all classes; do not change seed after seeing scores")
        visits[test] += 1
    if not np.all(visits == 1):
        raise ValueError("Every input row must be tested exactly once")
    return splits, type(splitter).__name__


def training_weights(y, groups):
    """Class-balanced subject contributions, computed exclusively from train rows."""
    import numpy as np
    from collections import Counter

    subjects = dict(zip(groups, y))
    counts = Counter(subjects.values())
    rows_per_subject = Counter(groups)
    if set(counts) != {0, 1, 2}:
        raise ValueError("Training requires all classes")
    weights = np.array([len(subjects) / (3 * counts[label] * rows_per_subject[group])
                        for label, group in zip(y, groups)])
    # Mean one keeps regularization scale comparable if repeated rows are present.
    weights /= weights.mean()
    return weights, {str(c): int(counts[c]) for c in (0, 1, 2)}


def preprocess_fold(raw_train, raw_test, fold=0):
    """Deterministic derivation on each side, then train-only median fit."""
    import pandas as pd
    from sklearn.impute import SimpleImputer

    X_train, train_audit = derive_wearable_features(raw_train, fold, "train")
    X_test, test_audit = derive_wearable_features(raw_test, fold, "test")
    empty = X_train.columns[X_train.isna().all()].tolist()
    if empty:
        raise ValueError(f"All-missing train-fold features cannot be imputed: {empty}")
    imputer = SimpleImputer(strategy="median")
    train = pd.DataFrame(imputer.fit_transform(X_train), columns=DOMAIN_FEATURES)
    test = pd.DataFrame(imputer.transform(X_test), columns=DOMAIN_FEATURES)
    return train, test, imputer, train_audit + test_audit


def fit_fold(X_train, y_train, groups_train, X_test, fold=0):
    import numpy as np
    from xgboost import XGBClassifier

    train, test, imputer, derived_audit = preprocess_fold(X_train, X_test, fold)
    weights, counts = training_weights(y_train, groups_train)
    model = XGBClassifier(**MODEL_PARAMS)
    # No eval_set, early stopping, feature search or tuning.
    model.fit(train, y_train, sample_weight=weights)
    if not np.array_equal(model.classes_, [0, 1, 2]):
        raise ValueError("Unexpected probability column order")
    probabilities = model.predict_proba(test)
    audit = {
        "train_subject_class_counts": counts,
        "train_rows": len(train), "test_rows": len(test),
        "sample_weight_min": float(weights.min()), "sample_weight_max": float(weights.max()),
        "imputer_medians": dict(zip(DOMAIN_FEATURES, imputer.statistics_.tolist())),
        "training_feature_names": model.get_booster().feature_names,
        "subject_overlap": 0,
        "derived_feature_audit": derived_audit,
    }
    return probabilities, model, audit


def aggregate_subjects(row_predictions):
    """Repeated rows receive one subject-level prediction (mean probabilities)."""
    grouping = row_predictions.groupby("subject_index", sort=True)
    if (grouping["y_true"].nunique() != 1).any() or (grouping["fold"].nunique() != 1).any():
        raise ValueError("Subject labels or fold assignments are inconsistent")
    result = grouping.agg(fold=("fold", "first"), y_true=("y_true", "first"),
                          p_CN=("p_CN", "mean"), p_MCI=("p_MCI", "mean"),
                          p_Dementia=("p_Dementia", "mean"), n_rows=("row_index", "size")).reset_index()
    result["prediction"] = result[["p_CN", "p_MCI", "p_Dementia"]].to_numpy().argmax(axis=1)
    return result


def train_and_evaluate(args, out):
    import numpy as np
    import pandas as pd
    from evaluation import metrics, bootstrap_intervals

    X, y, groups, audit = load_dataset(args.data, allow_derived=args.allow_derived)
    write_json(out / "input_audit.json", audit)
    pd.DataFrame(audit["excluded_columns"]).to_csv(out / "excluded_columns.csv", index=False)
    splits, split_name = make_splits(y, groups)
    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "fixed_single_model_subject_5fold_oof_no_nested_cv",
        "splitter": split_name, "n_splits": 5, "split_seed": 42,
        "features": DOMAIN_FEATURES, "target": "original_label", "model_params": MODEL_PARAMS,
        "sample_weight": "N_train_subjects/(3*N_train_subjects_in_class), divided by subject row count, normalized mean=1",
        "prediction_rule": "argmax, probability column order CN/MCI/Dementia",
        "bootstrap_repetitions": 2000, "bootstrap_seed": 7319,
        "python": platform.python_version(), "platform": platform.platform(), "packages": versions,
        "input_sha256": audit["input_sha256"], "allow_derived": args.allow_derived,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")},
        "limitations": [
            "This cohort and these features have been studied previously; this is not an untouched external holdout.",
            "The input contains 33 all-zero circadian rows in the current dataset; zeros are preserved.",
            "Bootstrap intervals are conditional on fixed OOF predictions, not repeated fitting uncertainty.",
        ],
    }
    write_json(out / "manifest.json", manifest)  # frozen before fitting any model
    (out / "environment.lock.txt").write_text("\n".join(f"{k}=={v}" for k, v in sorted(versions.items(), key=lambda x: x[0].lower())) + "\n", encoding="utf-8")
    subject_indices = pd.factorize(groups, sort=False)[0]
    folds = np.zeros(len(y), dtype=int)
    probabilities = np.full((len(y), 3), np.nan)
    split_rows = []
    for fold, (train, test) in enumerate(splits, 1):
        folds[test] = fold
        split_rows.extend({"fold": fold, "role": role, "row_index": int(i), "subject_index": int(subject_indices[i])}
                          for role, indices in (("train", train), ("test", test)) for i in indices)
    pd.DataFrame(split_rows).to_csv(out / "splits.csv", index=False)
    audits = []
    derived_audits = []
    for fold, (train, test) in enumerate(splits, 1):
        p, model, fold_audit = fit_fold(X.iloc[train], y[train], groups[train], X.iloc[test], fold=fold)
        derived_audits.extend(fold_audit["derived_feature_audit"])
        pd.DataFrame(derived_audits).to_csv(out / "derived_feature_audit.csv", index=False)
        probabilities[test] = p
        fold_dir = out / f"fold_{fold}"
        fold_dir.mkdir()
        model.save_model(fold_dir / "model.ubj")
        write_json(fold_dir / "preprocessing.json", fold_audit)
        audits.append(dict(fold=fold, **fold_audit))
        print(f"Fold {fold}/5 trained; held-out predictions stored.", flush=True)
    derived_frame = pd.DataFrame(derived_audits)
    # Held-out rows partition the input: sum their audit counts once per input row.
    unique_counts = derived_frame[derived_frame.role == "test"].groupby(["kind", "feature"], as_index=False).sum(numeric_only=True)
    unique_counts["fold"] = 0
    unique_counts["role"] = "oof_unique_input"
    pd.concat([derived_frame, unique_counts], ignore_index=True).to_csv(out / "derived_feature_audit.csv", index=False)
    if not np.isfinite(probabilities).all():
        raise ValueError("OOF coverage failure")
    row_predictions = pd.DataFrame({"row_index": np.arange(len(y)), "subject_index": subject_indices,
                                    "fold": folds, "y_true": y, "p_CN": probabilities[:, 0],
                                    "p_MCI": probabilities[:, 1], "p_Dementia": probabilities[:, 2]})
    row_predictions["prediction"] = probabilities.argmax(axis=1)
    row_predictions.to_csv(out / "oof_rows.csv", index=False)
    subject_predictions = aggregate_subjects(row_predictions)
    if len(subject_predictions) != len(np.unique(groups)):
        raise ValueError("Subject OOF coverage failure")
    subject_predictions.to_csv(out / "oof_subjects.csv", index=False)
    write_json(out / "leakage_audit.json", {"all_folds_subject_disjoint": True,
               "every_subject_evaluated_once": True, "all_models_use_exactly_14_features": True,
               "no_test_eval_set": True, "no_search": True, "folds": audits})
    prediction_path = out / "oof_subjects.csv"
    write_json(out / "PREDICTIONS_FROZEN.json", {"sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest()})
    # No choices are changed based on any test-fold scores. Scoring begins after all predictions freeze.
    proba_columns = ["p_CN", "p_MCI", "p_Dementia"]
    subject_y = subject_predictions.y_true.to_numpy()
    subject_p = subject_predictions[proba_columns].to_numpy()
    scores, cm = metrics(subject_y, subject_p)
    fold_scores = []
    for fold, frame in subject_predictions.groupby("fold"):
        score, fold_cm = metrics(frame.y_true.to_numpy(), frame[proba_columns].to_numpy())
        fold_scores.append(dict(fold=int(fold), n_subjects=len(frame), **score))
        pd.DataFrame(fold_cm, index=["CN", "MCI", "Dementia"], columns=["CN", "MCI", "Dementia"]).to_csv(out / f"fold_{fold}" / "confusion_matrix.csv")
    pd.DataFrame(fold_scores).to_csv(out / "fold_metrics.csv", index=False)
    write_json(out / "metrics.json", {"aggregation": "pooled_subject_oof", "n_subjects": len(subject_y),
                                     **scores, "confusion_matrix": cm.tolist()})
    from sklearn.metrics import classification_report
    report_args = dict(labels=[0, 1, 2], target_names=["CN", "MCI", "Dementia"], zero_division=0)
    report = classification_report(subject_y, subject_p.argmax(axis=1), output_dict=True, **report_args)
    write_json(out / "classification_report.json", report)
    (out / "classification_report.txt").write_text(classification_report(
        subject_y, subject_p.argmax(axis=1), digits=4, **report_args), encoding="utf-8")
    pd.DataFrame(report).T.to_csv(out / "classification_report.csv")
    pd.DataFrame(cm, index=["CN", "MCI", "Dementia"], columns=["CN", "MCI", "Dementia"]).to_csv(out / "confusion_matrix.csv")
    print("Computing 2,000 subject-stratified bootstrap draws...", flush=True)
    intervals = bootstrap_intervals(subject_y, subject_p)
    pd.DataFrame(intervals).to_csv(out / "bootstrap_ci.csv", index=False)
    write_json(out / "summary_metrics.json", {"aggregation": "pooled_subject_oof",
               "n_subjects": len(subject_y), **scores, "conditional_bootstrap_95_ci": intervals})
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["CN", "MCI", "Dementia"]).plot(ax=ax, colorbar=False)
    ax.set_title("Fixed 14-feature XGBoost: subject OOF")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    lines = ["# Fixed-domain 3-class XGBoost result", "", "Single fixed model, five-fold subject-disjoint OOF; no nested CV or tuning.", "",
             "| Metric | Estimate | Conditional 95% CI |", "|---|---:|---:|"]
    lines += [f"| {r['metric']} | {r['estimate']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] |" for r in intervals]
    lines += ["", "Confusion matrix: rows=true, columns=predicted, order CN/MCI/Dementia.", "", "```", str(cm), "```", "",
              "Only 12 Dementia subjects; bootstrap does not include retraining uncertainty.",
              "The cohort has been used in earlier research; these are not external validation results.",
              "All-zero circadian rows are preserved. No changes were made after observing scores."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    final_input_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if final_input_hash != audit["input_sha256"]:
        raise ValueError("Input changed during execution")
    write_json(out / "input_integrity.json", {"sha256_before": audit["input_sha256"],
               "sha256_after": final_input_hash, "unchanged": True})
    write_json(out / "COMPLETE.json", {"completed_utc": datetime.now(timezone.utc).isoformat(), "metrics": scores})
    print(json.dumps(scores, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--allow-derived", action="store_true", help="Explicitly permit the two documented row-local formulas; no substitution")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = args.output or (HERE / "outputs" / stamp if args.audit_only else HERE / "results")
    out = out.resolve()
    if not out.is_relative_to(HERE):
        parser.error("All outputs must stay inside Jaehwang/three_class_xgb_baseline")
    out.mkdir(parents=True, exist_ok=False)
    try:
        audit = inspect_input(args.data)
        write_json(out / "input_audit.json", audit)
        write_json(out / "excluded_columns.json", audit["excluded_columns"])
        print(json.dumps({key: audit[key] for key in ["n_rows", "n_subjects", "subject_class_counts", "missing_domain_features"]}, indent=2))
        if args.audit_only:
            print(f"Audit only; no training. Output: {out}")
            return
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            try:
                train_and_evaluate(args, out)
            finally:
                write_json(out / "runtime_warnings.json", [
                    {"category": item.category.__name__, "message": str(item.message),
                     "file": item.filename, "line": item.lineno} for item in captured])
        print(f"Python runtime warnings captured: {len(captured)}", flush=True)
    except Exception as exc:
        write_json(out / "FAILED.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
