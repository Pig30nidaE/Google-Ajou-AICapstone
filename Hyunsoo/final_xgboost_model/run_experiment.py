"""Fixed V44 ablations. Requires the original patient-level input; never invents data."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

CIRCADIAN = ["circadian_IV", "circadian_IS", "circadian_RA", "sleep_wake_bouts_avg",
             "HR_drop_ratio", "Circadian_Strain", "sleep_hr_5min_max_std"]
SLEEP = ["sleep_score_alignment", "sleep_awake_std", "sleep_breath_average",
         "activity_score_std", "activity_class_3_count_std", "activity_met_min_low_std",
         "sleep_restless_std"]
# Preserve original column order: random feature subsampling can depend on it.
ALL = ["sleep_score_alignment", "sleep_hr_5min_max_std", "sleep_awake_std",
       "sleep_breath_average", "activity_score_std", "activity_class_3_count_std",
       "activity_met_min_low_std", "sleep_restless_std", "circadian_IV", "circadian_IS",
       "circadian_RA", "sleep_wake_bouts_avg", "HR_drop_ratio", "Circadian_Strain"]
BASE = ["cb_circadian", "lgb_sleep", "cb_all", "xgb_all", "svm_all", "rf_all", "lr_all"]
WEIGHTS = np.array([.25, .15, .25, .20, .10, .05])
TWO_WEIGHTS = WEIGHTS[:2] / WEIGHTS[:2].sum()


def load_data(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Original V44 input required: {path}")
    df = pd.read_csv(path)
    required = set(ALL) - {"HR_drop_ratio", "Circadian_Strain"}
    required |= {"EMAIL", "label", "sleep_hr_average", "sleep_hr_lowest"}
    missing = sorted(required - set(df))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df.EMAIL.isna().any() or df.EMAIL.duplicated().any():
        raise ValueError("Requires one nonmissing, unique EMAIL per subject; do not silently aggregate.")
    if df.label.isna().any() or set(df.label.unique()) != {0, 1}:
        raise ValueError("label must contain both 0 and 1, with no missing labels")
    numeric = list(required - {"EMAIL", "label"})
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    df["HR_drop_ratio"] = (df.sleep_hr_average - df.sleep_hr_lowest) / (df.sleep_hr_average + 1e-5)
    df["Circadian_Strain"] = df.circadian_IV / (df.circadian_IS + 1e-5)
    if np.isinf(df[ALL].to_numpy()).any():
        raise ValueError("Derived features contain infinity; resolve before comparing with original V44")
    if df[ALL].isna().all().any():
        raise ValueError("Entirely missing feature in input")
    audit = {"input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "n_subjects": len(df), "label_counts": df.label.value_counts().sort_index().to_dict(),
             "missing_per_feature": df[ALL].isna().sum().to_dict(),
             "cohort_matches_report_counts": bool(len(df) == 174 and (df.label == 0).sum() == 111 and (df.label == 1).sum() == 63)}
    # Do not export email addresses or derive diagnosis from label without source metadata.
    if "DIAG_NM" in df:
        audit["diagnosis_by_label"] = pd.crosstab(df.DIAG_NM, df.label).to_dict()
    return df[ALL], df.label.to_numpy(dtype=int), audit


def make_models(weighted=True):
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    balance = "balanced" if weighted else None
    cb_common = dict(depth=3, verbose=False, random_state=42, thread_count=1,
                     allow_writing_files=False)
    if weighted:
        cb_common["auto_class_weights"] = "Balanced"
    return {
        "cb_circadian": CatBoostClassifier(**cb_common, l2_leaf_reg=6., learning_rate=.04, iterations=130),
        "lgb_sleep": LGBMClassifier(max_depth=3, num_leaves=7, learning_rate=.04, n_estimators=110,
                                   reg_alpha=.2, reg_lambda=.2, min_child_samples=18,
                                   class_weight=balance, verbosity=-1, random_state=42, n_jobs=1),
        "cb_all": CatBoostClassifier(**cb_common, l2_leaf_reg=4., learning_rate=.035, iterations=140),
        "xgb_all": XGBClassifier(max_depth=3, learning_rate=.04, n_estimators=100, subsample=.8,
                                 colsample_bytree=.8, eval_metric="auc", random_state=42, n_jobs=1),
        "svm_all": SVC(C=1., kernel="rbf", probability=True, class_weight=balance, random_state=42),
        "rf_all": RandomForestClassifier(max_depth=3, n_estimators=180, max_features="sqrt",
                                          min_samples_leaf=2, class_weight=balance, random_state=42, n_jobs=1),
        "lr_all": LogisticRegression(C=1., solver="lbfgs", class_weight=balance, max_iter=2000, random_state=42),
    }


def fit_predict(train, y, test, smote, weighted=True):
    from imblearn.over_sampling import BorderlineSMOTE
    models = make_models(weighted)
    predictions, audit = {}, []
    for group, features, names in [
        ("circadian", CIRCADIAN, ["cb_circadian"]),
        ("sleep", SLEEP, ["lgb_sleep"]),
        ("all", ALL, BASE[2:]),
    ]:
        if train[features].isna().all().any():
            raise ValueError("All-missing feature within training fold")
        imputer = SimpleImputer(strategy="median")
        scaler = RobustScaler()
        x_train = scaler.fit_transform(imputer.fit_transform(train[features]))
        x_test = scaler.transform(imputer.transform(test[features]))
        y_train = y.copy()
        before = len(y_train)
        if smote:
            x_train, y_train = BorderlineSMOTE(random_state=42).fit_resample(x_train, y_train)
        audit.append({"group": group, "real_training_rows": before,
                      "synthetic_rows": len(y_train) - before,
                      "training_class_0": int((y_train == 0).sum()),
                      "training_class_1": int((y_train == 1).sum())})
        for name in names:
            models[name].fit(x_train, y_train)
            predictions[name] = models[name].predict_proba(x_test)[:, 1]
    return np.column_stack([predictions[n] for n in BASE]), audit


def reference_percentile(reference, values):
    """Midrank empirical CDF against training OOF predictions only; batch independent."""
    result = np.empty_like(values)
    for col in range(reference.shape[1]):
        ordered = np.sort(reference[:, col])
        left = np.searchsorted(ordered, values[:, col], side="left")
        right = np.searchsorted(ordered, values[:, col], side="right")
        result[:, col] = (left + right) / (2 * len(ordered))
    return result


def combine(predictions, reference):
    values = {n: predictions[:, i] for i, n in enumerate(BASE)}
    values["soft_2"] = predictions[:, :2] @ TWO_WEIGHTS
    values["soft_6"] = predictions[:, :6] @ WEIGHTS
    ranks = reference_percentile(reference, predictions)
    values["rank_2_reference"] = ranks[:, :2] @ TWO_WEIGHTS
    values["rank_6_reference"] = ranks[:, :6] @ WEIGHTS
    # Historical reproduction only: depends on the other subjects in each test batch.
    batch_ranks = rankdata(predictions, axis=0, method="average") / len(predictions)
    values["rank_6_legacy"] = batch_ranks[:, :6] @ WEIGHTS
    return values


def metrics(y, scores, predicted):
    positive, negative = y == 1, y == 0
    return {"auc": roc_auc_score(y, scores), "accuracy": accuracy_score(y, predicted),
            "recall": predicted[positive].mean(), "specificity": (1-predicted[negative]).mean(),
            "f1": f1_score(y, predicted, zero_division=0)}


def threshold(y, scores):
    fpr, tpr, cutoffs = roc_curve(y, scores)
    return cutoffs[np.argmax(tpr-fpr)]


def run(X, y, seeds, outer_splits=5, inner_splits=5, weighted=True, progress=None):
    rows, assignments, preprocessing, fold_metrics = [], [], [], []
    for seed in seeds:
        outer = StratifiedKFold(outer_splits, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(outer.split(X, y), 1):
            assert not set(train_idx) & set(test_idx)
            assignments.extend({"seed": seed, "fold": fold, "subject_row": int(i)} for i in test_idx)
            inner = list(StratifiedKFold(inner_splits, shuffle=True, random_state=seed+fold)
                         .split(X.iloc[train_idx], y[train_idx]))
            for smote in [False, True]:
                oof = np.full((len(train_idx), len(BASE)), np.nan)
                for inner_fold, (tr, va) in enumerate(inner, 1):
                    assert not set(train_idx[tr]) & set(test_idx)
                    p, audit = fit_predict(X.iloc[train_idx[tr]], y[train_idx[tr]],
                                           X.iloc[train_idx[va]], smote, weighted)
                    oof[va] = p
                    preprocessing.extend(dict(a, seed=seed, fold=fold, inner_fold=inner_fold, smote=smote) for a in audit)
                assert np.isfinite(oof).all()
                p, audit = fit_predict(X.iloc[train_idx], y[train_idx], X.iloc[test_idx], smote, weighted)
                preprocessing.extend(dict(a, seed=seed, fold=fold, inner_fold=0, smote=smote) for a in audit)
                inner_scores, test_scores = combine(oof, oof), combine(p, oof)
                for name, scores in test_scores.items():
                    cutoff = threshold(y[train_idx], inner_scores[name])
                    predicted = (scores >= cutoff).astype(int)
                    for pos, subject in enumerate(test_idx):
                        rows.append(dict(seed=seed, fold=fold, smote=smote, model=name,
                                         subject_row=int(subject), y=int(y[subject]), score=float(scores[pos]),
                                         prediction=int(predicted[pos]), threshold=float(cutoff)))
                    fold_metrics.append(dict(seed=seed, fold=fold, smote=smote, model=name,
                                             **metrics(y[test_idx], scores, predicted)))
                print(f"seed={seed} fold={fold}/{outer_splits} SMOTE={smote} complete", flush=True)
                if progress is not None:
                    pd.DataFrame(rows).to_csv(progress / "predictions.partial.csv", index=False)
    return tuple(map(pd.DataFrame, [rows, assignments, preprocessing, fold_metrics]))


def paired_intervals(predictions, repetitions=2000):
    """Same resampled subjects in all seeds/models, conditional on fitted OOF predictions."""
    cells = {}
    for (model, smote), grp in predictions.groupby(["model", "smote"]):
        cells[(model, smote)] = grp.pivot(index="subject_row", columns="seed", values="score").to_numpy()
    y = predictions.drop_duplicates("subject_row").sort_values("subject_row").y.to_numpy()
    rng = np.random.default_rng(7319)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    comparisons = [(("cb_all", True), ("cb_all", False)),
                   (("rank_6_reference", True), ("rank_6_reference", False))]
    for smote in [False, True]:
        comparisons.extend(((model, smote), ("cb_all", smote))
                           for model in ["soft_2", "soft_6", "rank_2_reference", "rank_6_reference", "rank_6_legacy"])
    diffs = {pair: [] for pair in comparisons}
    for _ in range(repetitions):
        sample = np.r_[rng.choice(negative, len(negative)), rng.choice(positive, len(positive))]
        aucs = {key: np.mean([roc_auc_score(y[sample], matrix[sample, j]) for j in range(matrix.shape[1])])
                for key, matrix in cells.items() if any(key in pair for pair in comparisons)}
        for pair in comparisons:
            diffs[pair].append(aucs[pair[0]]-aucs[pair[1]])
    results = []
    for a, b in comparisons:
        observed = np.mean([roc_auc_score(y, cells[a][:, j])-roc_auc_score(y, cells[b][:, j])
                            for j in range(cells[a].shape[1])])
        low, high = np.quantile(diffs[(a, b)], [.025, .975])
        results.append(dict(model_a=a[0], smote_a=a[1], model_b=b[0], smote_b=b[1],
                            auc_difference=observed, conditional_ci_low=low, conditional_ci_high=high))
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 13, 73, 101, 2026])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--no-class-weights", action="store_true")
    args = parser.parse_args()
    X, y, audit = load_data(args.data)
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique")
    if args.output.exists():
        raise FileExistsError("Choose a new output directory to preserve previous runs")
    args.output.mkdir(parents=True)
    audit.update(seeds=args.seeds, original_class_weights=not args.no_class_weights,
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 versions={n: importlib.metadata.version(n) for n in
                           ["numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "catboost", "lightgbm", "xgboost"]})
    (args.output / "provenance.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    pred, splits, prep, folds = run(X, y, args.seeds, weighted=not args.no_class_weights, progress=args.output)
    for name, df in [("predictions", pred), ("outer_splits", splits), ("preprocessing", prep), ("fold_metrics", folds)]:
        df.to_csv(args.output / f"{name}.csv", index=False)
    seed_metrics = []
    for (seed, smote, model), grp in pred.groupby(["seed", "smote", "model"]):
        assert grp.subject_row.nunique() == len(X) and len(grp) == len(X)
        seed_metrics.append(dict(seed=seed, smote=smote, model=model,
                                 **metrics(grp.y.to_numpy(), grp.score.to_numpy(), grp.prediction.to_numpy())))
    result = pd.DataFrame(seed_metrics)
    result.to_csv(args.output / "seed_metrics.csv", index=False)
    result.groupby(["model", "smote"]).auc.agg(["mean", "std", "min", "max"]).to_csv(args.output / "auc_summary.csv")
    paired_intervals(pred, args.bootstrap).to_csv(args.output / "paired_auc_differences.csv", index=False)
    (args.output / "COMPLETE.txt").write_text("Completed. Interpret intervals as conditional on fitted predictions; see README.\n")
    print(result.groupby(["model", "smote"]).auc.agg(["mean", "std", "min", "max"]).to_string())


if __name__ == "__main__":
    main()
