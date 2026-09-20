"""Sensitivity analysis for XGBoost positive-class weights.

Everything except ``scale_pos_weight`` is held fixed. Each candidate uses the
same repeated nested-CV splits and fold-local preprocessing. Classification
thresholds are selected from inner out-of-fold predictions with Youden's J.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from fixed_xgb_confidence import (
    BOOTSTRAP_SEED,
    auc_from_fixed_class_bootstrap,
    safe_ratio,
    summarize,
)
from run_experiment import load_data, threshold


DEFAULT_SEEDS = (42, 13, 73, 101, 2026)
DEFAULT_WEIGHTS = tuple(np.round(np.arange(1.0, 3.01, 0.25), 2))
METRICS = ("auc", "accuracy", "precision", "recall", "specificity", "f1")
CONFUSION_CELLS = ("tn", "fp", "fn", "tp")


def make_model(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        max_depth=3,
        learning_rate=0.04,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=1,
    )


def preprocess(
    training: pd.DataFrame, validation: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Fit every preprocessing step on the training partition only."""
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    transformed_training = scaler.fit_transform(imputer.fit_transform(training))
    transformed_validation = scaler.transform(imputer.transform(validation))
    return transformed_training, transformed_validation


def run_predictions(
    features: pd.DataFrame,
    labels: np.ndarray,
    weights: tuple[float, ...],
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict] = []
    for seed in seeds:
        outer_cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        for fold, (training_indices, test_indices) in enumerate(
            outer_cv.split(features, labels), start=1
        ):
            inner_cv = StratifiedKFold(5, shuffle=True, random_state=seed + fold)
            inner_splits = list(
                inner_cv.split(
                    features.iloc[training_indices], labels[training_indices]
                )
            )
            for class_weight in weights:
                inner_scores = np.full(len(training_indices), np.nan)
                for inner_training, inner_validation in inner_splits:
                    train_rows = training_indices[inner_training]
                    validation_rows = training_indices[inner_validation]
                    x_train, x_validation = preprocess(
                        features.iloc[train_rows], features.iloc[validation_rows]
                    )
                    model = make_model(class_weight)
                    model.fit(x_train, labels[train_rows])
                    inner_scores[inner_validation] = model.predict_proba(
                        x_validation
                    )[:, 1]
                if not np.isfinite(inner_scores).all():
                    raise AssertionError("Incomplete inner out-of-fold predictions")

                cutoff = threshold(labels[training_indices], inner_scores)
                x_train, x_test = preprocess(
                    features.iloc[training_indices], features.iloc[test_indices]
                )
                model = make_model(class_weight)
                model.fit(x_train, labels[training_indices])
                scores = model.predict_proba(x_test)[:, 1]
                predictions = (scores >= cutoff).astype(int)
                rows.extend(
                    {
                        "scale_pos_weight": class_weight,
                        "seed": seed,
                        "fold": fold,
                        "subject_row": int(subject),
                        "y": int(labels[subject]),
                        "score": float(scores[position]),
                        "prediction": int(predictions[position]),
                        "threshold": float(cutoff),
                    }
                    for position, subject in enumerate(test_indices)
                )
            print(f"seed={seed} fold={fold}/5 complete", flush=True)
    return pd.DataFrame(rows)


def confusion_counts(
    labels: np.ndarray, predictions: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "tn": ((predictions == 0) & (labels[:, None] == 0)).sum(axis=0),
        "fp": ((predictions == 1) & (labels[:, None] == 0)).sum(axis=0),
        "fn": ((predictions == 0) & (labels[:, None] == 1)).sum(axis=0),
        "tp": ((predictions == 1) & (labels[:, None] == 1)).sum(axis=0),
    }


def observed_statistics(
    labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray
) -> tuple[dict[str, float], dict[str, float]]:
    counts = confusion_counts(labels, predictions)
    tn, fp, fn, tp = (counts[name].astype(float) for name in CONFUSION_CELLS)
    metrics = {
        "auc": float(
            np.mean(
                [roc_auc_score(labels, scores[:, column]) for column in range(scores.shape[1])]
            )
        ),
        "accuracy": float(np.mean((tp + tn) / len(labels))),
        "precision": float(np.mean(safe_ratio(tp, tp + fp))),
        "recall": float(np.mean(tp / (tp + fn))),
        "specificity": float(np.mean(tn / (tn + fp))),
        "f1": float(np.mean(2 * tp / (2 * tp + fp + fn))),
    }
    mean_counts = {name: float(values.mean()) for name, values in counts.items()}
    return metrics, mean_counts


def bootstrap_statistics(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    samples: np.ndarray,
    negative_count: int,
    positive_count: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    repetitions, seed_count = samples.shape[0], scores.shape[1]
    auc_by_seed = np.empty((repetitions, seed_count))
    for column in range(seed_count):
        auc_by_seed[:, column] = auc_from_fixed_class_bootstrap(
            scores[samples, column], negative_count, positive_count
        )

    sampled_predictions = predictions[samples, :]
    fp = sampled_predictions[:, :negative_count, :].sum(axis=1).astype(float)
    tp = sampled_predictions[:, negative_count:, :].sum(axis=1).astype(float)
    tn = negative_count - fp
    fn = positive_count - tp
    metric_draws = {
        "auc": auc_by_seed.mean(axis=1),
        "accuracy": ((tp + tn) / len(labels)).mean(axis=1),
        "precision": np.nanmean(safe_ratio(tp, tp + fp), axis=1),
        "recall": (tp / positive_count).mean(axis=1),
        "specificity": (tn / negative_count).mean(axis=1),
        "f1": (2 * tp / (2 * tp + fp + fn)).mean(axis=1),
    }
    confusion_draws = {
        "tn": tn.mean(axis=1),
        "fp": fp.mean(axis=1),
        "fn": fn.mean(axis=1),
        "tp": tp.mean(axis=1),
    }
    return metric_draws, confusion_draws


def matrices_for_weight(
    predictions: pd.DataFrame, class_weight: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    chosen = predictions[np.isclose(predictions.scale_pos_weight, class_weight)].copy()
    if chosen.duplicated(["subject_row", "seed"]).any():
        raise ValueError("Duplicate subject/seed predictions")
    seeds = np.array(sorted(chosen.seed.unique()), dtype=int)
    scores = chosen.pivot(index="subject_row", columns="seed", values="score")
    predicted = chosen.pivot(index="subject_row", columns="seed", values="prediction")
    label_frame = chosen.pivot(index="subject_row", columns="seed", values="y")
    if scores.isna().any().any() or predicted.isna().any().any():
        raise ValueError("Incomplete repeated out-of-fold predictions")
    if not label_frame.nunique(axis=1).eq(1).all():
        raise ValueError("Inconsistent labels across CV repetitions")
    return (
        label_frame.iloc[:, 0].to_numpy(dtype=int),
        scores[seeds].to_numpy(dtype=float),
        predicted[seeds].to_numpy(dtype=int),
        seeds,
    )


def summarize_results(
    predictions: pd.DataFrame,
    weights: tuple[float, ...],
    repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels, _, _, seeds = matrices_for_weight(predictions, weights[0])
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.column_stack(
        [
            rng.choice(negative, size=(repetitions, len(negative)), replace=True),
            rng.choice(positive, size=(repetitions, len(positive)), replace=True),
        ]
    )

    observed: dict[float, dict[str, float]] = {}
    observed_counts: dict[float, dict[str, float]] = {}
    metric_draws: dict[float, dict[str, np.ndarray]] = {}
    confusion_draws: dict[float, dict[str, np.ndarray]] = {}
    seed_rows: list[dict] = []

    for class_weight in weights:
        current_labels, scores, predicted, current_seeds = matrices_for_weight(
            predictions, class_weight
        )
        if not np.array_equal(labels, current_labels) or not np.array_equal(seeds, current_seeds):
            raise ValueError("Weights do not share identical subjects and CV seeds")
        observed[class_weight], observed_counts[class_weight] = observed_statistics(
            labels, scores, predicted
        )
        metric_draws[class_weight], confusion_draws[class_weight] = bootstrap_statistics(
            labels,
            scores,
            predicted,
            samples,
            len(negative),
            len(positive),
        )
        for column, seed in enumerate(seeds):
            seed_metrics, seed_counts = observed_statistics(
                labels, scores[:, [column]], predicted[:, [column]]
            )
            seed_rows.append(
                {
                    "scale_pos_weight": class_weight,
                    "seed": int(seed),
                    **seed_metrics,
                    **seed_counts,
                }
            )

    baseline = weights[0]
    metric_rows: list[dict] = []
    confusion_rows: list[dict] = []
    for class_weight in weights:
        for metric in METRICS:
            level = summarize(
                metric_draws[class_weight][metric], observed[class_weight][metric]
            )
            difference = summarize(
                metric_draws[class_weight][metric] - metric_draws[baseline][metric],
                observed[class_weight][metric] - observed[baseline][metric],
            )
            metric_rows.append(
                {
                    "scale_pos_weight": class_weight,
                    "metric": metric,
                    **level,
                    "difference_vs_weight_1": difference["estimate"],
                    "difference_ci_95_low": difference["ci_95_percentile_low"],
                    "difference_ci_95_high": difference["ci_95_percentile_high"],
                }
            )
        for cell in CONFUSION_CELLS:
            level = summarize(
                confusion_draws[class_weight][cell], observed_counts[class_weight][cell]
            )
            difference = summarize(
                confusion_draws[class_weight][cell] - confusion_draws[baseline][cell],
                observed_counts[class_weight][cell] - observed_counts[baseline][cell],
            )
            confusion_rows.append(
                {
                    "scale_pos_weight": class_weight,
                    "cell": cell,
                    **level,
                    "difference_vs_weight_1": difference["estimate"],
                    "difference_ci_95_low": difference["ci_95_percentile_low"],
                    "difference_ci_95_high": difference["ci_95_percentile_high"],
                }
            )

    metrics_long = pd.DataFrame(metric_rows)
    summary = metrics_long.pivot(
        index="scale_pos_weight", columns="metric", values="estimate"
    ).reset_index()
    summary = summary[["scale_pos_weight", *METRICS]]
    return summary, metrics_long, pd.DataFrame(confusion_rows), pd.DataFrame(seed_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--weights", nargs="+", type=float, default=list(DEFAULT_WEIGHTS))
    args = parser.parse_args()

    weights = tuple(float(weight) for weight in args.weights)
    seeds = tuple(int(seed) for seed in args.seeds)
    if weights[0] != 1.0 or any(weight <= 0 for weight in weights):
        raise ValueError("The first weight must be the positive baseline 1.0")
    if len(set(weights)) != len(weights) or len(set(seeds)) != len(seeds):
        raise ValueError("Weights and seeds must be unique")
    if args.output.exists():
        raise FileExistsError("Choose a new output directory to preserve prior runs")
    args.output.mkdir(parents=True)

    features, labels, data_audit = load_data(args.data)
    predictions = run_predictions(features, labels, weights, seeds)
    summary, metrics_long, confusion, seed_metrics = summarize_results(
        predictions, weights, args.bootstrap
    )

    predictions.to_csv(args.output / "predictions.csv", index=False)
    summary.to_csv(args.output / "metric_summary.csv", index=False)
    metrics_long.to_csv(args.output / "metric_intervals.csv", index=False)
    confusion.to_csv(args.output / "confusion_matrix_intervals.csv", index=False)
    seed_metrics.to_csv(args.output / "seed_metrics.csv", index=False)
    details = {
        "purpose": "XGBoost scale_pos_weight sensitivity analysis",
        "weights": list(weights),
        "approximate_inverse_frequency_weight": float((labels == 0).sum() / (labels == 1).sum()),
        "seeds": list(seeds),
        "outer_folds": 5,
        "inner_folds": 5,
        "bootstrap_repetitions": args.bootstrap,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_interpretation": "conditional on completed repeated nested-CV predictions",
        "threshold_rule": "inner OOF Youden J within each outer fold",
        "data_audit": data_audit,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {
            package: importlib.metadata.version(package)
            for package in ["numpy", "pandas", "scikit-learn", "xgboost"]
        },
    }
    (args.output / "details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "COMPLETE.txt").write_text(
        "Completed class-weight sensitivity analysis.\n", encoding="utf-8"
    )
    print("\nRepeated-CV mean metrics")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
