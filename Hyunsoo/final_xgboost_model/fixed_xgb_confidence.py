"""Conditional subject-level bootstrap intervals for the frozen V44 XGBoost model.

The same resampled subject indices are used across all repeated cross-validation
partitions. This preserves the subject as the statistical unit and avoids
treating the five predictions per subject as independent observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


MODEL = "xgb_all"
SMOTE = False
EXPECTED_SEEDS = (42, 13, 73, 101, 2026)
EXPECTED_SUBJECTS = 174
BOOTSTRAP_SEED = 20260919
DEFAULT_REPETITIONS = 10_000


def load_fixed_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return labels, score matrix, class matrix, and ordered CV seeds."""
    frame = pd.read_csv(path)
    required = {
        "seed", "fold", "smote", "model", "subject_row", "y", "score",
        "prediction", "threshold",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")

    chosen = frame[(frame["model"] == MODEL) & (frame["smote"] == SMOTE)].copy()
    if chosen.empty:
        raise ValueError(f"No rows found for model={MODEL!r}, smote={SMOTE}")
    if chosen.duplicated(["subject_row", "seed"]).any():
        raise ValueError("Expected exactly one out-of-fold prediction per subject and seed")
    expected_class = (chosen["score"] >= chosen["threshold"]).astype(int)
    if not expected_class.equals(chosen["prediction"].astype(int)):
        raise ValueError("Stored classes do not match the stored nested-CV thresholds")

    seeds = np.array(sorted(chosen["seed"].unique()))
    if tuple(seeds) != tuple(sorted(EXPECTED_SEEDS)):
        raise ValueError(f"Unexpected split seeds: {seeds.tolist()}")

    score_frame = chosen.pivot(index="subject_row", columns="seed", values="score")
    class_frame = chosen.pivot(index="subject_row", columns="seed", values="prediction")
    label_frame = chosen.pivot(index="subject_row", columns="seed", values="y")
    if len(score_frame) != EXPECTED_SUBJECTS or score_frame.isna().any().any():
        raise ValueError("Incomplete repeated out-of-fold score matrix")
    if class_frame.isna().any().any() or label_frame.isna().any().any():
        raise ValueError("Incomplete repeated out-of-fold class or label matrix")
    if not label_frame.nunique(axis=1).eq(1).all():
        raise ValueError("A subject has inconsistent labels across CV repetitions")

    labels = label_frame.iloc[:, 0].to_numpy(dtype=int)
    if set(labels) != {0, 1}:
        raise ValueError("Both binary outcome classes are required")
    return (
        labels,
        score_frame[seeds].to_numpy(dtype=float),
        class_frame[seeds].to_numpy(dtype=int),
        seeds,
    )


def auc_from_fixed_class_bootstrap(
    sampled_scores: np.ndarray,
    negative_count: int,
    positive_count: int,
) -> np.ndarray:
    """Vectorized AUC; samples contain all negatives followed by all positives."""
    ranks = rankdata(sampled_scores, axis=1, method="average")
    positive_rank_sum = ranks[:, negative_count:].sum(axis=1)
    minimum_positive_rank_sum = positive_count * (positive_count + 1) / 2
    return (
        positive_rank_sum - minimum_positive_rank_sum
    ) / (negative_count * positive_count)


def summarize(values: np.ndarray, observed: float) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise ValueError("No finite bootstrap values to summarize")
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return {
        "estimate": float(observed),
        "bootstrap_median": float(np.median(finite)),
        "bootstrap_standard_error": float(np.std(finite, ddof=1)),
        "ci_95_percentile_low": float(lower),
        "ci_95_percentile_high": float(upper),
        "valid_bootstrap_repetitions": int(len(finite)),
    }


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide while retaining undefined diagnostic metrics as NaN."""
    return np.divide(
        numerator,
        denominator,
        out=np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan),
        where=denominator != 0,
    )


def calculate_intervals(
    labels: np.ndarray,
    scores: np.ndarray,
    predicted: np.ndarray,
    seeds: np.ndarray,
    repetitions: int,
) -> dict:
    """Compute repeated-CV mean and seed-specific conditional intervals."""
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.column_stack(
        [
            rng.choice(negative, size=(repetitions, len(negative)), replace=True),
            rng.choice(positive, size=(repetitions, len(positive)), replace=True),
        ]
    )

    seed_auc_draws = np.empty((repetitions, len(seeds)))
    seed_ap_draws = np.empty((repetitions, len(seeds)))
    seed_brier_draws = np.empty((repetitions, len(seeds)))
    observed_auc = np.empty(len(seeds))
    observed_ap = np.empty(len(seeds))
    observed_brier = np.empty(len(seeds))
    sampled_labels = labels[samples]
    for column in range(len(seeds)):
        sampled_scores = scores[samples, column]
        seed_auc_draws[:, column] = auc_from_fixed_class_bootstrap(
            sampled_scores, len(negative), len(positive)
        )
        seed_ap_draws[:, column] = np.fromiter(
            (
                average_precision_score(sampled_labels[row], sampled_scores[row])
                for row in range(repetitions)
            ),
            dtype=float,
            count=repetitions,
        )
        seed_brier_draws[:, column] = np.mean(
            (sampled_scores - sampled_labels) ** 2, axis=1
        )
        observed_auc[column] = roc_auc_score(labels, scores[:, column])
        observed_ap[column] = average_precision_score(labels, scores[:, column])
        observed_brier[column] = np.mean((scores[:, column] - labels) ** 2)

    # Verify the vectorized rank-sum implementation against sklearn on a sample.
    crosscheck_labels = labels[samples[0]]
    for column in range(len(seeds)):
        expected = roc_auc_score(crosscheck_labels, scores[samples[0], column])
        if not np.isclose(seed_auc_draws[0, column], expected, atol=1e-12):
            raise AssertionError("Vectorized bootstrap AUC failed sklearn cross-check")

    repeated_auc_draws = seed_auc_draws.mean(axis=1)
    result = {
        "method": {
            "name": "stratified subject-level percentile bootstrap",
            "statistical_unit": "subject",
            "repetitions": repetitions,
            "random_seed": BOOTSTRAP_SEED,
            "resampling": (
                "111 negative and 63 positive subjects sampled with replacement; "
                "the same sampled subject indices are used in all five CV repetitions"
            ),
            "interpretation": (
                "conditional on the completed repeated-CV predictions; no model refitting "
                "and no correction for feature/model selection"
            ),
        },
        "frozen_candidate": {
            "model": MODEL,
            "smote": SMOTE,
            "feature_count": 14,
            "subject_count": int(len(labels)),
            "class_counts": {"negative": int(len(negative)), "positive": int(len(positive))},
            "cv_seeds": [int(seed) for seed in seeds],
        },
        "primary_repeated_cv_mean_auc": summarize(
            repeated_auc_draws, float(observed_auc.mean())
        ),
        "threshold_free_metrics": {
            "roc_auc": summarize(repeated_auc_draws, float(observed_auc.mean())),
            "average_precision": summarize(
                seed_ap_draws.mean(axis=1), float(observed_ap.mean())
            ),
            "brier_score_lower_is_better": summarize(
                seed_brier_draws.mean(axis=1), float(observed_brier.mean())
            ),
            "brier_skill_score_vs_prevalence_only": summarize(
                1 - seed_brier_draws.mean(axis=1) / (labels.mean() * (1 - labels.mean())),
                float(1 - observed_brier.mean() / (labels.mean() * (1 - labels.mean()))),
            ),
            "positive_prevalence": float(labels.mean()),
        },
        "seed_specific_auc": {
            str(seed): summarize(seed_auc_draws[:, column], observed_auc[column])
            for column, seed in enumerate(seeds)
        },
    }

    # Threshold-dependent metrics use each outer fold's nested-CV threshold.
    sampled_classes = predicted[samples, :]
    true_positive = sampled_classes[:, len(negative):, :].sum(axis=1)
    false_positive = sampled_classes[:, :len(negative), :].sum(axis=1)
    false_negative = len(positive) - true_positive
    true_negative = len(negative) - false_positive
    metric_draws = {
        "accuracy": ((true_positive + true_negative) / len(labels)).mean(axis=1),
        "recall": (true_positive / len(positive)).mean(axis=1),
        "specificity": (true_negative / len(negative)).mean(axis=1),
        "precision_ppv": np.nanmean(
            safe_ratio(true_positive, true_positive + false_positive), axis=1
        ),
        "negative_predictive_value": np.nanmean(safe_ratio(
            true_negative, true_negative + false_negative
        ), axis=1),
        "balanced_accuracy": (
            true_positive / len(positive) + true_negative / len(negative)
        ).mean(axis=1) / 2,
        "f1": (2 * true_positive / (2 * true_positive + false_positive + false_negative)).mean(axis=1),
        "matthews_correlation_coefficient": np.nanmean(safe_ratio(
            true_positive * true_negative - false_positive * false_negative,
            np.sqrt(
                (true_positive + false_positive)
                * (true_positive + false_negative)
                * (true_negative + false_positive)
                * (true_negative + false_negative)
            ),
        ), axis=1),
        "positive_likelihood_ratio": np.nanmean(safe_ratio(
            true_positive / len(positive), 1 - true_negative / len(negative)
        ), axis=1),
        "negative_likelihood_ratio": np.nanmean(safe_ratio(
            1 - true_positive / len(positive), true_negative / len(negative)
        ), axis=1),
    }
    observed_true_positive = ((predicted == 1) & (labels[:, None] == 1)).sum(axis=0)
    observed_false_positive = ((predicted == 1) & (labels[:, None] == 0)).sum(axis=0)
    observed_false_negative = ((predicted == 0) & (labels[:, None] == 1)).sum(axis=0)
    observed_true_negative = ((predicted == 0) & (labels[:, None] == 0)).sum(axis=0)
    observed_metrics = {
        "accuracy": ((observed_true_positive + observed_true_negative) / len(labels)).mean(),
        "recall": (observed_true_positive / len(positive)).mean(),
        "specificity": (observed_true_negative / len(negative)).mean(),
        "precision_ppv": safe_ratio(
            observed_true_positive, observed_true_positive + observed_false_positive
        ).mean(),
        "negative_predictive_value": safe_ratio(
            observed_true_negative, observed_true_negative + observed_false_negative
        ).mean(),
        "balanced_accuracy": (
            observed_true_positive / len(positive) + observed_true_negative / len(negative)
        ).mean() / 2,
        "f1": (
            2 * observed_true_positive
            / (2 * observed_true_positive + observed_false_positive + observed_false_negative)
        ).mean(),
        "matthews_correlation_coefficient": safe_ratio(
            observed_true_positive * observed_true_negative
            - observed_false_positive * observed_false_negative,
            np.sqrt(
                (observed_true_positive + observed_false_positive)
                * (observed_true_positive + observed_false_negative)
                * (observed_true_negative + observed_false_positive)
                * (observed_true_negative + observed_false_negative)
            ),
        ).mean(),
        "positive_likelihood_ratio": safe_ratio(
            observed_true_positive / len(positive),
            1 - observed_true_negative / len(negative),
        ).mean(),
        "negative_likelihood_ratio": safe_ratio(
            1 - observed_true_positive / len(positive),
            observed_true_negative / len(negative),
        ).mean(),
    }
    result["secondary_threshold_metrics"] = {
        metric: summarize(draws, float(observed_metrics[metric]))
        for metric, draws in metric_draws.items()
    }
    result["confusion_matrix_by_cv_seed"] = {
        str(seed): {
            "true_negative": int(observed_true_negative[column]),
            "false_positive": int(observed_false_positive[column]),
            "false_negative": int(observed_false_negative[column]),
            "true_positive": int(observed_true_positive[column]),
        }
        for column, seed in enumerate(seeds)
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(__file__).parent / "results_original_weights" / "predictions.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "fixed_xgb_confidence.json",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path(__file__).parent / "fixed_xgb_metrics.csv",
    )
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    args = parser.parse_args()
    if args.repetitions < 1_000:
        raise ValueError("Use at least 1,000 bootstrap repetitions")

    labels, scores, predicted, seeds = load_fixed_predictions(args.predictions)
    results = calculate_intervals(labels, scores, predicted, seeds, args.repetitions)
    prediction_rows = pd.read_csv(args.predictions)
    chosen_rows = prediction_rows[
        (prediction_rows["model"] == MODEL) & (prediction_rows["smote"] == SMOTE)
    ]
    thresholds = chosen_rows[["seed", "fold", "threshold"]].drop_duplicates()
    if len(thresholds) != len(seeds) * 5:
        raise ValueError("Expected one nested-CV threshold for each of 25 outer folds")
    results["nested_cv_thresholds"] = {
        "selection_rule": "Youden J on inner out-of-fold predictions within each outer fold",
        "count": int(len(thresholds)),
        "minimum": float(thresholds["threshold"].min()),
        "median": float(thresholds["threshold"].median()),
        "maximum": float(thresholds["threshold"].max()),
        "note": "These are evaluation thresholds, not one frozen deployment cutoff.",
    }
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    metric_rows = []
    for metric, values in results["threshold_free_metrics"].items():
        if isinstance(values, dict):
            metric_rows.append({"metric_group": "threshold_free", "metric": metric, **values})
    for metric, values in results["secondary_threshold_metrics"].items():
        metric_rows.append({"metric_group": "threshold_dependent", "metric": metric, **values})
    pd.DataFrame(metric_rows).to_csv(args.metrics_output, index=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
