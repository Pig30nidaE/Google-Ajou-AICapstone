"""Paired metric comparison: fixed XGBoost versus the former 6-model ensemble."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from fixed_xgb_confidence import (
    BOOTSTRAP_SEED,
    DEFAULT_REPETITIONS,
    EXPECTED_SEEDS,
    EXPECTED_SUBJECTS,
    auc_from_fixed_class_bootstrap,
    safe_ratio,
    summarize,
)


MODEL_SPECS = {
    "xgboost_no_smote": {"model": "xgb_all", "smote": False},
    "rank6_ensemble_smote": {"model": "rank_6_reference", "smote": True},
}
COMPARABLE_METRICS = (
    "roc_auc",
    "average_precision",
    "accuracy",
    "balanced_accuracy",
    "recall_sensitivity",
    "specificity",
    "precision_ppv",
    "negative_predictive_value",
    "f1",
    "matthews_correlation_coefficient",
    "positive_likelihood_ratio",
    "negative_likelihood_ratio",
)


def load_model_predictions(
    frame: pd.DataFrame, model: str, smote: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chosen = frame[(frame["model"] == model) & (frame["smote"] == smote)].copy()
    if chosen.empty:
        raise ValueError(f"No rows for model={model}, smote={smote}")
    if chosen.duplicated(["subject_row", "seed"]).any():
        raise ValueError(f"Duplicate subject/seed predictions for {model}, smote={smote}")
    expected_class = (chosen["score"] >= chosen["threshold"]).astype(int)
    if not expected_class.equals(chosen["prediction"].astype(int)):
        raise ValueError(f"Stored predictions do not match thresholds for {model}")

    seeds = np.array(sorted(chosen["seed"].unique()))
    if tuple(seeds) != tuple(sorted(EXPECTED_SEEDS)):
        raise ValueError(f"Unexpected CV seeds for {model}: {seeds.tolist()}")
    score_frame = chosen.pivot(index="subject_row", columns="seed", values="score")
    class_frame = chosen.pivot(index="subject_row", columns="seed", values="prediction")
    label_frame = chosen.pivot(index="subject_row", columns="seed", values="y")
    if len(score_frame) != EXPECTED_SUBJECTS:
        raise ValueError(f"Unexpected subject count for {model}: {len(score_frame)}")
    if score_frame.isna().any().any() or class_frame.isna().any().any():
        raise ValueError(f"Incomplete repeated OOF predictions for {model}")
    if not label_frame.nunique(axis=1).eq(1).all():
        raise ValueError(f"Inconsistent repeated labels for {model}")
    return (
        label_frame.iloc[:, 0].to_numpy(dtype=int),
        score_frame[seeds].to_numpy(dtype=float),
        class_frame[seeds].to_numpy(dtype=int),
    )


def observed_metrics(labels: np.ndarray, scores: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    negative_count = int((labels == 0).sum())
    positive_count = int((labels == 1).sum())
    true_positive = ((predicted == 1) & (labels[:, None] == 1)).sum(axis=0)
    false_positive = ((predicted == 1) & (labels[:, None] == 0)).sum(axis=0)
    false_negative = positive_count - true_positive
    true_negative = negative_count - false_positive
    prevalence_brier = labels.mean() * (1 - labels.mean())
    brier_by_seed = np.mean((scores - labels[:, None]) ** 2, axis=0)

    return {
        "roc_auc": float(np.mean([
            roc_auc_score(labels, scores[:, column]) for column in range(scores.shape[1])
        ])),
        "average_precision": float(np.mean([
            average_precision_score(labels, scores[:, column])
            for column in range(scores.shape[1])
        ])),
        "brier_score": float(brier_by_seed.mean()),
        "brier_skill_score": float((1 - brier_by_seed / prevalence_brier).mean()),
        "accuracy": float(((true_positive + true_negative) / len(labels)).mean()),
        "balanced_accuracy": float((
            true_positive / positive_count + true_negative / negative_count
        ).mean() / 2),
        "recall_sensitivity": float((true_positive / positive_count).mean()),
        "specificity": float((true_negative / negative_count).mean()),
        "precision_ppv": float(np.nanmean(safe_ratio(
            true_positive, true_positive + false_positive
        ))),
        "negative_predictive_value": float(np.nanmean(safe_ratio(
            true_negative, true_negative + false_negative
        ))),
        "f1": float(np.mean(
            2 * true_positive / (2 * true_positive + false_positive + false_negative)
        )),
        "matthews_correlation_coefficient": float(np.nanmean(safe_ratio(
            true_positive * true_negative - false_positive * false_negative,
            np.sqrt(
                (true_positive + false_positive)
                * (true_positive + false_negative)
                * (true_negative + false_positive)
                * (true_negative + false_negative)
            ),
        ))),
        "positive_likelihood_ratio": float(np.nanmean(safe_ratio(
            true_positive / positive_count, 1 - true_negative / negative_count
        ))),
        "negative_likelihood_ratio": float(np.nanmean(safe_ratio(
            1 - true_positive / positive_count, true_negative / negative_count
        ))),
    }


def bootstrap_metric_draws(
    labels: np.ndarray,
    scores: np.ndarray,
    predicted: np.ndarray,
    samples: np.ndarray,
) -> dict[str, np.ndarray]:
    negative_count = int((labels == 0).sum())
    positive_count = int((labels == 1).sum())
    repetitions, sample_size = samples.shape
    sampled_labels = labels[samples]
    auc_by_seed = np.empty((repetitions, scores.shape[1]))
    ap_by_seed = np.empty_like(auc_by_seed)
    brier_by_seed = np.empty_like(auc_by_seed)

    for column in range(scores.shape[1]):
        sampled_scores = scores[samples, column]
        auc_by_seed[:, column] = auc_from_fixed_class_bootstrap(
            sampled_scores, negative_count, positive_count
        )
        ap_by_seed[:, column] = np.fromiter(
            (
                average_precision_score(sampled_labels[row], sampled_scores[row])
                for row in range(repetitions)
            ),
            dtype=float,
            count=repetitions,
        )
        brier_by_seed[:, column] = np.mean(
            (sampled_scores - sampled_labels) ** 2, axis=1
        )

    sampled_classes = predicted[samples, :]
    true_positive = sampled_classes[:, negative_count:, :].sum(axis=1)
    false_positive = sampled_classes[:, :negative_count, :].sum(axis=1)
    false_negative = positive_count - true_positive
    true_negative = negative_count - false_positive
    prevalence_brier = labels.mean() * (1 - labels.mean())

    return {
        "roc_auc": auc_by_seed.mean(axis=1),
        "average_precision": ap_by_seed.mean(axis=1),
        "brier_score": brier_by_seed.mean(axis=1),
        "brier_skill_score": (1 - brier_by_seed / prevalence_brier).mean(axis=1),
        "accuracy": ((true_positive + true_negative) / sample_size).mean(axis=1),
        "balanced_accuracy": (
            true_positive / positive_count + true_negative / negative_count
        ).mean(axis=1) / 2,
        "recall_sensitivity": (true_positive / positive_count).mean(axis=1),
        "specificity": (true_negative / negative_count).mean(axis=1),
        "precision_ppv": np.nanmean(safe_ratio(
            true_positive, true_positive + false_positive
        ), axis=1),
        "negative_predictive_value": np.nanmean(safe_ratio(
            true_negative, true_negative + false_negative
        ), axis=1),
        "f1": (
            2 * true_positive / (2 * true_positive + false_positive + false_negative)
        ).mean(axis=1),
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
            true_positive / positive_count, 1 - true_negative / negative_count
        ), axis=1),
        "negative_likelihood_ratio": np.nanmean(safe_ratio(
            1 - true_positive / positive_count, true_negative / negative_count
        ), axis=1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(__file__).parent / "results_original_weights" / "predictions.csv",
    )
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "xgb_vs_rank6_smote_metrics.csv",
    )
    parser.add_argument(
        "--details-output",
        type=Path,
        default=Path(__file__).parent / "xgb_vs_rank6_smote_details.json",
    )
    parser.add_argument(
        "--confusion-output",
        type=Path,
        default=Path(__file__).parent / "xgb_vs_rank6_smote_confusion_matrix.csv",
    )
    args = parser.parse_args()
    if args.repetitions < 1_000:
        raise ValueError("Use at least 1,000 bootstrap repetitions")

    frame = pd.read_csv(args.predictions)
    required = {
        "seed", "fold", "smote", "model", "subject_row", "y", "score",
        "prediction", "threshold",
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    model_data = {}
    reference_labels = None
    for label, spec in MODEL_SPECS.items():
        labels, scores, predicted = load_model_predictions(frame, **spec)
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(labels, reference_labels):
            raise ValueError("Models do not have identical subject labels/order")
        model_data[label] = (scores, predicted)
    assert reference_labels is not None

    negative = np.flatnonzero(reference_labels == 0)
    positive = np.flatnonzero(reference_labels == 1)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.column_stack([
        rng.choice(negative, size=(args.repetitions, len(negative)), replace=True),
        rng.choice(positive, size=(args.repetitions, len(positive)), replace=True),
    ])

    observed = {}
    draws = {}
    for label, (scores, predicted) in model_data.items():
        observed[label] = observed_metrics(reference_labels, scores, predicted)
        draws[label] = bootstrap_metric_draws(
            reference_labels, scores, predicted, samples
        )

    xgb_label = "xgboost_no_smote"
    ensemble_label = "rank6_ensemble_smote"
    rows = []
    for metric in COMPARABLE_METRICS:
        xgb_summary = summarize(draws[xgb_label][metric], observed[xgb_label][metric])
        ensemble_summary = summarize(
            draws[ensemble_label][metric], observed[ensemble_label][metric]
        )
        difference = draws[xgb_label][metric] - draws[ensemble_label][metric]
        difference_summary = summarize(
            difference,
            observed[xgb_label][metric] - observed[ensemble_label][metric],
        )
        rows.append({
            "metric": metric,
            "xgboost": xgb_summary["estimate"],
            "xgboost_ci_low": xgb_summary["ci_95_percentile_low"],
            "xgboost_ci_high": xgb_summary["ci_95_percentile_high"],
            "ensemble": ensemble_summary["estimate"],
            "ensemble_ci_low": ensemble_summary["ci_95_percentile_low"],
            "ensemble_ci_high": ensemble_summary["ci_95_percentile_high"],
            "xgboost_minus_ensemble": difference_summary["estimate"],
            "difference_ci_low": difference_summary["ci_95_percentile_low"],
            "difference_ci_high": difference_summary["ci_95_percentile_high"],
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.output, index=False)

    negative_count = int((reference_labels == 0).sum())
    confusion_draws = {}
    confusion_observed = {}
    for label, (_, predicted) in model_data.items():
        sampled_classes = predicted[samples, :]
        confusion_draws[label] = {
            "true_negative": (sampled_classes[:, :negative_count, :] == 0).sum(axis=1).mean(axis=1),
            "false_positive": (sampled_classes[:, :negative_count, :] == 1).sum(axis=1).mean(axis=1),
            "false_negative": (sampled_classes[:, negative_count:, :] == 0).sum(axis=1).mean(axis=1),
            "true_positive": (sampled_classes[:, negative_count:, :] == 1).sum(axis=1).mean(axis=1),
        }
        confusion_observed[label] = {
            "true_negative": float(((predicted == 0) & (reference_labels[:, None] == 0)).sum(axis=0).mean()),
            "false_positive": float(((predicted == 1) & (reference_labels[:, None] == 0)).sum(axis=0).mean()),
            "false_negative": float(((predicted == 0) & (reference_labels[:, None] == 1)).sum(axis=0).mean()),
            "true_positive": float(((predicted == 1) & (reference_labels[:, None] == 1)).sum(axis=0).mean()),
        }

    confusion_rows = []
    for cell in ["true_negative", "false_positive", "false_negative", "true_positive"]:
        xgb_summary = summarize(
            confusion_draws[xgb_label][cell], confusion_observed[xgb_label][cell]
        )
        ensemble_summary = summarize(
            confusion_draws[ensemble_label][cell], confusion_observed[ensemble_label][cell]
        )
        difference_summary = summarize(
            confusion_draws[xgb_label][cell] - confusion_draws[ensemble_label][cell],
            confusion_observed[xgb_label][cell] - confusion_observed[ensemble_label][cell],
        )
        confusion_rows.append({
            "cell": cell,
            "xgboost_mean_count": xgb_summary["estimate"],
            "xgboost_ci_low": xgb_summary["ci_95_percentile_low"],
            "xgboost_ci_high": xgb_summary["ci_95_percentile_high"],
            "ensemble_mean_count": ensemble_summary["estimate"],
            "ensemble_ci_low": ensemble_summary["ci_95_percentile_low"],
            "ensemble_ci_high": ensemble_summary["ci_95_percentile_high"],
            "xgboost_minus_ensemble": difference_summary["estimate"],
            "difference_ci_low": difference_summary["ci_95_percentile_low"],
            "difference_ci_high": difference_summary["ci_95_percentile_high"],
        })
    pd.DataFrame(confusion_rows).to_csv(args.confusion_output, index=False)

    details = {
        "comparison": "XGBoost without SMOTE minus 6-model reference-rank ensemble with SMOTE",
        "bootstrap_repetitions": args.repetitions,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "method": (
            "stratified subject-level paired percentile bootstrap; identical sampled "
            "subjects used for both models and all five repeated CV partitions"
        ),
        "difference_interpretation": (
            "positive favors XGBoost except brier_score and negative_likelihood_ratio, "
            "where lower values are preferable"
        ),
        "conditional_limit": (
            "conditional on stored OOF predictions; does not correct prior model/feature "
            "selection or include full model-refitting uncertainty"
        ),
        "excluded_metrics": {
            "brier_score_and_skill": (
                "not compared because the reference-rank ensemble output is a relative-rank "
                "score rather than a calibrated probability"
            )
        },
        "confusion_matrix_counts": (
            "mean count per 174-subject OOF evaluation across the five CV seeds; "
            "the same subjects are not treated as 870 independent people"
        ),
        "rows": comparison.to_dict(orient="records"),
    }
    args.details_output.write_text(json.dumps(details, indent=2) + "\n")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
