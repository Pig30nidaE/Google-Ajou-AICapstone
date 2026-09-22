"""Prespecified subject-level metrics and conditional bootstrap intervals."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    recall_score, roc_auc_score,
)


def metrics(y, probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(y), 3) or not np.isfinite(probabilities).all():
        raise ValueError("Expected finite N x 3 probabilities")
    if (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Invalid probability simplex")
    if set(np.unique(y)) != {0, 1, 2}:
        raise ValueError("Every evaluation split must contain all three classes")
    pred = probabilities.argmax(axis=1)
    recall = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_ovr_auc": float(roc_auc_score(y, probabilities, labels=[0, 1, 2], multi_class="ovr", average="macro")),
        "recall_CN": float(recall[0]), "recall_MCI": float(recall[1]),
        "recall_Dementia": float(recall[2]),
    }, confusion_matrix(y, pred, labels=[0, 1, 2])


def bootstrap_intervals(y, probabilities, repetitions=2000, seed=7319):
    if repetitions < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    point, _ = metrics(y, probabilities)
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(y == c) for c in (0, 1, 2)]
    values = {key: [] for key in point}
    for _ in range(repetitions):
        sample = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in strata])
        scores, _ = metrics(y[sample], probabilities[sample])
        for key in values:
            values[key].append(scores[key])
    return [{"metric": key, "estimate": point[key],
             "ci_low": float(np.quantile(value, .025)), "ci_high": float(np.quantile(value, .975)),
             "repetitions": repetitions, "method": "subject_stratified_percentile",
             "scope": "conditional_on_fixed_oof_predictions"}
            for key, value in values.items()]
