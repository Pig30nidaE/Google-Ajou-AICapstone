"""평가 지표.

주 평가 단위는 **피험자**다 (사용자 지시 10절). 기록 단위는 보조 결과다.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ..data.schema import CLASS_ORDER, CODE_TO_CLASS

log = logging.getLogger(__name__)

__all__ = ["compute_metrics", "multiclass_brier_score", "METRIC_COLUMNS"]

METRIC_COLUMNS = (
    "macro_f1",
    "balanced_accuracy",
    "macro_roc_auc_ovr",
    "macro_pr_auc",
    "accuracy",
    "weighted_f1",
    "log_loss",
    "brier",
)


def multiclass_brier_score(y_true: np.ndarray, proba: np.ndarray, n_classes: int = 3) -> float:
    """다중 클래스 Brier score (원-핫과 확률의 평균 제곱 오차 합)."""
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _safe_auc(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, list[str]]:
    """macro OvR ROC-AUC. 평가셋에 없는 클래스는 제외하고 그 사실을 반환한다."""
    present = sorted(set(int(v) for v in y_true))
    skipped = [CODE_TO_CLASS[c] for c in range(proba.shape[1]) if c not in present]
    if len(present) < 2:
        return float("nan"), skipped
    if len(present) == proba.shape[1]:
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro")), skipped
    aucs = []
    for c in present:
        binary = (y_true == c).astype(int)
        if 0 < binary.sum() < len(binary):
            aucs.append(roc_auc_score(binary, proba[:, c]))
    return (float(np.mean(aucs)) if aucs else float("nan")), skipped


def _safe_pr_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    present = sorted(set(int(v) for v in y_true))
    aps = []
    for c in present:
        binary = (y_true == c).astype(int)
        if 0 < binary.sum() < len(binary):
            aps.append(average_precision_score(binary, proba[:, c]))
    return float(np.mean(aps)) if aps else float("nan")


def compute_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    unit: str = "subject",
    n_classes: int = 3,
) -> dict:
    """전 지표를 한 번에 계산한다.

    Args:
        y_true: (n,) 정수 라벨.
        proba: (n, 3) 클래스 확률.
        unit: "subject" 또는 "record". 결과 dict에 기록된다.

    Returns:
        주·보조 지표와 클래스별 P/R/F1, 혼동행렬, Dem 전용 수치.
    """
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    y_pred = np.argmax(proba, axis=1)

    auc, auc_skipped = _safe_auc(y_true, proba)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    out: dict = {
        "unit": unit,
        "n": int(len(y_true)),
        # ---- 주 지표
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_roc_auc_ovr": auc,
        "macro_pr_auc": _safe_pr_auc(y_true, proba),
        # ---- 보조 지표
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "brier": multiclass_brier_score(y_true, proba, n_classes),
        "confusion_matrix": cm.tolist(),
        "auc_skipped_classes": auc_skipped,
    }
    try:
        out["log_loss"] = float(log_loss(y_true, proba, labels=list(range(n_classes))))
    except ValueError:
        out["log_loss"] = float("nan")

    for code, cls in CODE_TO_CLASS.items():
        out[f"{cls}_precision"] = float(p[code])
        out[f"{cls}_recall"] = float(r[code])
        out[f"{cls}_f1"] = float(f1[code])
        out[f"n_{cls}"] = int(support[code])
        out[f"n_{cls}_correct"] = int(cm[code, code])

    # Dem은 사용자 지시 10절에 따라 별도로 강조한다.
    out["dem_recall"] = out["Dem_recall"]
    out["dem_precision"] = out["Dem_precision"]
    out["dem_f1"] = out["Dem_f1"]
    out["n_dem_units"] = out["n_Dem"]
    out["n_dem_units_correct"] = out["n_Dem_correct"]
    out["class_order"] = list(CLASS_ORDER)
    return out
