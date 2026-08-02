"""기록 단위 예측을 피험자 단위로 집계한다.

주 평가 단위는 피험자다. 합성행은 집계 대상에서 **구조적으로 배제**된다
(synthetic_data_risk.md §2 금지 2·5).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..audit.checks import check_subject_aggregation_excludes_synthetic
from ..data.schema import SYNTHETIC_SUBJECT_SENTINEL

log = logging.getLogger(__name__)

__all__ = ["SubjectPredictions", "aggregate_to_subject"]


class SubjectPredictions:
    """피험자 단위 예측 결과."""

    __slots__ = ("subject", "y", "proba", "n_records")

    def __init__(self, subject: np.ndarray, y: np.ndarray, proba: np.ndarray, n_records: np.ndarray):
        self.subject, self.y, self.proba, self.n_records = subject, y, proba, n_records

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.proba, columns=[f"proba_{i}" for i in range(self.proba.shape[1])])
        df.insert(0, "n_records", self.n_records)
        df.insert(0, "y_true", self.y)
        df.insert(0, "subject", self.subject)
        df["y_pred"] = np.argmax(self.proba, axis=1)
        return df


def aggregate_to_subject(
    subject: np.ndarray,
    y: np.ndarray,
    proba: np.ndarray,
    *,
    is_synthetic: np.ndarray | None = None,
    method: str = "mean",
) -> SubjectPredictions:
    """일별 예측확률을 피험자별로 집계한다.

    Args:
        method: ``mean`` (기본, 산술평균) | ``median`` | ``logit_mean`` | ``majority_vote``.

    Raises:
        ValueError: 합성행이 집계 대상에 남아 있는 경우.
    """
    subject = np.asarray(subject, dtype=object)
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)

    if is_synthetic is not None:
        keep = ~np.asarray(is_synthetic, dtype=bool)
        if not keep.all():
            log.info("피험자 집계에서 합성행 %d건을 제외한다", int((~keep).sum()))
        subject, y, proba = subject[keep], y[keep], proba[keep]

    violations = check_subject_aggregation_excludes_synthetic(subject)
    if violations:
        raise ValueError(str(violations[0]))
    if (subject == SYNTHETIC_SUBJECT_SENTINEL).any():  # pragma: no cover - 위에서 이미 차단
        raise ValueError("합성행이 피험자 집계에 남아 있다")

    df = pd.DataFrame(proba, columns=[f"p{i}" for i in range(proba.shape[1])])
    df["subject"], df["y"] = subject, y
    pcols = [c for c in df.columns if c.startswith("p")]

    if method == "mean":
        agg = df.groupby("subject", sort=True)[pcols].mean()
    elif method == "median":
        agg = df.groupby("subject", sort=True)[pcols].median()
    elif method == "logit_mean":
        eps = 1e-12
        logits = np.log(np.clip(proba, eps, 1.0))
        tmp = pd.DataFrame(logits, columns=pcols)
        tmp["subject"] = subject
        agg = tmp.groupby("subject", sort=True)[pcols].mean()
        e = np.exp(agg.to_numpy() - agg.to_numpy().max(axis=1, keepdims=True))
        agg = pd.DataFrame(e / e.sum(axis=1, keepdims=True), index=agg.index, columns=pcols)
    elif method == "majority_vote":
        hard = np.zeros_like(proba)
        hard[np.arange(len(proba)), np.argmax(proba, axis=1)] = 1.0
        tmp = pd.DataFrame(hard, columns=pcols)
        tmp["subject"] = subject
        agg = tmp.groupby("subject", sort=True)[pcols].mean()
    else:
        raise ValueError(f"unknown aggregate method {method!r}")

    labels = df.groupby("subject", sort=True)["y"].first()
    counts = df.groupby("subject", sort=True)["y"].size()
    order = agg.index.to_numpy()
    p = agg.to_numpy()
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    return SubjectPredictions(order, labels.to_numpy(), p, counts.to_numpy())
