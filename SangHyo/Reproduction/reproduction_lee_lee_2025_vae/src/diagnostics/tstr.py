"""TSTR / TRTS 진단 (synthetic_data_risk.md §3.4).

TSTR (Train on Synthetic, Test on Real)
    합성 Dem + 실제 CN/MCI로 학습하고 **실제 outer test**로 평가한다. 주 진단.

TRTS (Train on Real, Test on Synthetic)
    실제 train으로 학습하고 합성 Dem으로 평가한다.
    **보조 진단만** — 암기해도 높게 나오므로 품질 근거가 되지 못한다 (사용자 지시 9절 9번).
"""

from __future__ import annotations

import logging

import numpy as np

from ..audit.leakage import LeakageAuditor
from ..data.loader import LifelogData
from ..data.schema import CLASS_TO_CODE
from ..evaluation.aggregate import aggregate_to_subject
from ..evaluation.metrics import compute_metrics
from ..models.registry import fit_classifier

log = logging.getLogger(__name__)

__all__ = ["run_tstr", "run_trts"]

TRTS_CAVEAT = (
    "TRTS는 보조 진단이다. 합성자료가 학습 표본을 암기한 경우에도 높게 나오므로 "
    "합성자료 품질의 근거로 쓸 수 없다."
)


def run_tstr(
    augmented_train: LifelogData,
    test: LifelogData,
    cfg: dict,
    *,
    model_name: str,
    auditor: LeakageAuditor,
    fold_id: str,
    seed: int = 42,
) -> dict:
    """소수 클래스를 합성행으로만 대체해 학습하고 실제 test로 평가한다."""
    target = CLASS_TO_CODE[(cfg.get("augmentation") or {}).get("target_class", "Dem")]
    keep = (augmented_train.y != target) | augmented_train.is_synthetic
    subset = augmented_train.take(keep)
    n_syn = int(subset.is_synthetic.sum())
    if n_syn == 0:
        return {"skipped": True, "reason": "합성행이 없다"}
    log.info(
        "[%s] TSTR: 실제 Dem 제거 후 합성 %d행으로 학습 -> 실제 test %d행 평가",
        fold_id, n_syn, test.n,
    )
    model = fit_classifier(
        model_name, subset, cfg, auditor=auditor, fold_id=fold_id, seed=seed
    )
    proba = model.predict_proba(test.X.to_numpy())
    auditor.record_eval(fold_id, is_synthetic=test.is_synthetic, where="tstr_test")
    subj = aggregate_to_subject(
        test.subject, test.y, proba,
        is_synthetic=test.is_synthetic,
        method=(cfg.get("aggregate") or {}).get("method", "mean"),
    )
    return {
        "skipped": False,
        "n_synthetic_train_rows": n_syn,
        "record_level": compute_metrics(test.y, proba, unit="record"),
        "subject_level": compute_metrics(subj.y, subj.proba, unit="subject"),
    }


def run_trts(
    real_train: LifelogData,
    synthetic_X,
    cfg: dict,
    *,
    model_name: str,
    auditor: LeakageAuditor,
    fold_id: str,
    seed: int = 42,
) -> dict:
    """실제 train으로 학습하고 합성 Dem으로 평가한다 (보조 진단)."""
    target = CLASS_TO_CODE[(cfg.get("augmentation") or {}).get("target_class", "Dem")]
    if synthetic_X is None or len(synthetic_X) == 0:
        return {"skipped": True, "reason": "합성행이 없다", "caveat": TRTS_CAVEAT}
    model = fit_classifier(
        model_name, real_train.real_only(), cfg, auditor=auditor, fold_id=fold_id, seed=seed
    )
    proba = model.predict_proba(np.asarray(synthetic_X, dtype=float))
    y_syn = np.full(len(synthetic_X), target, dtype=int)
    pred = np.argmax(proba, axis=1)
    return {
        "skipped": False,
        "n_synthetic": int(len(synthetic_X)),
        "accuracy_on_synthetic": float((pred == y_syn).mean()),
        "mean_target_proba": float(proba[:, target].mean()),
        "caveat": TRTS_CAVEAT,
    }
