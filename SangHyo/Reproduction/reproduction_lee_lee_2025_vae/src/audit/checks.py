"""개별 누수 검사 함수.

각 함수는 순수 함수이며 위반 목록(``list[Violation]``)을 반환한다.
예외를 던지는 것은 ``LeakageAuditor``의 책임이다 — 이렇게 분리하면
unit test가 검사 로직을 직접 호출할 수 있고, 실험 A의 ``observe`` 모드에서
위반을 "측정만" 할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from ..data.schema import (
    SYNTHETIC_SUBJECT_SENTINEL,
    FORBIDDEN_COLUMNS,
    FORBIDDEN_PATTERNS,
)

__all__ = [
    "Violation",
    "check_subject_overlap",
    "check_row_overlap",
    "check_fit_scope",
    "check_fit_row_scope",
    "check_vae_fit_scope",
    "check_no_synthetic_in_eval",
    "check_synthetic_source_subjects",
    "check_forbidden_features",
    "check_early_stopping_scope",
    "check_selection_scope",
    "check_preprocessing_after_split",
    "check_subject_aggregation_excludes_synthetic",
]


@dataclass
class Violation:
    """누수 검사 위반 1건."""

    code: str
    message: str
    fold_id: str | None = None
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - 표시용
        where = f"[{self.fold_id}] " if self.fold_id else ""
        return f"{where}{self.code}: {self.message}"


def _to_set(values: Iterable) -> set:
    return {v for v in np.asarray(list(values), dtype=object).ravel()}


# --------------------------------------------------------------------------------------
def check_subject_overlap(
    train_subjects: Iterable, eval_subjects: Iterable, *, fold_id: str | None = None
) -> list[Violation]:
    """train과 eval의 피험자 ID 교집합이 비어 있어야 한다."""
    tr = _to_set(train_subjects) - {SYNTHETIC_SUBJECT_SENTINEL}
    ev = _to_set(eval_subjects) - {SYNTHETIC_SUBJECT_SENTINEL}
    shared = tr & ev
    if shared:
        return [
            Violation(
                "SUBJECT_OVERLAP",
                f"train/eval에 동일 피험자 {len(shared)}명이 있다",
                fold_id,
                {"n_shared": len(shared), "n_train": len(tr), "n_eval": len(ev)},
            )
        ]
    return []


def check_row_overlap(
    train_row_ids: Iterable, eval_row_ids: Iterable, *, fold_id: str | None = None
) -> list[Violation]:
    """train과 eval의 원시 row가 겹치면 안 된다."""
    tr, ev = set(np.asarray(list(train_row_ids)).ravel()), set(np.asarray(list(eval_row_ids)).ravel())
    shared = tr & ev
    if shared:
        return [
            Violation(
                "ROW_OVERLAP",
                f"train/eval에 동일 row가 {len(shared)}건 있다",
                fold_id,
                {"n_shared": len(shared)},
            )
        ]
    return []


def check_fit_scope(
    component: str,
    fit_subjects: Iterable,
    train_subjects: Iterable,
    *,
    fold_id: str | None = None,
) -> list[Violation]:
    """전처리기가 train 피험자 밖의 자료를 보지 않았는지 확인한다.

    scaler / imputer / outlier detector 모두에 동일하게 적용된다 (사용자 지시 9).
    """
    fit = _to_set(fit_subjects) - {SYNTHETIC_SUBJECT_SENTINEL}
    tr = _to_set(train_subjects) - {SYNTHETIC_SUBJECT_SENTINEL}
    outside = fit - tr
    if outside:
        return [
            Violation(
                "FIT_SCOPE",
                f"'{component}'가 train 밖 피험자 {len(outside)}명의 자료로 fit되었다",
                fold_id,
                {"component": component, "n_outside": len(outside)},
            )
        ]
    return []


def check_fit_row_scope(
    component: str,
    fit_row_ids: Iterable,
    train_row_ids: Iterable,
    *,
    fold_id: str | None = None,
) -> list[Violation]:
    """전처리/VAE fit 원시행이 등록된 train 행의 부분집합인지 확인한다.

    행 단위 split에서는 같은 피험자가 train과 eval에 모두 있어 subject-set 검사만으로
    all-data fit을 놓칠 수 있다. 원시 row ID 검사는 그 사각지대를 닫는다.
    """
    fit = set(np.asarray(list(fit_row_ids)).ravel().tolist())
    # 합성행은 음수 sentinel row ID를 가지며 원시 eval 행을 뜻하지 않는다.
    fit = {
        row_id for row_id in fit
        if not isinstance(row_id, (int, np.integer)) or int(row_id) >= 0
    }
    train = set(np.asarray(list(train_row_ids)).ravel().tolist())
    outside = fit - train
    if outside:
        return [
            Violation(
                "FIT_ROW_SCOPE",
                f"'{component}'가 train 밖 원시행 {len(outside)}건으로 fit되었다",
                fold_id,
                {"component": component, "n_outside_rows": len(outside)},
            )
        ]
    return []


def check_vae_fit_scope(
    fit_subjects: Iterable,
    fit_labels: Iterable,
    train_subjects: Iterable,
    *,
    expected_label: int | None = None,
    fold_id: str | None = None,
) -> list[Violation]:
    """VAE가 현재 fold의 train 자료로만, 지정된 클래스에서만 학습되었는지 확인한다.

    (사용자 지시 9·10 — 평가 자료로 합성자료를 생성하면 안 된다.)
    """
    out = check_fit_scope("vae", fit_subjects, train_subjects, fold_id=fold_id)
    if out:
        out[0].code = "VAE_FIT_SCOPE"
    if expected_label is not None:
        labels = set(int(v) for v in np.asarray(list(fit_labels)).ravel())
        wrong = labels - {int(expected_label)}
        if wrong:
            out.append(
                Violation(
                    "VAE_FIT_LABEL",
                    f"VAE 학습자료에 기대 클래스({expected_label}) 밖 라벨 {sorted(wrong)}가 있다",
                    fold_id,
                    {"labels": sorted(labels)},
                )
            )
    return out


def check_no_synthetic_in_eval(
    is_synthetic: Sequence[bool], *, fold_id: str | None = None, where: str = "eval"
) -> list[Violation]:
    """평가셋에 합성행이 단 한 건도 없어야 한다 (사용자 지시 11)."""
    arr = np.asarray(is_synthetic, dtype=bool)
    n = int(arr.sum())
    if n:
        return [
            Violation(
                "SYNTHETIC_IN_EVAL",
                f"{where}에 합성행이 {n}건 포함되었다",
                fold_id,
                {"n_synthetic": n, "where": where},
            )
        ]
    return []


def check_synthetic_source_subjects(
    source_subjects: Iterable, train_subjects: Iterable, *, fold_id: str | None = None
) -> list[Violation]:
    """합성행 생성에 평가 fold의 피험자가 쓰이지 않았는지 확인한다."""
    out = check_fit_scope("synthetic_generator", source_subjects, train_subjects, fold_id=fold_id)
    if out:
        out[0].code = "SYNTHETIC_SOURCE_SCOPE"
    return out


def check_forbidden_features(columns: Iterable[str]) -> list[Violation]:
    """feature에 subject ID·MMSE·진단 파생이 없는지 확인한다 (사용자 지시 12·13)."""
    bad = []
    for col in columns:
        if col in FORBIDDEN_COLUMNS:
            bad.append(col)
            continue
        if any(p.search(col) for p in FORBIDDEN_PATTERNS):
            bad.append(col)
    if bad:
        return [
            Violation(
                "FORBIDDEN_FEATURE",
                f"입력에 금지 변수가 있다: {bad}",
                None,
                {"columns": bad},
            )
        ]
    return []


def check_early_stopping_scope(
    es_subjects: Iterable, train_subjects: Iterable, *, fold_id: str | None = None
) -> list[Violation]:
    """early stopping이 train 내부 validation에서만 이루어졌는지 확인한다."""
    out = check_fit_scope("early_stopping", es_subjects, train_subjects, fold_id=fold_id)
    if out:
        out[0].code = "EARLY_STOPPING_SCOPE"
        out[0].message = "early stopping이 train 밖 자료(=outer test)를 참조했다"
    return out


def check_selection_scope(
    what: str, selection_subjects: Iterable, train_subjects: Iterable, *, fold_id: str | None = None
) -> list[Violation]:
    """하이퍼파라미터·임계값 선택이 outer test를 참조하지 않았는지 확인한다.

    사용자 지시 11의 "test 성능으로 이상치 임계값/latent dimension을 선택했는지"에 해당한다.
    """
    out = check_fit_scope(f"selection:{what}", selection_subjects, train_subjects, fold_id=fold_id)
    if out:
        out[0].code = "SELECTION_SCOPE"
        out[0].message = f"'{what}' 선택에 train 밖 자료(=outer test)가 사용되었다"
    return out


def check_preprocessing_after_split(
    split_registered: bool, component: str, *, fold_id: str | None = None
) -> list[Violation]:
    """전처리 fit이 split 등록 **이후**에 일어났는지 확인한다.

    split을 등록하지 않은 채 fit이 호출되면 "split 전에 전처리했다"는 뜻이다.
    """
    if not split_registered:
        return [
            Violation(
                "PREPROCESSING_BEFORE_SPLIT",
                f"'{component}'가 split 등록 전에 fit되었다",
                fold_id,
                {"component": component},
            )
        ]
    return []


def check_subject_aggregation_excludes_synthetic(
    subjects: Iterable, *, fold_id: str | None = None
) -> list[Violation]:
    """피험자 단위 집계 대상에 합성행 센티널이 섞이지 않았는지 확인한다.

    사용자 지시 15·16 — 합성 row를 독립 피험자로 취급하면 안 된다.
    """
    arr = np.asarray(list(subjects), dtype=object)
    n = int((arr == SYNTHETIC_SUBJECT_SENTINEL).sum())
    if n:
        return [
            Violation(
                "SYNTHETIC_AS_SUBJECT",
                f"피험자 단위 집계에 합성행 {n}건이 포함되었다",
                fold_id,
                {"n_synthetic": n},
            )
        ]
    return []
