"""누수 감사기.

파이프라인의 각 단계가 감사기에 **자기 행위를 신고**하고, 감사기가 불변식을 검증한다.

두 가지 모드가 있다.

``enforce``
    위반 시 즉시 :class:`LeakageError`. 실험 B·C가 사용한다.

``observe``
    위반을 기록만 하고 진행한다. **실험 A 전용**이다.
    논문 절차는 설계상 누수를 포함하므로 (행 단위 분할, 전체 데이터 전처리),
    이를 오류로 막으면 재현 자체가 불가능하다. 대신 어떤 불변식이 몇 건
    위반되는지 **정량 측정해서 보고**하는 것이 실험 A의 산출물이다.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from . import checks
from .checks import Violation

log = logging.getLogger(__name__)

__all__ = ["LeakageAuditor", "LeakageError", "FoldRegistration"]


class LeakageError(RuntimeError):
    """enforce 모드에서 누수 불변식이 깨졌을 때."""

    def __init__(self, violations: Sequence[Violation]):
        self.violations = list(violations)
        body = "\n  - ".join(str(v) for v in violations)
        super().__init__(f"데이터 누수 검사 실패 ({len(violations)}건):\n  - {body}")


@dataclass
class FoldRegistration:
    """한 fold의 train/eval 경계."""

    fold_id: str
    train_subjects: frozenset
    eval_subjects: frozenset
    train_row_ids: frozenset
    eval_row_ids: frozenset
    validation_subjects: frozenset = field(default_factory=frozenset)
    validation_row_ids: frozenset = field(default_factory=frozenset)
    events: list[dict] = field(default_factory=list)


class LeakageAuditor:
    """파이프라인 전 구간의 누수 불변식을 추적한다."""

    def __init__(self, mode: str = "enforce", *, name: str = "run") -> None:
        if mode not in {"enforce", "observe"}:
            raise ValueError(f"mode must be 'enforce' or 'observe', got {mode!r}")
        self.mode = mode
        self.name = name
        self.folds: dict[str, FoldRegistration] = {}
        self.violations: list[Violation] = []
        self.observations: list[dict] = []

    # ------------------------------------------------------------------ 내부
    def _handle(self, violations: Sequence[Violation]) -> None:
        if not violations:
            return
        self.violations.extend(violations)
        if self.mode == "enforce":
            raise LeakageError(violations)
        for v in violations:
            log.warning("[observe] %s", v)

    def _fold(self, fold_id: str) -> FoldRegistration:
        if fold_id not in self.folds:
            raise LeakageError(
                [
                    Violation(
                        "SPLIT_NOT_REGISTERED",
                        f"fold '{fold_id}'가 등록되기 전에 파이프라인 단계가 호출되었다. "
                        "split보다 전처리가 먼저 수행되었다는 뜻이다.",
                        fold_id,
                    )
                ]
            )
        return self.folds[fold_id]

    def _log_event(self, fold_id: str, kind: str, **payload) -> None:
        if fold_id in self.folds:
            self.folds[fold_id].events.append({"kind": kind, **payload})

    # ------------------------------------------------------------------ 등록
    def register_split(
        self,
        fold_id: str,
        *,
        train_subjects: Iterable,
        eval_subjects: Iterable,
        train_row_ids: Iterable,
        eval_row_ids: Iterable,
        validation_subjects: Iterable = (),
        validation_row_ids: Iterable = (),
        require_disjoint_subjects: bool = True,
    ) -> None:
        """fold 경계를 등록하고 즉시 중복을 검사한다."""
        reg = FoldRegistration(
            fold_id=fold_id,
            train_subjects=frozenset(np.asarray(list(train_subjects), dtype=object).ravel()),
            eval_subjects=frozenset(np.asarray(list(eval_subjects), dtype=object).ravel()),
            train_row_ids=frozenset(np.asarray(list(train_row_ids)).ravel().tolist()),
            eval_row_ids=frozenset(np.asarray(list(eval_row_ids)).ravel().tolist()),
            validation_subjects=frozenset(
                np.asarray(list(validation_subjects), dtype=object).ravel()
            ),
            validation_row_ids=frozenset(
                np.asarray(list(validation_row_ids)).ravel().tolist()
            ),
        )
        self.folds[fold_id] = reg

        found = checks.check_row_overlap(reg.train_row_ids, reg.eval_row_ids, fold_id=fold_id)
        found += checks.check_row_overlap(
            reg.validation_row_ids, reg.eval_row_ids, fold_id=fold_id
        )
        found += checks.check_row_overlap(
            reg.train_row_ids, reg.validation_row_ids, fold_id=fold_id
        )
        if require_disjoint_subjects:
            found += checks.check_subject_overlap(
                reg.train_subjects | reg.validation_subjects,
                reg.eval_subjects,
                fold_id=fold_id,
            )
        else:
            # 실험 A: 행 단위 분할이라 피험자 중복이 설계상 발생한다. 측정만 한다.
            shared = (reg.train_subjects & reg.eval_subjects) - {"__SYNTHETIC__"}
            self.observations.append(
                {
                    "fold_id": fold_id,
                    "kind": "subject_overlap_measured",
                    "n_shared_subjects": len(shared),
                    "n_train_subjects": len(reg.train_subjects),
                    "n_eval_subjects": len(reg.eval_subjects),
                }
            )
        self._handle(found)

    # ------------------------------------------------------------------ 신고 API
    def record_fit(
        self,
        component: str,
        fold_id: str,
        *,
        subjects: Iterable,
        row_ids: Iterable,
        n_rows: int | None = None,
        occurred_before_split: bool = False,
    ) -> None:
        """전처리기(scaler/imputer/outlier)의 fit 범위를 신고한다."""
        reg = self._fold(fold_id)
        subs = list(subjects)
        rows = list(row_ids)
        self._log_event(
            fold_id,
            "fit",
            component=component,
            n_subjects=len(set(subs)),
            n_rows=n_rows,
            row_scope_recorded=True,
            occurred_before_split=occurred_before_split,
        )
        found = checks.check_preprocessing_after_split(
            not occurred_before_split, component, fold_id=fold_id
        )
        found += checks.check_fit_scope(component, subs, reg.train_subjects, fold_id=fold_id)
        found += checks.check_fit_row_scope(
            component, rows, reg.train_row_ids, fold_id=fold_id
        )
        # observe 모드에서도 "평가 자료를 몇 행 보았는가"를 정량화한다.
        outside = set(subs) - set(reg.train_subjects)
        if outside:
            self.observations.append(
                {
                    "fold_id": fold_id,
                    "kind": "fit_saw_eval_subjects",
                    "component": component,
                    "n_eval_subjects_seen": len(outside & set(reg.eval_subjects)),
                    "n_rows_fit": n_rows,
                }
            )
        eval_rows_seen = set(rows) & set(reg.eval_row_ids)
        if eval_rows_seen:
            self.observations.append(
                {
                    "fold_id": fold_id,
                    "kind": "fit_saw_eval_rows",
                    "component": component,
                    "n_eval_rows_seen": len(eval_rows_seen),
                    "n_rows_fit": n_rows,
                }
            )
        self._handle(found)

    def record_vae_fit(
        self,
        fold_id: str,
        *,
        subjects: Iterable,
        labels: Iterable,
        row_ids: Iterable,
        expected_label: int | None = None,
        n_rows: int | None = None,
    ) -> None:
        """VAE 학습 범위를 신고한다."""
        reg = self._fold(fold_id)
        subs = list(subjects)
        rows = list(row_ids)
        self._log_event(
            fold_id,
            "vae_fit",
            n_subjects=len(set(subs)),
            n_rows=n_rows,
            expected_label=expected_label,
            row_scope_recorded=True,
        )
        found = checks.check_vae_fit_scope(
            subs, list(labels), reg.train_subjects, expected_label=expected_label, fold_id=fold_id
        )
        row_violations = checks.check_fit_row_scope(
            "vae", rows, reg.train_row_ids, fold_id=fold_id
        )
        for violation in row_violations:
            violation.code = "VAE_FIT_ROW_SCOPE"
        found += row_violations
        eval_rows_seen = set(rows) & set(reg.eval_row_ids)
        if eval_rows_seen:
            self.observations.append(
                {
                    "fold_id": fold_id,
                    "kind": "vae_fit_saw_eval_rows",
                    "n_eval_rows_seen": len(eval_rows_seen),
                    "n_rows_fit": n_rows,
                }
            )
        overlap_eval = set(subs) & set(reg.eval_subjects)
        if overlap_eval:
            self.observations.append(
                {
                    "fold_id": fold_id,
                    "kind": "vae_fit_saw_eval_subjects",
                    "n_eval_subjects_seen": len(overlap_eval),
                    "n_rows_fit": n_rows,
                }
            )
        self._handle(found)

    def record_synthetic(
        self,
        fold_id: str,
        *,
        source_subjects: Iterable,
        n_rows: int,
        target: str = "train",
    ) -> None:
        """합성행 생성과 투입 대상을 신고한다."""
        reg = self._fold(fold_id)
        self._log_event(fold_id, "synthetic", n_rows=n_rows, target=target)
        found = checks.check_synthetic_source_subjects(
            source_subjects, reg.train_subjects, fold_id=fold_id
        )
        if target != "train":
            found.append(
                Violation(
                    "SYNTHETIC_TARGET",
                    f"합성행이 '{target}'에 투입되었다. train만 허용된다 (사용자 지시 11).",
                    fold_id,
                    {"target": target, "n_rows": n_rows},
                )
            )
        self._handle(found)

    def record_eval(
        self,
        fold_id: str,
        *,
        is_synthetic: Sequence[bool],
        subjects: Iterable | None = None,
        where: str = "eval",
    ) -> None:
        """평가셋 구성을 신고한다. 합성행이 섞였으면 위반이다."""
        self._fold(fold_id)
        found = checks.check_no_synthetic_in_eval(is_synthetic, fold_id=fold_id, where=where)
        if subjects is not None:
            found += checks.check_subject_aggregation_excludes_synthetic(subjects, fold_id=fold_id)
        self._log_event(fold_id, "eval", n_rows=len(is_synthetic), where=where)
        self._handle(found)

    def record_early_stopping(
        self,
        fold_id: str,
        *,
        subjects: Iterable,
        row_ids: Iterable,
    ) -> None:
        """early stopping 범위를 신고하고 outer eval 행 사용을 차단한다.

        A의 명시적 validation은 허용하되, 행 단위 split에서 피험자 ID가 겹쳐도
        test 행을 validation으로 잘못 넘기는 오류를 원시 row ID로 검출한다.
        """
        reg = self._fold(fold_id)
        subs = list(subjects)
        rows = list(row_ids)
        self._log_event(
            fold_id,
            "early_stopping",
            n_subjects=len(set(subs)),
            n_rows=len(rows),
            row_scope_recorded=True,
        )
        allowed_subjects = reg.train_subjects | reg.validation_subjects
        allowed_rows = reg.train_row_ids | reg.validation_row_ids
        found = checks.check_early_stopping_scope(
            subs, allowed_subjects, fold_id=fold_id
        )
        row_violations = checks.check_fit_row_scope(
            "early_stopping", rows, allowed_rows, fold_id=fold_id
        )
        for violation in row_violations:
            violation.code = "EARLY_STOPPING_ROW_SCOPE"
            violation.message = (
                "early stopping이 허용된 train/validation 밖 원시행을 참조했다"
            )
        found += row_violations
        self._handle(found)

    def record_selection(self, what: str, fold_id: str, *, subjects: Iterable) -> None:
        """하이퍼파라미터/임계값 선택에 쓰인 자료 범위를 신고한다."""
        reg = self._fold(fold_id)
        self._log_event(fold_id, "selection", what=what, n_subjects=len(set(subjects)))
        self._handle(
            checks.check_selection_scope(what, subjects, reg.train_subjects, fold_id=fold_id)
        )

    def check_features(self, columns: Iterable[str]) -> None:
        """feature 컬럼 검사는 모드와 무관하게 **항상 강제**한다.

        subject ID나 MMSE가 입력에 들어가는 것은 논문 재현과 무관한 순수 구현 오류다.
        """
        found = checks.check_forbidden_features(columns)
        if found:
            raise LeakageError(found)

    # ------------------------------------------------------------------ 보고
    def summary(self) -> dict:
        by_code: dict[str, int] = {}
        for v in self.violations:
            by_code[v.code] = by_code.get(v.code, 0) + 1
        return {
            "name": self.name,
            "mode": self.mode,
            "n_folds": len(self.folds),
            "n_violations": len(self.violations),
            "violations_by_code": by_code,
            "violations": [
                {"code": v.code, "fold_id": v.fold_id, "message": v.message, "detail": v.detail}
                for v in self.violations
            ],
            "observations": self.observations,
            "fold_events": {fid: reg.events for fid, reg in self.folds.items()},
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.summary(), fh, ensure_ascii=False, indent=2, default=str)
        log.info("leakage audit -> %s", path)

    def assert_clean(self) -> None:
        """enforce 모드에서 최종 확인용."""
        if self.violations:
            raise LeakageError(self.violations)
