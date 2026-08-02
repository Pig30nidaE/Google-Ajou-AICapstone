"""피험자 단위 group cross-validation.

Dem 피험자가 12명뿐이므로 fold 구성이 실험 설계 전체를 지배한다.
모든 split은 **각 fold의 train·eval에 세 클래스 피험자가 모두 존재하는지**를 검증한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from ..data.loader import LifelogData
from ..data.schema import CODE_TO_CLASS

log = logging.getLogger(__name__)

__all__ = ["Fold", "SplitError", "make_group_folds", "describe_folds"]


class SplitError(RuntimeError):
    """fold가 클래스 존재 조건을 만족하지 못할 때."""


@dataclass(frozen=True)
class Fold:
    """한 fold의 행 인덱스와 메타."""

    fold_id: str
    repeat: int
    index: int
    train_idx: np.ndarray
    eval_idx: np.ndarray

    def subjects(self, data: LifelogData) -> tuple[np.ndarray, np.ndarray]:
        return data.subject[self.train_idx], data.subject[self.eval_idx]


def _subject_table(data: LifelogData) -> pd.DataFrame:
    """피험자 1행 테이블 (subject, label, n_records)."""
    real = data.real_only()
    df = pd.DataFrame({"subject": real.subject, "y": real.y})
    tab = df.groupby("subject", sort=True).agg(y=("y", "first"), n_records=("y", "size"))
    return tab.reset_index()


def make_group_folds(
    data: LifelogData,
    *,
    method: str = "stratified_group_kfold",
    n_splits: int = 3,
    n_repeats: int = 1,
    seed: int = 42,
    prefix: str = "outer",
    require_all_classes: bool = True,
) -> list[Fold]:
    """피험자를 group으로 하는 층화 K-fold를 만든다.

    Args:
        method:
            ``stratified_group_kfold``
                ``sklearn.model_selection.StratifiedGroupKFold`` (사용자 지시의 기본값).
                **행 단위** 클래스 비율을 맞추므로 fold별 Dem *피험자* 수가 균등하다는
                보장이 없다.
            ``subject_stratified``
                피험자 테이블(174행)에 ``StratifiedKFold``를 적용한다.
                fold당 Dem 4명을 **보장**한다 (assumptions.md C-2).
        require_all_classes: 모든 fold의 train·eval에 세 클래스가 있어야 하는지.

    Returns:
        fold 목록. 반복(repeat)이 여러 번이면 ``n_repeats * n_splits`` 개.

    Raises:
        SplitError: 클래스 존재 조건 위반.
    """
    real = data.real_only()
    if real.n != data.n:
        log.info("합성행 %d건을 split 대상에서 제외한다", data.n - real.n)

    folds: list[Fold] = []
    for rep in range(n_repeats):
        rep_seed = seed + 1000 * rep
        if method == "stratified_group_kfold":
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rep_seed)
            pairs = splitter.split(np.zeros(real.n), real.y, groups=real.subject)
        elif method == "subject_stratified":
            pairs = _subject_stratified_split(real, n_splits=n_splits, seed=rep_seed)
        else:
            raise ValueError(f"unknown split method {method!r}")

        for i, (tr, ev) in enumerate(pairs):
            fold = Fold(
                fold_id=f"{prefix}_r{rep}_f{i}",
                repeat=rep,
                index=i,
                train_idx=np.asarray(tr, dtype=np.int64),
                eval_idx=np.asarray(ev, dtype=np.int64),
            )
            if require_all_classes:
                _validate_fold(real, fold, method=method)
            folds.append(fold)
    return folds


def _subject_stratified_split(data: LifelogData, *, n_splits: int, seed: int):
    """피험자 테이블을 층화 분할한 뒤 행 인덱스로 되돌린다."""
    tab = _subject_table(data)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    subject_to_rows: dict[object, np.ndarray] = {
        s: np.flatnonzero(data.subject == s) for s in tab["subject"]
    }
    for tr_s, ev_s in skf.split(tab, tab["y"]):
        train_subs = tab["subject"].to_numpy()[tr_s]
        eval_subs = tab["subject"].to_numpy()[ev_s]
        tr = np.concatenate([subject_to_rows[s] for s in train_subs])
        ev = np.concatenate([subject_to_rows[s] for s in eval_subs])
        yield np.sort(tr), np.sort(ev)


def _validate_fold(data: LifelogData, fold: Fold, *, method: str) -> None:
    """train·eval 모두에 세 클래스 피험자가 존재하는지 확인한다."""
    problems = []
    for name, idx in (("train", fold.train_idx), ("eval", fold.eval_idx)):
        sub_y = pd.DataFrame({"s": data.subject[idx], "y": data.y[idx]}).drop_duplicates("s")
        present = set(sub_y["y"].unique())
        missing = set(CODE_TO_CLASS) - present
        if missing:
            problems.append(
                f"{name}에 {[CODE_TO_CLASS[m] for m in sorted(missing)]} 피험자가 없다"
            )
    if problems:
        hint = (
            "split.method: subject_stratified 로 바꾸면 fold당 Dem 피험자 수가 보장된다 "
            "(assumptions.md C-2)."
            if method == "stratified_group_kfold"
            else "n_splits를 줄여라."
        )
        raise SplitError(f"[{fold.fold_id}] " + "; ".join(problems) + f". {hint}")


def describe_folds(data: LifelogData, folds: list[Fold]) -> pd.DataFrame:
    """fold별 피험자·기록 구성표. --dry-run이 이것을 출력한다."""
    rows = []
    for f in folds:
        row: dict[str, object] = {"fold_id": f.fold_id}
        for name, idx in (("train", f.train_idx), ("eval", f.eval_idx)):
            sub_y = pd.DataFrame({"s": data.subject[idx], "y": data.y[idx]}).drop_duplicates("s")
            for code, cls in CODE_TO_CLASS.items():
                row[f"{name}_{cls}_subjects"] = int((sub_y["y"] == code).sum())
                row[f"{name}_{cls}_records"] = int((data.y[idx] == code).sum())
            row[f"{name}_n_rows"] = int(len(idx))
        rows.append(row)
    return pd.DataFrame(rows)
