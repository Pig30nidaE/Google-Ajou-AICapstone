"""AI-Hub 라이프로그 로더와 핵심 데이터 컨테이너."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .schema import (
    CLASS_ORDER,
    CLASS_TO_CODE,
    CODE_TO_CLASS,
    DATA_CONTRACT,
    KNOWN_DUPLICATE_COLUMN_PAIRS,
    PAPER_ACTIVITY_FEATURES,
    PAPER_FEATURES,
    PAPER_SLEEP_FEATURES,
    SYNTHETIC_SUBJECT_SENTINEL,
    SchemaError,
    assert_no_forbidden_features,
)

log = logging.getLogger(__name__)

__all__ = ["LifelogData", "load_lifelog", "DEFAULT_SOURCE_FILES"]


#: partition -> (activity, sleep, label) 상대 경로.
DEFAULT_SOURCE_FILES: dict[str, dict[str, str]] = {
    "training": {
        "activity": "1.Training/SourceData/1.Gait/train_activity.csv",
        "sleep": "1.Training/SourceData/2.Sleep/train_sleep.csv",
        "label": "1.Training/LabelingData/1.Gait/training_label.csv",
    },
    "validation": {
        "activity": "2.Validation/SourceData/1.Gait/val_activity.csv",
        "sleep": "2.Validation/SourceData/2.Sleep/val_sleep.csv",
        "label": "2.Validation/LabelingData/1.Gait/val_label.csv",
    },
}


@dataclass
class LifelogData:
    """feature 행렬과 메타데이터를 **분리해서** 보유하는 컨테이너.

    메타(피험자 ID·라벨)를 feature 프레임에 섞지 않는 것이 설계의 핵심이다.
    이 분리 덕분에 ``X``에 식별자가 흘러드는 사고가 구조적으로 불가능하다.

    Attributes:
        X: (n, 46) feature 프레임. 식별자·타깃·MMSE를 포함하지 않는다.
        y: (n,) 정수 라벨 (0=CN, 1=MCI, 2=Dem).
        subject: (n,) 피험자 ID. 합성행은 ``SYNTHETIC_SUBJECT_SENTINEL``.
        row_id: (n,) 고유 행 ID. 실제 행은 >= 0, 합성행은 < 0.
        is_synthetic: (n,) 합성 여부.
        provenance: 합성행 메타데이터 (없으면 None).
        source_partition: (n,) 원본 AI-Hub 폴더 ("training"/"validation"/"synthetic").
    """

    X: pd.DataFrame
    y: np.ndarray
    subject: np.ndarray
    row_id: np.ndarray
    is_synthetic: np.ndarray
    provenance: pd.DataFrame | None = None
    source_partition: np.ndarray | None = None
    _next_synthetic_id: int = field(default=-1, repr=False)

    def __post_init__(self) -> None:
        n = len(self.X)
        for name in ("y", "subject", "row_id", "is_synthetic"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise SchemaError(f"길이 불일치: X={n}, {name}={len(arr)}")
        if self.source_partition is None:
            self.source_partition = np.array(["unknown"] * n, dtype=object)
        assert_no_forbidden_features(self.X.columns)

    # ---------------------------------------------------------------- 기본 속성
    @property
    def n(self) -> int:
        return len(self.X)

    @property
    def features(self) -> list[str]:
        return list(self.X.columns)

    @property
    def n_real(self) -> int:
        return int((~self.is_synthetic).sum())

    def subjects(self, *, real_only: bool = True) -> np.ndarray:
        """고유 피험자 ID 배열. 기본적으로 합성행 센티널을 제외한다."""
        mask = ~self.is_synthetic if real_only else np.ones(self.n, bool)
        subs = pd.unique(self.subject[mask])
        return np.asarray([s for s in subs if s != SYNTHETIC_SUBJECT_SENTINEL], dtype=object)

    def subject_labels(self) -> pd.Series:
        """피험자 -> 라벨 코드. 합성행은 제외한다."""
        mask = ~self.is_synthetic
        s = pd.Series(self.y[mask], index=self.subject[mask])
        grouped = s.groupby(level=0).agg(lambda v: v.iloc[0])
        n_multi = s.groupby(level=0).nunique()
        if (n_multi > 1).any():
            bad = list(n_multi[n_multi > 1].index)
            raise SchemaError(f"한 피험자가 여러 라벨을 갖는다: {bad}")
        return grouped

    def class_counts(self, *, by: str = "record") -> dict[str, int]:
        """클래스별 기록 수 또는 피험자 수."""
        if by == "record":
            vc = pd.Series(self.y).value_counts()
            return {CODE_TO_CLASS[int(k)]: int(v) for k, v in vc.items()}
        if by == "subject":
            vc = self.subject_labels().value_counts()
            return {CODE_TO_CLASS[int(k)]: int(v) for k, v in vc.items()}
        raise ValueError(f"by must be 'record' or 'subject', got {by!r}")

    # ---------------------------------------------------------------- 조작
    def take(self, idx: np.ndarray | Sequence[int]) -> "LifelogData":
        """위치 인덱스로 부분집합을 만든다."""
        idx = np.asarray(idx)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        prov = None
        if self.provenance is not None and len(self.provenance) == self.n:
            prov = self.provenance.iloc[idx].reset_index(drop=True)
        return LifelogData(
            X=self.X.iloc[idx].reset_index(drop=True),
            y=self.y[idx],
            subject=self.subject[idx],
            row_id=self.row_id[idx],
            is_synthetic=self.is_synthetic[idx],
            provenance=prov,
            source_partition=self.source_partition[idx],
            _next_synthetic_id=self._next_synthetic_id,
        )

    def real_only(self) -> "LifelogData":
        return self.take(~self.is_synthetic)

    def with_features(self, X: pd.DataFrame) -> "LifelogData":
        """feature 프레임만 교체한다 (스케일링 등)."""
        if len(X) != self.n:
            raise SchemaError(f"길이 불일치: {len(X)} != {self.n}")
        return replace(self, X=X.reset_index(drop=True))

    def append_synthetic(
        self,
        X_syn: pd.DataFrame,
        label: int,
        provenance_rows: pd.DataFrame,
    ) -> "LifelogData":
        """합성행을 덧붙인다.

        합성행의 ``subject``는 센티널로 고정되고 ``row_id``는 음수로 채번된다.
        (사용자 지시 15·16 — 합성행에 가짜 피험자 ID를 부여하지 않는다.)
        """
        if list(X_syn.columns) != self.features:
            raise SchemaError("합성 feature 컬럼이 원본과 다르다")
        k = len(X_syn)
        new_ids = np.arange(self._next_synthetic_id, self._next_synthetic_id - k, -1)
        base_prov = self.provenance
        if base_prov is None:
            base_prov = pd.DataFrame(index=range(self.n))
        prov = pd.concat(
            [base_prov.reset_index(drop=True), provenance_rows.reset_index(drop=True)],
            ignore_index=True,
        )
        return LifelogData(
            X=pd.concat([self.X, X_syn], ignore_index=True),
            y=np.concatenate([self.y, np.full(k, label, dtype=self.y.dtype)]),
            subject=np.concatenate(
                [self.subject, np.full(k, SYNTHETIC_SUBJECT_SENTINEL, dtype=object)]
            ),
            row_id=np.concatenate([self.row_id, new_ids]),
            is_synthetic=np.concatenate([self.is_synthetic, np.ones(k, bool)]),
            provenance=prov,
            source_partition=np.concatenate(
                [self.source_partition, np.full(k, "synthetic", dtype=object)]
            ),
            _next_synthetic_id=self._next_synthetic_id - k,
        )

    def describe(self) -> str:
        rec = self.class_counts(by="record")
        sub = self.class_counts(by="subject")
        return (
            f"LifelogData(n={self.n}, real={self.n_real}, synthetic={self.n - self.n_real}, "
            f"features={len(self.features)}, subjects={len(self.subjects())}, "
            f"records={rec}, subjects_per_class={sub})"
        )


# --------------------------------------------------------------------------------------
def _read_partition(root: Path, paths: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    act = pd.read_csv(root / paths["activity"])
    slp = pd.read_csv(root / paths["sleep"])
    lab = pd.read_csv(root / paths["label"])
    return act, slp, lab


def load_lifelog(
    data_root: str | Path,
    *,
    partitions: Iterable[str] = ("training", "validation"),
    drop_duplicate_columns: bool = False,
    extra_features: Sequence[str] = (),
    validate_contract: bool = True,
) -> LifelogData:
    """AI-Hub Training/Validation을 합쳐 논문 코호트를 재구성한다.

    논문은 AI-Hub의 폴더 구분을 쓰지 않고 합본에서 자체적으로 8:1:1 재분할을 했다.
    합본이 174명 / 12,183행 / (7737, 3661, 785)로 논문 표 3과 정확히 일치한다.

    Args:
        data_root: ``Data/`` 디렉터리 경로.
        partitions: 읽을 partition. 기본은 둘 다 (= 논문 코호트).
        drop_duplicate_columns: 완전 동일한 컬럼쌍의 뒤쪽을 제거할지 (기본 False = 논문대로 46개).
        extra_features: 논문 표 1·2 밖의 컬럼을 추가할 때 (민감도 분석용).
        validate_contract: 실측 계약 검증 여부.

    Returns:
        LifelogData.

    Raises:
        SchemaError: 컬럼 누락, 행 정렬 불일치, 계약 위반.
    """
    root = Path(data_root)
    acts, slps, labs, parts = [], [], [], []
    for name in partitions:
        if name not in DEFAULT_SOURCE_FILES:
            raise ValueError(f"unknown partition {name!r}")
        a, s, l = _read_partition(root, DEFAULT_SOURCE_FILES[name])
        if len(a) != len(s):
            raise SchemaError(f"[{name}] activity({len(a)})와 sleep({len(s)})의 행 수가 다르다")
        # assumptions.md A-2: 날짜 컬럼이 없으므로 위치 결합만 가능하다.
        # 두 파일의 EMAIL이 행 단위로 동일한 순서임을 매번 확인한다.
        if not (a["EMAIL"].values == s["EMAIL"].values).all():
            raise SchemaError(
                f"[{name}] activity와 sleep의 EMAIL 행 순서가 다르다. "
                "위치 기반 결합(assumptions.md A-2)이 성립하지 않는다."
            )
        acts.append(a)
        slps.append(s)
        labs.append(l)
        parts.append(np.full(len(a), name, dtype=object))

    activity = pd.concat(acts, ignore_index=True)
    sleep = pd.concat(slps, ignore_index=True)
    labels = pd.concat(labs, ignore_index=True)
    source_partition = np.concatenate(parts)

    # ---- 계약: 원본 컬럼 수 (논문 §3.1과 대조)
    n_act_cols = sum(c.startswith("activity_") for c in activity.columns)
    n_slp_cols = sum(c.startswith("sleep_") for c in sleep.columns)
    if validate_contract:
        if n_act_cols != DATA_CONTRACT["n_activity_columns"]:
            raise SchemaError(
                f"activity_* 컬럼이 {n_act_cols}개다 (기대 {DATA_CONTRACT['n_activity_columns']})"
            )
        if n_slp_cols != DATA_CONTRACT["n_sleep_columns"]:
            log.warning(
                "sleep_* 컬럼이 %d개다 (실측 계약 %d, 논문 §3.1은 33개로 기재 "
                "→ report_inconsistencies.md I-13)",
                n_slp_cols,
                DATA_CONTRACT["n_sleep_columns"],
            )

    # ---- feature 선택
    missing_a = [c for c in PAPER_ACTIVITY_FEATURES if c not in activity.columns]
    missing_s = [c for c in PAPER_SLEEP_FEATURES if c not in sleep.columns]
    if missing_a or missing_s:
        raise SchemaError(
            "논문 표 1·2의 변수가 실제 데이터에 없다. 유사 변수로 임의 대체하지 않는다 "
            f"(사용자 지시 3). activity 누락={missing_a}, sleep 누락={missing_s}"
        )

    feat_cols = list(PAPER_FEATURES)
    for col in extra_features:
        if col in activity.columns or col in sleep.columns:
            feat_cols.append(col)
        else:
            raise SchemaError(f"extra_features의 {col!r}가 데이터에 없다")

    src = pd.concat([activity, sleep.drop(columns=["EMAIL"])], axis=1)
    X = src[feat_cols].copy()

    # ---- 완전 동일 컬럼쌍 경고 (paper_data_mapping.md §4)
    for left, right in KNOWN_DUPLICATE_COLUMN_PAIRS:
        if left in X.columns and right in X.columns:
            if np.array_equal(X[left].to_numpy(), X[right].to_numpy()):
                log.warning(
                    "'%s'와 '%s'는 원소 단위로 완전히 동일하다. 논문의 46개 변수는 실질 45개다 "
                    "(paper_data_mapping.md §4). drop_duplicate_columns=%s",
                    left,
                    right,
                    drop_duplicate_columns,
                )
                if drop_duplicate_columns:
                    X = X.drop(columns=[right])

    # ---- 라벨
    label_map = labels.drop_duplicates("SAMPLE_EMAIL").set_index("SAMPLE_EMAIL")["DIAG_NM"]
    subject = activity["EMAIL"].to_numpy(dtype=object)
    y_name = pd.Series(subject).map(label_map)
    if y_name.isna().any():
        n_bad = int(y_name.isna().sum())
        raise SchemaError(f"라벨을 찾지 못한 행이 {n_bad}건 있다")
    unknown = set(y_name.unique()) - set(CLASS_ORDER)
    if unknown:
        raise SchemaError(
            f"예상치 못한 DIAG_NM 값: {sorted(unknown)}. "
            f"기대 = {CLASS_ORDER} (AAD는 배포본에서 이미 CN으로 병합됨, I-16)"
        )
    y = y_name.map(CLASS_TO_CODE).to_numpy(dtype=np.int64)

    data = LifelogData(
        X=X.reset_index(drop=True),
        y=y,
        subject=subject,
        row_id=np.arange(len(X), dtype=np.int64),
        is_synthetic=np.zeros(len(X), dtype=bool),
        source_partition=source_partition,
    )

    if validate_contract and set(partitions) == {"training", "validation"}:
        _validate_contract(data)
    log.info("loaded: %s", data.describe())
    return data


def _validate_contract(data: LifelogData) -> None:
    """실측 계약(2026-08-02)과 대조한다. 어긋나면 조용히 넘어가지 않는다."""
    problems: list[str] = []
    if data.n != DATA_CONTRACT["n_rows"]:
        problems.append(f"행 수 {data.n} != {DATA_CONTRACT['n_rows']}")
    n_sub = len(data.subjects())
    if n_sub != DATA_CONTRACT["n_subjects"]:
        problems.append(f"피험자 수 {n_sub} != {DATA_CONTRACT['n_subjects']}")
    rec = data.class_counts(by="record")
    if rec != DATA_CONTRACT["records_per_class"]:
        problems.append(f"클래스별 기록 수 {rec} != {DATA_CONTRACT['records_per_class']}")
    sub = data.class_counts(by="subject")
    if sub != DATA_CONTRACT["subjects_per_class"]:
        problems.append(f"클래스별 피험자 수 {sub} != {DATA_CONTRACT['subjects_per_class']}")
    n_missing = int(data.X.isna().sum().sum())
    if n_missing != DATA_CONTRACT["n_missing_cells"]:
        problems.append(f"결측 셀 {n_missing} != {DATA_CONTRACT['n_missing_cells']} (imputer가 동작한다)")
    if problems:
        raise SchemaError(
            "실측 데이터 계약 위반. 데이터 버전이 다르거나 로더가 잘못되었다:\n  - "
            + "\n  - ".join(problems)
        )
