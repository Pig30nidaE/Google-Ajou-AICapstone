"""논문 §5.1의 행 단위 8:1:1 분할 (실험 A 전용).

**의도적으로 피험자 누수를 포함한다.** 이것이 논문 절차의 재현이며,
실험 B·C가 통제하려는 대상이다. 누수 감사기는 ``observe`` 모드로
피험자 중복을 측정해 보고한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..data.loader import LifelogData
from ..data.paper_reference import TABLE5_SPLIT
from ..data.schema import CODE_TO_CLASS

log = logging.getLogger(__name__)

__all__ = ["RowSplit", "paper_row_split", "compare_with_table5", "InfeasibleSplitError"]


class InfeasibleSplitError(RuntimeError):
    """남은 표본이 너무 적어 8:1:1 층화 분할이 불가능할 때."""


@dataclass(frozen=True)
class RowSplit:
    """행 단위 train/valid/test 인덱스."""

    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray
    seed: int

    @property
    def fold_id(self) -> str:
        return f"paper_row_split_seed{self.seed}"


def paper_row_split(
    data: LifelogData,
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    stratify: bool = True,
) -> RowSplit:
    """클래스 층화 행 단위 분할 (피험자는 고려하지 않는다).

    논문은 분할 방법·seed를 보고하지 않았다 (assumptions.md C-1).
    표 5와 정확히 같은 정수가 나오지 않을 수 있으며, 그 차이는
    :func:`compare_with_table5`로 보고한다.
    """
    train_r, valid_r, test_r = ratios
    if not np.isclose(train_r + valid_r + test_r, 1.0):
        raise ValueError(f"ratios must sum to 1, got {ratios}")

    idx = np.arange(data.n)
    strat = data.y if stratify else None
    _assert_split_feasible(data, ratios)
    train_idx, rest_idx = train_test_split(
        idx, test_size=valid_r + test_r, random_state=seed, stratify=strat
    )
    rest_strat = data.y[rest_idx] if stratify else None
    # 논문 표 5의 열 순서(Train / Test / Vaild)를 따라 test를 먼저 떼어낸다.
    test_idx, valid_idx = train_test_split(
        rest_idx,
        test_size=valid_r / (valid_r + test_r),
        random_state=seed,
        stratify=rest_strat,
    )
    split = RowSplit(np.sort(train_idx), np.sort(valid_idx), np.sort(test_idx), seed)

    n_sub_train = len(pd.unique(data.subject[split.train_idx]))
    n_sub_test = len(pd.unique(data.subject[split.test_idx]))
    shared = set(data.subject[split.train_idx]) & set(data.subject[split.test_idx])
    log.warning(
        "행 단위 분할(논문 절차 재현): train 피험자 %d명, test 피험자 %d명, "
        "**중복 %d명** — 피험자 누수는 설계상 의도된 것이다 (report_inconsistencies.md I-6)",
        n_sub_train,
        n_sub_test,
        len(shared),
    )
    return split


def _assert_split_feasible(data: LifelogData, ratios: tuple[float, float, float]) -> None:
    """층화 8:1:1 분할이 가능한 표본이 남아 있는지 확인한다.

    논문 §5.1 본문의 이상치 방식(변수별 [p10, p90] 절단)을 그대로 적용하면
    12,183행 중 372행(Dem 6행)만 남아 이 분할이 **원리적으로 불가능**하다.
    sklearn의 원문 오류 대신 그 사실을 설명하는 오류를 던진다.
    """
    counts = data.class_counts(by="record")
    smallest = min(counts.values()) if counts else 0
    min_needed = int(np.ceil(1 / min(ratios[1], ratios[2]))) if min(ratios[1:]) > 0 else 0
    if smallest >= max(min_needed, 4):
        return
    raise InfeasibleSplitError(
        f"층화 {ratios[0]:.0%}:{ratios[1]:.0%}:{ratios[2]:.0%} 분할이 불가능하다. "
        f"이상치 제거 후 남은 기록: {counts} (최소 클래스 {smallest}행, "
        f"각 분할에 1행씩 넣으려면 최소 {min_needed}행 필요).\n"
        "이것은 구현 오류가 아니라 논문 §5.1 본문 방식의 산술적 귀결이다: "
        "46개 변수 각각에서 상·하위 10%를 제거하면 잔존율이 3.05%(372행, Dem 6행)가 된다. "
        "report_inconsistencies.md I-1 증거 B 참조.\n"
        "대안:\n"
        "  - configs/paper_isoforest_latent500.yaml (§4.2·그림 1의 Isolation Forest 해석)\n"
        "  - outlier.percentile.action: clip  (행을 지우지 않는 해석)\n"
        "  - outlier.percentile.q 를 더 작은 값으로 (예: 0.003 — 실측 최근접)"
    )


def compare_with_table5(data: LifelogData, split: RowSplit, n_synthetic_dem: int = 0) -> pd.DataFrame:
    """실제 분할 결과를 논문 표 5와 나란히 놓는다.

    Args:
        n_synthetic_dem: train에 더해질 합성 Dem 행 수 (논문 유도값 4,000).
    """
    rows = []
    parts = {"Train": split.train_idx, "Test": split.test_idx, "Vaild": split.valid_idx}
    for part, idx in parts.items():
        for code, cls in CODE_TO_CLASS.items():
            got = int((data.y[idx] == code).sum())
            if part == "Train" and cls == "Dem":
                got += n_synthetic_dem
            paper = TABLE5_SPLIT[part][cls]
            rows.append(
                {
                    "split": part,
                    "class": cls,
                    "paper_table5": paper,
                    "reproduction": got,
                    "diff": got - paper,
                    "includes_synthetic": part == "Train" and cls == "Dem" and n_synthetic_dem > 0,
                }
            )
    return pd.DataFrame(rows)
