"""테스트 공통 픽스처.

무거운 의존성(torch, xgboost, pytorch-tabnet) 없이 실행되도록 설계했다.
실제 데이터가 있으면 통합 테스트도 함께 돈다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.audit.leakage import LeakageAuditor  # noqa: E402
from src.data.loader import LifelogData  # noqa: E402
from src.data.schema import PAPER_FEATURES  # noqa: E402

DATA_ROOT = (REPO_ROOT / "../../../Data").resolve()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def make_synthetic_lifelog(
    *,
    n_subjects_per_class: tuple[int, int, int] = (12, 6, 4),
    records_per_subject: int = 8,
    seed: int = 0,
) -> LifelogData:
    """46개 변수를 가진 소형 가짜 데이터셋.

    실제 데이터 없이도 누수 로직을 검증할 수 있게 한다.
    """
    rng = np.random.default_rng(seed)
    rows, subs, ys = [], [], []
    sid = 0
    for code, n_sub in enumerate(n_subjects_per_class):
        for _ in range(n_sub):
            name = f"subj_{sid:03d}"
            sid += 1
            base = rng.normal(code, 0.5, size=len(PAPER_FEATURES))
            for _ in range(records_per_subject):
                rows.append(np.abs(base + rng.normal(0, 0.1, size=len(PAPER_FEATURES))))
                subs.append(name)
                ys.append(code)
    X = pd.DataFrame(np.vstack(rows), columns=list(PAPER_FEATURES))
    return LifelogData(
        X=X,
        y=np.array(ys, dtype=np.int64),
        subject=np.array(subs, dtype=object),
        row_id=np.arange(len(X), dtype=np.int64),
        is_synthetic=np.zeros(len(X), dtype=bool),
    )


@pytest.fixture
def fake_data() -> LifelogData:
    return make_synthetic_lifelog()


@pytest.fixture
def enforcing_auditor() -> LeakageAuditor:
    return LeakageAuditor(mode="enforce", name="test")


@pytest.fixture
def observing_auditor() -> LeakageAuditor:
    return LeakageAuditor(mode="observe", name="test")


@pytest.fixture(scope="session")
def real_data():
    """실제 AI-Hub 데이터. 없으면 해당 테스트를 건너뛴다."""
    from src.data.loader import load_lifelog

    if not (DATA_ROOT / "1.Training").exists():
        pytest.skip(f"실제 데이터가 없다: {DATA_ROOT}")
    return load_lifelog(DATA_ROOT)
