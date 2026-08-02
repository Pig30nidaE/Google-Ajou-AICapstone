"""subject ID·MMSE·진단 파생이 입력 feature에 들어가지 않는지 검증한다.

사용자 지시 12·13.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.audit.checks import check_forbidden_features
from src.audit.leakage import LeakageError
from src.data.loader import LifelogData
from src.data.schema import (
    DATA_CONTRACT,
    PAPER_ACTIVITY_FEATURES,
    PAPER_FEATURES,
    PAPER_SLEEP_FEATURES,
    SchemaError,
    assert_no_forbidden_features,
)


@pytest.mark.parametrize(
    "col",
    [
        "EMAIL",
        "SAMPLE_EMAIL",
        "DIAG_NM",
        "DIAG_SEQ",
        "TOTAL",
        "MMSE_NUM",
        "Q01",
        "Q11_2",
        "Q12_TOTAL",
        "CONVERT(activity_met_1min USING utf8)",
    ],
)
def test_forbidden_columns_are_rejected(col):
    v = check_forbidden_features([col])
    assert len(v) == 1 and v[0].code == "FORBIDDEN_FEATURE"
    with pytest.raises(SchemaError):
        assert_no_forbidden_features([col])


def test_paper_features_pass():
    assert check_forbidden_features(PAPER_FEATURES) == []
    assert_no_forbidden_features(PAPER_FEATURES)


def test_feature_counts_match_paper():
    assert len(PAPER_ACTIVITY_FEATURES) == 22   # 논문 표 1
    assert len(PAPER_SLEEP_FEATURES) == 24      # 논문 표 2
    assert len(PAPER_FEATURES) == 46
    assert len(set(PAPER_FEATURES)) == 46       # 이름 중복 없음


def test_lifelog_data_rejects_forbidden_column_at_construction(fake_data):
    X = fake_data.X.copy()
    X["EMAIL"] = fake_data.subject
    with pytest.raises(SchemaError, match="금지 변수"):
        LifelogData(
            X=X, y=fake_data.y, subject=fake_data.subject,
            row_id=fake_data.row_id, is_synthetic=fake_data.is_synthetic,
        )


def test_lifelog_data_rejects_mmse_total(fake_data):
    X = fake_data.X.copy()
    X["TOTAL"] = 25.0
    with pytest.raises(SchemaError):
        LifelogData(
            X=X, y=fake_data.y, subject=fake_data.subject,
            row_id=fake_data.row_id, is_synthetic=fake_data.is_synthetic,
        )


def test_auditor_always_enforces_feature_check_even_in_observe_mode(observing_auditor):
    """feature 오염은 논문 재현과 무관한 순수 구현 오류이므로 observe 모드에서도 막는다."""
    with pytest.raises(LeakageError, match="FORBIDDEN_FEATURE"):
        observing_auditor.check_features(["activity_steps", "DIAG_NM"])


# ── 실제 데이터 통합 검증 ────────────────────────────────────────────────────
def test_real_data_matches_paper_table3(real_data):
    assert real_data.n == DATA_CONTRACT["n_rows"] == 12_183
    assert len(real_data.subjects()) == 174
    assert real_data.class_counts(by="record") == {"CN": 7_737, "MCI": 3_661, "Dem": 785}
    assert real_data.class_counts(by="subject") == {"CN": 111, "MCI": 51, "Dem": 12}


def test_real_data_has_all_46_paper_features(real_data):
    assert list(real_data.X.columns) == list(PAPER_FEATURES)
    assert_no_forbidden_features(real_data.X.columns)


def test_real_data_has_no_missing_values(real_data):
    assert int(real_data.X.isna().sum().sum()) == 0


def test_real_data_temperature_columns_are_identical(real_data):
    """paper_data_mapping.md §4 — 논문의 46개 변수는 실질 45개다."""
    a = real_data.X["sleep_temperature_delta"].to_numpy()
    b = real_data.X["sleep_temperature_deviation"].to_numpy()
    assert np.array_equal(a, b)


def test_real_data_nonnegativity_profile(real_data):
    """43개 변수가 항상 >= 0이라는 계약 (VAE 생성값 검사의 근거)."""
    negative_cols = {c for c in real_data.X.columns if (real_data.X[c] < 0).any()}
    assert negative_cols == {
        "sleep_midpoint_at_delta",
        "sleep_temperature_delta",
        "sleep_temperature_deviation",
    }
