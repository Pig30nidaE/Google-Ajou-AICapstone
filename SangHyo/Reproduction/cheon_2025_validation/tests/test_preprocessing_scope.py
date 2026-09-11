"""No learned preprocessing exists; feature construction is row-wise deterministic; no NaN so imputation is vacuous."""
import re

import numpy as np
import pytest

from conftest import DATA_ROOT, EXP_DIR


def test_no_imputer_or_scaler_in_pipeline_source():
    src = "\n".join(p.read_text() for p in (EXP_DIR / "src" / "cheon").glob("*.py"))
    for pat in (r"SimpleImputer", r"KNNImputer", r"IterativeImputer", r"StandardScaler", r"MinMaxScaler", r"RobustScaler", r"\.fillna\("):
        assert re.search(pat, src) is None, f"learned/dataset-wide preprocessing found: {pat}"


def test_feature_builder_is_row_wise(real_data_available):
    if not real_data_available:
        pytest.skip("raw data not available")
    from cheon.data import load_daily_records
    from cheon.features import build_features, feature_matrix
    raw = load_daily_records(DATA_ROOT)
    full = feature_matrix(build_features(raw))
    sub = feature_matrix(build_features(raw.iloc[500:700]))
    assert np.allclose(sub, full[500:700], equal_nan=True)  # a record's features never depend on other records


def test_no_missing_values_in_paper_features(real_data_available):
    if not real_data_available:
        pytest.skip("raw data not available")
    from cheon.data import load_daily_records
    from cheon.features import build_features, feature_matrix
    X = feature_matrix(build_features(load_daily_records(DATA_ROOT)))
    assert int(np.isnan(X).sum()) == 0 and int(np.isinf(X).sum()) == 0
