"""The paper's inputs are lifelog features only (PAPER_PROTOCOL.md, Classification task): no MMSE, no diagnosis columns."""
import pandas as pd
import pytest

from conftest import DATA_ROOT
from cheon.data import _check_path, load_daily_records
from cheon.features import PAPER_FEATURES, check_no_forbidden


def test_paper_feature_list_is_72_lifelog_names():
    assert len(PAPER_FEATURES) == 72 and len(set(PAPER_FEATURES)) == 72
    check_no_forbidden(PAPER_FEATURES)
    assert all(n.startswith(("activity_", "sleep_")) for n in PAPER_FEATURES)
    for figure6_name in ("sleep_breath_average", "sleep_hr_average", "sleep_hr_lowest", "sleep_bedtime_end",
                         "activity_class_5min_count_3", "activity_met_1min_kurtosis", "activity_met_1min_autocorrelation"):
        assert figure6_name in PAPER_FEATURES


def test_forbidden_names_are_rejected():
    for bad in (["MMSE_TOTAL"], ["Q01"], ["DIAG_NM"], ["SAMPLE_EMAIL"]):
        with pytest.raises(RuntimeError):
            check_no_forbidden(bad)


def test_loader_never_opens_cognitive_function_files(real_data_available, monkeypatch):
    if not real_data_available:
        pytest.skip("raw data not available")
    opened = []
    orig = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda p, *a, **k: (opened.append(str(p)), orig(p, *a, **k))[1])
    df = load_daily_records(DATA_ROOT)
    assert opened and not any(("CognitiveFunction" in p) or ("mmse" in p.lower()) for p in opened)
    assert "EMAIL" not in df.columns and "SAMPLE_EMAIL" not in df.columns and "DIAG_NM" not in df.columns
    assert not any(c.upper().startswith("Q0") or c.upper().startswith("Q1") or "MMSE" in c.upper() for c in df.columns)


def test_check_path_refuses_mmse():
    with pytest.raises(RuntimeError):
        _check_path(DATA_ROOT / "1.Training/SourceData/3.CognitiveFunction/train_mmse.csv")
