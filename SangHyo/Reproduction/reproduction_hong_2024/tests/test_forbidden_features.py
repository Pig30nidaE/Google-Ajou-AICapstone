"""Identifiers, the label, and diagnosis-defining cognitive tests are never inputs."""

from __future__ import annotations

import pytest

from src.audit.leakage import LeakageError, assert_no_forbidden_features, audit_dataset
from src.data import schema
from src.utils.config import Config, ConfigError, load_config, validate_config


@pytest.mark.parametrize(
    "name",
    ["EMAIL", "SAMPLE_EMAIL", "DIAG_NM", "DIAG_SEQ", "MMSE_TOTAL", "mmse_q1",
     "subject_id", "label", "DOCTOR_NM", "y_true"],
)
def test_forbidden_names_are_rejected(name):
    assert schema.is_forbidden_feature(name)
    with pytest.raises(LeakageError):
        assert_no_forbidden_features([name])


@pytest.mark.parametrize("name", list(schema.PAPER_FEATURES))
def test_every_paper_feature_is_allowed(name):
    assert not schema.is_forbidden_feature(name)


def test_the_paper_feature_set_is_exactly_32():
    assert len(schema.PAPER_FEATURES) == 32
    assert len(set(schema.PAPER_FEATURES)) == 32
    assert_no_forbidden_features(schema.PAPER_FEATURES)


def test_table_4_categories_are_all_present():
    """Table 4 lists 26 sleep-quality and 6 statistical features."""
    statistical = {"sleep_breath_average", "sleep_hr_average", "sleep_hr_min",
                   "sleep_hr_max", "sleep_hr_median", "rmssd_average"}
    assert statistical <= set(schema.PAPER_FEATURES)
    assert len(set(schema.PAPER_FEATURES) - statistical) == 26
    for i in range(1, 7):
        assert f"start{i}" in schema.PAPER_FEATURES
        assert f"end{i}" in schema.PAPER_FEATURES


def test_start4_not_strat4():
    """Table 4 prints 'strat4'; Table A1 says start1-6.  The typo must not ship."""
    assert "start4" in schema.PAPER_FEATURES
    assert "strat4" not in schema.PAPER_FEATURES


def test_diagnosis_mapping_collapses_mci_and_dem():
    assert schema.diagnosis_to_label("CN") == 0
    assert schema.diagnosis_to_label("MCI") == 1
    assert schema.diagnosis_to_label("Dem") == 1
    with pytest.raises(ValueError):
        schema.diagnosis_to_label("Unknown")


def test_dataset_audit_flags_a_forbidden_feature(synthetic_data):
    synthetic_data.feature_columns = synthetic_data.feature_columns + ("MMSE_TOTAL",)
    synthetic_data.daily["MMSE_TOTAL"] = 25.0
    log = audit_dataset(synthetic_data)
    assert "no_forbidden_features" in {r["check"] for r in log.failures}


def test_clean_dataset_passes_the_audit(synthetic_data):
    log = audit_dataset(synthetic_data)
    log.raise_if_failed()
    assert log.passed


def test_config_cannot_enable_cognitive_tests():
    raw = {
        "experiment": "fixed_subject_independent",
        "split": {"mode": "stratified_group_kfold"},
        "features": {"include_cognitive_tests": True},
        "models": {"enabled": ["lstm"]},
    }
    with pytest.raises(ConfigError, match="cognitive tests"):
        validate_config(Config(experiment=raw["experiment"], raw=raw))


def test_config_cannot_enable_subject_id():
    raw = {
        "experiment": "fixed_subject_independent",
        "split": {"mode": "stratified_group_kfold"},
        "features": {"include_subject_id": True},
        "models": {"enabled": ["lstm"]},
    }
    with pytest.raises(ConfigError, match="subject id"):
        validate_config(Config(experiment=raw["experiment"], raw=raw))


def test_every_shipped_config_forbids_cognitive_tests(config_dir):
    paths = sorted(config_dir.glob("*.yaml"))
    assert len(paths) == 7
    for path in paths:
        config = load_config(path)
        assert config.get("features.include_cognitive_tests") is False, path.name
        assert config.get("features.include_subject_id") is False, path.name
        assert config.scaler_scope == "train_only", path.name


def test_real_data_carries_only_paper_features(real_data_root):
    from src.data.loader import load_lifelog

    data = load_lifelog(real_data_root)
    assert set(data.feature_columns) == set(schema.PAPER_FEATURES)
    assert_no_forbidden_features(data.feature_columns)
    audit_dataset(data).raise_if_failed()
