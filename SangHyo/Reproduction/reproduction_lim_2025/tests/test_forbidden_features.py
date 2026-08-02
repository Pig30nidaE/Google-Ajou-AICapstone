"""Target, identifier and cognitive-test columns must never become features."""

from __future__ import annotations

import pytest

from src.audit.leakage import AuditLog, LeakageError, check_forbidden_features
from src.data import schema
from src.utils.config import Config, ConfigError, validate_config


@pytest.mark.parametrize("column", ["DIAG_NM", "DIAG_NIM", "DIAG_SEQ", "label", "target"])
def test_target_columns_are_rejected(column: str) -> None:
    log = AuditLog()
    with pytest.raises(LeakageError, match="forbidden_features|cognitive_like"):
        check_forbidden_features(["activity_score", column], log)


@pytest.mark.parametrize(
    "column", ["EMAIL", "SAMPLE_EMAIL", "subject_id", "DOCTOR_NM", "MMSE_NUM", "MMSE_KIND"]
)
def test_identifier_and_admin_columns_are_rejected(column: str) -> None:
    log = AuditLog()
    with pytest.raises(LeakageError):
        check_forbidden_features(["activity_score", column], log)


@pytest.mark.parametrize("column", ["TOTAL", "Q01", "Q13_3", "Q12_TOTAL", "Q19"])
def test_cognitive_columns_are_rejected_in_the_main_analysis(column: str) -> None:
    log = AuditLog()
    with pytest.raises(LeakageError, match="forbidden_features|cognitive_like"):
        check_forbidden_features(["activity_score", column], log)


@pytest.mark.parametrize(
    "column", ["mmse_total", "MMSE_delta", "snsb_score", "kmmse_sum", "diag_prob"]
)
def test_cognitive_like_names_are_caught_by_the_heuristic(column: str) -> None:
    log = AuditLog()
    with pytest.raises(LeakageError):
        check_forbidden_features(["activity_score", column], log)


def test_cognitive_columns_allowed_only_in_the_secondary_analysis() -> None:
    log = AuditLog()
    check_forbidden_features(
        ["activity_score", "TOTAL", "Q13_3"], log, include_cognitive=True
    )
    assert log.passed


def test_identifiers_stay_forbidden_even_in_the_secondary_analysis() -> None:
    log = AuditLog()
    with pytest.raises(LeakageError):
        check_forbidden_features(["TOTAL", "SAMPLE_EMAIL"], log, include_cognitive=True)


def test_clean_lifelog_features_pass() -> None:
    log = AuditLog()
    check_forbidden_features(schema.PAPER_CODE_FEATURES, log)
    assert log.passed


def test_paper_code_feature_set_is_49_and_clean() -> None:
    """The paper claims 58; running its own code yields 49."""
    assert len(schema.PAPER_CODE_FEATURES) == 49
    assert len(schema.PAPER_CODE_ACTIVITY_FEATURES) == 22
    assert len(schema.PAPER_CODE_SLEEP_FEATURES) == 27
    assert len(set(schema.PAPER_CODE_FEATURES)) == 49, "no duplicates"
    forbidden = schema.forbidden_feature_columns(include_cognitive=False)
    assert set(schema.PAPER_CODE_FEATURES).isdisjoint(forbidden)


def test_config_rejects_cognitive_features_in_the_main_analysis() -> None:
    raw = {
        "experiment": "paper_reported_reconstruction",
        "analysis": "main_lifelog_only",
        "split": {"mode": "official_partition"},
        "features": {"include_cognitive_tests": True},
        "models": ["random_forest"],
        "representations": {"random_forest": "tabular_subject_aggregate"},
    }
    with pytest.raises(ConfigError, match="cognitive test variables"):
        validate_config(Config(experiment=raw["experiment"], raw=raw))

    raw["analysis"] = "secondary_lifelog_plus_cognitive"
    validate_config(Config(experiment=raw["experiment"], raw=raw))


def test_shipped_configs_keep_the_main_analysis_lifelog_only(config_dir) -> None:
    from src.utils.config import load_config

    for name in (
        "paper_reproduction.yaml",
        "leakage_controlled_non_nested.yaml",
        "nested_subject_independent.yaml",
    ):
        config = load_config(config_dir / name)
        assert config.get("analysis") == "main_lifelog_only"
        assert config.get("features.include_cognitive_tests") is False


def test_loader_never_puts_cognitive_data_in_the_daily_frame(synthetic_data) -> None:
    assert "TOTAL" not in synthetic_data.daily.columns
    assert set(synthetic_data.feature_columns).isdisjoint(schema.COGNITIVE_TEST_COLUMNS)
