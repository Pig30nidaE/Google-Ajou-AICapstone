"""Feature sets, including the paper's own Table 16 variable list.

Section 3.2 says variable selection "removed multicollinearity and considered
both statistical significance and clinical importance to form the final analysis
variable set", but never prints that set for the ML models.  The 12 variables of
Table 16 (thesis) / Table 13 (journal) are the only concrete lifelog list either
paper gives, so `paper_table16_lifelog` reconstructs the sentence with them.
"""

from __future__ import annotations

import pytest

from src.data import schema
from src.utils.config import Config, load_config, validate_config


def test_table16_has_the_twelve_printed_variables() -> None:
    assert len(schema.PAPER_TABLE16_LIFELOG_FEATURES) == 12
    assert len(set(schema.PAPER_TABLE16_LIFELOG_FEATURES)) == 12
    assert set(schema.PAPER_TABLE16_WALD_P) == set(
        schema.PAPER_TABLE16_LIFELOG_FEATURES
    )


def test_table16_variables_all_exist_in_the_paper_code_feature_set() -> None:
    """Every Table 16 name must be a real column the loader can produce."""
    missing = set(schema.PAPER_TABLE16_LIFELOG_FEATURES) - set(
        schema.PAPER_CODE_FEATURES
    )
    assert not missing, f"Table 16 names absent from the data: {sorted(missing)}"


def test_table16_is_not_a_significance_filter() -> None:
    """`sleep_efficiency` has p=0.756, so the table is a model, not a filter.

    This matters: reconstructing the set by thresholding p-values would drop it
    and produce a different feature set from the paper's own list.
    """
    assert schema.PAPER_TABLE16_WALD_P["sleep_efficiency"] > 0.05
    assert "sleep_efficiency" in schema.PAPER_TABLE16_LIFELOG_FEATURES


def test_table16_set_carries_no_forbidden_column() -> None:
    forbidden = schema.forbidden_feature_columns(include_cognitive=False)
    assert set(schema.PAPER_TABLE16_LIFELOG_FEATURES).isdisjoint(forbidden)


def test_registered_feature_sets() -> None:
    assert set(schema.FEATURE_SETS) == {
        "paper_code_verbatim", "paper_table16_lifelog"
    }
    assert len(schema.FEATURE_SETS["paper_code_verbatim"]) == 49
    assert len(schema.FEATURE_SETS["paper_table16_lifelog"]) == 12


def test_unknown_feature_set_is_rejected(synthetic_data) -> None:
    from src.data.loader import _select_features

    with pytest.raises(NotImplementedError, match="unknown"):
        _select_features(
            synthetic_data.daily,
            feature_set="does_not_exist",
            drop_zero_variance=False,
            drop_administrative=False,
            notes={},
        )


def test_paper_literal_config_selects_the_table16_set(config_dir) -> None:
    config = load_config(config_dir / "paper_literal_table16.yaml")
    assert config.get("features.feature_set") == "paper_table16_lifelog"
    assert config.get("analysis") == "main_lifelog_only"
    assert config.get("features.include_cognitive_tests") is False
    assert config.split_mode == "official_partition"
    assert config.scaler_scope == "train_only"


def test_paper_literal_config_uses_the_single_conv_block(config_dir) -> None:
    config = load_config(config_dir / "paper_literal_table16.yaml")
    overrides = config.get("model_overrides.cnn1d")
    assert list(overrides["filters"]) == [64], "paper describes one Conv1D stage"
    assert overrides["conv_padding"] == "valid", "Keras Conv1D default"
    assert int(config.get("training.batch_size")) == 32, "Keras fit default"
    assert config.get("training.early_stopping") is False


def test_paper_literal_config_still_forbids_cognitive_features() -> None:
    from src.utils.config import ConfigError

    raw = {
        "experiment": "paper_reported_reconstruction",
        "analysis": "main_lifelog_only",
        "split": {"mode": "official_partition"},
        "features": {
            "feature_set": "paper_table16_lifelog",
            "include_cognitive_tests": True,
        },
        "models": ["random_forest"],
        "representations": {"random_forest": "tabular_subject_aggregate"},
    }
    with pytest.raises(ConfigError, match="cognitive test variables"):
        validate_config(Config(experiment=raw["experiment"], raw=raw))
