"""Label / diagnosis / MMSE containment. No model fit, no API call."""

from __future__ import annotations

from pathlib import Path
import re

import pytest

from SangHyo.Binary.GeminiFeaturePipeline import guards
from SangHyo.Binary.GeminiFeaturePipeline.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    prompt_hash,
)
from SangHyo.Binary.GeminiFeaturePipeline.schema import (
    FEATURE_NAMES,
    design_matrix_names,
    response_schema,
    schema_hash,
    validate_feature_payload,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


# --- 1/2. payload must contain no target and no diagnosis ------------------- #
@pytest.mark.parametrize(
    "poisoned",
    [
        {"observation": {"n_days": 40}, "target": 1},
        {"observation": {"n_days": 40, "diagnosis": "unknown"}},
        {"channels": {"act_steps": {"mean": 1.0}}, "note": "subject is MCI"},
        {"channels": {"act_steps": {"mean": 1.0}}, "note": "possible dementia pattern"},
        {"y_true": 0},
        {"clinical_label": "x"},
        {"groups": [{"class_index": 2}]},
    ],
)
def test_label_like_payloads_are_rejected(poisoned):
    with pytest.raises(guards.LeakageError):
        guards.assert_payload_is_label_free(poisoned, context="test")


def test_legitimate_payload_passes_both_guards():
    payload = {
        "payload_version": "gfp-payload-1",
        "subject_ref": "0123456789abcdef",
        "observation": {"n_days": 66, "coverage_ratio": 0.91},
        "channels": {
            "act_steps": {"mean": 9800.0, "sd": 3100.0, "missing_rate": 0.0},
            "slp_efficiency": {"mean": 83.2, "sd": 5.1, "missing_rate": 0.0},
        },
        "clock": {"slp_midsleep_hour": {"mean_hour": 3.4, "circular_sd_hours": 1.1}},
        "series": {"day_index": [0, 7, 14], "channels": {"act_steps": [9000, 10000, 8000]}},
    }
    guards.assert_payload_is_label_free(payload, context="test")
    guards.assert_payload_is_mmse_free(payload, context="test")


def test_short_tokens_do_not_false_positive():
    """'cn'/'dem'/'ad'/'y' must never match as substrings of ordinary names."""

    benign = [
        "act_daily_movement",
        "slp_efficiency",
        "day_index",
        "academic_free_field",
        "cadence_mean",
        "modem_free_name",
        "yesterday_delta",
    ]
    assert guards.find_label_like_names(benign) == []


# --- 3. MMSE containment ---------------------------------------------------- #
@pytest.mark.parametrize(
    "name", ["mmse__total", "MMSE_TOTAL", "Q13_2", "mini_mental_score", "cognitivefunction_block"]
)
def test_mmse_names_are_rejected_where_forbidden(name):
    with pytest.raises(guards.LeakageError):
        guards.assert_names_are_mmse_free([name], context="test")


def test_mmse_payload_is_rejected():
    with pytest.raises(guards.LeakageError):
        guards.assert_payload_is_mmse_free({"cognitive": {"mmse_total": 27}}, context="test")


# --- 5/6. Gemini response validation --------------------------------------- #
def _valid_response() -> dict[str, float]:
    return {name: 0.5 for name in FEATURE_NAMES}


def test_valid_response_is_accepted():
    assert set(validate_feature_payload(_valid_response())) == set(FEATURE_NAMES)


def test_missing_field_is_rejected():
    payload = _valid_response()
    payload.pop(FEATURE_NAMES[0])
    with pytest.raises(ValueError, match="missing required fields"):
        validate_feature_payload(payload)


@pytest.mark.parametrize("bad_value", [1.5, -0.01, float("nan"), "0.4", None, True])
def test_out_of_contract_values_are_rejected(bad_value):
    payload = _valid_response()
    payload[FEATURE_NAMES[0]] = bad_value
    with pytest.raises(ValueError):
        validate_feature_payload(payload)


def test_unexpected_extra_field_is_rejected():
    payload = _valid_response()
    payload["free_text_explanation"] = "..."
    with pytest.raises(ValueError, match="unexpected fields"):
        validate_feature_payload(payload)


# --- 9. feature names must not imply a diagnosis ---------------------------- #
def test_gemini_feature_names_are_neutral():
    guards.assert_names_are_label_free(FEATURE_NAMES, context="schema")
    guards.assert_names_are_mmse_free(FEATURE_NAMES, context="schema")
    guards.assert_names_are_label_free(design_matrix_names(), context="design matrix")
    assert 8 <= len(FEATURE_NAMES) <= 15
    assert set(response_schema()["required"]) == set(FEATURE_NAMES)


def test_prompts_contain_no_class_or_test_names():
    guards.assert_prompt_has_no_class_names(SYSTEM_PROMPT, context="system")
    guards.assert_prompt_has_no_class_names(USER_PROMPT_TEMPLATE, context="user")
    assert len(schema_hash()) == 64
    assert len(prompt_hash()) == 64


# --- 12. no hard-coded credentials ----------------------------------------- #
def test_no_api_key_is_hard_coded():
    suspicious = re.compile(
        r"(AIza[0-9A-Za-z_\-]{10,})|(api_key\s*=\s*[\"'][^\"'{}\s]{8,}[\"'])", re.I
    )
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert not suspicious.search(text), f"possible hard-coded credential in {path.name}"
    client = (PACKAGE_ROOT / "gemini_client.py").read_text(encoding="utf-8")
    assert "os.environ.get(self.config.api_key_env" in client
