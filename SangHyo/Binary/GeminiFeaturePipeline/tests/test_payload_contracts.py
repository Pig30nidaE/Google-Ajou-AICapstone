"""Payload construction, caching and dry-run behaviour. No API call, no fit.

Every test builds a small synthetic daily table, so the suite runs without the
real ``Data/`` tree and without touching Google Drive.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from SangHyo.Binary.GeminiFeaturePipeline.config import GeminiConfig, PayloadConfig
from SangHyo.Binary.GeminiFeaturePipeline.data import (
    DAILY_CHANNELS,
    HOURLY_MET_COLUMNS,
    INTENSITY_SHARE_COLUMNS,
    SLEEP_PHASE_SHARE_COLUMNS,
    parse_slash_series,
)
from SangHyo.Binary.GeminiFeaturePipeline.gemini_client import GeminiFeatureExtractor
from SangHyo.Binary.GeminiFeaturePipeline.guards import hash_subject_id
from SangHyo.Binary.GeminiFeaturePipeline.payload import (
    build_subject_payload,
    payload_hash,
    payload_size_bytes,
)


def synthetic_daily_frame(n_days: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(n_days):
        row: dict[str, float] = {
            "subject_id": "subject-a",
            "day_index": day,
            "day_of_week": day % 7,
            "is_weekend": int(day % 7 >= 5),
        }
        for channel in DAILY_CHANNELS:
            if channel.endswith("_hour"):
                row[channel] = float((23.0 + rng.normal(0, 0.5)) % 24.0)
            elif channel.endswith("_ratio") or "relative_amplitude" in channel:
                row[channel] = float(np.clip(rng.normal(0.4, 0.1), 0, 1))
            else:
                row[channel] = float(abs(rng.normal(100, 20)))
        for column in HOURLY_MET_COLUMNS:
            row[column] = float(abs(rng.normal(1.4, 0.4)))
        for column in INTENSITY_SHARE_COLUMNS + SLEEP_PHASE_SHARE_COLUMNS:
            row[column] = float(np.clip(rng.normal(0.25, 0.05), 0, 1))
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture()
def payload() -> dict:
    frame = synthetic_daily_frame()
    return build_subject_payload(
        frame, subject_ref=hash_subject_id("subject-a", salt="t"), payload_config=PayloadConfig()
    )


def test_payload_has_no_identifier_and_no_absolute_date(payload):
    text = json.dumps(payload)
    assert "@" not in text and "rowan" not in text
    assert "2020" not in text and "2021" not in text
    assert payload["subject_ref"] == hash_subject_id("subject-a", salt="t")
    assert len(payload["subject_ref"]) == 16


def test_payload_structure_is_complete(payload):
    for section in (
        "observation",
        "channels",
        "clock",
        "hourly_profile",
        "intensity_profile",
        "sleep_phase_profile",
        "weekly_summary",
        "series",
    ):
        assert section in payload, section
    assert payload["observation"]["n_days"] == 60
    assert len(payload["hourly_profile"]["mean_met_by_hour"]) == 24
    for statistics in payload["channels"].values():
        assert {"mean", "sd", "n_valid", "missing_rate", "trend_per_week"} <= set(statistics)


def test_series_is_downsampled_and_order_preserving(payload):
    day_index = payload["series"]["day_index"]
    assert day_index == sorted(day_index)
    assert len(day_index) <= PayloadConfig().max_series_points
    for values in payload["series"]["channels"].values():
        assert len(values) == len(day_index)


def test_clock_channels_use_circular_statistics(payload):
    """Values around midnight must not produce a ~12h spread."""

    assert "slp_midsleep_hour" not in payload["channels"]
    circular = payload["clock"]["slp_bedtime_hour"]
    assert circular["circular_sd_hours"] is not None
    assert circular["circular_sd_hours"] < 2.0


def test_missing_days_are_reported_not_invented():
    frame = synthetic_daily_frame(n_days=40)
    frame = frame[frame["day_index"] % 4 != 0].reset_index(drop=True)  # punch holes
    frame.loc[frame.index[:5], "act_steps"] = np.nan
    built = build_subject_payload(
        frame, subject_ref="0" * 16, payload_config=PayloadConfig()
    )
    assert built["observation"]["n_days"] == len(frame)
    assert built["observation"]["coverage_ratio"] < 1.0
    assert built["channels"]["act_steps"]["missing_rate"] > 0


def test_intraday_parser_handles_ellipsis_and_garbage():
    assert parse_slash_series("...").size == 0
    assert parse_slash_series("1/2/3").tolist() == [1.0, 2.0, 3.0]
    parsed = parse_slash_series("1//x/3")
    assert parsed.size == 3 and np.isnan(parsed[1])


# --- 7. identical payloads must not trigger a second call -------------------- #
def test_identical_payloads_share_one_cache_fingerprint(tmp_path, payload):
    extractor = GeminiFeatureExtractor(GeminiConfig(dry_run=True), cache_root=tmp_path)
    first = extractor.request_fingerprint(payload)
    second = extractor.request_fingerprint(json.loads(json.dumps(payload)))
    assert first == second
    assert extractor.cache_path(first).name == f"{first}.json"
    assert payload_hash(payload) == payload_hash(payload)


def test_fingerprint_changes_when_the_contract_changes(tmp_path, payload):
    base = GeminiFeatureExtractor(GeminiConfig(dry_run=True), cache_root=tmp_path)
    other_model = GeminiFeatureExtractor(
        GeminiConfig(dry_run=True, model="gemini-2.5-pro"), cache_root=tmp_path
    )
    hotter = GeminiFeatureExtractor(
        GeminiConfig(dry_run=True, temperature=0.7), cache_root=tmp_path
    )
    more_thinking = GeminiFeatureExtractor(
        GeminiConfig(dry_run=True, thinking_level="low"), cache_root=tmp_path
    )
    assert base.request_fingerprint(payload) != other_model.request_fingerprint(payload)
    assert base.request_fingerprint(payload) != hotter.request_fingerprint(payload)
    assert base.request_fingerprint(payload) != more_thinking.request_fingerprint(payload)


def test_gemini3_thinking_level_is_sent_as_an_enum(tmp_path):
    class _ThinkingLevel:
        MINIMAL = "MINIMAL"

    class _ThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Types:
        ThinkingLevel = _ThinkingLevel
        ThinkingConfig = _ThinkingConfig

    extractor = GeminiFeatureExtractor(
        GeminiConfig(thinking_level="minimal", thinking_budget=None),
        cache_root=tmp_path,
    )
    thinking = extractor._thinking_config(_Types)
    assert thinking.kwargs == {"thinking_level": "MINIMAL"}


def test_thinking_config_fails_closed_with_an_old_sdk(tmp_path):
    extractor = GeminiFeatureExtractor(
        GeminiConfig(thinking_level="minimal"), cache_root=tmp_path
    )
    with pytest.raises(RuntimeError, match="google-genai"):
        extractor._thinking_config(object())


def test_dry_run_reports_without_calling_the_api(tmp_path, payload):
    extractor = GeminiFeatureExtractor(GeminiConfig(dry_run=True), cache_root=tmp_path)
    report = extractor.dry_run_report({"subject-a": payload})
    assert report["api_calls_executed"] == 0
    assert report["requests_that_would_be_sent"] == 1
    assert report["payload_bytes_total"] == payload_size_bytes(payload)
    assert report["cache_directory"].startswith(str(tmp_path))


def test_offline_mode_reports_cache_miss_instead_of_calling(tmp_path, payload):
    extractor = GeminiFeatureExtractor(GeminiConfig(offline=True), cache_root=tmp_path)
    results, summary = extractor.extract({"subject-a": payload})
    assert results["subject-a"].status == "cache_miss"
    assert summary.api_calls == 0 and summary.fresh == 0


def test_max_tokens_failure_is_recorded_and_not_retried_unchanged(
    tmp_path, payload, monkeypatch
):
    extractor = GeminiFeatureExtractor(
        GeminiConfig(
            max_retries=6,
            min_interval_seconds=0,
            initial_backoff_seconds=0,
        ),
        cache_root=tmp_path,
    )
    calls = 0

    def truncated(_prompt):
        nonlocal calls
        calls += 1
        return (
            '{"routine_regularity": "unterminated',
            {
                "prompt_tokens": 100,
                "output_tokens": 20,
                "thinking_tokens": 8170,
                "billed_output_tokens": 8190,
                "total_tokens": 8290,
            },
            "MAX_TOKENS",
        )

    monkeypatch.setattr(extractor, "_call_api", truncated)
    results, summary = extractor.extract({"subject-a": payload})

    result = results["subject-a"]
    assert calls == 1
    assert summary.api_calls == 1 and summary.failed == 1
    assert result.status == "failed"
    assert result.record["last_finish_reason"] == "MAX_TOKENS"
    assert result.record["attempts"][0]["retryable"] is False
    assert "response truncated" in result.error
    assert summary.output_tokens == 20
    assert summary.thinking_tokens == 8170
    assert summary.to_dict()["billed_output_tokens"] == 8190


def test_transient_503_retries_then_counts_thinking_tokens(
    tmp_path, payload, monkeypatch
):
    extractor = GeminiFeatureExtractor(
        GeminiConfig(
            max_retries=2,
            min_interval_seconds=0,
            initial_backoff_seconds=0,
            price_per_million_input_tokens=1.0,
            price_per_million_output_tokens=2.0,
        ),
        cache_root=tmp_path,
    )
    calls = 0
    body = json.dumps(
        {
            name: 0.5
            for name in (
                "routine_regularity",
                "sleep_timing_variability",
                "sleep_continuity",
                "activity_volume_stability",
                "sustained_exertion",
                "diurnal_contrast",
                "long_term_trend_direction",
                "short_term_volatility",
                "weekday_weekend_divergence",
                "cross_domain_coherence",
                "atypical_day_frequency",
                "observation_reliability",
            )
        }
    )

    def flaky(_prompt):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("503 UNAVAILABLE: temporarily overloaded")
        return (
            body,
            {
                "prompt_tokens": 100,
                "output_tokens": 50,
                "thinking_tokens": 25,
                "billed_output_tokens": 75,
                "total_tokens": 175,
            },
            "STOP",
        )

    monkeypatch.setattr(extractor, "_call_api", flaky)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    results, summary = extractor.extract({"subject-a": payload})

    assert calls == 2
    assert results["subject-a"].status == "fresh"
    assert summary.api_calls == 2
    assert summary.output_tokens == 50
    assert summary.thinking_tokens == 25
    assert summary.to_dict()["billed_output_tokens"] == 75
    assert summary.estimated_cost_usd == pytest.approx(0.00025)
