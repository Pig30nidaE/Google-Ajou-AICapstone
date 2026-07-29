"""Fixed Gemini output contract: 12 diagnosis-neutral continuous features.

Design decisions and their sources:

* **A single fixed schema shared by every subject** - "global rubric" idea from
  *LLMs can construct powerful representations...* (Demirel et al.), Section 2.1.
  Unlike that paper we do **not** synthesise the schema from labelled examples:
  <critical_label_rule> forbids showing any label to the model, so the field list
  is fixed a priori from the wearable literature and from the observed data
  layout, and is frozen before any label is touched.
* **Bounded 0.0-1.0 numeric fields returned as strict JSON** - the free-dialogue
  cognitive-decline paper (de Arriba-Perez et al., Listing 2) uses exactly this
  contract; their features are also model-scored quantities in (0, 1) that are
  then consumed by a conventional classifier.
* **"Extract facts only, no predictions/risk levels/conclusions"** - wording
  adapted from the same rubric paper's Appendix D.1/D.2 prompts.

Every field is a *description of behaviour*, never a statement about health.
``schema_hash()`` binds a cache entry to this exact contract, so changing a
definition invalidates previously cached Gemini answers instead of silently
mixing two feature generations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

from . import SCHEMA_VERSION
from .guards import assert_names_are_label_free, assert_names_are_mmse_free

__all__ = [
    "FeatureSpec",
    "FEATURE_SPECS",
    "FEATURE_NAMES",
    "DESIGN_MATRIX_PREFIX",
    "response_schema",
    "schema_hash",
    "feature_instructions",
    "design_matrix_names",
    "validate_feature_payload",
]

DESIGN_MATRIX_PREFIX = "gemini__"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    definition: str
    high_means: str
    low_means: str
    required_inputs: tuple[str, ...]
    missing_policy: str
    minimum: float = 0.0
    maximum: float = 1.0

    @property
    def directionality(self) -> str:
        return f"1.0 = {self.high_means}; 0.0 = {self.low_means}"


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="routine_regularity",
        definition=(
            "How repeatable the person's 24-hour pattern is across the observed days, "
            "judged from the hourly activity profile, the spread of bed/wake clock hours "
            "and the day-to-day dispersion of the daily channels."
        ),
        high_means="the same daily pattern repeats almost every day",
        low_means="the daily pattern differs strongly from day to day",
        required_inputs=("clock", "hourly_profile", "channels"),
        missing_policy="Judge only from days that are present; lower observation_reliability instead of guessing.",
    ),
    FeatureSpec(
        name="sleep_timing_variability",
        definition=(
            "Dispersion of sleep timing: circular spread of bedtime, wake time and "
            "mid-sleep hour, and the variability of sleep duration."
        ),
        high_means="sleep timing moves a lot between days",
        low_means="the person falls asleep and wakes at nearly the same clock time",
        required_inputs=("clock", "channels.slp_total_minutes", "series.slp_midsleep_hour"),
        missing_policy="If clock statistics are absent, return 0.5 and lower observation_reliability.",
    ),
    FeatureSpec(
        name="sleep_continuity",
        definition=(
            "How uninterrupted the sleep episodes are, from efficiency, awake minutes, "
            "restless counts, sleep-phase transition rate and awake share."
        ),
        high_means="sleep is consolidated with few interruptions",
        low_means="sleep is fragmented with much wake time and many transitions",
        required_inputs=("channels.slp_efficiency", "channels.slp_awake_minutes", "sleep_phase_profile"),
        missing_policy="Use the subset of sleep statistics that is present.",
    ),
    FeatureSpec(
        name="activity_volume_stability",
        definition=(
            "Stability of the amount of daily movement (steps, movement, active minutes) "
            "across days, i.e. low relative dispersion rather than a high average."
        ),
        high_means="daily activity volume is nearly constant across days",
        low_means="daily activity volume swings widely between days",
        required_inputs=("channels.act_steps", "channels.act_daily_movement", "series"),
        missing_policy="Ignore channels with missing_rate above 0.5.",
    ),
    FeatureSpec(
        name="sustained_exertion",
        definition=(
            "How much of the observed time is spent above light intensity and how often "
            "such effort is maintained rather than appearing as isolated spikes."
        ),
        high_means="moderate/high intensity effort occurs regularly and is maintained",
        low_means="the days consist almost entirely of rest and inactive time",
        required_inputs=("channels.act_medium_minutes", "channels.act_high_minutes", "intensity_profile"),
        missing_policy="If intensity shares are absent, use MET-based statistics only.",
    ),
    FeatureSpec(
        name="diurnal_contrast",
        definition=(
            "Separation between the most active 10 hours and the least active 5 hours of "
            "the day, i.e. how sharply active and rest periods are separated."
        ),
        high_means="a strong, clearly separated active period and rest period",
        low_means="activity is flat across the whole 24 hours",
        required_inputs=("hourly_profile", "channels.act_intraday_relative_amplitude"),
        missing_policy="If the hourly profile is missing, return 0.5 and lower observation_reliability.",
    ),
    FeatureSpec(
        name="long_term_trend_direction",
        definition=(
            "Direction of the slow change across the whole observation window, combining "
            "the per-channel weekly trends of activity volume and sleep quality. "
            "This is an ordered direction, not a magnitude."
        ),
        high_means="activity and sleep quality clearly increase over the window",
        low_means="activity and sleep quality clearly decrease over the window",
        required_inputs=("channels.*.trend_per_week", "series", "weekly_summary"),
        missing_policy="Return 0.5 (flat) when fewer than three weeks are observed.",
    ),
    FeatureSpec(
        name="short_term_volatility",
        definition=(
            "Size of day-to-day fluctuations after the slow trend is set aside, i.e. how "
            "jumpy consecutive days are relative to the person's own typical level."
        ),
        high_means="consecutive days differ strongly from each other",
        low_means="consecutive days are very similar",
        required_inputs=("series", "channels.*.cv"),
        missing_policy="Use only channels with at least ten valid days.",
    ),
    FeatureSpec(
        name="weekday_weekend_divergence",
        definition=(
            "Magnitude of the systematic difference between weekday and weekend behaviour "
            "relative to the person's overall dispersion."
        ),
        high_means="weekends look clearly different from weekdays",
        low_means="weekdays and weekends are indistinguishable",
        required_inputs=("channels.*.weekend_minus_weekday", "observation.weekend_days"),
        missing_policy="Return 0.5 when fewer than four weekend days are observed.",
    ),
    FeatureSpec(
        name="cross_domain_coherence",
        definition=(
            "Degree to which the activity domain and the sleep domain move together, "
            "e.g. whether more active periods coincide with better sleep statistics "
            "within the same person."
        ),
        high_means="the activity and sleep domains change together consistently",
        low_means="the two domains change independently or in opposite directions",
        required_inputs=("series", "weekly_summary"),
        missing_policy="Return 0.5 when one of the two domains is largely missing.",
    ),
    FeatureSpec(
        name="atypical_day_frequency",
        definition=(
            "Share of observed days that deviate markedly from this person's own usual "
            "pattern, judged against their own distribution rather than any external norm."
        ),
        high_means="many days depart strongly from the person's own routine",
        low_means="almost every day matches the person's own routine",
        required_inputs=("series", "channels.*.iqr", "channels.*.p10", "channels.*.p90"),
        missing_policy="Count only days that are actually present in the series.",
    ),
    FeatureSpec(
        name="observation_reliability",
        definition=(
            "How well the supplied statistics support the other eleven answers: number of "
            "observed days, coverage of the observation window, missing rates and how much "
            "of the intraday profile is present."
        ),
        high_means="many days, dense coverage and almost no missing channels",
        low_means="few days, sparse coverage or heavy missingness",
        required_inputs=("observation", "channels.*.missing_rate", "channels.*.n_valid"),
        missing_policy="This field must always be answerable; it describes the input itself.",
    ),
)

FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)

assert_names_are_label_free(FEATURE_NAMES, context="Gemini output schema")
assert_names_are_mmse_free(FEATURE_NAMES, context="Gemini output schema")
if len(set(FEATURE_NAMES)) != len(FEATURE_NAMES):
    raise ValueError("Duplicate Gemini feature name")
if not 8 <= len(FEATURE_NAMES) <= 15:
    raise ValueError("The initial schema is specified as 8-15 continuous features")


def design_matrix_names() -> tuple[str, ...]:
    return tuple(f"{DESIGN_MATRIX_PREFIX}{name}" for name in FEATURE_NAMES)


def response_schema() -> dict[str, Any]:
    """JSON schema handed to Gemini structured output (google-genai ``response_schema``)."""

    return {
        "type": "object",
        "properties": {
            spec.name: {
                "type": "number",
                "description": (
                    f"{spec.definition} Scale 0.0-1.0. {spec.directionality}."
                ),
            }
            for spec in FEATURE_SPECS
        },
        "required": list(FEATURE_NAMES),
        "propertyOrdering": list(FEATURE_NAMES),
    }


def schema_hash() -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "specs": [asdict(spec) for spec in FEATURE_SPECS],
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def feature_instructions() -> str:
    """Human-readable field list embedded verbatim in the user prompt."""

    lines: list[str] = []
    for index, spec in enumerate(FEATURE_SPECS, start=1):
        lines.append(
            f"{index}. {spec.name} [{spec.minimum:.1f}-{spec.maximum:.1f}]\n"
            f"   definition: {spec.definition}\n"
            f"   direction: {spec.directionality}\n"
            f"   inputs to use: {', '.join(spec.required_inputs)}\n"
            f"   missing data: {spec.missing_policy}"
        )
    return "\n".join(lines)


def validate_feature_payload(payload: Mapping[str, Any]) -> dict[str, float]:
    """Check required fields, types and ranges; raise ``ValueError`` otherwise."""

    if not isinstance(payload, Mapping):
        raise ValueError(f"Gemini response is not a JSON object: {type(payload).__name__}")
    missing = [name for name in FEATURE_NAMES if name not in payload]
    if missing:
        raise ValueError(f"Gemini response is missing required fields: {missing}")
    unexpected = [str(key) for key in payload if key not in set(FEATURE_NAMES)]
    if unexpected:
        raise ValueError(f"Gemini response contains unexpected fields: {sorted(unexpected)}")
    values: dict[str, float] = {}
    for spec in FEATURE_SPECS:
        raw = payload[spec.name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{spec.name} is not numeric: {raw!r}")
        value = float(raw)
        if not (value == value) or value in (float("inf"), float("-inf")):
            raise ValueError(f"{spec.name} is not finite: {raw!r}")
        if not spec.minimum <= value <= spec.maximum:
            raise ValueError(
                f"{spec.name}={value} is outside [{spec.minimum}, {spec.maximum}]"
            )
        values[spec.name] = value
    return values
