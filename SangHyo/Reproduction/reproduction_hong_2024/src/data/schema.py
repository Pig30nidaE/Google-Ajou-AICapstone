"""Column names, the paper's 32-feature list, and the fail-closed denylist.

Everything here is traceable to Hong et al. (2024) Table 4 (the feature list) and
Table A1 (the per-feature description and format).  Where Table 4 and Table A1
disagree, Table A1 wins, because Table A1 is the one that matches the AI-Hub
column names.  ``paper_data_mapping.md`` records every one of those decisions.

No name here is a guess: each derived feature was checked numerically against the
raw file before it was written down (see ``paper_data_mapping.md`` §3).
"""

from __future__ import annotations

# --- canonical internal columns ----------------------------------------------
SUBJECT_ID = "subject_id"
RAW_ROW_ID = "raw_row_id"
DATE_COL = "sleep_date"
LABEL_COL = "label"
DIAGNOSIS_COL = "DIAG_NM"
SPLIT_ORIGIN = "split_origin"

#: Raw keys in the AI-Hub CSVs.
RAW_SUBJECT_KEY = "EMAIL"
RAW_LABEL_KEY = "SAMPLE_EMAIL"

#: ``DIAG_NM`` values.  The paper writes NC / MCI / DE; AI-Hub writes CN / MCI / Dem.
NEGATIVE_DIAGNOSIS = "CN"
POSITIVE_DIAGNOSES = ("MCI", "Dem")


# --- Hong et al. (2024) Table 4: the 32 input features ------------------------
#: Features that are read straight out of the AI-Hub sleep CSV, mapped
#: ``paper_name -> aihub_column``.  Table 4 prints ``skin_temperature_*`` but
#: Table A1 prints ``sleep_temperature_*``, which is the actual column name.
PASSTHROUGH_FEATURES: dict[str, str] = {
    "sleep_awake": "sleep_awake",
    "sleep_deep": "sleep_deep",
    "sleep_duration": "sleep_duration",
    "sleep_efficiency": "sleep_efficiency",
    "sleep_light": "sleep_light",
    "sleep_rem": "sleep_rem",
    "sleep_midpoint_time": "sleep_midpoint_time",
    "sleep_midpoint_at_delta": "sleep_midpoint_at_delta",
    "sleep_onset_latency": "sleep_onset_latency",
    "sleep_restless": "sleep_restless",
    "skin_temperature_delta": "sleep_temperature_delta",
    "skin_temperature_deviation": "sleep_temperature_deviation",
    "sleep_total": "sleep_total",
    "sleep_breath_average": "sleep_breath_average",
    "sleep_hr_average": "sleep_hr_average",
    # Table A1 defines sleep_hr_min as "Minimum value of heart rate per minute".
    # sleep_hr_lowest equals min(non-zero sleep_hr_5min) on 100% of rows, so this
    # is an identity, not a name-similarity substitution.
    "sleep_hr_min": "sleep_hr_lowest",
}

#: Features the paper lists but the CSV does not contain; they are computed in
#: ``loader.py`` from the intraday ``CONVERT(... USING utf8)`` series or from the
#: bedtime timestamps.  Mapped ``paper_name -> source column``.
DERIVED_FEATURES: dict[str, str] = {
    "sleep_hypnogram_average": "CONVERT(sleep_hypnogram_5min USING utf8)",
    "sleep_hr_max": "CONVERT(sleep_hr_5min USING utf8)",
    "sleep_hr_median": "CONVERT(sleep_hr_5min USING utf8)",
    "rmssd_average": "CONVERT(sleep_rmssd_5min USING utf8)",
}

#: Table 4 prints ``strat4``; Table A1 prints ``start1-6``.  ``start4`` is used.
START_ONEHOT = tuple(f"start{i}" for i in range(1, 7))
END_ONEHOT = tuple(f"end{i}" for i in range(1, 7))

#: Table A1: six four-hour bins, 0-4, 4-8, 8-12, 12-16, 16-20, 20-24 o'clock.
TIME_BIN_HOURS = 4
N_TIME_BINS = 6

#: The full 32, in Table 4's order.
PAPER_FEATURES: tuple[str, ...] = (
    "sleep_awake",
    "sleep_deep",
    "sleep_duration",
    "sleep_efficiency",
    "sleep_light",
    "sleep_rem",
    "sleep_midpoint_time",
    "sleep_midpoint_at_delta",
    "sleep_onset_latency",
    "sleep_restless",
    "skin_temperature_delta",
    "skin_temperature_deviation",
    "sleep_total",
    "sleep_hypnogram_average",
    *START_ONEHOT,
    *END_ONEHOT,
    "sleep_breath_average",
    "sleep_hr_average",
    "sleep_hr_min",
    "sleep_hr_max",
    "sleep_hr_median",
    "rmssd_average",
)

#: Intraday series columns.  The plain ``sleep_hr_5min`` / ``sleep_hypnogram_5min``
#: / ``sleep_rmssd_5min`` columns hold a literal "..." placeholder in every row;
#: the real series live in the ``CONVERT(... USING utf8)`` twins.
INTRADAY_SOURCE = {
    "hr": "CONVERT(sleep_hr_5min USING utf8)",
    "hypnogram": "CONVERT(sleep_hypnogram_5min USING utf8)",
    "rmssd": "CONVERT(sleep_rmssd_5min USING utf8)",
}
INTRADAY_PLACEHOLDER_COLUMNS = ("sleep_hr_5min", "sleep_hypnogram_5min", "sleep_rmssd_5min")

#: Hypnogram code book from Table A1.
HYPNOGRAM_CODES = {1: "deep", 2: "light", 3: "rem", 4: "awake"}

#: 0 is the device's "no reading" sentinel in the hr and rmssd series.  It is not
#: a plausible heart rate, so it is excluded from the summary statistics.
INTRADAY_MISSING_SENTINEL = 0


# --- fail-closed denylist -----------------------------------------------------
#: Anything that identifies the subject, restates the label, or is a cognitive
#: test used to make the diagnosis.  ``audit/leakage.py`` raises if any of these
#: reaches a feature matrix, in every experiment, with no config override.
FORBIDDEN_FEATURE_SUBSTRINGS: tuple[str, ...] = (
    "email",
    "diag",
    "mmse",
    "snsb",
    "subject",
    "label",
    "sample_",
    "doctor",
    "target",
    "y_true",
)

#: Sleep-score columns.  Oura derives these from the same sleep stages the paper
#: already uses, and Table 4 does not list them, so they stay out of the paper
#: feature set.  Not a leak, just not the paper's 32.
NON_PAPER_SLEEP_COLUMNS: tuple[str, ...] = (
    "sleep_score",
    "sleep_score_alignment",
    "sleep_score_deep",
    "sleep_score_disturbances",
    "sleep_score_efficiency",
    "sleep_score_latency",
    "sleep_score_rem",
    "sleep_score_total",
    "sleep_is_longest",
    "sleep_period_id",
    "sleep_rmssd",
    "sleep_hr_lowest",
)


def is_forbidden_feature(name: str) -> bool:
    """True when *name* must never appear in a feature matrix."""
    lowered = str(name).lower()
    return any(token in lowered for token in FORBIDDEN_FEATURE_SUBSTRINGS)


def diagnosis_to_label(diagnosis: str) -> int:
    """CN -> 0, MCI/Dem -> 1 (the paper's NC vs {MCI, DE} collapse)."""
    text = str(diagnosis).strip()
    if text == NEGATIVE_DIAGNOSIS:
        return 0
    if text in POSITIVE_DIAGNOSES:
        return 1
    raise ValueError(
        f"unknown diagnosis {diagnosis!r}; expected one of "
        f"{(NEGATIVE_DIAGNOSIS, *POSITIVE_DIAGNOSES)}"
    )
