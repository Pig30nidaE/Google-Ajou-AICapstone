"""Column contracts for the AI-Hub dementia lifelog dataset.

Everything here was verified against the distributed CSVs on 2026-08-02; see
``paper_data_mapping.md`` for the paper-vs-data comparison that produced it.
"""

from __future__ import annotations

# --- keys --------------------------------------------------------------------
SOURCE_PERSON_KEY = "EMAIL"          # activity / sleep
LABEL_PERSON_KEY = "SAMPLE_EMAIL"    # mmse / label files
SUBJECT_ID = "subject_id"            # our pseudonymous id
DATE_COL = "date"
LABEL_COL = "label"
DIAGNOSIS_COL = "DIAG_NM"

POSITIVE_DIAGNOSES = ("MCI", "Dem")
NEGATIVE_DIAGNOSES = ("CN",)

# --- timestamp sources used to derive the missing `date` column ---------------
# The paper's Table 1 lists a `date` column. The distributed CSVs do not have one.
ACTIVITY_DATE_SOURCE = "activity_day_start"
SLEEP_DATE_SOURCES = {
    "bedtime_end": "sleep_bedtime_end",
    "bedtime_start": "sleep_bedtime_start",
}

# --- the paper's own drop list (Table 12 / Table 9), verbatim ------------------
# Only `activity_non_wear` and `activity_inactivity_alerts` exist in the real data;
# the other three are no-ops that pandas swallows via errors="ignore".
PAPER_DROP_COLUMNS = (
    "activity_non_wear",
    "nonwear",
    "activity_inactivity_alerts",
    "check",
    "timezone",
)

# --- features produced by running the paper's code verbatim -------------------
# groupby(EMAIL).mean(numeric_only=True) then drop(PAPER_DROP_COLUMNS).
PAPER_CODE_ACTIVITY_FEATURES = (
    "activity_average_met",
    "activity_cal_active",
    "activity_cal_total",
    "activity_daily_movement",
    "activity_high",
    "activity_inactive",
    "activity_low",
    "activity_medium",
    "activity_met_min_high",
    "activity_met_min_inactive",
    "activity_met_min_low",
    "activity_met_min_medium",
    "activity_rest",
    "activity_score",
    "activity_score_meet_daily_targets",
    "activity_score_move_every_hour",
    "activity_score_recovery_time",
    "activity_score_stay_active",
    "activity_score_training_frequency",
    "activity_score_training_volume",
    "activity_steps",
    "activity_total",
)

PAPER_CODE_SLEEP_FEATURES = (
    "sleep_awake",
    "sleep_breath_average",
    "sleep_deep",
    "sleep_duration",
    "sleep_efficiency",
    "sleep_hr_average",
    "sleep_hr_lowest",
    "sleep_is_longest",
    "sleep_light",
    "sleep_midpoint_at_delta",
    "sleep_midpoint_time",
    "sleep_onset_latency",
    "sleep_period_id",
    "sleep_rem",
    "sleep_restless",
    "sleep_rmssd",
    "sleep_score",
    "sleep_score_alignment",
    "sleep_score_deep",
    "sleep_score_disturbances",
    "sleep_score_efficiency",
    "sleep_score_latency",
    "sleep_score_rem",
    "sleep_score_total",
    "sleep_temperature_delta",
    "sleep_temperature_deviation",
    "sleep_total",
)

#: 49 features -- what the paper's code actually yields, against its claim of 58.
PAPER_CODE_FEATURES = PAPER_CODE_ACTIVITY_FEATURES + PAPER_CODE_SLEEP_FEATURES

#: Names the paper lists that do not exist in the distributed CSVs.
PAPER_NAMES_ABSENT_FROM_DATA = {
    "active_low": "activity_low",              # spelling error in the paper
    "sleep_temperature_trend_deviation": None,  # no counterpart at all
}

#: Names the paper lists that are strings, so mean(numeric_only=True) drops them.
PAPER_NAMES_DROPPED_AS_NON_NUMERIC = (
    "activity_class_5min",
    "sleep_bedtime_start",
    "sleep_bedtime_end",
    "sleep_hr_5min",
    "sleep_hypnogram_5min",
    "sleep_rmssd_5min",
)

#: The real 5-minute / 1-minute series, which the paper never mentions.
CONVERT_SERIES_COLUMNS = (
    "CONVERT(activity_class_5min USING utf8)",
    "CONVERT(activity_met_1min USING utf8)",
    "CONVERT(sleep_hr_5min USING utf8)",
    "CONVERT(sleep_hypnogram_5min USING utf8)",
    "CONVERT(sleep_rmssd_5min USING utf8)",
)

#: Constant across all 12,183 rows -- carries no information.
ZERO_VARIANCE_FEATURES = ("sleep_is_longest",)

#: Session bookkeeping rather than physiology.
ADMINISTRATIVE_FEATURES = ("sleep_period_id",)

# --- forbidden features (fail closed) -----------------------------------------
TARGET_LEAKING_COLUMNS = frozenset(
    {"DIAG_NM", "DIAG_NIM", "DIAG_SEQ", "label", "y", "target", "diagnosis"}
)

IDENTIFIER_COLUMNS = frozenset(
    {
        "EMAIL",
        "SAMPLE_EMAIL",
        "email",
        "subject_id",
        "subject_key",
        "DOCTOR_NM",
        "MMSE_NUM",
        "MMSE_KIND",
        "source_file",
        "split_origin",
    }
)

MMSE_ITEM_COLUMNS = (
    "Q01", "Q02", "Q03", "Q04", "Q05", "Q06", "Q07", "Q08", "Q09", "Q10",
    "Q11_1", "Q11_2", "Q11_3",
    "Q12_1", "Q12_2", "Q12_3", "Q12_4", "Q12_5", "Q12_TOTAL",
    "Q13_1", "Q13_2", "Q13_3",
    "Q14_1", "Q14_2", "Q15",
    "Q16_1", "Q16_2", "Q16_3",
    "Q17", "Q18", "Q19",
)

COGNITIVE_TEST_COLUMNS = frozenset({"TOTAL", *MMSE_ITEM_COLUMNS})

#: MMSE items are coded 2 = correct, 1 = incorrect (verified against the CSVs).
MMSE_CORRECT_CODE = 2
MMSE_INCORRECT_CODE = 1

#: 0 is out of scale.  Exactly one Validation subject (a Dem case) has 0 on every
#: item and TOTAL = 0, which reads as "not administered", not "scored zero".
#: Treat it as missing rather than as a very low score.
MMSE_MISSING_SENTINEL = 0

#: Constant 0 for all 174 subjects despite Table 7 describing it as the Q12
#: subtotal.  It is not the sum of Q12_1..Q12_5 and carries no information.
MMSE_BROKEN_COLUMNS = ("Q12_TOTAL",)

#: TOTAL == number of items scored 2, over the 30 items excluding Q12_TOTAL.
#: Verified to hold for 100% of subjects in both splits.
MMSE_TOTAL_IS_COUNT_OF_CORRECT = True

#: Zero variance within the Training split -- everyone answered correctly.
MMSE_CONSTANT_IN_TRAINING = ("Q11_1", "Q11_2", "Q11_3", "Q14_2")


def forbidden_feature_columns(*, include_cognitive: bool = False) -> frozenset[str]:
    """Columns that must never appear in a main-analysis feature matrix."""
    forbidden = TARGET_LEAKING_COLUMNS | IDENTIFIER_COLUMNS
    if not include_cognitive:
        forbidden = forbidden | COGNITIVE_TEST_COLUMNS
    return frozenset(forbidden)


def is_cognitive_like(column: str) -> bool:
    """Heuristic guard for MMSE/SNSB-derived names beyond the exact allowlist."""
    lowered = str(column).lower()
    if column in COGNITIVE_TEST_COLUMNS:
        return True
    markers = ("mmse", "snsb", "kmmse", "k_mmse", "cognitive_score", "diag")
    return any(marker in lowered for marker in markers)
