"""Frozen, label-free wearable primitive catalogue supplied to Gemma.

Every value is computed deterministically from one subject's own recordings.
The catalogue contains names, units and measurement semantics only: it has no
class directions, cohort statistics, cut-points, patient examples or MMSE
fields.  Gemma can therefore compose meaningful feature programs without
seeing any outcome-bearing data.
"""

from __future__ import annotations

from .program_schema import PrimitiveSpec


PRIMITIVE_CATALOG: tuple[PrimitiveSpec, ...] = (
    # Circadian rhythm -------------------------------------------------------
    PrimitiveSpec("wi_met_IS", "circadian", "Repeatability of the aligned 24-hour MET profile across recorded days.", "unitless"),
    PrimitiveSpec("wi_met_IV", "circadian", "Successive-hour fragmentation of the aligned MET rhythm.", "unitless"),
    PrimitiveSpec("wi_met_RA", "circadian", "Relative amplitude between the most-active 10-hour and least-active 5-hour windows.", "ratio"),
    PrimitiveSpec("wi_met_M10", "circadian", "Mean MET in the most-active rolling 10-hour window.", "MET"),
    PrimitiveSpec("wi_met_L5", "circadian", "Mean MET in the least-active rolling 5-hour window.", "MET"),
    PrimitiveSpec("wi_met_M10_onset_sin", "circadian", "Sine coordinate of the local-clock onset of the most-active 10-hour window.", "unit circle"),
    PrimitiveSpec("wi_met_M10_onset_cos", "circadian", "Cosine coordinate of the local-clock onset of the most-active 10-hour window.", "unit circle"),
    PrimitiveSpec("wi_met_L5_onset_sin", "circadian", "Sine coordinate of the local-clock onset of the least-active 5-hour window.", "unit circle"),
    PrimitiveSpec("wi_met_L5_onset_cos", "circadian", "Cosine coordinate of the local-clock onset of the least-active 5-hour window.", "unit circle"),
    PrimitiveSpec("wi_met_daily_profile_std", "circadian", "Dispersion across the subject's mean 24 hourly MET bins.", "MET"),
    PrimitiveSpec("wd_circ_bedtime__circsd_h", "circadian", "Circular day-to-day spread of bedtime.", "hours"),
    PrimitiveSpec("wd_circ_waketime__circsd_h", "circadian", "Circular day-to-day spread of wake time.", "hours"),
    PrimitiveSpec("wd_circ_midpoint__circsd_h", "circadian", "Day-to-day spread of sleep midpoint.", "hours"),

    # Daily activity and fragmentation --------------------------------------
    PrimitiveSpec("wd_activity_steps__mean", "activity", "Mean daily step count.", "steps/day"),
    PrimitiveSpec("wd_activity_steps__std", "activity", "Day-to-day standard deviation of step count.", "steps/day"),
    PrimitiveSpec("wd_activity_steps__cv", "activity", "Relative day-to-day dispersion of step count.", "ratio"),
    PrimitiveSpec("wd_activity_steps__slope", "activity", "Linear change in daily step count over the observation order.", "steps/day per day"),
    PrimitiveSpec("wd_activity_steps__weekend_delta", "activity", "Weekend mean step count minus weekday mean step count.", "steps/day"),
    PrimitiveSpec("wd_activity_average_met__mean", "activity", "Mean of daily average MET.", "MET"),
    PrimitiveSpec("wd_activity_average_met__cv", "activity", "Relative day-to-day dispersion of average MET.", "ratio"),
    PrimitiveSpec("wd_activity_average_met__slope", "activity", "Linear change in daily average MET over the observation order.", "MET/day"),
    PrimitiveSpec("wd_activity_inactive__mean", "activity", "Mean daily inactive duration in the source representation.", "source-native duration/day"),
    PrimitiveSpec("wd_activity_inactive__cv", "activity", "Relative day-to-day dispersion of inactive duration.", "ratio"),
    PrimitiveSpec("wd_activity_high__mean", "activity", "Mean daily high-intensity duration in the source representation.", "source-native duration/day"),
    PrimitiveSpec("wd_activity_high__cv", "activity", "Relative day-to-day dispersion of high-intensity duration.", "ratio"),
    PrimitiveSpec("wi_met_active_bout_mean__mean", "activity", "Across-day mean of within-day active-bout mean length.", "minutes"),
    PrimitiveSpec("wi_met_sed_bout_mean__mean", "activity", "Across-day mean of within-day sedentary-bout mean length.", "minutes"),
    PrimitiveSpec("wi_met_transition_rate__mean", "activity", "Across-day mean transition rate between sedentary and active states.", "transitions/hour"),
    PrimitiveSpec("wi_met_entropy__mean", "activity", "Across-day mean entropy of minute-level MET values.", "nats"),
    PrimitiveSpec("wi_met_active_frac__mean", "activity", "Across-day mean fraction of minute-level samples marked active by the deterministic parser.", "ratio"),

    # Sleep timing, continuity and architecture -----------------------------
    PrimitiveSpec("wd_sleep_duration__mean", "sleep", "Mean daily sleep-duration field.", "source-native duration/night"),
    PrimitiveSpec("wd_sleep_duration__std", "sleep", "Day-to-day standard deviation of sleep duration.", "source-native duration/night"),
    PrimitiveSpec("wd_sleep_duration__cv", "sleep", "Relative day-to-day dispersion of sleep duration.", "ratio"),
    PrimitiveSpec("wd_sleep_duration__slope", "sleep", "Linear change in sleep duration over the observation order.", "source-native duration/night per day"),
    PrimitiveSpec("wd_sleep_efficiency__mean", "sleep", "Mean nightly sleep efficiency.", "source-native score or ratio"),
    PrimitiveSpec("wd_sleep_efficiency__std", "sleep", "Day-to-day standard deviation of sleep efficiency.", "source-native score or ratio"),
    PrimitiveSpec("wd_sleep_efficiency__cv", "sleep", "Relative day-to-day dispersion of sleep efficiency.", "ratio"),
    PrimitiveSpec("wd_sleep_efficiency__slope", "sleep", "Linear change in sleep efficiency over the observation order.", "source-native value/day"),
    PrimitiveSpec("wd_sleep_awake__mean", "sleep", "Mean nightly awake-duration field.", "source-native duration/night"),
    PrimitiveSpec("wd_sleep_awake__std", "sleep", "Day-to-day standard deviation of awake duration.", "source-native duration/night"),
    PrimitiveSpec("wd_sleep_awake__cv", "sleep", "Relative day-to-day dispersion of awake duration.", "ratio"),
    PrimitiveSpec("wd_sleep_restless__mean", "sleep", "Mean nightly restless-event field.", "events/night"),
    PrimitiveSpec("wd_sleep_restless__std", "sleep", "Day-to-day standard deviation of restless events.", "events/night"),
    PrimitiveSpec("wd_sleep_restless__cv", "sleep", "Relative day-to-day dispersion of restless events.", "ratio"),
    PrimitiveSpec("wd_arch_ratio_sleep_deep__mean", "sleep", "Across-night mean deep-sleep share.", "ratio"),
    PrimitiveSpec("wd_arch_ratio_sleep_rem__mean", "sleep", "Across-night mean REM-sleep share.", "ratio"),
    PrimitiveSpec("wd_arch_ratio_sleep_awake__mean", "sleep", "Across-night mean awake share.", "ratio"),
    PrimitiveSpec("wd_arch_fragmentation__mean", "sleep", "Across-night mean awake-duration to sleep-duration ratio.", "ratio"),
    PrimitiveSpec("wi_hyp_waso_min__mean", "sleep", "Across-night mean wake after sleep onset from the 5-minute hypnogram.", "minutes"),
    PrimitiveSpec("wi_hyp_awakenings__mean", "sleep", "Across-night mean number of awake bouts after sleep onset.", "bouts/night"),
    PrimitiveSpec("wi_hyp_frag_index__mean", "sleep", "Across-night mean sleep-stage transition rate.", "transitions/hour"),
    PrimitiveSpec("wi_hyp_deep_bout_max__mean", "sleep", "Across-night mean of the longest deep-sleep bout.", "minutes"),
    PrimitiveSpec("wi_hyp_rem_bout_max__mean", "sleep", "Across-night mean of the longest REM-sleep bout.", "minutes"),
    PrimitiveSpec("wi_hyp_rem_frac__mean", "sleep", "Across-night mean REM share from the hypnogram.", "ratio"),
    PrimitiveSpec("wi_hyp_trans_light_awake__mean", "sleep", "Across-night mean light-to-awake transition rate.", "transitions/hour"),

    # Nocturnal autonomic measurements --------------------------------------
    PrimitiveSpec("wd_sleep_hr_average__mean", "autonomic", "Mean of nightly average heart rate.", "beats/minute"),
    PrimitiveSpec("wd_sleep_hr_average__std", "autonomic", "Day-to-day standard deviation of nightly average heart rate.", "beats/minute"),
    PrimitiveSpec("wd_sleep_hr_average__cv", "autonomic", "Relative day-to-day dispersion of nightly average heart rate.", "ratio"),
    PrimitiveSpec("wd_sleep_hr_lowest__mean", "autonomic", "Mean nightly lowest heart rate.", "beats/minute"),
    PrimitiveSpec("wd_sleep_rmssd__mean", "autonomic", "Mean nightly RMSSD summary.", "milliseconds"),
    PrimitiveSpec("wd_sleep_rmssd__std", "autonomic", "Day-to-day standard deviation of nightly RMSSD.", "milliseconds"),
    PrimitiveSpec("wd_sleep_rmssd__cv", "autonomic", "Relative day-to-day dispersion of nightly RMSSD.", "ratio"),
    PrimitiveSpec("wd_sleep_breath_average__mean", "autonomic", "Mean nightly respiration-rate summary.", "breaths/minute"),
    PrimitiveSpec("wd_sleep_breath_average__std", "autonomic", "Day-to-day standard deviation of nightly respiration rate.", "breaths/minute"),
    PrimitiveSpec("wi_hr_night_min__mean", "autonomic", "Across-night mean of the minimum valid 5-minute heart-rate sample.", "beats/minute"),
    PrimitiveSpec("wi_hr_night_dip__mean", "autonomic", "Across-night mean early-night to nightly-minimum heart-rate difference.", "beats/minute"),
    PrimitiveSpec("wi_hr_night_slope__mean", "autonomic", "Across-night mean linear heart-rate slope.", "beats/minute per hour"),
    PrimitiveSpec("wi_hr_night_cv__mean", "autonomic", "Across-night mean within-night relative heart-rate dispersion.", "ratio"),
    PrimitiveSpec("wi_rmssd_night_mean__mean", "autonomic", "Across-night mean of within-night RMSSD.", "milliseconds"),
    PrimitiveSpec("wi_rmssd_night_cv__mean", "autonomic", "Across-night mean within-night relative RMSSD dispersion.", "ratio"),
    PrimitiveSpec("wi_rmssd_night_slope__mean", "autonomic", "Across-night mean linear RMSSD slope.", "milliseconds/hour"),
)


def primitive_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in PRIMITIVE_CATALOG)


if len(set(primitive_names())) != len(PRIMITIVE_CATALOG):
    raise ValueError("Duplicate name in frozen wearable primitive catalogue")


__all__ = ["PRIMITIVE_CATALOG", "primitive_names"]
