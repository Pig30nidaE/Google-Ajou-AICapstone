"""Central configuration: paths, column contracts, feature blocks, CV profiles.

Every path and column name used anywhere in this experiment is declared here so
that a reviewer can audit the leakage surface from a single file.

Task
----
``positive = Dem`` (dementia, n=12), ``negative = CN + MCI`` (n=162), pooled over
the 141 Training and 33 Validation subjects (174 total).  This is the same task
definition as ``Binary_Google_DemScreen`` so the numbers are directly
comparable; it is **not** the ``CN vs MCI+Dem`` task used by the other Binary
folders.

Why this folder exists
----------------------
``Binary_Google_DemScreen`` (full run 20260728_051820_utc) reported repeated
nested-CV subject-level ROC-AUC of

    wearable_only  (174 subjects) : 0.7184 +- 0.0363
    wearable+MMSE  (174 subjects) : 0.8284 +- 0.0450   <- benchmark to beat

while the *single, unfitted* feature ``mmse_TOTAL`` scores 0.947 on the same
cohort.  A trained 151-feature ensemble that lands 0.12 AUC below one raw column
is losing information, and the reasons are identifiable:

1. ``select_features`` ranked 151 candidates by training-fold AUC with ~10
   positives, so the chosen subset was noise-dominated and unstable.
2. Blend weights were fitted by 4000-draw Dirichlet search on an inner OOF with
   ~10 positives -- more free parameters than the signal can support.
3. Blending happened in log-odds space.  ROC-AUC depends only on ranks, and
   averaging probabilities from models with very different score scales moves
   ranks in ways log-odds averaging does not control.
4. Every wearable feature was a *daily summary*.  The dataset also ships the
   intraday series (1-min MET, 5-min activity class, 5-min hypnogram, 5-min
   sleep HR and RMSSD) in the ``CONVERT(... USING utf8)`` columns, and no prior
   experiment in this repository parsed them.

This folder addresses all four, with (4) as the pre-registered hypothesis: the
canonical non-parametric circadian statistics (IS, IV, RA, M10, L5) and sleep
fragmentation / nocturnal autonomic measures are established dementia markers
and are continuous, which also breaks the score ties that cap tree models at 12
positives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------- identity ----
EXPERIMENT_NAME = "Binary_Google_DemRankAUC"
TASK_DESCRIPTION = "CN+MCI (0) vs Dem (1) - subject-level dementia screening"
SEED = 20260728

# Prior best on *this* task, from the stored FINAL_REPORT.json of
# Binary_Google_DemScreen/Binary_Google_DemScreen_result/20260728_051820_utc.
# Used only for reporting the delta; never for selecting anything.
BENCHMARK = {
    "source": "Binary_Google_DemScreen/20260728_051820_utc/training/FINAL_REPORT.json",
    "protocol": "repeated nested CV (20 repeats x 5 outer folds), pooled OOF per repeat",
    "wearable_only__full": 0.7184027777777777,
    "wearable_plus_mmse__full": 0.8283822016460907,
    "wearable_only__filtered": 0.8303650686832222,
    "wearable_plus_mmse__filtered": 0.9225113464447806,
    "mmse_total_single_feature_full_cohort": 0.947,
}

# ------------------------------------------------------------------ paths ----
SPLIT_DIRS = {"train": "1.Training", "val": "2.Validation"}
SOURCE_FILES = {
    "train": {
        "activity": "SourceData/1.Gait/train_activity.csv",
        "sleep": "SourceData/2.Sleep/train_sleep.csv",
        "mmse": "SourceData/3.CognitiveFunction/train_mmse.csv",
    },
    "val": {
        "activity": "SourceData/1.Gait/val_activity.csv",
        "sleep": "SourceData/2.Sleep/val_sleep.csv",
        "mmse": "SourceData/3.CognitiveFunction/val_mmse.csv",
    },
}
# The three label copies are byte-different files with identical content; the
# loader reads Gait and Sleep and asserts they agree (AGENTS.md contract).
LABEL_FILES = {
    "train": {
        "gait": "LabelingData/1.Gait/training_label.csv",
        "sleep": "LabelingData/2.Sleep/training_label.csv",
    },
    "val": {
        "gait": "LabelingData/1.Gait/val_label.csv",
        "sleep": "LabelingData/2.Sleep/val_label.csv",
    },
}

LOCAL_TZ = "Asia/Seoul"
ACTIVITY_DAY_START_HOUR = 4  # activity_day_start is always 04:00 local

# --------------------------------------------------------------- contracts ---
# Fail-closed: any of these reaching the feature matrix aborts the run.
FORBIDDEN_COLUMNS = frozenset(
    {
        "DIAG_NM",      # the label itself
        "DIAG_SEQ",     # diagnosis ordering -> post-diagnosis variable
        "DOCTOR_NM",    # administrative, encodes recruitment site
        "MMSE_NUM",     # administrative
        "MMSE_KIND",    # administrative
        "SAMPLE_EMAIL",
        "EMAIL",
    }
)
FORBIDDEN_SUBSTRINGS = ("diag", "label", "doctor", "email")

COHORT_CONTRACT = {"n_subjects": 174, "CN": 111, "MCI": 51, "Dem": 12}
SPLIT_CONTRACT = {
    "train": {"n": 141, "CN": 85, "MCI": 47, "Dem": 9},
    "val": {"n": 33, "CN": 26, "MCI": 4, "Dem": 3},
}
DIAG_ORDER = ("CN", "MCI", "Dem")
SEVERITY = {"CN": 0, "MCI": 1, "Dem": 2}

# ------------------------------------------------------------------- MMSE ----
# Items are coded 1 = failed, 2 = passed (verified: observed values are exactly
# {1, 2} for all 30 items).  These are *structural* constants of the instrument,
# not statistics estimated from the data, so no fold-locality question arises.
MMSE_ITEM_FAIL = 1.0
MMSE_ITEM_PASS = 2.0
MMSE_DOMAINS = {
    "orient_time": ("Q01", "Q02", "Q03", "Q04", "Q05"),
    "orient_place": ("Q06", "Q07", "Q08", "Q09", "Q10"),
    "registration": ("Q11_1", "Q11_2", "Q11_3"),  # constant in this cohort
    "attention": ("Q12_1", "Q12_2", "Q12_3", "Q12_4", "Q12_5"),
    "recall": ("Q13_1", "Q13_2", "Q13_3"),
    "language": ("Q14_1", "Q14_2", "Q15", "Q16_1", "Q16_2", "Q16_3", "Q17", "Q18", "Q19"),
}
MMSE_ITEMS = tuple(item for items in MMSE_DOMAINS.values() for item in items)
# Q12_TOTAL is all-zero in both splits; carrying it would only add a constant.
MMSE_EXCLUDED_ITEMS = ("Q12_TOTAL",)

# ------------------------------------------------------- wearable channels ---
# "rich" channels get the full statistic set, "lite" channels get mean/std/cv.
RICH_SLEEP = (
    "sleep_duration", "sleep_efficiency", "sleep_awake", "sleep_deep", "sleep_light",
    "sleep_rem", "sleep_restless", "sleep_onset_latency", "sleep_midpoint_time",
    "sleep_hr_average", "sleep_hr_lowest", "sleep_rmssd", "sleep_breath_average",
    "sleep_score", "sleep_score_deep", "sleep_temperature_deviation",
)
RICH_ACTIVITY = (
    "activity_score", "activity_steps", "activity_rest", "activity_inactive",
    "activity_low", "activity_medium", "activity_high", "activity_daily_movement",
    "activity_average_met", "activity_cal_active",
)
LITE_SLEEP = (
    "sleep_score_alignment", "sleep_score_disturbances", "sleep_score_efficiency",
    "sleep_score_latency", "sleep_score_rem", "sleep_score_total", "sleep_total",
)
LITE_ACTIVITY = (
    "activity_score_meet_daily_targets", "activity_score_move_every_hour",
    "activity_score_recovery_time", "activity_score_stay_active",
    "activity_score_training_frequency", "activity_score_training_volume",
    "activity_cal_total", "activity_inactivity_alerts", "activity_met_min_high",
    "activity_met_min_medium", "activity_met_min_low", "activity_met_min_inactive",
    "activity_non_wear",
)

# Intraday series live in these columns.  The plain ``*_5min`` / ``*_1min``
# columns contain the literal string "..." and are unusable.
INTRADAY_COLUMNS = {
    "met_1min": "CONVERT(activity_met_1min USING utf8)",
    "activity_class_5min": "CONVERT(activity_class_5min USING utf8)",
    "hypnogram_5min": "CONVERT(sleep_hypnogram_5min USING utf8)",
    "sleep_hr_5min": "CONVERT(sleep_hr_5min USING utf8)",
    "sleep_rmssd_5min": "CONVERT(sleep_rmssd_5min USING utf8)",
}
MET_MINUTES_PER_DAY = 1440
ACTIVITY_CLASS_EPOCHS_PER_DAY = 288
# Oura encodings.
ACTIVITY_CLASSES = {0: "nonwear", 1: "rest", 2: "inactive", 3: "low", 4: "medium", 5: "high"}
HYPNOGRAM_STAGES = {1: "deep", 2: "light", 3: "rem", 4: "awake"}

# Adherence / wear-time descriptors.  They can carry real signal (apathy) but can
# equally be a recruitment-wave artifact, so the pipeline reports an ablation.
SUSPECT_FEATURE_PREFIXES = ("wd_cover_", "wd_n_days", "wd_span_days", "wd_gap_")

# Label-blind data-quality rules, copied verbatim from Binary_Google_DemScreen so
# the sensitivity arm stays comparable.  These never consult y.
QUALITY_RULES = {
    "implausible_sustained_steps": {
        "feature": "wd_activity_steps__mean",
        "op": "gt",
        "value": 25000.0,
        "reason": "일 평균 25,000보 초과 - 장기 지속이 어려운 값(기기 오사용/기록 오류 가능)",
    },
    "insufficient_wear": {
        "feature": "wd_n_days_activity",
        "op": "lt",
        "value": 7.0,
        "reason": "착용 일수 7일 미만 - 일별 변동성 피처를 신뢰할 수 없음",
    },
}

# ---------------------------------------------------------- feature blocks ---
# Prefixes:  mmse_ = cognitive test, wd_ = wearable daily summary,
#            wi_ = wearable intraday (new in this folder).
#
# "core" blocks are pre-specified from clinical reasoning and this repo's prior
# EDA; they are NOT chosen by looking at this cohort's AUCs.  The repeated lesson
# in AGENTS.md is that small pre-specified sets beat in-fold selection over
# hundreds of candidates when there are ~10 positives, so both are compared.
MMSE_CORE = (
    "mmse_TOTAL", "mmse_orient_time", "mmse_orient_place", "mmse_attention",
    "mmse_recall", "mmse_language", "mmse_recall_deficit",
)
WD_CORE = (
    "wd_sleep_light__mean", "wd_sleep_restless__mean", "wd_activity_rest__mean",
    "wd_activity_low__std", "wd_sleep_score_deep__std", "wd_activity_average_met__mean",
    "wd_sleep_efficiency__mean", "wd_sleep_duration__std", "wd_activity_steps__mean",
    "wd_arch_ratio_sleep_deep__mean", "wd_arch_fragmentation__mean",
    "wd_circ_midpoint__circsd_h", "wd_circ_bedtime__circsd_h",
)
WI_CORE = (
    # canonical non-parametric circadian rhythm statistics
    "wi_met_IS", "wi_met_IV", "wi_met_RA", "wi_met_M10", "wi_met_L5",
    "wi_met_M10_onset_h", "wi_met_L5_onset_h",
    # activity fragmentation
    "wi_met_active_bout_mean__mean", "wi_met_sed_bout_mean__mean",
    "wi_met_transition_rate__mean", "wi_met_entropy__mean",
    "wi_class_nonwear_frac__mean", "wi_class_high_frac__mean",
    # sleep micro-architecture
    "wi_hyp_waso_min__mean", "wi_hyp_awakenings__mean", "wi_hyp_frag_index__mean",
    "wi_hyp_deep_bout_max__mean", "wi_hyp_rem_frac__mean",
    "wi_hyp_trans_light_awake__mean",
    # nocturnal autonomic function
    "wi_hr_night_min__mean", "wi_hr_night_dip__mean", "wi_hr_night_slope__mean",
    "wi_hr_night_cv__mean", "wi_rmssd_night_mean__mean", "wi_rmssd_night_cv__mean",
)

FEATURE_BLOCK_PREFIXES = {
    "mmse_full": ("mmse_",),
    "wd_full": ("wd_",),
    "wi_full": ("wi_",),
    "wearable_full": ("wd_", "wi_"),
    "fused_full": ("mmse_", "wd_", "wi_"),
}
FEATURE_BLOCK_EXPLICIT = {
    "mmse_core": MMSE_CORE,
    "wd_core": WD_CORE,
    "wi_core": WI_CORE,
    "wearable_core": WD_CORE + WI_CORE,
    "fused_core": MMSE_CORE + WD_CORE + WI_CORE,
}
ALL_BLOCKS = tuple(FEATURE_BLOCK_EXPLICIT) + tuple(FEATURE_BLOCK_PREFIXES)

# Candidate families carried into the *nested* loop.
#
# Pre-specified, not screening-derived: the point of the nested loop is that
# nothing in it was chosen with the outer-test subjects in view, so its candidate
# list must not depend on this cohort's scores either.  The seven entries are
# picked to span the hypothesis space -- a parameter-free rank ranker, a
# regularised linear model, a kernel method, a bagged-tree ensemble, a
# class-balanced tree ensemble, Google's boosted trees and a strong non-Google
# boosting baseline -- rather than to be the seven that happened to screen well.
#
# The full zoo (18 families on a stock Colab image) still runs in the screening
# phase.  Carrying all of it into the nested loop costs roughly an order of
# magnitude more compute for candidates that are near-duplicates of each other.
NESTED_CANDIDATE_FAMILIES = (
    "rank_mean",       # zero fitted magnitudes; strongest low-variance option at n_pos=12
    "logreg_en",       # elastic-net linear
    "svm_rbf",         # kernel
    "extra_trees",     # bagged trees
    "balanced_rf",     # class-balanced trees (falls back out if imblearn is absent)
    "ydf_gbt",         # Google
    "lightgbm",        # strong tree baseline
)
# Used when an entry above is unavailable in this environment.
NESTED_FALLBACKS = {
    "balanced_rf": "random_forest",
    "ydf_gbt": "hist_gb",
    "lightgbm": "hist_gb",
}

# Tracks evaluated by default.  ``wd_full`` reproduces the representation the
# prior best used, so the comparison isolates the intraday contribution.
DEFAULT_TRACKS = (
    "mmse_core", "mmse_full",
    "wd_core", "wd_full",
    "wi_core", "wi_full",
    "wearable_core", "wearable_full",
    "fused_core", "fused_full",
)


# ------------------------------------------------------------- CV profiles ---
@dataclass(frozen=True)
class CVConfig:
    """Repeated stratified subject-level CV.

    One row per subject after aggregation, so stratified K-fold over subjects is
    group-aware by construction; ``splits.py`` still asserts subject disjointness
    explicitly rather than relying on that argument.

    ``outer_k`` is capped by the positive count so every test fold contains at
    least two Dem subjects (12 positives / 5 folds).
    """

    outer_k: int = 5
    inner_k: int = 4
    repeats: int = 20
    seed: int = SEED
    min_positives_per_fold: int = 2


@dataclass(frozen=True)
class Profile:
    name: str
    cv: CVConfig
    screen_repeats: int
    tune_trials: int
    shortlist: int
    seeds: tuple[int, ...]
    use_sequence_arm: bool = False
    description: str = ""


PROFILES = {
    # Wiring check only.  AGENTS.md contract: smoke numbers are never reported as
    # performance.  Tiny CV, no tuning, two models.
    "smoke": Profile(
        name="smoke",
        cv=CVConfig(outer_k=3, inner_k=2, repeats=1),
        screen_repeats=1,
        tune_trials=0,
        shortlist=2,
        seeds=(SEED,),
        description="wiring only - not a performance measurement",
    ),
    "standard": Profile(
        name="standard",
        cv=CVConfig(outer_k=5, inner_k=4, repeats=5),
        screen_repeats=3,
        tune_trials=25,
        shortlist=4,
        seeds=(SEED, SEED + 1),
        description="~30 min on CPU high-RAM",
    ),
    "full": Profile(
        name="full",
        cv=CVConfig(outer_k=5, inner_k=4, repeats=20),
        screen_repeats=5,
        tune_trials=60,
        shortlist=6,
        seeds=(SEED, SEED + 1, SEED + 2),
        description="matches the 20-repeat protocol of the DemScreen benchmark",
    ),
    "max": Profile(
        name="max",
        cv=CVConfig(outer_k=5, inner_k=4, repeats=30),
        screen_repeats=8,
        tune_trials=120,
        shortlist=8,
        seeds=(SEED, SEED + 1, SEED + 2, SEED + 3, SEED + 4),
        use_sequence_arm=True,
        description="adds the TSMixer sequence arm; needs GPU to be worth it",
    ),
}

HARD_RUNTIME_SECONDS = 6 * 60 * 60


@dataclass
class RunConfig:
    """Everything ``train.run_experiment`` needs, resolved by ``run.py``."""

    data_root: str
    output_dir: str
    profile: str = "full"
    tracks: tuple[str, ...] = DEFAULT_TRACKS
    models: tuple[str, ...] | None = None       # None -> every available model
    cohort: str = "both"                        # full | filtered | both
    drop_suspect: bool = False
    resamplers: tuple[str, ...] = ("none", "class_weight")
    use_sequence_arm: bool = False
    tune: bool = True
    hard_runtime_seconds: int = HARD_RUNTIME_SECONDS
    seed: int = SEED
    n_bootstrap: int = 4000
    feature_cache: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def profile_config(self) -> Profile:
        if self.profile not in PROFILES:
            raise ValueError(f"Unknown profile {self.profile!r}; choose from {sorted(PROFILES)}")
        return PROFILES[self.profile]

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def training_dir(self) -> Path:
        return self.output_path / "training"


__all__ = [
    "ACTIVITY_CLASSES", "ACTIVITY_CLASS_EPOCHS_PER_DAY", "ACTIVITY_DAY_START_HOUR",
    "ALL_BLOCKS", "BENCHMARK", "COHORT_CONTRACT", "CVConfig", "DEFAULT_TRACKS",
    "DIAG_ORDER", "EXPERIMENT_NAME", "FEATURE_BLOCK_EXPLICIT", "FEATURE_BLOCK_PREFIXES",
    "FORBIDDEN_COLUMNS", "FORBIDDEN_SUBSTRINGS", "HARD_RUNTIME_SECONDS",
    "HYPNOGRAM_STAGES", "INTRADAY_COLUMNS", "LABEL_FILES", "LITE_ACTIVITY", "LITE_SLEEP",
    "LOCAL_TZ", "MET_MINUTES_PER_DAY", "MMSE_DOMAINS", "MMSE_EXCLUDED_ITEMS",
    "MMSE_ITEMS", "MMSE_ITEM_FAIL", "MMSE_ITEM_PASS", "PROFILES", "Profile",
    "QUALITY_RULES", "RICH_ACTIVITY", "RICH_SLEEP", "RunConfig", "SEED", "SEVERITY",
    "SOURCE_FILES", "SPLIT_CONTRACT", "SPLIT_DIRS", "SUSPECT_FEATURE_PREFIXES",
    "TASK_DESCRIPTION",
]
