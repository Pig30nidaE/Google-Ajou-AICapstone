"""Central configuration for Binary_Google_CircadianNested.

Task
----
``CN = 0`` vs ``MCI or Dem = 1`` on the 141 Training subjects (the repository's
main task).  The 33 historical Validation subjects are never used for any
selection; they receive one frozen prediction pass at the very end.

Pre-registered hypothesis (fixed before any label was scored)
-------------------------------------------------------------
The intraday series shipped inside the ``CONVERT(... USING utf8)`` columns
(1-min MET, 5-min activity class, 5-min hypnogram, 5-min nocturnal HR/RMSSD)
have never been used for this task in this repository -- every prior wearable
representation was a daily summary, and honest subject-level OOF AUC for those
stayed near 0.5.  The actigraphy literature associates non-parametric circadian
statistics (IS, IV, RA, M10, L5), sleep micro-architecture (WASO, fragmentation,
stage transitions) and nocturnal autonomic measures (HR dip, RMSSD) with
cognitive decline.  The hypothesis is that a *small, pre-registered* circadian
block adds signal on top of the MMSE anchor.  The candidate set below therefore
contains matched pairs: every learner appears once with the MMSE view and once
with the MMSE+circadian view, and inner CV -- never the outer test -- picks.

Why the candidate set is small
------------------------------
Repeated lesson recorded in AGENTS.md: with 141 subjects, wide feature banks
and large hyperparameter searches raise inner scores that do not transfer to
outer folds (measured optimism up to +0.084).  Nine fixed candidates with a
simplicity-biased tolerance rule is the deliberate response to that record.

Google technology
-----------------
Google Yggdrasil Decision Forests (YDF) is the core learning engine: six of the
nine nested candidates and the circadian-only diagnostic arm train YDF gradient
boosted trees, including the sparse-oblique variant.  There is **no fallback**:
if ``ydf`` cannot honor the exact learner, the run fails instead of silently
substituting a scikit-learn model.
"""

from __future__ import annotations

from dataclasses import dataclass

EXPERIMENT_NAME = "Binary_Google_CircadianNested"
TASK_DESCRIPTION = "CN (0) vs MCI+Dem (1), subject-level, MMSE + intraday circadian fusion"
SEED = 20260813

# Anchors from completed formal runs, used only to report deltas -- never to
# select anything in this run.
BENCHMARK = {
    "nested_best_mmse_only": {
        "source": "Binary_MMSE_MaxAUC/20260727_042357_utc",
        "pooled_oof_roc_auc": 0.765756,
    },
    "wearable_daily_summary_range": "0.45-0.57 (multiple folders, honest subject OOF)",
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
FORBIDDEN_COLUMNS = frozenset(
    {"DIAG_NM", "DIAG_SEQ", "DOCTOR_NM", "MMSE_NUM", "MMSE_KIND",
     "SAMPLE_EMAIL", "EMAIL"}
)
FORBIDDEN_SUBSTRINGS = ("diag", "label", "doctor", "email")
# Collection-quantity proxies must never become features (recruitment-wave
# artifact risk); the curated lists below contain none, and this guard makes
# the claim mechanical.
FORBIDDEN_FEATURE_TOKENS = ("n_days", "span_days", "cover", "gap_", "nonwear", "non_wear")

SPLIT_CONTRACT = {
    "train": {"n": 141, "CN": 85, "MCI": 47, "Dem": 9},
    "val": {"n": 33, "CN": 26, "MCI": 4, "Dem": 3},
}
DIAG_ORDER = ("CN", "MCI", "Dem")
POSITIVE_DIAGS = ("MCI", "Dem")
CLASS_NAMES = ("CN", "MCI_DEM")  # YDF label classes; index 1 is the positive class

# ------------------------------------------------------------------- MMSE ----
# Structural constants of the instrument (1 = failed, 2 = passed); asserted at
# build time, not estimated from data.
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
MMSE_EXCLUDED_ITEMS = ("Q12_TOTAL",)  # all-zero in both splits

# ----------------------------------------------------------- intraday input --
INTRADAY_COLUMNS = {
    "met_1min": "CONVERT(activity_met_1min USING utf8)",
    "activity_class_5min": "CONVERT(activity_class_5min USING utf8)",
    "hypnogram_5min": "CONVERT(sleep_hypnogram_5min USING utf8)",
    "sleep_hr_5min": "CONVERT(sleep_hr_5min USING utf8)",
    "sleep_rmssd_5min": "CONVERT(sleep_rmssd_5min USING utf8)",
}
MET_MINUTES_PER_DAY = 1440
ACTIVITY_CLASS_EPOCHS_PER_DAY = 288
ACTIVITY_CLASSES = {0: "nonwear", 1: "rest", 2: "inactive", 3: "low", 4: "medium", 5: "high"}
HYPNOGRAM_STAGES = {1: "deep", 2: "light", 3: "rem", 4: "awake"}
SEDENTARY_MET = 1.5
NIGHT_EPOCH_MINUTES = 5

# ------------------------------------------------------------ feature views --
# The MMSE view mirrors the Binary_MMSE_MaxAUC anchor: TOTAL, 6 domain sums,
# 30 items, recall_deficit.  ``num_failed`` is excluded because it equals
# ``30 - TOTAL`` exactly (verified on 141/141 subjects) and a perfectly
# collinear duplicate adds nothing.
MMSE_VIEW = (
    ("mmse_TOTAL",)
    + tuple(f"mmse_{domain}" for domain in MMSE_DOMAINS)
    + tuple(f"mmse_{item}" for item in MMSE_ITEMS)
    + ("mmse_recall_deficit",)
)

# Pre-registered circadian block.  Chosen from the actigraphy/sleep literature
# and the DemRankAUC feature vocabulary BEFORE any CN-vs-MCI+Dem label was
# scored against them; no per-feature screening happened outside folds.
CIRCADIAN_VIEW = (
    # canonical non-parametric circadian rhythm statistics (1-min MET)
    "wi_met_IS", "wi_met_IV", "wi_met_RA", "wi_met_M10", "wi_met_L5",
    "wi_met_M10_onset_h", "wi_met_L5_onset_h", "wi_met_daily_profile_std",
    # within-day activity fragmentation
    "wi_met_entropy__mean", "wi_met_transition_rate__mean", "wi_met_transition_rate__std",
    "wi_met_active_frac__mean", "wi_met_active_bout_mean__mean", "wi_met_sed_bout_mean__mean",
    # sleep micro-architecture (5-min hypnogram)
    "wi_hyp_waso_min__mean", "wi_hyp_waso_min__std", "wi_hyp_awakenings__mean",
    "wi_hyp_frag_index__mean", "wi_hyp_frag_index__std",
    "wi_hyp_deep_frac__mean", "wi_hyp_rem_frac__mean", "wi_hyp_light_frac__mean",
    "wi_hyp_deep_bout_max__mean", "wi_hyp_trans_light_awake__mean",
    "wi_hyp_sleep_onset_epochs__mean",
    # nocturnal autonomic function (5-min HR / RMSSD)
    "wi_hr_night_mean__mean", "wi_hr_night_dip__mean", "wi_hr_night_slope__mean",
    "wi_hr_night_cv__mean", "wi_rmssd_night_mean__mean", "wi_rmssd_night_cv__mean",
    # sleep-timing regularity (daily bedtime/waketime clock values)
    "wd_circ_bedtime__mean_h", "wd_circ_bedtime__circsd_h", "wd_circ_waketime__circsd_h",
)

VIEWS = {
    "mmse": MMSE_VIEW,
    "mmse_circ": MMSE_VIEW + CIRCADIAN_VIEW,
    "circ": CIRCADIAN_VIEW,
}

# ------------------------------------------------------------- model params --
# L2 is sklearn's default penalty; leaving it implicit keeps the pipeline
# compatible with both sklearn <1.8 and the 1.8+ deprecation of ``penalty``.
LR_PARAMS = {"solver": "lbfgs", "max_iter": 4000, "class_weight": "balanced"}
YDF_AXIS_GBT = {
    "num_trees": 250, "max_depth": 4, "min_examples": 5, "shrinkage": 0.05,
    "subsample": 0.9, "num_candidate_attributes_ratio": 1.0,
    "l2_regularization": 0.1,
}
YDF_OBLIQUE_GBT = {
    "num_trees": 200, "max_depth": 4, "min_examples": 5, "shrinkage": 0.05,
    "subsample": 0.9, "num_candidate_attributes_ratio": 1.0,
    "l2_regularization": 0.1,
    "sparse_oblique_normalization": "STANDARD_DEVIATION",
    "sparse_oblique_num_projections_exponent": 1.0,
    "sparse_oblique_projection_density_factor": 2.0,
}

# ---------------------------------------------------------------- candidates -
# ``complexity`` implements the tolerance rule: among candidates whose inner
# mean AUC is within SELECTION_TOLERANCE of the best, the LOWEST complexity is
# selected.  Ordering encodes "prefer the anchor; adopt circadian fusion or a
# tree/blend only when inner evidence clearly supports it".
SELECTION_TOLERANCE = 0.005


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    view: str
    learner: str            # lr | ydf_gbt | ydf_oblique | blend_lr_oblique
    complexity: int
    params: dict

    def describe(self) -> str:
        return f"{self.candidate_id} (view={self.view}, learner={self.learner})"


CANDIDATES = (
    Candidate("lr_mmse_c001", "mmse", "lr", 0, {"C": 0.01}),
    Candidate("lr_mmse_c01", "mmse", "lr", 1, {"C": 0.1}),
    Candidate("lr_fusion_c001", "mmse_circ", "lr", 2, {"C": 0.01}),
    Candidate("gbt_mmse", "mmse", "ydf_gbt", 3, dict(YDF_AXIS_GBT)),
    Candidate("obl_mmse", "mmse", "ydf_oblique", 4, dict(YDF_OBLIQUE_GBT)),
    Candidate("gbt_fusion", "mmse_circ", "ydf_gbt", 5, dict(YDF_AXIS_GBT)),
    Candidate("obl_fusion", "mmse_circ", "ydf_oblique", 6, dict(YDF_OBLIQUE_GBT)),
    Candidate("blend_mmse", "mmse", "blend_lr_oblique", 7,
              {"C": 0.01, "oblique": dict(YDF_OBLIQUE_GBT)}),
    Candidate("blend_fusion", "mmse_circ", "blend_lr_oblique", 8,
              {"C": 0.01, "oblique": dict(YDF_OBLIQUE_GBT)}),
)

# Fixed arms evaluated on the identical outer folds, without selection.
FIXED_ARMS = {
    # paired baseline: the strongest known honest configuration
    "anchor_mmse_lr": Candidate("lr_mmse_c001", "mmse", "lr", 0, {"C": 0.01}),
    # diagnostic: do the intraday circadian features alone beat the ~0.5 record
    # of daily-summary wearable features?  (Google YDF engine)
    "circadian_ydf": Candidate("gbt_circ", "circ", "ydf_gbt", 9, dict(YDF_AXIS_GBT)),
}


# ------------------------------------------------------------- CV profiles ---
@dataclass(frozen=True)
class Profile:
    name: str
    outer_k: int
    outer_repeats: int
    inner_k: int
    inner_repeats: int
    n_bootstrap: int
    candidate_ids: tuple[str, ...]   # subset for smoke; full set otherwise
    ydf_trees_override: int | None = None
    description: str = ""


_ALL_IDS = tuple(candidate.candidate_id for candidate in CANDIDATES)

PROFILES = {
    # Wiring check only.  AGENTS.md contract: smoke numbers are never reported
    # as performance.
    "smoke": Profile(
        name="smoke", outer_k=3, outer_repeats=1, inner_k=2, inner_repeats=1,
        n_bootstrap=200, candidate_ids=("lr_mmse_c001", "gbt_mmse"),
        ydf_trees_override=30,
        description="wiring only - not a performance measurement",
    ),
    "default": Profile(
        name="default", outer_k=5, outer_repeats=10, inner_k=4, inner_repeats=2,
        n_bootstrap=4000, candidate_ids=_ALL_IDS,
        description="primary protocol: 5 outer folds x 10 repeats, inner 4x2",
    ),
    "max": Profile(
        name="max", outer_k=5, outer_repeats=20, inner_k=4, inner_repeats=2,
        n_bootstrap=4000, candidate_ids=_ALL_IDS,
        description="20 outer repeats for tighter repeat variance",
    ),
}

YDF_NUM_THREADS = 2
REQUIRED_YDF_VERSION = "0.16.1"

__all__ = [
    "ACTIVITY_CLASSES", "ACTIVITY_CLASS_EPOCHS_PER_DAY", "ACTIVITY_DAY_START_HOUR",
    "BENCHMARK", "CANDIDATES", "CIRCADIAN_VIEW", "CLASS_NAMES", "Candidate",
    "DIAG_ORDER", "EXPERIMENT_NAME", "FIXED_ARMS", "FORBIDDEN_COLUMNS",
    "FORBIDDEN_FEATURE_TOKENS", "FORBIDDEN_SUBSTRINGS", "HYPNOGRAM_STAGES",
    "INTRADAY_COLUMNS", "LABEL_FILES", "LOCAL_TZ", "LR_PARAMS",
    "MET_MINUTES_PER_DAY", "MMSE_DOMAINS", "MMSE_EXCLUDED_ITEMS", "MMSE_ITEMS",
    "MMSE_ITEM_FAIL", "MMSE_ITEM_PASS", "MMSE_VIEW", "NIGHT_EPOCH_MINUTES",
    "POSITIVE_DIAGS", "PROFILES", "Profile", "REQUIRED_YDF_VERSION",
    "SEDENTARY_MET", "SEED", "SELECTION_TOLERANCE", "SOURCE_FILES",
    "SPLIT_CONTRACT", "SPLIT_DIRS", "TASK_DESCRIPTION", "VIEWS",
    "YDF_AXIS_GBT", "YDF_NUM_THREADS", "YDF_OBLIQUE_GBT",
]
