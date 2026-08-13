"""Central configuration for Binary_Google_TabFM_Nested.

Task
----
``CN = 0`` vs ``MCI or Dem = 1`` on the 141 Training subjects.  The 33
historical Validation subjects receive one frozen prediction pass at the end.

Question this folder answers (fixed before any label was scored)
----------------------------------------------------------------
Does **Google TabFM** -- the tabular foundation model released by Google
Research in July 2026 (in-context learning, zero dataset-specific training) --
beat the honest MMSE anchor (regularized logistic regression) on this task?
TabPFN-style in-context learners are strongest exactly in this regime
(n < 1000 rows), which makes this the first genuinely new *model family*
hypothesis in this repository since the tree/linear era.

The candidate set is deliberately tiny (three entries, one question):

    lr_mmse_c001      the anchor          (complexity 0)
    tabfm_mmse        Google TabFM        (complexity 1)
    blend_tabfm_lr    rank-average blend  (complexity 2)

Inner CV -- never the outer test -- picks per outer fold, with the same
simplicity-biased tolerance rule as Binary_Google_CircadianNested.

A fixed audit arm ``tabfm_fusion`` (MMSE + circadian view) re-tests the
circadian hypothesis that Binary_Google_CircadianNested rejected with LR/YDF,
now under a different model family.  It never competes in selection.

Cross-run consistency check
---------------------------
Seed, label loading and fold construction are identical to
``Binary_Google_CircadianNested`` run 20260813_060156_utc, so the anchor track
``lr_mmse_c001`` must reproduce its published numbers (repeat mean 0.7617,
sd 0.0135) up to library-version noise.  A large deviation means the
environment -- not the model -- changed, and the run should be treated as
suspect.

Google technology
-----------------
Google TabFM (weights: tabfm-non-commercial-v1.0 license, research use) is the
core engine: it is the model under test, half of the blend candidate, and the
audit arm.  There is **no fallback**: if the pinned package or its checkpoint
cannot be loaded the run fails instead of substituting another learner.  The
only exception is an explicit, loudly-marked wiring stub that activates ONLY
when BGTF_WIRING_STUB=1 AND profile=smoke (local wiring checks on machines
that cannot install TabFM); it refuses to run any other profile.
"""

from __future__ import annotations

from dataclasses import dataclass

EXPERIMENT_NAME = "Binary_Google_TabFM_Nested"
TASK_DESCRIPTION = "CN (0) vs MCI+Dem (1), subject-level, Google TabFM vs MMSE anchor"
SEED = 20260813  # identical to Binary_Google_CircadianNested for fold parity

# Completed formal anchors, used only to report deltas -- never to select.
BENCHMARK = {
    "nested_best_mmse_only": {
        "source": "Binary_MMSE_MaxAUC/20260727_042357_utc",
        "pooled_oof_roc_auc": 0.765756,
    },
    "circadian_nested_run": {
        "source": "Binary_Google_CircadianNested/20260813_060156_utc",
        "nested_repeat_mean": 0.7426,
        "anchor_lr_mmse_c001_repeat_mean": 0.7617,
        "anchor_lr_mmse_c001_repeat_sd": 0.0135,
        "anchor_subject_mean": 0.7630,
        "note": "same seed & fold construction; anchor track must reproduce",
    },
}

# ------------------------------------------------------------------- TabFM ---
# Reproducibility pin: release tag v1.0.1 of google-research/tabfm.
TABFM_GIT_URL = "https://github.com/google-research/tabfm.git"
TABFM_PIN_TAG = "v1.0.1"
TABFM_PIN_COMMIT = "d8678b6895f1428a468d4cc299c1ff4cf704e726"
TABFM_PIP_SPEC = f"tabfm @ git+{TABFM_GIT_URL}@{TABFM_PIN_TAG}"
# Constructor kwargs we *want*; the adapter passes only those the installed
# TabFMClassifier signature accepts and records accepted/dropped ones.
# max_num_rows is critical: the documented default (100 context rows) would
# silently subsample our 113-141 training subjects.
TABFM_DESIRED_KWARGS = {
    "max_num_rows": 256,
    "random_state": SEED,
}
WIRING_STUB_ENV = "BGTF_WIRING_STUB"

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
ACTIVITY_DAY_START_HOUR = 4

# --------------------------------------------------------------- contracts ---
FORBIDDEN_COLUMNS = frozenset(
    {"DIAG_NM", "DIAG_SEQ", "DOCTOR_NM", "MMSE_NUM", "MMSE_KIND",
     "SAMPLE_EMAIL", "EMAIL"}
)
FORBIDDEN_SUBSTRINGS = ("diag", "label", "doctor", "email")
FORBIDDEN_FEATURE_TOKENS = ("n_days", "span_days", "cover", "gap_", "nonwear", "non_wear")

SPLIT_CONTRACT = {
    "train": {"n": 141, "CN": 85, "MCI": 47, "Dem": 9},
    "val": {"n": 33, "CN": 26, "MCI": 4, "Dem": 3},
}
DIAG_ORDER = ("CN", "MCI", "Dem")
POSITIVE_DIAGS = ("MCI", "Dem")
CLASS_NAMES = ("CN", "MCI_DEM")

# ------------------------------------------------------------------- MMSE ----
MMSE_ITEM_FAIL = 1.0
MMSE_ITEM_PASS = 2.0
MMSE_DOMAINS = {
    "orient_time": ("Q01", "Q02", "Q03", "Q04", "Q05"),
    "orient_place": ("Q06", "Q07", "Q08", "Q09", "Q10"),
    "registration": ("Q11_1", "Q11_2", "Q11_3"),
    "attention": ("Q12_1", "Q12_2", "Q12_3", "Q12_4", "Q12_5"),
    "recall": ("Q13_1", "Q13_2", "Q13_3"),
    "language": ("Q14_1", "Q14_2", "Q15", "Q16_1", "Q16_2", "Q16_3", "Q17", "Q18", "Q19"),
}
MMSE_ITEMS = tuple(item for items in MMSE_DOMAINS.values() for item in items)
MMSE_EXCLUDED_ITEMS = ("Q12_TOTAL",)

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
MMSE_VIEW = (
    ("mmse_TOTAL",)
    + tuple(f"mmse_{domain}" for domain in MMSE_DOMAINS)
    + tuple(f"mmse_{item}" for item in MMSE_ITEMS)
    + ("mmse_recall_deficit",)
)
# Same pre-registered circadian block as Binary_Google_CircadianNested (kept
# verbatim so the TabFM re-test of the rejected hypothesis is exactly paired).
CIRCADIAN_VIEW = (
    "wi_met_IS", "wi_met_IV", "wi_met_RA", "wi_met_M10", "wi_met_L5",
    "wi_met_M10_onset_h", "wi_met_L5_onset_h", "wi_met_daily_profile_std",
    "wi_met_entropy__mean", "wi_met_transition_rate__mean", "wi_met_transition_rate__std",
    "wi_met_active_frac__mean", "wi_met_active_bout_mean__mean", "wi_met_sed_bout_mean__mean",
    "wi_hyp_waso_min__mean", "wi_hyp_waso_min__std", "wi_hyp_awakenings__mean",
    "wi_hyp_frag_index__mean", "wi_hyp_frag_index__std",
    "wi_hyp_deep_frac__mean", "wi_hyp_rem_frac__mean", "wi_hyp_light_frac__mean",
    "wi_hyp_deep_bout_max__mean", "wi_hyp_trans_light_awake__mean",
    "wi_hyp_sleep_onset_epochs__mean",
    "wi_hr_night_mean__mean", "wi_hr_night_dip__mean", "wi_hr_night_slope__mean",
    "wi_hr_night_cv__mean", "wi_rmssd_night_mean__mean", "wi_rmssd_night_cv__mean",
    "wd_circ_bedtime__mean_h", "wd_circ_bedtime__circsd_h", "wd_circ_waketime__circsd_h",
)
VIEWS = {
    "mmse": MMSE_VIEW,
    "mmse_circ": MMSE_VIEW + CIRCADIAN_VIEW,
}

# ------------------------------------------------------------- model params --
LR_PARAMS = {"solver": "lbfgs", "max_iter": 4000, "class_weight": "balanced"}

SELECTION_TOLERANCE = 0.005


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    view: str
    learner: str            # lr | tabfm | blend_tabfm_lr
    complexity: int
    params: dict

    def describe(self) -> str:
        return f"{self.candidate_id} (view={self.view}, learner={self.learner})"


CANDIDATES = (
    Candidate("lr_mmse_c001", "mmse", "lr", 0, {"C": 0.01}),
    Candidate("tabfm_mmse", "mmse", "tabfm", 1, {}),
    Candidate("blend_tabfm_lr", "mmse", "blend_tabfm_lr", 2, {"C": 0.01}),
)

# Fixed audit arms: evaluated on the identical outer folds, never competing.
FIXED_ARMS = {
    # TabFM re-test of the circadian-fusion hypothesis rejected by
    # Binary_Google_CircadianNested under LR/YDF.  Paired contrast:
    # tabfm_fusion minus tabfm_mmse.
    "tabfm_fusion": Candidate("tabfm_fusion", "mmse_circ", "tabfm", 9, {}),
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
    candidate_ids: tuple[str, ...]
    description: str = ""


_ALL_IDS = tuple(candidate.candidate_id for candidate in CANDIDATES)

PROFILES = {
    # Wiring check only; never reported as performance.
    "smoke": Profile(
        name="smoke", outer_k=3, outer_repeats=1, inner_k=2, inner_repeats=1,
        n_bootstrap=200, candidate_ids=("lr_mmse_c001", "tabfm_mmse"),
        description="wiring only - not a performance measurement",
    ),
    # Cheap first pass: TabFM inference cost on the Colab runtime is unknown
    # until measured, so run this before default and read the projection line.
    "quick": Profile(
        name="quick", outer_k=5, outer_repeats=2, inner_k=4, inner_repeats=2,
        n_bootstrap=2000, candidate_ids=_ALL_IDS,
        description="2 outer repeats - projection/feasibility pass",
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

__all__ = [
    "ACTIVITY_CLASSES", "ACTIVITY_CLASS_EPOCHS_PER_DAY", "ACTIVITY_DAY_START_HOUR",
    "BENCHMARK", "CANDIDATES", "CIRCADIAN_VIEW", "CLASS_NAMES", "Candidate",
    "DIAG_ORDER", "EXPERIMENT_NAME", "FIXED_ARMS", "FORBIDDEN_COLUMNS",
    "FORBIDDEN_FEATURE_TOKENS", "FORBIDDEN_SUBSTRINGS", "HYPNOGRAM_STAGES",
    "INTRADAY_COLUMNS", "LABEL_FILES", "LOCAL_TZ", "LR_PARAMS",
    "MET_MINUTES_PER_DAY", "MMSE_DOMAINS", "MMSE_EXCLUDED_ITEMS", "MMSE_ITEMS",
    "MMSE_ITEM_FAIL", "MMSE_ITEM_PASS", "MMSE_VIEW", "NIGHT_EPOCH_MINUTES",
    "POSITIVE_DIAGS", "PROFILES", "Profile", "SEDENTARY_MET", "SEED",
    "SELECTION_TOLERANCE", "SOURCE_FILES", "SPLIT_CONTRACT", "SPLIT_DIRS",
    "TABFM_DESIRED_KWARGS", "TABFM_GIT_URL", "TABFM_PIN_COMMIT", "TABFM_PIN_TAG",
    "TABFM_PIP_SPEC", "TASK_DESCRIPTION", "VIEWS", "WIRING_STUB_ENV",
]
