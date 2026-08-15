"""Central configuration for Binary_Google_SensorFM_Nested.

Task
----
``CN = 0`` vs ``MCI or Dem = 1`` on the 141 Training subjects, **wearable-only**
(modality contract 1: the MMSE SourceData file is never opened; any feature
name containing "mmse" fails closed).  The 33 historical Validation subjects
receive one frozen prediction pass at the end.

Question this folder answers (fixed before any label is scored)
---------------------------------------------------------------
Google's **SensorFM** (arXiv:2605.22759, "Towards a General Intelligence and
Interface for Wearable Health Data") reports that a masked-autoencoder
foundation model over minute-level wearable features, followed by a frozen
encoder + PCA-50 + linear probe, beats supervised feature-engineered baselines
on 34 of 35 person-level health tasks.  Its checkpoints are NOT public and its
input schema (34 Fitbit/Pixel features) does not match our Oura export, so the
only honest test available to us is the **recipe at our own scale**:

    Does SensorFM-style self-supervised pretraining (ViT-1D MAE + AIM masking,
    pretrained from scratch on the outer-training subjects' own unlabeled
    days) beat the paper's own supervised baseline (engineered daily features
    -> PCA -> logistic regression) on our CN vs MCI+Dem task, under
    subject-level repeated nested CV?

The paper's scaling result cuts against us and is part of the pre-registered
expectation: SensorFM-XXS (10^5 params, the size matched to our data volume)
ranks LAST among their variants on 33/35 tasks, and prior wearable-only
attempts in this repository sit at OOF ROC-AUC 0.45~0.57.  A null result is an
informative outcome, not a failure of wiring.

Candidates (deliberately tiny):

    fe_paper_lr       paper M.3.6 baseline: engineered stats -> PCA -> LR (c=0)
    sensorfm_lr       frozen SSL encoder -> mean+std -> PCA -> LR      (c=1)
    sensorfm_fe_blend rank-average of the two, fixed configs           (c=2)

Inner CV -- never the outer test -- picks per outer fold with the same
simplicity-biased tolerance rule as Binary_Google_TabFM_Nested.

Leakage rules specific to SSL (stricter than the paper)
-------------------------------------------------------
The paper pretrains on 5M external subjects, so its 5-fold downstream CV never
mixes pretraining and test people.  We have no external cohort, therefore:

* the MAE encoder is pretrained **inside each outer fold on the outer-training
  subjects' days only** -- outer-test subjects contribute zero minutes to
  pretraining (labels are never used by SSL anywhere);
* per-channel z-score stats are computed from outer-training days only
  (fold-local analogue of the paper's global pretraining stats), clip [-5, 5];
* PCA / imputer / scaler / probe / threshold are fold-local as always.

Cross-run consistency
---------------------
Seed, label loading and outer-fold construction are identical to
``Binary_Google_TabFM_Nested`` / ``Binary_Google_CircadianNested``
(SEED 20260813), so nested tracks are paired fold-by-fold across runs.

Google technology
-----------------
SensorFM is a Google Research / Google DeepMind model.  Since neither weights
nor code are released, the Google technology here is the published SensorFM
*architecture and training recipe* (ViT-1D MAE, AIM masking from LSM-2
arXiv:2506.05321, patch [20 min x 1 feature], Table ED.4 model sizes, the
M.3.4 downstream protocol, and the M.3.6 engineered baseline), re-implemented
in PyTorch and documented deviation-by-deviation in README_KO.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

EXPERIMENT_NAME = "Binary_Google_SensorFM_Nested"
TASK_DESCRIPTION = (
    "CN (0) vs MCI+Dem (1), subject-level, wearable-only: "
    "SensorFM-recipe SSL encoder vs the paper's engineered-feature baseline"
)
SEED = 20260813  # identical to TabFM/CircadianNested for outer-fold parity

# Completed formal wearable-only anchors -- used only to report deltas.
BENCHMARK = {
    "wearable_only_history": {
        "Binary_Wearable_SequenceFusion_Google": 0.5664,
        "Binary_Wearable_GoogleModels": 0.5370,
        "Binary_PaperLGBM_NoMMSE": 0.5214,
        "single_sequence_transformer_hint": 0.6254,
        "note": "honest subject-level OOF ROC-AUC; 0.45~0.57 band",
    },
    "mmse_reference_do_not_compete": {
        "Binary_MMSE_MaxAUC": 0.765756,
        "note": "different modality contract; reported for context only",
    },
}

# SensorFM has no public checkpoint or code; recorded so the report is honest.
SENSORFM_PAPER = {
    "title": "Towards a General Intelligence and Interface for Wearable Health Data",
    "arxiv": "2605.22759",
    "aim_masking_paper": "LSM-2, arXiv:2506.05321",
    "checkpoint_public": False,
    "code_public": False,
}

# ------------------------------------------------------------------ paths ----
SPLIT_DIRS = {"train": "1.Training", "val": "2.Validation"}
# Wearable-only contract: the CognitiveFunction (MMSE) file is deliberately
# absent from this mapping and must never be read by this experiment.
SOURCE_FILES = {
    "train": {
        "activity": "SourceData/1.Gait/train_activity.csv",
        "sleep": "SourceData/2.Sleep/train_sleep.csv",
    },
    "val": {
        "activity": "SourceData/1.Gait/val_activity.csv",
        "sleep": "SourceData/2.Sleep/val_sleep.csv",
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
FORBIDDEN_SUBSTRINGS = ("diag", "label", "doctor", "email", "mmse", "cognitive")
# Collection-process proxies (device-wear behavior) are not physiology.  This
# is stricter than SensorFM's M.3.6 baseline, which includes a missingness
# rate feature; we drop it (documented deviation D6 in README_KO.md).
FORBIDDEN_FEATURE_TOKENS = ("n_days", "span_days", "cover", "gap_", "nonwear",
                            "non_wear", "missing", "wearfrac")

SPLIT_CONTRACT = {
    "train": {"n": 141, "CN": 85, "MCI": 47, "Dem": 9},
    "val": {"n": 33, "CN": 26, "MCI": 4, "Dem": 3},
}
DIAG_ORDER = ("CN", "MCI", "Dem")
POSITIVE_DIAGS = ("MCI", "Dem")
CLASS_NAMES = ("CN", "MCI_DEM")

# ----------------------------------------------------------- intraday input --
INTRADAY_COLUMNS = {
    "met_1min": "CONVERT(activity_met_1min USING utf8)",
    "activity_class_5min": "CONVERT(activity_class_5min USING utf8)",
    "hypnogram_5min": "CONVERT(sleep_hypnogram_5min USING utf8)",
    "sleep_hr_5min": "CONVERT(sleep_hr_5min USING utf8)",
    "sleep_rmssd_5min": "CONVERT(sleep_rmssd_5min USING utf8)",
}
MINUTES_PER_DAY = 1440
ACTIVITY_CLASS_EPOCHS_PER_DAY = 288
ACTIVITY_EPOCH_MINUTES = 5
SLEEP_EPOCH_MINUTES = 5
ACTIVITY_CLASSES = {0: "nonwear", 1: "rest", 2: "inactive", 3: "low", 4: "medium", 5: "high"}
HYPNOGRAM_STAGES = {1: "deep", 2: "light", 3: "rem", 4: "awake"}

# The day-grid channels (our analogue of the paper's 34 minutely features;
# Table ED.1 equivalents noted).  Order is the channel axis everywhere.
CHANNELS = (
    "met",          # 1-min MET               ~ paper MTN block
    "act_class",    # 5-min intensity class    ~ paper MTN block (ordinal 1..5)
    "stage_deep",   # 5-min hypnogram one-hot  ~ paper SLP "Stage Deep"
    "stage_light",  # ~ paper SLP "Stage Light"
    "stage_rem",    # ~ paper SLP "Stage REM"
    "stage_awake",  # ~ paper SLP "Stage Awake"
    "sleep_hr",     # 5-min nocturnal HR       ~ paper CRD "Heart Rate" (night only)
    "sleep_rmssd",  # 5-min nocturnal RMSSD    ~ paper CRD "RMSSD" (night only)
)
N_CHANNELS = len(CHANNELS)

# Day-window admission rules (paper M.3.2: windows >80% missing removed).
MAX_DAY_MISSING_FRACTION = 0.80
MIN_OBSERVED_TOKENS = 8          # guards the embedding mean against empty days

# ------------------------------------------------------------ MAE geometry ---
PATCH_MINUTES = 20               # paper M.3.3: patch size [20, 1]
TOKENS_PER_CHANNEL = MINUTES_PER_DAY // PATCH_MINUTES   # 72
TOKENS_PER_DAY = TOKENS_PER_CHANNEL * N_CHANNELS        # 576
TOKEN_OBSERVED_MIN_FRACTION = 0.5  # token counts as observed if >=50% minutes are
CLIP_SIGMA = 5.0                 # paper M.3.2: z-score then clip [-5, 5]
N_CYCLIC_DIMS = 8                # paper M.3.1: 8 cyclic datetime embedding dims

# AIM artificial masking (paper M.3.3): one mode drawn per sample.
AIM_RANDOM_MASK_RATIO = 0.80
AIM_TEMPORAL_BLOCK_RATIO = 0.50
AIM_MODALITY_BLOCK_RATIO = 0.50


@dataclass(frozen=True)
class ModelVariant:
    """One row of SensorFM Table ED.4 (encoder/decoder ViT-1D dimensions)."""

    name: str
    enc_dim: int
    enc_mlp: int
    enc_heads: int
    enc_layers: int
    dec_dim: int
    dec_mlp: int
    dec_heads: int
    dec_layers: int


# Verbatim from SensorFM Table ED.4.  XXS/XS are the sizes whose pretraining
# data volume (5K-subject tier) is closest to our 141-subject cohort; B is
# unreachable and listed only for documentation.
MODEL_VARIANTS = {
    "XXS": ModelVariant("XXS", 64, 256, 1, 2, 48, 192, 1, 1),
    "XS": ModelVariant("XS", 128, 512, 2, 4, 96, 384, 2, 1),
    "S": ModelVariant("S", 256, 1024, 4, 8, 192, 768, 4, 2),
}

# ------------------------------------------------------------- pretraining ---
@dataclass(frozen=True)
class PretrainBudget:
    epochs: int
    batch_size: int
    base_lr: float          # paper: 5e-4 at batch 4096; scaled for our batch
    weight_decay: float     # paper: 1e-4
    warmup_fraction: float  # paper: 5% of steps
    val_subject_fraction: float
    patience: int           # early stop on held-out-subject reconstruction MSE
    min_epochs: int


PRETRAIN_BUDGETS = {
    "smoke": PretrainBudget(2, 64, 3e-4, 1e-4, 0.05, 0.2, 999, 1),
    "quick": PretrainBudget(40, 256, 3e-4, 1e-4, 0.05, 0.1, 8, 10),
    "default": PretrainBudget(120, 256, 3e-4, 1e-4, 0.05, 0.1, 12, 20),
    "max": PretrainBudget(200, 256, 3e-4, 1e-4, 0.05, 0.1, 15, 30),
}

# ------------------------------------------------------------- candidates ----
@dataclass(frozen=True)
class ProbeConfig:
    pca_k: int
    lr_c: float

    def key(self) -> str:
        return f"k{self.pca_k}_c{self.lr_c:g}"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    view: str               # "fe" | "emb" | "blend"
    complexity: int
    configs: tuple[ProbeConfig, ...] = field(default=())

    def describe(self) -> str:
        return f"{self.candidate_id} (view={self.view}, complexity={self.complexity})"


LR_PARAMS = {"solver": "lbfgs", "max_iter": 4000, "class_weight": "balanced"}
SELECTION_TOLERANCE = 0.005

# The paper fixes PCA-50 + a plain logistic head; our inner grid brackets that
# choice with one smaller K (141 subjects may not support 50 components) and
# two regularization strengths.  Pre-registered, never extended mid-run.
FE_CONFIGS = (ProbeConfig(50, 0.01), ProbeConfig(50, 0.1))
EMB_CONFIGS = (
    ProbeConfig(16, 0.01), ProbeConfig(16, 0.1),
    ProbeConfig(50, 0.01), ProbeConfig(50, 0.1),
)
# The blend uses fixed member configs (paper defaults) to avoid coupling the
# blend's inner score to the members' inner config search.
BLEND_FE_CONFIG = ProbeConfig(50, 0.01)
BLEND_EMB_CONFIG = ProbeConfig(50, 0.01)

CANDIDATES = (
    Candidate("fe_paper_lr", "fe", 0, FE_CONFIGS),
    Candidate("sensorfm_lr", "emb", 1, EMB_CONFIGS),
    Candidate("sensorfm_fe_blend", "blend", 2),
)

# Pre-registered paired contrasts (subject bootstrap on identical folds).
PAIRED_CONTRASTS = (
    ("nested", "fe_paper_lr"),
    ("sensorfm_lr", "fe_paper_lr"),
    ("sensorfm_fe_blend", "sensorfm_lr"),
)

# ------------------------------------------------------------- CV profiles ---
@dataclass(frozen=True)
class Profile:
    name: str
    outer_k: int
    outer_repeats: int
    inner_k: int
    inner_repeats: int
    n_bootstrap: int
    model_variant: str
    candidate_ids: tuple[str, ...]
    max_days_per_subject: int | None = None  # smoke only; None = all days
    description: str = ""


_ALL_IDS = tuple(candidate.candidate_id for candidate in CANDIDATES)

PROFILES = {
    # Wiring check only; tiny day cap + 2 epochs; never reported as performance.
    "smoke": Profile(
        name="smoke", outer_k=3, outer_repeats=1, inner_k=2, inner_repeats=1,
        n_bootstrap=200, model_variant="XXS", candidate_ids=_ALL_IDS,
        max_days_per_subject=10,
        description="wiring only - not a performance measurement",
    ),
    # Feasibility pass: measures real pretraining cost, prints the projection.
    "quick": Profile(
        name="quick", outer_k=5, outer_repeats=2, inner_k=4, inner_repeats=2,
        n_bootstrap=2000, model_variant="XXS", candidate_ids=_ALL_IDS,
        description="2 outer repeats, 40-epoch budget - projection pass",
    ),
    "default": Profile(
        name="default", outer_k=5, outer_repeats=10, inner_k=4, inner_repeats=2,
        n_bootstrap=4000, model_variant="XXS", candidate_ids=_ALL_IDS,
        description="primary protocol: 5 outer folds x 10 repeats, inner 4x2, XXS",
    ),
    # XS re-run of the identical protocol; run ONLY if default finishes in
    # budget and the XXS result motivates a capacity check.
    "max": Profile(
        name="max", outer_k=5, outer_repeats=10, inner_k=4, inner_repeats=2,
        n_bootstrap=4000, model_variant="XS", candidate_ids=_ALL_IDS,
        description="same protocol at XS capacity (paper joint-scaling check)",
    ),
}

__all__ = [
    "ACTIVITY_CLASSES", "ACTIVITY_CLASS_EPOCHS_PER_DAY", "ACTIVITY_DAY_START_HOUR",
    "ACTIVITY_EPOCH_MINUTES", "AIM_MODALITY_BLOCK_RATIO", "AIM_RANDOM_MASK_RATIO",
    "AIM_TEMPORAL_BLOCK_RATIO", "BENCHMARK", "BLEND_EMB_CONFIG", "BLEND_FE_CONFIG",
    "CANDIDATES", "CHANNELS", "CLASS_NAMES", "CLIP_SIGMA", "Candidate",
    "DIAG_ORDER", "EMB_CONFIGS", "EXPERIMENT_NAME", "FE_CONFIGS",
    "FORBIDDEN_COLUMNS", "FORBIDDEN_FEATURE_TOKENS", "FORBIDDEN_SUBSTRINGS",
    "HYPNOGRAM_STAGES", "INTRADAY_COLUMNS", "LABEL_FILES", "LOCAL_TZ",
    "LR_PARAMS", "MAX_DAY_MISSING_FRACTION", "MINUTES_PER_DAY",
    "MIN_OBSERVED_TOKENS", "MODEL_VARIANTS", "ModelVariant", "N_CHANNELS",
    "N_CYCLIC_DIMS", "PAIRED_CONTRASTS", "PATCH_MINUTES", "POSITIVE_DIAGS",
    "PRETRAIN_BUDGETS", "PROFILES", "PretrainBudget", "ProbeConfig", "Profile",
    "SEED", "SELECTION_TOLERANCE", "SENSORFM_PAPER", "SLEEP_EPOCH_MINUTES",
    "SOURCE_FILES", "SPLIT_CONTRACT", "SPLIT_DIRS", "TASK_DESCRIPTION",
    "TOKENS_PER_CHANNEL", "TOKENS_PER_DAY", "TOKEN_OBSERVED_MIN_FRACTION",
]
