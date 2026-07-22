"""Fold-local preprocessing for fixed wearable sequence views.

The sequence branch receives one equally sized recent-event view per subject.
This module deliberately has no target argument: every fitted statistic is
estimated from the subjects in the current training fold and their wearable
values only.  It never creates availability masks, collection-count features,
calendar features, identifiers, or diagnosis-derived features.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Sequence

import numpy as np


DEFAULT_VIEW_DAYS = 28
DEFAULT_RECENT_DAYS = 7

# Substring checks are intentionally fail-closed for signals known to reveal a
# subject, target, collection protocol, or absolute collection time.  ``count``
# itself is checked as a name token below so words such as ``discount`` do not
# fail accidentally.
FORBIDDEN_FEATURE_TOKENS = (
    "email",
    "subject_id",
    "patient_id",
    "diag",
    "label",
    "target",
    "mmse",
    "cognitive",
    "doctor",
    "period_id",
    "sample_order",
    "observed_count",
    "observed_day",
    "observed_night",
    "valid_count",
    "valid_ratio",
    "raw_length",
    "sequence_length",
    "missing_ratio",
    "coverage",
    "calendar_gap",
    "span_day",
    "nonwear",
    "non_wear",
    "mask",
    "delta_since",
    "absolute_date",
    "timestamp",
    "activity_day_start",
    "activity_day_end",
    "cn_abs",
)

# Only these explicitly named biological measurements receive signed-log1p.
# Ratios, scores, clock sin/cos values, entropy, efficiency and temperature are
# intentionally absent.  Matching is by final semantic component, not an
# arbitrary user-provided list, so this transformation contract is stable.
HEAVY_TAILED_SAFE_COMPONENTS = frozenset(
    {
        "activity_cal_active",
        "activity_cal_total",
        "activity_daily_movement",
        "activity_high",
        "activity_inactive",
        "activity_inactivity_alerts",
        "activity_low",
        "activity_medium",
        "activity_met_min_high",
        "activity_met_min_inactive",
        "activity_met_min_low",
        "activity_met_min_medium",
        "activity_rest",
        "activity_steps",
        "activity_total",
        "sleep_awake",
        "sleep_deep",
        "sleep_duration",
        "sleep_light",
        "sleep_onset_latency",
        "sleep_rem",
        "sleep_restless",
        "sleep_total",
        # The self-contained daily builder stores the same raw scalars below
        # ``activity__scalar`` / ``sleep__scalar`` without repeating the
        # modality prefix in the final semantic component.
        "cal_active",
        "cal_total",
        "daily_movement",
        "high",
        "inactive",
        "inactivity_alerts",
        "low",
        "medium",
        "met_min_high",
        "met_min_inactive",
        "met_min_low",
        "met_min_medium",
        "rest",
        "steps",
        "total",
        "awake",
        "deep",
        "duration",
        "light",
        "onset_latency",
        "rem",
        "restless",
    }
)

SUMMARY_STATISTICS = (
    "median",
    "iqr",
    "mad",
    "p10",
    "p90",
    "normalized_theil_sen_rank_slope",
    "recent7_minus_previous21_median",
)


def _schema_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(map(str, names)).encode("utf-8")).hexdigest()


@dataclass
class StableSummarySelector:
    """Fold-local bootstrap selector for the semantic summary bank.

    This is deliberately separate from :class:`SequencePreprocessor`: value
    preprocessing is label-free, while this optional supervised screen is fit
    only on the current outer-fold training subjects.  Modality and statistic
    quotas prevent another alphabetic/event28-median collapse.
    """

    max_features: int = 160
    bootstrap_rounds: int = 32
    correlation_threshold: float = 0.985
    minimum_per_modality: int = 40
    minimum_per_statistic: int = 8
    seed: int = 20260722
    fit_scope: str = "current outer-fold training subjects only"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str],
    ) -> "StableSummarySelector":
        values = _as_float_array(X, ndim=2, role="summary_features")
        target = np.asarray(y, dtype=np.int64)
        names = [str(name) for name in feature_names]
        if values.shape != (len(target), len(names)):
            raise ValueError("Summary matrix, labels, and names are not aligned")
        if set(np.unique(target)) != {0, 1}:
            raise ValueError("StableSummarySelector requires both binary classes")
        if not all("__summary__" in name for name in names):
            raise ValueError("Selector accepts only traceable semantic summary features")
        base_names = list(
            dict.fromkeys(name.split("__summary__", 1)[0] for name in names)
        )
        assert_sequence_feature_contract(base_names)

        clean = np.asarray(values, dtype=np.float64).copy()
        clean[~np.isfinite(clean)] = np.nan
        medians = np.nanmedian(clean, axis=0)
        medians[~np.isfinite(medians)] = 0.0
        clean = np.where(np.isnan(clean), medians[None, :], clean)
        centers = np.median(clean, axis=0)
        q25, q75 = np.quantile(clean, [0.25, 0.75], axis=0)
        scales = q75 - q25
        fallback = np.std(clean, axis=0)
        bad_scale = ~np.isfinite(scales) | (np.abs(scales) < 1e-8)
        scales[bad_scale] = fallback[bad_scale]
        scales[~np.isfinite(scales) | (np.abs(scales) < 1e-8)] = 1.0
        standardized = (clean - centers) / scales

        class_indices = [np.flatnonzero(target == class_id) for class_id in (0, 1)]
        rng = np.random.default_rng(self.seed)
        rounds = max(1, int(self.bootstrap_rounds))
        effects = np.zeros((rounds, values.shape[1]), dtype=np.float64)
        frequency = np.zeros(values.shape[1], dtype=np.float64)
        top_per_round = min(values.shape[1], max(int(self.max_features), 96))
        for round_index in range(rounds):
            sampled = np.concatenate(
                [
                    rng.choice(
                        indices,
                        size=max(3, int(np.ceil(0.85 * len(indices)))),
                        replace=True,
                    )
                    for indices in class_indices
                ]
            )
            sampled_y = target[sampled]
            negative = standardized[sampled][sampled_y == 0]
            positive = standardized[sampled][sampled_y == 1]
            delta = np.abs(positive.mean(axis=0) - negative.mean(axis=0))
            pooled = np.sqrt(0.5 * (positive.var(axis=0) + negative.var(axis=0)))
            effect = np.divide(
                delta,
                pooled,
                out=np.zeros_like(delta),
                where=pooled > 1e-8,
            )
            effect = np.nan_to_num(effect, nan=0.0, posinf=1e6, neginf=0.0)
            effects[round_index] = effect
            chosen = np.argsort(-effect, kind="stable")[:top_per_round]
            frequency[chosen] += 1.0
        frequency /= rounds
        lower_effect = np.quantile(effects, 0.25, axis=0)
        median_effect = np.median(effects, axis=0)
        composite = frequency + 0.20 * lower_effect + 0.10 * median_effect
        feature_std = np.std(clean, axis=0)
        eligible_indices = [
            index
            for index in range(len(names))
            if np.isfinite(feature_std[index]) and feature_std[index] > 1e-8
        ]
        order = sorted(
            eligible_indices, key=lambda index: (-composite[index], names[index])
        )
        if not order:
            raise ValueError("Every semantic summary feature is constant")

        centered = standardized - standardized.mean(axis=0, keepdims=True)
        norms = np.sqrt(np.square(centered).sum(axis=0))
        unit = np.divide(
            centered,
            norms[None, :],
            out=np.zeros_like(centered),
            where=norms[None, :] > 1e-12,
        )
        selected: list[int] = []
        selected_set: set[int] = set()

        def nonredundant(candidate: int) -> bool:
            if not selected:
                return True
            correlation = np.abs(unit[:, selected].T @ unit[:, candidate])
            return bool(float(correlation.max(initial=0.0)) < self.correlation_threshold)

        def add(candidate: int) -> bool:
            if candidate in selected_set or len(selected) >= int(self.max_features):
                return False
            if not nonredundant(candidate):
                return False
            selected.append(candidate)
            selected_set.add(candidate)
            return True

        modality_quota = min(
            int(self.minimum_per_modality), max(1, int(self.max_features) // 2)
        )
        for modality in ("activity__", "sleep__"):
            for candidate in order:
                if sum(names[index].startswith(modality) for index in selected) >= modality_quota:
                    break
                if names[candidate].startswith(modality):
                    add(candidate)
        for statistic in SUMMARY_STATISTICS:
            candidates = [
                index
                for index in order
                if names[index].endswith(f"__summary__{statistic}")
            ]
            for candidate in candidates:
                observed = sum(
                    names[index].endswith(f"__summary__{statistic}")
                    for index in selected
                )
                if observed >= int(self.minimum_per_statistic):
                    break
                add(candidate)
        for candidate in order:
            if len(selected) >= int(self.max_features):
                break
            add(candidate)
        if not selected:
            raise ValueError("Summary selection removed every feature")

        self.feature_names_in_ = names
        self.selected_indices_ = np.asarray(selected, dtype=np.int64)
        self.selected_feature_names_ = [names[index] for index in selected]
        self.selection_frequency_ = {
            names[index]: float(frequency[index]) for index in selected
        }
        self.selection_effect_ = {
            names[index]: float(median_effect[index]) for index in selected
        }
        self.constant_features_removed_ = int(len(names) - len(eligible_indices))
        self.achieved_modality_counts_ = {
            modality.removesuffix("__"): int(
                sum(names[index].startswith(modality) for index in selected)
            )
            for modality in ("activity__", "sleep__")
        }
        self.achieved_statistic_counts_ = {
            statistic: int(
                sum(
                    names[index].endswith(f"__summary__{statistic}")
                    for index in selected
                )
            )
            for statistic in SUMMARY_STATISTICS
        }
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "selected_indices_"):
            raise RuntimeError("StableSummarySelector must be fitted before transform")
        values = _as_float_array(X, ndim=2, role="summary_features")
        if values.shape[1] != len(self.feature_names_in_):
            raise ValueError("Summary input width differs from fitted selector")
        result = values[:, self.selected_indices_]
        if not np.isfinite(result).all():
            raise FloatingPointError("Non-finite selected summary value")
        return result.astype(np.float32, copy=False)

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray, feature_names: Sequence[str]
    ) -> np.ndarray:
        return self.fit(X, y, feature_names).transform(X)

    @property
    def selected_feature_names(self) -> list[str]:
        if not hasattr(self, "selected_feature_names_"):
            raise RuntimeError("StableSummarySelector must be fitted first")
        return list(self.selected_feature_names_)

    def manifest(self) -> dict:
        if not hasattr(self, "selected_feature_names_"):
            raise RuntimeError("StableSummarySelector must be fitted first")
        return {
            "fit_scope": str(self.fit_scope),
            "selection": (
                "stratified bootstrap effect stability with soft modality/statistic "
                "quota attempts and correlation pruning"
            ),
            "bootstrap_rounds": int(self.bootstrap_rounds),
            "input_feature_count": len(self.feature_names_in_),
            "input_feature_schema_sha256": _schema_sha256(self.feature_names_in_),
            "selected_feature_count": len(self.selected_feature_names_),
            "selected_feature_schema_sha256": _schema_sha256(
                self.selected_feature_names_
            ),
            "constant_features_removed": int(self.constant_features_removed_),
            "correlation_threshold": float(self.correlation_threshold),
            "requested_soft_modality_minimum": int(self.minimum_per_modality),
            "achieved_modality_counts": dict(self.achieved_modality_counts_),
            "requested_soft_statistic_minimum": int(self.minimum_per_statistic),
            "achieved_statistic_counts": dict(self.achieved_statistic_counts_),
            "selected_features": list(self.selected_feature_names_),
            "selection_frequency": dict(self.selection_frequency_),
            "median_bootstrap_effect": dict(self.selection_effect_),
        }


def _name_tokens(name: str) -> set[str]:
    normalized = str(name).lower()
    for delimiter in ("__", "-", "/", ".", " "):
        normalized = normalized.replace(delimiter, "_")
    return {token for token in normalized.split("_") if token}


def assert_sequence_feature_contract(feature_names: Iterable[str]) -> None:
    """Reject non-wearable, identifying, target, or protocol-proxy channels."""

    names = [str(name) for name in feature_names]
    if not names:
        raise ValueError("At least one sequence feature name is required")
    if len(names) != len(set(names)):
        raise AssertionError("Sequence feature names must be unique")
    offenders: list[str] = []
    for name in names:
        lowered = name.lower()
        tokens = _name_tokens(name)
        # ``activity_score_meet_daily_targets`` is a genuine device score, so
        # the singular model-target word is checked as an exact name token.
        forbidden_substring = any(
            token in lowered
            for token in FORBIDDEN_FEATURE_TOKENS
            if token != "target"
        )
        forbidden_target = "target" in tokens
        forbidden_count = "count" in tokens or "counts" in tokens
        wearable = lowered.startswith("activity__") or lowered.startswith("sleep__")
        if forbidden_substring or forbidden_target or forbidden_count or not wearable:
            offenders.append(name)
    if offenders:
        raise AssertionError(
            "Forbidden/non-wearable sequence feature(s): " + repr(sorted(offenders)[:20])
        )


def _semantic_component(feature_name: str) -> str:
    """Return the stable raw/derived component used by the log allow-list."""

    parts = [part for part in str(feature_name).lower().split("__") if part]
    if not parts:
        return ""
    # Daily channels normally look like ``activity__raw__activity_steps`` or
    # ``sleep__raw__sleep_duration``.  The last part is therefore exact and
    # avoids applying a logarithm to similarly named ratios.
    return parts[-1]


def is_predeclared_heavy_tailed(feature_name: str) -> bool:
    """Whether a wearable channel belongs to the fixed signed-log allow-list."""

    return _semantic_component(feature_name) in HEAVY_TAILED_SAFE_COMPONENTS


def _as_float_array(values: np.ndarray, *, ndim: int, role: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{role} must be {ndim}-D; received shape {array.shape}")
    if any(size <= 0 for size in array.shape):
        raise ValueError(f"{role} must have no empty dimension; received {array.shape}")
    return array


def _normalized_theil_sen(values: np.ndarray) -> np.ndarray:
    """Median pairwise slope on a rank axis normalized to the fixed window."""

    n_subjects, n_steps, n_features = values.shape
    if n_steps < 3:
        raise ValueError("Theil-Sen summaries require at least three time steps")
    ranks = np.linspace(0.0, 1.0, n_steps, dtype=np.float64)
    slopes: list[np.ndarray] = []
    for left in range(n_steps - 1):
        denominator = ranks[left + 1 :] - ranks[left]
        difference = values[:, left + 1 :, :] - values[:, left : left + 1, :]
        slopes.append(difference / denominator[None, :, None])
    joined = np.concatenate(slopes, axis=1)
    result = np.median(joined, axis=1)
    if result.shape != (n_subjects, n_features):
        raise AssertionError("Unexpected Theil-Sen summary shape")
    return result


def build_subject_summary_features(
    transformed_fixed_windows: np.ndarray,
    feature_names: Sequence[str],
    *,
    recent_days: int = DEFAULT_RECENT_DAYS,
) -> tuple[np.ndarray, list[str]]:
    """Summarize equally sized transformed subject windows without labels.

    For the default 28-event tail, the last statistic is the median of the
    recent seven events minus the median of the preceding 21.  A caller may
    supply another *fixed and equal* subject window; in that case the same
    recent-seven-versus-all-previous definition is used and the emitted name
    records the actual previous-window size.
    """

    windows = _as_float_array(
        transformed_fixed_windows, ndim=3, role="transformed_fixed_windows"
    )
    names = [str(name) for name in feature_names]
    assert_sequence_feature_contract(names)
    if windows.shape[2] != len(names):
        raise ValueError(
            "Summary feature names do not match the window feature dimension: "
            f"{len(names)} != {windows.shape[2]}"
        )
    if not np.isfinite(windows).all():
        raise ValueError("Summaries require already transformed finite windows")
    n_steps = windows.shape[1]
    if not 1 <= int(recent_days) < n_steps:
        raise ValueError("recent_days must leave at least one previous observation")

    q10, q25, median, q75, q90 = np.quantile(
        windows, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1
    )
    mad = np.median(np.abs(windows - median[:, None, :]), axis=1)
    slope = _normalized_theil_sen(windows)
    previous_days = n_steps - int(recent_days)
    recent_shift = np.median(windows[:, -int(recent_days) :, :], axis=1) - np.median(
        windows[:, :previous_days, :], axis=1
    )
    statistics = (
        median,
        q75 - q25,
        mad,
        q10,
        q90,
        slope,
        recent_shift,
    )
    statistic_names = list(SUMMARY_STATISTICS)
    statistic_names[-1] = (
        f"recent{int(recent_days)}_minus_previous{previous_days}_median"
    )
    # Feature-major ordering keeps every derived value adjacent to its source
    # channel and makes manifests easy to audit.
    values = np.stack(statistics, axis=2).reshape(windows.shape[0], -1)
    output_names = [
        f"{feature_name}__summary__{statistic}"
        for feature_name in names
        for statistic in statistic_names
    ]
    if values.shape[1] != len(output_names) or not np.isfinite(values).all():
        raise AssertionError("Subject summary construction produced an invalid matrix")
    return values.astype(np.float32, copy=False), output_names


@dataclass
class SequencePreprocessor:
    """Robust preprocessing learned from fixed views in one training fold."""

    view_days: int = DEFAULT_VIEW_DAYS
    apply_signed_log1p: bool = True
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    scale_epsilon: float = 1e-8
    fit_scope: str = "current fold training subjects only"

    def fit(
        self,
        fixed_train_views: np.ndarray,
        feature_names: Sequence[str],
    ) -> "SequencePreprocessor":
        """Fit value-only statistics; labels are intentionally not accepted."""

        views = _as_float_array(fixed_train_views, ndim=3, role="fixed_train_views")
        names = [str(name) for name in feature_names]
        assert_sequence_feature_contract(names)
        if int(self.view_days) < 3:
            raise ValueError("view_days must be at least three")
        if views.shape[1] != int(self.view_days):
            raise ValueError(
                "Every input row must be the same fixed view length: "
                f"expected [N,{self.view_days},F], received {views.shape}"
            )
        if views.shape[2] != len(names):
            raise ValueError(
                f"Feature-name width {len(names)} differs from view width {views.shape[2]}"
            )
        if not 0.0 <= float(self.lower_quantile) < float(self.upper_quantile) <= 1.0:
            raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1")

        self.feature_names_ = names
        self.n_features_in_ = len(names)
        self.heavy_tailed_indices_ = np.asarray(
            [index for index, name in enumerate(names) if is_predeclared_heavy_tailed(name)],
            dtype=np.int64,
        )
        prepared = self._signed_log_copy(views)
        flattened = prepared.reshape(-1, self.n_features_in_)
        flattened[~np.isfinite(flattened)] = np.nan
        all_missing = np.flatnonzero(np.isnan(flattened).all(axis=0))
        if len(all_missing):
            bad = [names[index] for index in all_missing]
            raise ValueError(f"All-missing sequence feature(s): {bad[:20]}")

        self.medians_ = np.nanmedian(flattened, axis=0)
        filled = np.where(np.isnan(flattened), self.medians_[None, :], flattened)
        self.lower_ = np.quantile(filled, float(self.lower_quantile), axis=0)
        self.upper_ = np.quantile(filled, float(self.upper_quantile), axis=0)
        clipped = np.clip(filled, self.lower_[None, :], self.upper_[None, :])
        self.centers_ = np.median(clipped, axis=0)
        q25, q75 = np.quantile(clipped, [0.25, 0.75], axis=0)
        scale = q75 - q25
        fallback = np.std(clipped, axis=0)
        use_fallback = ~np.isfinite(scale) | (np.abs(scale) < float(self.scale_epsilon))
        scale[use_fallback] = fallback[use_fallback]
        scale[~np.isfinite(scale) | (np.abs(scale) < float(self.scale_epsilon))] = 1.0
        self.scales_ = scale
        self.fit_views_ = int(views.shape[0])
        self.fit_observations_per_view_ = int(views.shape[1])
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "feature_names_"):
            raise RuntimeError("SequencePreprocessor must be fitted before transform")

    def _signed_log_copy(self, values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64).copy()
        indices = getattr(self, "heavy_tailed_indices_", np.empty(0, dtype=np.int64))
        if bool(self.apply_signed_log1p) and len(indices):
            selected = result[..., indices]
            finite = np.isfinite(selected)
            selected[finite] = np.sign(selected[finite]) * np.log1p(
                np.abs(selected[finite])
            )
            result[..., indices] = selected
        return result

    def _transform_array(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if values.shape[-1] != self.n_features_in_:
            raise ValueError(
                f"Sequence width {values.shape[-1]} differs from fitted width "
                f"{self.n_features_in_}"
            )
        prepared = self._signed_log_copy(values)
        prepared[~np.isfinite(prepared)] = np.nan
        filled = np.where(np.isnan(prepared), self.medians_, prepared)
        clipped = np.clip(filled, self.lower_, self.upper_)
        transformed = (clipped - self.centers_) / self.scales_
        if not np.isfinite(transformed).all():
            raise FloatingPointError("Non-finite sequence value survived preprocessing")
        return transformed.astype(np.float32, copy=False)

    def transform_views(self, fixed_views: np.ndarray) -> np.ndarray:
        """Transform an [N, view_days, F] fixed-view tensor."""

        self._check_fitted()
        views = _as_float_array(fixed_views, ndim=3, role="fixed_views")
        if views.shape[1] != int(self.view_days):
            raise ValueError(
                f"Expected fixed view length {self.view_days}; received {views.shape[1]}"
            )
        return self._transform_array(views)

    def transform_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Transform one non-empty variable-length [T,F] sequence."""

        self._check_fitted()
        values = _as_float_array(sequence, ndim=2, role="sequence")
        return self._transform_array(values)

    def transform_sequences(self, sequences: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Transform variable sequences without padding or emitting masks."""

        self._check_fitted()
        if len(sequences) == 0:
            raise ValueError("At least one sequence is required")
        return [self.transform_sequence(sequence) for sequence in sequences]

    def fit_transform(
        self,
        fixed_train_views: np.ndarray,
        feature_names: Sequence[str],
    ) -> np.ndarray:
        return self.fit(fixed_train_views, feature_names).transform_views(
            fixed_train_views
        )

    def make_summary_features(
        self,
        transformed_fixed_windows: np.ndarray,
        *,
        recent_days: int = DEFAULT_RECENT_DAYS,
    ) -> tuple[np.ndarray, list[str]]:
        """Create the fixed label-free semantic summary bank."""

        self._check_fitted()
        return build_subject_summary_features(
            transformed_fixed_windows,
            self.feature_names_,
            recent_days=recent_days,
        )

    def manifest(self) -> dict:
        """Return a serializable, auditable description of fitted operations."""

        self._check_fitted()
        heavy_names = [self.feature_names_[index] for index in self.heavy_tailed_indices_]
        return {
            "fit_scope": str(self.fit_scope),
            "labels_consumed_by_preprocessor": False,
            "fit_view_count": int(self.fit_views_),
            "fixed_view_length_required": True,
            "subject_balance_verified_by_preprocessor": False,
            "subject_balance_verified_by": "data.make_fixed_views and train._view_subset",
            "view_days": int(self.view_days),
            "input_feature_count": int(self.n_features_in_),
            "input_features": list(self.feature_names_),
            "input_feature_schema_sha256": _schema_sha256(self.feature_names_),
            "signed_log1p_enabled": bool(self.apply_signed_log1p),
            "signed_log1p_predeclared_features": heavy_names,
            "imputation": f"value median fitted on: {self.fit_scope}",
            "winsorization": {
                "lower_quantile": float(self.lower_quantile),
                "upper_quantile": float(self.upper_quantile),
                "fit_scope": str(self.fit_scope),
            },
            "scaling": (
                "value median/IQR with standard-deviation fallback fitted on: "
                f"{self.fit_scope}"
            ),
            "variable_sequence_transform_emits_padding": False,
            "variable_sequence_transform_emits_mask": False,
            "summary_statistics": list(SUMMARY_STATISTICS),
            "summary_recent_window_days": DEFAULT_RECENT_DAYS,
            "forbidden_feature_signals": [
                "identifier",
                "diagnosis or cognitive score",
                "absolute date/time",
                "coverage/count/length",
                "missingness mask",
                "non-wear",
            ],
        }
