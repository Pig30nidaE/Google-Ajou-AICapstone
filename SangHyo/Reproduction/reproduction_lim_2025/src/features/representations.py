"""The three input representations, plus a preprocessor that records its fit scope.

The representations are deliberately separate objects because the paper feeds the
tree models and the deep models differently (Section 3.2: "전자는 시간 순서를 유지한
채 정규화 후 3차원 텐서, 후자는 각 환자의 시계열 데이터를 평균값 기반으로 요약").

Every fit here stores the subject ids it saw, so ``src/audit/leakage.py`` can prove
after the fact that no preprocessing touched held-out subjects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..data import schema

REPRESENTATIONS = ("tabular_subject_aggregate", "daily_record", "temporal_sequence")


class PreprocessingScopeError(RuntimeError):
    """Raised when a transform is used before fitting, or fitted on the wrong rows."""


@dataclass
class FoldPreprocessor:
    """Median imputation + optional standardisation, fitted on training rows only.

    ``fitted_subjects`` is the audit trail: it must be a subset of the fold's
    training subjects, and disjoint from the evaluation subjects.
    """

    standardize: bool = True
    impute: bool = True
    fitted_subjects: tuple[str, ...] = ()
    feature_names: tuple[str, ...] = ()
    medians_: np.ndarray | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    _is_fit: bool = False

    def fit(
        self,
        X: np.ndarray,
        *,
        subjects: Sequence[str],
        feature_names: Sequence[str],
        mask: np.ndarray | None = None,
    ) -> "FoldPreprocessor":
        """Fit on the given rows.

        For a padded (N, T, F) tensor, pass ``mask`` -- an (N, T) boolean array of
        real observations -- so padded timesteps are excluded: they are not
        observations, and with 60% padding they would drag every mean toward zero.
        This also matches the paper's stated order: normalise, *then* assemble the
        tensor.

        A boolean mask rather than lengths is deliberate.  Lengths silently assume
        the valid steps sit at the front, which is false for pre-padding and was
        the source of a bug that blanked every real observation.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 3:
            if mask is None:
                flat = X.reshape(-1, X.shape[-1])
            else:
                mask = np.asarray(mask, dtype=bool)
                if mask.shape != X.shape[:2]:
                    raise PreprocessingScopeError(
                        f"mask shape {mask.shape} does not match tensor {X.shape[:2]}"
                    )
                flat = X[mask]
        else:
            flat = X
        if flat.size == 0:
            raise PreprocessingScopeError("nothing to fit on: the training slice is empty")
        self.feature_names = tuple(feature_names)
        self.fitted_subjects = tuple(sorted(set(map(str, subjects))))

        self.medians_ = np.nanmedian(flat, axis=0)
        self.medians_ = np.where(np.isfinite(self.medians_), self.medians_, 0.0)

        filled = np.where(np.isfinite(flat), flat, self.medians_)
        self.mean_ = filled.mean(axis=0)
        scale = filled.std(axis=0)
        # A zero-variance column would divide by zero; leave it centred at 0.
        self.scale_ = np.where(scale > 1e-12, scale, 1.0)
        self._is_fit = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise PreprocessingScopeError(
                "transform() called before fit(); preprocessing must be fitted on the "
                "training part of the current fold first"
            )
        X = np.asarray(X, dtype=np.float64)
        if X.ndim not in (2, 3):
            raise PreprocessingScopeError(
                f"expected a 2D/3D feature array, got shape {X.shape}"
            )
        if X.shape[-1] != len(self.feature_names):
            raise PreprocessingScopeError(
                "feature count/order contract changed: state expects "
                f"{len(self.feature_names)} columns, input has {X.shape[-1]}"
            )
        shape = X.shape
        flat = X.reshape(-1, shape[-1]) if X.ndim == 3 else X
        if self.impute:
            flat = np.where(np.isfinite(flat), flat, self.medians_)
        if self.standardize:
            flat = (flat - self.mean_) / self.scale_
        flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        return flat.reshape(shape)

    def fit_transform(
        self,
        X: np.ndarray,
        *,
        subjects: Sequence[str],
        feature_names: Sequence[str],
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.fit(
            X, subjects=subjects, feature_names=feature_names, mask=mask
        ).transform(X)

    def audit_record(self) -> dict[str, Any]:
        return {
            "standardize": self.standardize,
            "impute": self.impute,
            "n_fitted_subjects": len(self.fitted_subjects),
            "fitted_subjects": list(self.fitted_subjects),
            "n_features": len(self.feature_names),
        }

    def state_record(self) -> dict[str, Any]:
        """Serializable transform state needed to reproduce raw-input inference."""
        if not self._is_fit:
            raise PreprocessingScopeError("cannot serialise an unfitted preprocessor")
        return {
            "standardize": bool(self.standardize),
            "impute": bool(self.impute),
            "feature_names": list(self.feature_names),
            "medians": self.medians_.tolist(),
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "fitted_subjects": list(self.fitted_subjects),
        }

    @classmethod
    def from_state_record(
        cls,
        state: dict[str, Any],
        *,
        expected_feature_names: Sequence[str] | None = None,
    ) -> "FoldPreprocessor":
        """Restore a fitted transform and fail closed on order/shape corruption."""
        required = {
            "standardize", "impute", "feature_names", "medians", "mean", "scale",
            "fitted_subjects",
        }
        missing = sorted(required - set(state))
        if missing:
            raise PreprocessingScopeError(
                f"preprocessor checkpoint is missing fields: {missing}"
            )

        names = tuple(map(str, state["feature_names"]))
        if expected_feature_names is not None:
            expected = tuple(map(str, expected_feature_names))
            if names != expected:
                raise PreprocessingScopeError(
                    "preprocessor feature order differs from the representation: "
                    f"checkpoint={names}, representation={expected}"
                )

        arrays = {
            "medians": np.asarray(state["medians"], dtype=np.float64),
            "mean": np.asarray(state["mean"], dtype=np.float64),
            "scale": np.asarray(state["scale"], dtype=np.float64),
        }
        for key, values in arrays.items():
            if values.ndim != 1 or len(values) != len(names):
                raise PreprocessingScopeError(
                    f"preprocessor {key} shape {values.shape} does not match "
                    f"{len(names)} feature names"
                )
            if not np.isfinite(values).all():
                raise PreprocessingScopeError(
                    f"preprocessor {key} contains a non-finite value"
                )
        if (arrays["scale"] <= 0).any():
            raise PreprocessingScopeError("preprocessor scale must be strictly positive")

        restored = cls(
            standardize=bool(state["standardize"]),
            impute=bool(state["impute"]),
            fitted_subjects=tuple(map(str, state["fitted_subjects"])),
            feature_names=names,
        )
        restored.medians_ = arrays["medians"]
        restored.mean_ = arrays["mean"]
        restored.scale_ = arrays["scale"]
        restored._is_fit = True
        return restored


@dataclass
class Representation:
    """A materialised model input.

    ``X`` is (N, F) for tabular/daily and (N, T, F) for sequences.  ``subjects``
    and ``y`` are aligned to axis 0, so a subject-wise split is always expressible
    as a boolean mask over rows.
    """

    kind: str
    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    feature_names: tuple[str, ...]
    lengths: np.ndarray | None = None      # represented calendar/observed span
    observation_mask: np.ndarray | None = None  # actual measured timesteps
    row_ids: np.ndarray | None = None      # source row ids, for overlap audits
    dates: np.ndarray | None = None        # source dates, for overlap audits
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_units(self) -> int:
        return int(self.X.shape[0])

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(self.X.shape[1:])

    @property
    def padding_side(self) -> str:
        return str(self.meta.get("padding", "pre"))

    def valid_mask(self) -> np.ndarray | None:
        """(N, T) boolean mask of real observations, or None if not a sequence.

        This is the single source of truth for where measured rows sit.  It also
        excludes internal missing calendar days.  Nothing downstream may
        re-derive it from ``lengths`` alone: lengths describe the represented time
        span, not necessarily the number of observed rows.
        """
        if self.kind != "temporal_sequence" or self.lengths is None:
            return None
        if self.observation_mask is not None:
            return np.asarray(self.observation_mask, dtype=bool)
        return self.span_mask()

    def span_mask(self) -> np.ndarray | None:
        """External-padding mask; internal calendar gaps remain inside the span."""
        if self.kind != "temporal_sequence" or self.lengths is None:
            return None
        timesteps = int(self.X.shape[1])
        index = np.arange(timesteps)[None, :]
        lengths = np.asarray(self.lengths, dtype=int)[:, None]
        if self.padding_side == "pre":
            return index >= (timesteps - lengths)
        return index < lengths

    def left_aligned(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(X, lengths)`` with every represented time span at the front.

        ``pack_padded_sequence`` requires this layout.  Internal calendar gaps
        stay in place as zero timesteps; only external pre-padding is moved.
        CNNs do not call this method because Flatten is position-sensitive and
        must preserve the configured padding side.
        """
        mask = self.span_mask()
        if mask is None:
            return self.X, np.full(len(self.X), self.X.shape[1], dtype=np.int64)
        if self.padding_side != "pre":
            return self.X, np.asarray(self.lengths, dtype=np.int64)

        aligned = np.zeros_like(self.X)
        lengths = np.asarray(self.lengths, dtype=np.int64)
        for i, valid in enumerate(lengths):
            if valid:
                aligned[i, :valid, :] = self.X[i, mask[i], :]
        return aligned, lengths

    def mask_for_subjects(self, subjects: Sequence[str]) -> np.ndarray:
        wanted = set(map(str, subjects))
        return np.array([str(s) in wanted for s in self.subjects], dtype=bool)

    def subset(self, mask: np.ndarray) -> "Representation":
        mask = np.asarray(mask, dtype=bool)
        return Representation(
            kind=self.kind,
            X=self.X[mask],
            y=self.y[mask],
            subjects=self.subjects[mask],
            feature_names=self.feature_names,
            lengths=None if self.lengths is None else self.lengths[mask],
            observation_mask=(
                None if self.observation_mask is None else self.observation_mask[mask]
            ),
            row_ids=None if self.row_ids is None else self.row_ids[mask],
            dates=None if self.dates is None else self.dates[mask],
            meta=dict(self.meta),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n_units": self.n_units,
            "input_shape": list(self.input_shape),
            "n_subjects": int(len(set(map(str, self.subjects)))),
            "n_features": len(self.feature_names),
            "positive_rate": round(float(np.mean(self.y)), 4) if len(self.y) else None,
            **self.meta,
        }


# --- builders -----------------------------------------------------------------

def build_tabular_subject_aggregate(
    daily: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    subjects: Sequence[str] | None = None,
    agg: str = "mean",
) -> Representation:
    """One row per subject -- the paper's ``groupby('EMAIL').mean()`` flattening."""
    if agg != "mean":
        raise NotImplementedError(
            f"agg={agg!r} is not the paper's method; only 'mean' is implemented"
        )
    frame = daily
    if subjects is not None:
        frame = frame[frame[schema.SUBJECT_ID].isin(set(map(str, subjects)))]

    columns = list(feature_columns)
    grouped = frame.groupby(schema.SUBJECT_ID, sort=True)
    X = grouped[columns].mean()
    labels = grouped[schema.LABEL_COL].first().loc[X.index]
    n_days = grouped.size().loc[X.index]

    return Representation(
        kind="tabular_subject_aggregate",
        X=X.to_numpy(dtype=np.float64),
        y=labels.to_numpy(dtype=np.int64),
        subjects=np.asarray(X.index, dtype=object),
        feature_names=tuple(columns),
        meta={"aggregation": agg, "days_per_subject_mean": round(float(n_days.mean()), 2)},
    )


def build_daily_record(
    daily: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    subjects: Sequence[str] | None = None,
) -> Representation:
    """One row per subject-day.  Predictions must be pooled back to subjects."""
    frame = daily
    if subjects is not None:
        frame = frame[frame[schema.SUBJECT_ID].isin(set(map(str, subjects)))]
    frame = frame.sort_values([schema.SUBJECT_ID, schema.DATE_COL])
    columns = list(feature_columns)
    return Representation(
        kind="daily_record",
        X=frame[columns].to_numpy(dtype=np.float64),
        y=frame[schema.LABEL_COL].to_numpy(dtype=np.int64),
        subjects=frame[schema.SUBJECT_ID].to_numpy(dtype=object),
        feature_names=tuple(columns),
        row_ids=frame["row_id"].to_numpy() if "row_id" in frame.columns else None,
        dates=frame[schema.DATE_COL].to_numpy() if schema.DATE_COL in frame.columns else None,
        meta={},
    )


def build_temporal_sequence(
    daily: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    subjects: Sequence[str] | None = None,
    sequence_length: int | str = "max",
    padding: str = "pre",
    truncation: str = "last",
    gap_handling: str = "compress",
    min_observations: int = 1,
) -> Representation:
    """One padded (T, F) sequence per subject, ordered by date.

    ``gap_handling='compress'`` stacks only observed days; ``'calendar'`` lays them
    on a daily grid and leaves missing days as padding.  The paper reports neither
    (``unresolved_questions.md`` Q13), so both are available.
    """
    if padding not in ("pre", "post"):
        raise ValueError(f"padding must be pre/post, got {padding!r}")
    if truncation not in ("last", "first"):
        raise ValueError(f"truncation must be last/first, got {truncation!r}")
    if gap_handling not in ("compress", "calendar"):
        raise ValueError(f"gap_handling must be compress/calendar, got {gap_handling!r}")

    frame = daily
    if subjects is not None:
        frame = frame[frame[schema.SUBJECT_ID].isin(set(map(str, subjects)))]
    frame = frame.sort_values([schema.SUBJECT_ID, schema.DATE_COL])
    columns = list(feature_columns)

    per_subject: list[
        tuple[str, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    skipped: list[str] = []
    for subject, group in frame.groupby(schema.SUBJECT_ID, sort=True):
        if len(group) < min_observations:
            skipped.append(str(subject))
            continue
        values = group[columns].to_numpy(dtype=np.float64)
        dates = group[schema.DATE_COL].to_numpy()
        row_ids = (
            group["row_id"].to_numpy() if "row_id" in group.columns
            else np.full(len(group), -1)
        )
        observed_mask = np.ones(len(values), dtype=bool)
        if gap_handling == "calendar":
            values, dates, row_ids, observed_mask = _to_calendar_grid(
                values, dates, row_ids
            )
        per_subject.append(
            (
                str(subject), int(group[schema.LABEL_COL].iloc[0]), values,
                dates, row_ids, observed_mask,
            )
        )

    if not per_subject:
        raise ValueError("no subject met min_observations; cannot build sequences")

    source_spans = [len(v) for _, _, v, _, _, _ in per_subject]
    source_observed = [int(mask.sum()) for _, _, _, _, _, mask in per_subject]
    if sequence_length == "max":
        length = int(max(source_spans))
    elif sequence_length == "median":
        length = int(np.median(source_spans))
    elif sequence_length == "min":
        length = int(min(source_spans))
    else:
        length = int(sequence_length)
    if length < 1:
        raise ValueError(f"sequence_length resolved to {length}")

    n_features = len(columns)
    X = np.zeros((len(per_subject), length, n_features), dtype=np.float64)
    observation_mask = np.zeros((len(per_subject), length), dtype=bool)
    lengths = np.zeros(len(per_subject), dtype=np.int64)
    y = np.zeros(len(per_subject), dtype=np.int64)
    ids = np.empty(len(per_subject), dtype=object)
    used_rows: list[np.ndarray] = []
    used_dates: list[np.ndarray] = []
    truncated_observations = 0
    truncated_span_steps = 0
    truncated_subjects: list[str] = []

    for i, (subject, label, values, dates, row_ids, observed_mask) in enumerate(per_subject):
        if len(values) > length:
            truncated_span_steps += int(len(values) - length)
            truncated_subjects.append(subject)
            if truncation == "last":
                truncated_observations += int(observed_mask[:-length].sum())
                values, dates, row_ids = values[-length:], dates[-length:], row_ids[-length:]
                observed_mask = observed_mask[-length:]
            else:
                truncated_observations += int(observed_mask[length:].sum())
                values, dates, row_ids = values[:length], dates[:length], row_ids[:length]
                observed_mask = observed_mask[:length]
        valid = len(values)
        if padding == "pre":
            X[i, length - valid:, :] = values
            observation_mask[i, length - valid:] = observed_mask
        else:
            X[i, :valid, :] = values
            observation_mask[i, :valid] = observed_mask
        lengths[i] = valid
        y[i] = label
        ids[i] = subject
        used_rows.append(np.asarray(row_ids))
        used_dates.append(np.asarray(dates))

    return Representation(
        kind="temporal_sequence",
        X=X,
        y=y,
        subjects=ids,
        feature_names=tuple(columns),
        lengths=lengths,
        observation_mask=observation_mask,
        row_ids=np.array(used_rows, dtype=object),
        dates=np.array(used_dates, dtype=object),
        meta={
            "sequence_length": length,
            "sequence_length_setting": sequence_length,
            "padding": padding,
            "truncation": truncation,
            "gap_handling": gap_handling,
            "observed_days_min": int(observation_mask.sum(axis=1).min()),
            "observed_days_max": int(observation_mask.sum(axis=1).max()),
            "source_observed_days_min": int(min(source_observed)),
            "source_observed_days_max": int(max(source_observed)),
            "sequence_span_min": int(lengths.min()),
            "sequence_span_max": int(lengths.max()),
            "source_sequence_span_min": int(min(source_spans)),
            "source_sequence_span_max": int(max(source_spans)),
            "padding_or_gap_fraction": round(
                float(1.0 - observation_mask.sum() / observation_mask.size), 4
            ),
            "truncated_observations": int(truncated_observations),
            "truncated_span_steps": int(truncated_span_steps),
            "truncated_subjects": truncated_subjects,
            "skipped_subjects": skipped,
        },
    )


def _to_calendar_grid(
    values: np.ndarray, dates: np.ndarray, row_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Place observations on a daily grid, leaving gaps as NaN rows."""
    stamps = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates)))
    if stamps.isna().any():
        raise ValueError("calendar gap handling requires a valid date for every row")
    if stamps.has_duplicates:
        duplicates = stamps[stamps.duplicated()].unique().astype(str).tolist()
        raise ValueError(
            "calendar gap handling requires one row per subject-date; duplicate "
            f"dates include {duplicates[:5]}"
        )
    grid = pd.date_range(stamps.min(), stamps.max(), freq="D")
    frame = pd.DataFrame(values, index=stamps)
    frame = frame.reindex(grid)
    id_series = pd.Series(row_ids, index=stamps).reindex(grid, fill_value=-1)
    observed = pd.Series(True, index=stamps).reindex(grid, fill_value=False)
    return (
        frame.to_numpy(dtype=np.float64),
        grid.to_numpy(),
        id_series.to_numpy(),
        observed.to_numpy(dtype=bool),
    )


def zero_padding(rep: Representation) -> None:
    """Reset padded timesteps to 0 in place, after standardisation.

    Standardising turns a padded 0 into ``-mean/scale``, which a Conv1d would
    convolve across as if it were a real observation.  The recurrent models mask
    padding anyway; this makes the convolutional one safe too.
    """
    mask = rep.valid_mask()
    if mask is None:
        return
    rep.X = rep.X * mask[:, :, None]


def fit_transform_pair(
    preprocessor: "FoldPreprocessor",
    train_rep: Representation,
    test_rep: Representation,
    *,
    fit_subjects: Sequence[str] | None = None,
    fit_on: Representation | None = None,
) -> None:
    """Fit on the training representation and transform both, in place.

    ``fit_on``/``fit_subjects`` exist only for the paper reconstruction's
    ``scaler_scope: all_data`` variant, which deliberately fits on everything so
    the resulting optimism can be measured.
    """
    source = fit_on if fit_on is not None else train_rep
    subjects = fit_subjects if fit_subjects is not None else source.subjects
    preprocessor.fit(
        source.X,
        subjects=subjects,
        feature_names=source.feature_names,
        mask=source.valid_mask(),
    )
    for rep in (train_rep, test_rep):
        rep.X = preprocessor.transform(rep.X)
        zero_padding(rep)


def build_representation(
    kind: str,
    daily: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    subjects: Sequence[str] | None = None,
    **kwargs: Any,
) -> Representation:
    """Dispatch to the requested representation builder."""
    if kind == "tabular_subject_aggregate":
        allowed = {"agg"}
        return build_tabular_subject_aggregate(
            daily, feature_columns, subjects=subjects,
            **{k: v for k, v in kwargs.items() if k in allowed},
        )
    if kind == "daily_record":
        return build_daily_record(daily, feature_columns, subjects=subjects)
    if kind == "temporal_sequence":
        allowed = {
            "sequence_length", "padding", "truncation", "gap_handling", "min_observations",
        }
        return build_temporal_sequence(
            daily, feature_columns, subjects=subjects,
            **{k: v for k, v in kwargs.items() if k in allowed},
        )
    raise ValueError(f"unknown representation: {kind!r}; expected one of {REPRESENTATIONS}")
