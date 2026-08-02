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
        lengths: Sequence[int] | None = None,
    ) -> "FoldPreprocessor":
        """Fit on the given rows.

        For a padded (N, T, F) tensor, pass ``lengths`` so the padded timesteps are
        excluded: they are not observations, and with 60% padding they would drag
        every mean toward zero.  This also matches the paper's stated order --
        normalise, *then* assemble the tensor.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 3:
            if lengths is None:
                flat = X.reshape(-1, X.shape[-1])
            else:
                valid = (
                    np.arange(X.shape[1])[None, :] < np.asarray(lengths, dtype=int)[:, None]
                )
                flat = X[valid]
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
        lengths: Sequence[int] | None = None,
    ) -> np.ndarray:
        return self.fit(
            X, subjects=subjects, feature_names=feature_names, lengths=lengths
        ).transform(X)

    def audit_record(self) -> dict[str, Any]:
        return {
            "standardize": self.standardize,
            "impute": self.impute,
            "n_fitted_subjects": len(self.fitted_subjects),
            "fitted_subjects": list(self.fitted_subjects),
            "n_features": len(self.feature_names),
        }


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
    lengths: np.ndarray | None = None      # valid timesteps per sequence
    row_ids: np.ndarray | None = None      # source row ids, for overlap audits
    dates: np.ndarray | None = None        # source dates, for overlap audits
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_units(self) -> int:
        return int(self.X.shape[0])

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(self.X.shape[1:])

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

    per_subject: list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray]] = []
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
        if gap_handling == "calendar":
            values, dates, row_ids = _to_calendar_grid(values, dates, row_ids)
        per_subject.append(
            (str(subject), int(group[schema.LABEL_COL].iloc[0]), values, dates, row_ids)
        )

    if not per_subject:
        raise ValueError("no subject met min_observations; cannot build sequences")

    observed = [len(v) for _, _, v, _, _ in per_subject]
    if sequence_length == "max":
        length = int(max(observed))
    elif sequence_length == "median":
        length = int(np.median(observed))
    elif sequence_length == "min":
        length = int(min(observed))
    else:
        length = int(sequence_length)
    if length < 1:
        raise ValueError(f"sequence_length resolved to {length}")

    n_features = len(columns)
    X = np.zeros((len(per_subject), length, n_features), dtype=np.float64)
    lengths = np.zeros(len(per_subject), dtype=np.int64)
    y = np.zeros(len(per_subject), dtype=np.int64)
    ids = np.empty(len(per_subject), dtype=object)
    used_rows: list[np.ndarray] = []
    used_dates: list[np.ndarray] = []

    for i, (subject, label, values, dates, row_ids) in enumerate(per_subject):
        if len(values) > length:
            if truncation == "last":
                values, dates, row_ids = values[-length:], dates[-length:], row_ids[-length:]
            else:
                values, dates, row_ids = values[:length], dates[:length], row_ids[:length]
        valid = len(values)
        if padding == "pre":
            X[i, length - valid:, :] = values
        else:
            X[i, :valid, :] = values
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
        row_ids=np.array(used_rows, dtype=object),
        dates=np.array(used_dates, dtype=object),
        meta={
            "sequence_length": length,
            "sequence_length_setting": sequence_length,
            "padding": padding,
            "truncation": truncation,
            "gap_handling": gap_handling,
            "observed_days_min": int(min(observed)),
            "observed_days_max": int(max(observed)),
            "padding_fraction": round(float(1.0 - lengths.sum() / (len(lengths) * length)), 4),
            "skipped_subjects": skipped,
        },
    )


def _to_calendar_grid(
    values: np.ndarray, dates: np.ndarray, row_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place observations on a daily grid, leaving gaps as NaN rows."""
    stamps = pd.to_datetime(pd.Series(dates))
    grid = pd.date_range(stamps.min(), stamps.max(), freq="D")
    frame = pd.DataFrame(values, index=pd.DatetimeIndex(stamps))
    frame = frame.reindex(grid)
    id_series = pd.Series(row_ids, index=pd.DatetimeIndex(stamps)).reindex(grid, fill_value=-1)
    return frame.to_numpy(dtype=np.float64), grid.to_numpy(), id_series.to_numpy()


def zero_padding(rep: Representation) -> None:
    """Reset padded timesteps to 0 in place, after standardisation.

    Standardising turns a padded 0 into ``-mean/scale``, which a Conv1d would
    convolve across as if it were a real observation.  The recurrent models mask
    padding anyway; this makes the convolutional one safe too.
    """
    if rep.kind != "temporal_sequence" or rep.lengths is None:
        return
    valid = np.arange(rep.X.shape[1])[None, :] < np.asarray(rep.lengths, dtype=int)[:, None]
    rep.X = rep.X * valid[:, :, None]


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
        lengths=source.lengths,
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
