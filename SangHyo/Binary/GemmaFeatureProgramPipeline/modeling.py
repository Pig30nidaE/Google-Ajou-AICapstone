"""Fold-local models for the Gemma feature-program experiment.

The important distinction in this module is between a *training reference*
rank and a test-batch rank.  Every base learner stores the empirical CDF of its
raw decision scores on the rows used to fit that learner.  A future subject is
then located in that frozen reference distribution.  Test subjects are never
ranked against one another, so the score is inductive and is also meaningful
when ``predict`` receives a single row.

The preprocessing contract is equally strict:

``median impute -> 1/99 percentile winsorise -> standardise``

Every statistic and every constant-column decision is fitted on the current
training rows only.  The wearable primitive and Gemma-program blocks share the
same fitted primitive transformer.  The program receives a finite,
standardised matrix in the original wearable-column order; inactive primitive
columns are represented by zero rather than removed so the frozen program's
dependencies remain stable across folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

__all__ = [
    "FoldLocalTransformer",
    "EmpiricalCDF",
    "InductiveRankEnsemble",
    "FittedBaseBlocks",
    "TrainingBundle",
    "fit_final_bundle",
]


def _as_float_matrix(value: Any, *, where: str) -> np.ndarray:
    """Convert a table to a two-dimensional float matrix, without fitting."""

    if isinstance(value, pd.DataFrame):
        matrix = value.to_numpy(dtype=np.float64)
    else:
        matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{where} must be a two-dimensional matrix")
    return matrix


def _column_names(frame: pd.DataFrame, *, where: str) -> tuple[str, ...]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{where} must be a pandas DataFrame so columns are auditable")
    names = tuple(map(str, frame.columns))
    if len(set(names)) != len(names):
        raise ValueError(f"{where} contains duplicate column names")
    return names


def _ordered_matrix(
    frame: pd.DataFrame,
    expected_names: Sequence[str],
    *,
    where: str,
) -> np.ndarray:
    """Return columns in their fitted order and reject a changed schema."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{where} must be a pandas DataFrame")
    current = tuple(map(str, frame.columns))
    if len(set(current)) != len(current):
        raise ValueError(f"{where} contains duplicate column names")
    missing = [name for name in expected_names if name not in current]
    if missing:
        raise ValueError(f"{where} is missing fitted column(s): {missing[:8]}")
    lookup = {str(name): name for name in frame.columns}
    return frame.loc[:, [lookup[name] for name in expected_names]].to_numpy(dtype=np.float64)


@dataclass
class FoldLocalTransformer:
    """Median imputation, winsorisation and scaling fitted on training rows.

    ``active_mask_`` is learned only from the fit matrix.  ``transform`` normally
    returns active columns for a classifier.  With ``active_only=False`` it
    returns every original column and fills inactive columns with zero; this is
    the representation supplied to the frozen Gemma feature program.
    """

    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    variance_epsilon: float = 1e-12
    feature_names_: tuple[str, ...] = field(default_factory=tuple, init=False)
    median_: np.ndarray | None = field(default=None, init=False, repr=False)
    lower_: np.ndarray | None = field(default=None, init=False, repr=False)
    upper_: np.ndarray | None = field(default=None, init=False, repr=False)
    center_: np.ndarray | None = field(default=None, init=False, repr=False)
    scale_: np.ndarray | None = field(default=None, init=False, repr=False)
    active_mask_: np.ndarray | None = field(default=None, init=False, repr=False)
    n_fit_rows_: int = field(default=0, init=False)

    def fit(
        self,
        X: Any,
        feature_names: Sequence[str],
    ) -> "FoldLocalTransformer":
        matrix = _as_float_matrix(X, where="FoldLocalTransformer.fit X")
        names = tuple(map(str, feature_names))
        if matrix.shape[0] == 0:
            raise ValueError("FoldLocalTransformer.fit needs at least one training row")
        if matrix.shape[1] != len(names):
            raise ValueError("feature_names length does not match the fit matrix")
        if len(set(names)) != len(names):
            raise ValueError("feature_names contains duplicates")
        if not 0.0 <= self.lower_quantile < self.upper_quantile <= 1.0:
            raise ValueError("winsorisation quantiles must satisfy 0 <= lower < upper <= 1")

        n_features = matrix.shape[1]
        if n_features == 0:
            empty = np.empty(0, dtype=np.float64)
            self.feature_names_ = names
            self.median_ = empty.copy()
            self.lower_ = empty.copy()
            self.upper_ = empty.copy()
            self.center_ = empty.copy()
            self.scale_ = empty.copy()
            self.active_mask_ = np.empty(0, dtype=bool)
            self.n_fit_rows_ = int(matrix.shape[0])
            return self

        finite = np.isfinite(matrix)
        has_finite = finite.any(axis=0)
        median = np.zeros(n_features, dtype=np.float64)
        for column in np.flatnonzero(has_finite):
            median[column] = float(np.median(matrix[finite[:, column], column]))

        filled = np.array(matrix, dtype=np.float64, copy=True)
        missing_rows, missing_columns = np.nonzero(~finite)
        if missing_rows.size:
            filled[missing_rows, missing_columns] = median[missing_columns]

        lower = np.quantile(filled, self.lower_quantile, axis=0)
        upper = np.quantile(filled, self.upper_quantile, axis=0)
        clipped = np.clip(filled, lower, upper)
        center = clipped.mean(axis=0)
        scale_raw = clipped.std(axis=0)
        active = has_finite & np.isfinite(scale_raw) & (scale_raw > self.variance_epsilon)
        scale = np.where(active, scale_raw, 1.0)

        self.feature_names_ = names
        self.median_ = median
        self.lower_ = lower
        self.upper_ = upper
        self.center_ = center
        self.scale_ = scale
        self.active_mask_ = active
        self.n_fit_rows_ = int(matrix.shape[0])
        return self

    @property
    def is_fitted(self) -> bool:
        return self.median_ is not None

    @property
    def active_feature_names(self) -> tuple[str, ...]:
        if self.active_mask_ is None:
            raise RuntimeError("active_feature_names requested before fit")
        return tuple(
            name for name, keep in zip(self.feature_names_, self.active_mask_) if bool(keep)
        )

    @property
    def n_active_features(self) -> int:
        if self.active_mask_ is None:
            raise RuntimeError("n_active_features requested before fit")
        return int(self.active_mask_.sum())

    def transform(self, X: Any, *, active_only: bool = True) -> np.ndarray:
        if not self.is_fitted or self.active_mask_ is None:
            raise RuntimeError("FoldLocalTransformer.transform called before fit")
        matrix = _as_float_matrix(X, where="FoldLocalTransformer.transform X")
        if matrix.shape[1] != len(self.feature_names_):
            raise ValueError(
                "transform matrix has a different number of columns than the fit matrix"
            )
        if matrix.shape[1] == 0:
            return np.empty((matrix.shape[0], 0), dtype=np.float64)

        out = np.array(matrix, dtype=np.float64, copy=True)
        missing_rows, missing_columns = np.nonzero(~np.isfinite(out))
        if missing_rows.size:
            out[missing_rows, missing_columns] = self.median_[missing_columns]
        out = np.clip(out, self.lower_, self.upper_)
        out = (out - self.center_) / self.scale_
        # Inactive columns are deliberately represented as a stable zero for
        # program dependencies.  The classifier path removes them below.
        out[:, ~self.active_mask_] = 0.0
        if not np.isfinite(out).all():
            raise FloatingPointError("fold-local transform produced a non-finite value")
        return out[:, self.active_mask_] if active_only else out

    def fit_transform(
        self,
        X: Any,
        feature_names: Sequence[str],
        *,
        active_only: bool = True,
    ) -> np.ndarray:
        return self.fit(X, feature_names).transform(X, active_only=active_only)


@dataclass(frozen=True)
class EmpiricalCDF:
    """Frozen empirical CDF of one learner's training decision scores."""

    sorted_reference: np.ndarray

    @classmethod
    def fit(cls, raw_training_scores: Sequence[float]) -> "EmpiricalCDF":
        values = np.asarray(raw_training_scores, dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("EmpiricalCDF needs finite training decision scores")
        return cls(sorted_reference=np.sort(values))

    def transform(self, raw_scores: Sequence[float]) -> np.ndarray:
        values = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("Cannot rank non-finite decision scores")
        reference = self.sorted_reference
        # Mid-CDF handles ties without depending on their input order.  Crucially,
        # every lookup is against `reference`, never against the prediction batch.
        left = np.searchsorted(reference, values, side="left")
        right = np.searchsorted(reference, values, side="right")
        return (left.astype(np.float64) + right.astype(np.float64)) / (
            2.0 * float(reference.size)
        )


@dataclass
class _RankedComponent:
    name: str
    estimator: Any
    reference: EmpiricalCDF


@dataclass
class InductiveRankEnsemble:
    """Regularised LR/linear-SVM ensemble on a frozen training-CDF scale."""

    c_values: tuple[float, ...] = (0.1, 1.0)
    components_: list[_RankedComponent] = field(default_factory=list, init=False, repr=False)
    n_features_in_: int | None = field(default=None, init=False)
    constant_score_: float | None = field(default=None, init=False)

    def fit(self, X: Any, y: Sequence[int], *, seed: int) -> "InductiveRankEnsemble":
        matrix = _as_float_matrix(X, where="InductiveRankEnsemble.fit X")
        target = np.asarray(y, dtype=np.int64).reshape(-1)
        if matrix.shape[0] != target.size:
            raise ValueError("X and y row counts differ")
        if matrix.shape[0] == 0 or len(np.unique(target)) != 2:
            raise ValueError("InductiveRankEnsemble requires a non-empty two-class fit set")
        if not np.isfinite(matrix).all():
            raise ValueError("InductiveRankEnsemble.fit requires finite preprocessed X")
        if any(float(c) <= 0 for c in self.c_values):
            raise ValueError("All regularisation C values must be positive")

        self.n_features_in_ = int(matrix.shape[1])
        self.components_ = []
        self.constant_score_ = None
        if matrix.shape[1] == 0:
            # A fold whose training rows make every feature constant must remain
            # scoreable, but it must not borrow a feature decision from another
            # fold.  0.5 is the neutral point of the rank scale.
            self.constant_score_ = 0.5
            return self

        estimators: list[tuple[str, Any]] = []
        for c_value in self.c_values:
            label = f"{float(c_value):g}"
            estimators.extend(
                [
                    (
                        f"logistic_c{label}",
                        LogisticRegression(
                            C=float(c_value),
                            penalty="l2",
                            solver="liblinear",
                            class_weight="balanced",
                            max_iter=10_000,
                            random_state=int(seed),
                        ),
                    ),
                    (
                        f"linear_svc_c{label}",
                        LinearSVC(
                            C=float(c_value),
                            class_weight="balanced",
                            dual="auto",
                            max_iter=20_000,
                            random_state=int(seed),
                        ),
                    ),
                ]
            )

        for name, estimator in estimators:
            estimator.fit(matrix, target)
            raw_training = np.asarray(estimator.decision_function(matrix), dtype=np.float64)
            self.components_.append(
                _RankedComponent(
                    name=name,
                    estimator=estimator,
                    reference=EmpiricalCDF.fit(raw_training),
                )
            )
        return self

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(component.name for component in self.components_)

    def predict_rank(self, X: Any) -> np.ndarray:
        if self.n_features_in_ is None:
            raise RuntimeError("InductiveRankEnsemble.predict_rank called before fit")
        matrix = _as_float_matrix(X, where="InductiveRankEnsemble.predict_rank X")
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("prediction feature count differs from fitted feature count")
        if not np.isfinite(matrix).all():
            raise ValueError("InductiveRankEnsemble.predict_rank requires finite X")
        if self.constant_score_ is not None:
            return np.full(matrix.shape[0], self.constant_score_, dtype=np.float64)
        ranked = [
            component.reference.transform(component.estimator.decision_function(matrix))
            for component in self.components_
        ]
        if not ranked:
            raise RuntimeError("InductiveRankEnsemble has no fitted components")
        return np.mean(np.vstack(ranked), axis=0)


def _apply_program_matrix(
    primitive_matrix: np.ndarray,
    primitive_names: Sequence[str],
    program: Mapping[str, Any],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Apply the validated row-local DSL and verify its static output schema."""

    # Delayed import keeps this module independently inspectable while the
    # schema module is generated, and avoids a module-level import cycle.
    from .program_schema import apply_program, program_feature_names

    names = tuple(map(str, program_feature_names(program)))
    output = apply_program(primitive_matrix, tuple(map(str, primitive_names)), program)
    matrix = _as_float_matrix(output, where="program_schema.apply_program output")
    if matrix.shape[0] != primitive_matrix.shape[0]:
        raise ValueError("Gemma program changed the number or order of subject rows")
    if matrix.shape[1] != len(names):
        raise ValueError(
            "Gemma program output width disagrees with program_feature_names(program)"
        )
    return matrix, names


@dataclass
class FittedBaseBlocks:
    """Three independently regularised blocks fitted on exactly the same rows."""

    program: Mapping[str, Any]
    mmse_columns: tuple[str, ...]
    wearable_columns: tuple[str, ...]
    program_columns: tuple[str, ...]
    mmse_transformer: FoldLocalTransformer
    wearable_transformer: FoldLocalTransformer
    program_transformer: FoldLocalTransformer
    mmse_model: InductiveRankEnsemble
    wearable_model: InductiveRankEnsemble
    program_model: InductiveRankEnsemble
    seed: int

    @classmethod
    def fit(
        cls,
        mmse: pd.DataFrame,
        wearable: pd.DataFrame,
        y: Sequence[int],
        program: Mapping[str, Any],
        *,
        seed: int,
    ) -> "FittedBaseBlocks":
        mmse_columns = _column_names(mmse, where="mmse")
        wearable_columns = _column_names(wearable, where="wearable")
        mmse_raw = _as_float_matrix(mmse, where="mmse")
        wearable_raw = _as_float_matrix(wearable, where="wearable")
        target = np.asarray(y, dtype=np.int64).reshape(-1)
        if mmse_raw.shape[0] != target.size or wearable_raw.shape[0] != target.size:
            raise ValueError("mmse, wearable and y must contain the same subjects")

        mmse_transformer = FoldLocalTransformer().fit(mmse_raw, mmse_columns)
        mmse_train = mmse_transformer.transform(mmse_raw, active_only=True)

        wearable_transformer = FoldLocalTransformer().fit(wearable_raw, wearable_columns)
        wearable_train = wearable_transformer.transform(wearable_raw, active_only=True)
        wearable_for_program = wearable_transformer.transform(
            wearable_raw, active_only=False
        )
        program_raw, program_columns = _apply_program_matrix(
            wearable_for_program, wearable_columns, program
        )
        program_transformer = FoldLocalTransformer().fit(program_raw, program_columns)
        program_train = program_transformer.transform(program_raw, active_only=True)

        mmse_model = InductiveRankEnsemble().fit(mmse_train, target, seed=int(seed) + 11)
        wearable_model = InductiveRankEnsemble().fit(
            wearable_train, target, seed=int(seed) + 23
        )
        program_model = InductiveRankEnsemble().fit(
            program_train, target, seed=int(seed) + 37
        )
        return cls(
            program=program,
            mmse_columns=mmse_columns,
            wearable_columns=wearable_columns,
            program_columns=program_columns,
            mmse_transformer=mmse_transformer,
            wearable_transformer=wearable_transformer,
            program_transformer=program_transformer,
            mmse_model=mmse_model,
            wearable_model=wearable_model,
            program_model=program_model,
            seed=int(seed),
        )

    def predict_base(
        self,
        mmse: pd.DataFrame,
        wearable: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        mmse_raw = _ordered_matrix(mmse, self.mmse_columns, where="prediction mmse")
        wearable_raw = _ordered_matrix(
            wearable, self.wearable_columns, where="prediction wearable"
        )
        if mmse_raw.shape[0] != wearable_raw.shape[0]:
            raise ValueError("prediction mmse and wearable row counts differ")

        mmse_matrix = self.mmse_transformer.transform(mmse_raw, active_only=True)
        wearable_matrix = self.wearable_transformer.transform(
            wearable_raw, active_only=True
        )
        wearable_for_program = self.wearable_transformer.transform(
            wearable_raw, active_only=False
        )
        program_raw, program_columns = _apply_program_matrix(
            wearable_for_program, self.wearable_columns, self.program
        )
        if program_columns != self.program_columns:
            raise ValueError("Gemma program output schema changed after fitting")
        program_matrix = self.program_transformer.transform(
            program_raw, active_only=True
        )
        return {
            "mmse": self.mmse_model.predict_rank(mmse_matrix),
            "wearable": self.wearable_model.predict_rank(wearable_matrix),
            "program": self.program_model.predict_rank(program_matrix),
        }

    def audit_summary(self) -> dict[str, Any]:
        return {
            "preprocessing": "train-fold median -> 1/99% winsorise -> standardise",
            "rank_mapping": "each learner raw score -> frozen training empirical CDF",
            "test_batch_ranking": False,
            "models": list(self.mmse_model.component_names),
            "input_features": {
                "mmse_total": len(self.mmse_columns),
                "mmse_active": self.mmse_transformer.n_active_features,
                "wearable_total": len(self.wearable_columns),
                "wearable_active": self.wearable_transformer.n_active_features,
                "program_total": len(self.program_columns),
                "program_active": self.program_transformer.n_active_features,
            },
        }


def _normalise_full_weight(
    selected_weight: Mapping[str, Any] | Sequence[float],
) -> tuple[float, float]:
    """Return ``(wearable_weight, program_weight)`` from report or tuple form."""

    if isinstance(selected_weight, Mapping):
        wearable = selected_weight.get(
            "wearable_weight", selected_weight.get("wearable", selected_weight.get("a"))
        )
        program = selected_weight.get(
            "program_weight", selected_weight.get("program", selected_weight.get("b"))
        )
        if wearable is None or program is None:
            raise ValueError("selected_weight mapping needs wearable and program weights")
    else:
        values = list(selected_weight)
        if len(values) != 2:
            raise ValueError("selected_weight sequence must be (wearable, program)")
        wearable, program = values
    a, b = float(wearable), float(program)
    if a < 0.0 or b < 0.0 or a + b > 1.0 + 1e-12:
        raise ValueError("full-fusion weights must be non-negative and sum to at most one")
    return a, b


@dataclass
class TrainingBundle:
    """Joblib-serialisable final refit for future, genuinely unseen subjects."""

    blocks: FittedBaseBlocks
    wearable_weight: float
    program_weight: float
    cohort_fingerprint: str

    @property
    def mmse_weight(self) -> float:
        return float(1.0 - self.wearable_weight - self.program_weight)

    def predict(
        self,
        mmse: pd.DataFrame,
        wearable: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        base = self.blocks.predict_base(mmse, wearable)
        full = (
            self.mmse_weight * base["mmse"]
            + self.wearable_weight * base["wearable"]
            + self.program_weight * base["program"]
        )
        return {**base, "full": np.asarray(full, dtype=np.float64)}

    def predict_score(
        self,
        mmse: pd.DataFrame,
        wearable: pd.DataFrame,
    ) -> np.ndarray:
        return self.predict(mmse, wearable)["full"]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "cohort_fingerprint": self.cohort_fingerprint,
            "weights": {
                "mmse": self.mmse_weight,
                "wearable": float(self.wearable_weight),
                "program": float(self.program_weight),
            },
            "threshold": 0.5,
            "audit": self.blocks.audit_summary(),
        }


def fit_final_bundle(
    data: Any,
    program: Mapping[str, Any],
    selected_weight: Mapping[str, Any] | Sequence[float],
    *,
    seed: int = 20260729,
) -> TrainingBundle:
    """Refit the fixed architecture on all available rows after evaluation.

    This function performs no evaluation and must not be used to overwrite OOF
    predictions.  ``selected_weight`` is expected to be the modal outer-fold
    selection reported by :func:`evaluation.evaluate_nested`.
    """

    wearable_weight, program_weight = _normalise_full_weight(selected_weight)
    blocks = FittedBaseBlocks.fit(
        data.mmse,
        data.wearable,
        data.y,
        program,
        seed=int(seed),
    )
    return TrainingBundle(
        blocks=blocks,
        wearable_weight=wearable_weight,
        program_weight=program_weight,
        cohort_fingerprint=str(getattr(data, "fingerprint", "")),
    )
