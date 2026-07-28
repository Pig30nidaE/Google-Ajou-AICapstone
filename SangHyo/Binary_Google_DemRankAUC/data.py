"""Cohort assembly: 174 subjects, Dem-positive labels, and the contract checks.

The label is read from the Gait *and* Sleep copies and the two are asserted
identical (AGENTS.md contract).  ``DIAG_NM`` never touches the feature matrix --
``features.py`` drops it fail-closed at read time, and ``assert_contract`` checks
again after assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import (
    COHORT_CONTRACT,
    DIAG_ORDER,
    LABEL_FILES,
    QUALITY_RULES,
    SEVERITY,
    SPLIT_CONTRACT,
    SPLIT_DIRS,
    SUSPECT_FEATURE_PREFIXES,
)
from .features import (
    assert_no_forbidden,
    build_cohort_features,
    drop_degenerate,
    feature_fingerprint,
    read_csv,
)


@dataclass(frozen=True)
class Cohort:
    """One row per subject.  ``y == 1`` is Dem (the positive class)."""

    subject_ids: np.ndarray
    diagnosis: np.ndarray
    severity: np.ndarray
    y: np.ndarray
    X: np.ndarray
    feature_names: tuple[str, ...]
    split_of: np.ndarray
    fingerprint: str

    @property
    def n_subjects(self) -> int:
        return len(self.subject_ids)

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_positive(self) -> int:
        return int(self.y.sum())

    def select(self, names: Sequence[str]) -> "Cohort":
        index = {name: position for position, name in enumerate(self.feature_names)}
        missing = [name for name in names if name not in index]
        if missing:
            raise KeyError(f"Requested features were not built: {missing[:8]}")
        columns = [index[name] for name in names]
        return Cohort(self.subject_ids, self.diagnosis, self.severity, self.y,
                      self.X[:, columns], tuple(names), self.split_of, self.fingerprint)

    def subset(self, mask: np.ndarray) -> "Cohort":
        mask = np.asarray(mask, dtype=bool)
        return Cohort(self.subject_ids[mask], self.diagnosis[mask], self.severity[mask],
                      self.y[mask], self.X[mask], self.feature_names, self.split_of[mask],
                      self.fingerprint)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.X, index=pd.Index(self.subject_ids, name="subject_id"),
                            columns=list(self.feature_names))


def _label_series(data_root: Path, split: str, copy: str) -> pd.Series:
    path = Path(data_root) / SPLIT_DIRS[split] / LABEL_FILES[split][copy]
    frame = read_csv(path)
    id_column = next((c for c in ("SAMPLE_EMAIL", "EMAIL") if c in frame.columns), None)
    if id_column is None:
        raise KeyError(f"Label file lacks SAMPLE_EMAIL/EMAIL: {path}")
    frame = frame.copy()
    frame["_sid"] = frame[id_column].astype(str).str.strip()
    series = frame.drop_duplicates("_sid").set_index("_sid")["DIAG_NM"].astype(str).str.strip()
    unknown = sorted(set(series.unique()) - set(DIAG_ORDER))
    if unknown:
        raise AssertionError(f"Unexpected DIAG_NM values in {path.name}: {unknown}")
    return series.sort_index()


def load_labels(data_root: str | Path) -> pd.Series:
    """Diagnosis per subject, cross-checked between the Gait and Sleep copies."""

    data_root = Path(data_root)
    parts = []
    for split in ("train", "val"):
        gait = _label_series(data_root, split, "gait")
        sleep = _label_series(data_root, split, "sleep")
        if not gait.equals(sleep):
            disagreeing = gait.index[gait.to_numpy() != sleep.reindex(gait.index).to_numpy()]
            raise AssertionError(
                f"{split}: Gait and Sleep label copies disagree for {len(disagreeing)} subjects"
            )
        counts = gait.value_counts().to_dict()
        expected = SPLIT_CONTRACT[split]
        observed = {"n": int(len(gait)), **{d: int(counts.get(d, 0)) for d in DIAG_ORDER}}
        if observed != expected:
            raise AssertionError(f"{split} label contract changed: {observed} != {expected}")
        parts.append(gait)
    return pd.concat(parts)


def load_cohort(data_root: str | Path, *, mmse_zero_as_missing: bool = True,
                drop_suspect: bool = False) -> Cohort:
    """Build the pooled 174-subject cohort.

    Train and Validation are pooled because there are only 12 Dem subjects in
    the entire dataset; holding any of them out would make both the fit and the
    estimate meaningless.  The consequence -- **there is no held-out test set,
    every number this folder reports is cross-validated** -- is stated in every
    report rather than left to be discovered.
    """

    frame, origin = build_cohort_features(data_root, mmse_zero_as_missing=mmse_zero_as_missing)
    frame = drop_degenerate(frame)
    if drop_suspect:
        frame = frame.loc[:, [c for c in frame.columns
                              if not str(c).startswith(SUSPECT_FEATURE_PREFIXES)]]
    assert_no_forbidden(frame.columns)

    labels = load_labels(data_root)
    subjects = [str(s) for s in frame.index]
    unlabelled = [s for s in subjects if s not in labels.index]
    if unlabelled:
        raise AssertionError(f"{len(unlabelled)} subjects have no diagnosis label")

    diagnosis = labels.reindex(subjects).to_numpy(dtype=str)
    severity = np.array([SEVERITY[d] for d in diagnosis], dtype=np.int64)
    cohort = Cohort(
        subject_ids=np.asarray(subjects, dtype=str),
        diagnosis=diagnosis,
        severity=severity,
        y=(severity == SEVERITY["Dem"]).astype(np.int64),
        X=frame.to_numpy(dtype=np.float64),
        feature_names=tuple(map(str, frame.columns)),
        split_of=np.asarray(origin, dtype=str),
        fingerprint=feature_fingerprint(frame),
    )
    assert_contract(cohort)
    return cohort


def assert_contract(cohort: Cohort) -> None:
    """Hard checks that must hold before any model sees the data."""

    counts = {d: int((cohort.diagnosis == d).sum()) for d in DIAG_ORDER}
    observed = {"n_subjects": cohort.n_subjects, **counts}
    if observed != COHORT_CONTRACT:
        raise AssertionError(f"Cohort contract changed: {observed} != {COHORT_CONTRACT}")
    if len(set(cohort.subject_ids.tolist())) != cohort.n_subjects:
        raise AssertionError("Duplicate subject id in the pooled cohort")
    if cohort.n_positive != COHORT_CONTRACT["Dem"]:
        raise AssertionError(f"Positive count changed: {cohort.n_positive}")
    if cohort.X.shape[0] != cohort.n_subjects:
        raise AssertionError("Feature matrix rows do not match subject count")
    assert_no_forbidden(cohort.feature_names)
    # Nothing in the matrix may reproduce the label exactly: a feature that is a
    # perfect separator at 12 positives is far more likely to be leakage than
    # signal, so it is surfaced rather than silently used.
    for position, name in enumerate(cohort.feature_names):
        column = cohort.X[:, position]
        finite = np.isfinite(column)
        if finite.sum() < cohort.n_subjects:
            continue
        if len(np.unique(column[cohort.y == 1])) == 1 and len(np.unique(column[cohort.y == 0])) == 1:
            if column[cohort.y == 1][0] != column[cohort.y == 0][0]:
                raise AssertionError(f"Feature {name!r} separates the classes perfectly")


def quality_flags(cohort: Cohort) -> dict[str, list[str]]:
    """Label-blind plausibility rules.  Never consults ``cohort.y``."""

    index = {name: position for position, name in enumerate(cohort.feature_names)}
    flagged: dict[str, list[str]] = {}
    for rule, spec in QUALITY_RULES.items():
        feature = spec["feature"]
        if feature not in index:
            continue
        column = cohort.X[:, index[feature]]
        if spec["op"] == "gt":
            hit = np.isfinite(column) & (column > spec["value"])
        elif spec["op"] == "lt":
            hit = np.isfinite(column) & (column < spec["value"])
        else:  # pragma: no cover - config typo guard
            raise ValueError(f"Unknown quality-rule operator {spec['op']!r}")
        for position in np.flatnonzero(hit):
            flagged.setdefault(str(cohort.subject_ids[position]), []).append(
                f"{rule}: {spec['reason']}"
            )
    return flagged


def hash_subject_id(subject_id: str) -> str:
    """Reports store hashed ids only; raw e-mail addresses never leave the run."""

    return hashlib.sha256(str(subject_id).encode("utf-8")).hexdigest()[:16]


def resolve_block(cohort: Cohort, block: str) -> tuple[str, ...]:
    """Feature names belonging to a named block, in cohort column order."""

    from .config import FEATURE_BLOCK_EXPLICIT, FEATURE_BLOCK_PREFIXES

    if block in FEATURE_BLOCK_EXPLICIT:
        wanted = set(FEATURE_BLOCK_EXPLICIT[block])
        names = tuple(n for n in cohort.feature_names if n in wanted)
        absent = wanted - set(names)
        if absent:
            # A pre-specified name that ``drop_degenerate`` removed (constant in
            # this cohort) is dropped from the block, and the run records it.
            names = tuple(n for n in cohort.feature_names if n in wanted)
        return names
    if block in FEATURE_BLOCK_PREFIXES:
        prefixes = FEATURE_BLOCK_PREFIXES[block]
        return tuple(n for n in cohort.feature_names if str(n).startswith(prefixes))
    raise KeyError(f"Unknown feature block: {block!r}")


__all__ = [
    "Cohort", "assert_contract", "hash_subject_id", "load_cohort", "load_labels",
    "quality_flags", "resolve_block",
]
