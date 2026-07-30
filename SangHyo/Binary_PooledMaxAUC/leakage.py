"""Direct-leakage guards.  These are the only hard constraints in this experiment.

Everything here fails closed: a violation raises instead of warning, because the
whole point of allowing aggressive non-nested selection elsewhere is that the
*direct* leakage boundary stays trustworthy.

Reference for the forbidden-column list: ``SangHyo/AGENTS.md`` section 1
(diagnosis ``DIAG_NM``/``DIAG_SEQ``, administrative ``DOCTOR_NM``/``MMSE_NUM``/
``MMSE_KIND``, identifiers ``SAMPLE_EMAIL``/``EMAIL``).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "LeakageError",
    "FORBIDDEN_EXACT",
    "FORBIDDEN_PATTERNS",
    "assert_no_forbidden_features",
    "assert_fold_disjoint",
    "assert_screening_is_train_local",
    "assert_finite_scores",
    "hash_subject_id",
    "LeakageAudit",
]


class LeakageError(RuntimeError):
    """Raised when a *direct* leakage contract is violated."""


#: Exact column names that may never become a model feature.
FORBIDDEN_EXACT = frozenset(
    {
        "DIAG_NM",
        "DIAG_SEQ",
        "DOCTOR_NM",
        "MMSE_NUM",
        "MMSE_KIND",
        "SAMPLE_EMAIL",
        "EMAIL",
        "label",
        "target",
        "y",
        "original_label",
        "split_origin",
        "subject_id",
        "fold",
    }
)

#: Long, unambiguous spellings.  Short tokens are handled by exact match above so
#: that legitimate names (``slp_deep_ratio``, ``act_cal_active``) never trip.
FORBIDDEN_PATTERNS = (
    re.compile(r"diag", re.I),
    re.compile(r"dementia|alzheim", re.I),
    re.compile(r"ground[_ ]?truth", re.I),
    re.compile(r"\by[_ ]?true\b", re.I),
    # Acquisition-protocol proxies: how much data a subject produced is a
    # collection artifact, not a biological signal, and correlates with cohort
    # membership.  AGENTS.md forbids these as features.
    re.compile(r"(^|_)(n_days|n_obs|n_rows|observation_count|coverage|missing_rate|nonwear|non_wear)($|_)", re.I),
)


def assert_no_forbidden_features(names: Iterable[str], *, context: str) -> None:
    """Reject diagnosis, identifier, target and acquisition-proxy columns."""

    bad: list[str] = []
    for raw in names:
        name = str(raw)
        if name in FORBIDDEN_EXACT:
            bad.append(f"{name} (exact)")
            continue
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(name):
                bad.append(f"{name} (~{pattern.pattern})")
                break
    if bad:
        raise LeakageError(f"{context}: forbidden feature name(s): {sorted(set(bad))[:20]}")


def assert_fold_disjoint(
    train_ids: Sequence[str], test_ids: Sequence[str], *, context: str
) -> None:
    """A subject may never be on both sides of a fold."""

    overlap = sorted(set(map(str, train_ids)) & set(map(str, test_ids)))
    if overlap:
        raise LeakageError(
            f"{context}: {len(overlap)} subject(s) appear in both train and test "
            f"(e.g. {overlap[:3]})"
        )


def assert_screening_is_train_local(
    n_screening_rows: int, n_train_rows: int, *, context: str
) -> None:
    """Feature screening must see exactly the training fold, never more.

    This is the specific mistake that inflates a score the most (a global SHAP
    or univariate ranking computed on all labels), so it gets its own assertion
    at the call site rather than relying on convention.
    """

    if int(n_screening_rows) != int(n_train_rows):
        raise LeakageError(
            f"{context}: feature screening saw {n_screening_rows} rows but the "
            f"training fold has {n_train_rows}; screening must be fold-local"
        )


def assert_finite_scores(scores: np.ndarray, *, context: str) -> None:
    values = np.asarray(scores, dtype=float)
    if values.size and not np.isfinite(values).all():
        raise LeakageError(
            f"{context}: {int((~np.isfinite(values)).sum())} non-finite score(s); "
            "an unscored subject would silently corrupt the pooled AUC"
        )


def hash_subject_id(subject_id: str, *, salt: str = "") -> str:
    """Pseudonymous id so raw e-mail identifiers never reach saved artifacts."""

    return hashlib.sha256(f"{salt}|{str(subject_id).strip()}".encode("utf-8")).hexdigest()[:16]


class LeakageAudit:
    """Accumulates every check performed so the run can prove what it enforced."""

    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.disclosures: list[str] = []

    def record(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"check": name, "passed": bool(passed), "detail": detail})

    def disclose(self, statement: str) -> None:
        """Record an *allowed* optimism source that must appear in the report."""

        self.disclosures.append(statement)

    def to_dict(self) -> dict[str, Any]:
        return {
            "direct_leakage_checks": self.checks,
            "all_direct_checks_passed": all(check["passed"] for check in self.checks),
            "n_checks": len(self.checks),
            "allowed_optimism_disclosures": self.disclosures,
            "interpretation": (
                "Direct-leakage checks are hard constraints. The disclosures list "
                "sources of selection optimism that were deliberately permitted; "
                "the headline score is a development score, not a clean "
                "generalization estimate."
            ),
        }
