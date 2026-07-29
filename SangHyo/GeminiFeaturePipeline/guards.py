"""Label / diagnosis / MMSE leakage guards.

Everything that is about to leave this process towards the Gemini API passes
through :func:`assert_payload_is_label_free`.  The checks are deliberately
split into three mechanisms so that short tokens such as ``cn``, ``dem`` or
``y`` do not produce false positives on legitimate names like ``efficiency``
(contains ``cn``? no) or ``day_index``:

1. **token match** - the name is split on non-alphanumeric boundaries and on
   camelCase, and each token is compared *exactly* against a short-token set.
2. **substring match** - only for long, unambiguous words (``diagnosis``,
   ``dementia``, ...).
3. **regex match** - for compound spellings such as ``cognitive_status``.

The guard walks nested dicts/lists and inspects **keys and string values**, so
an injected ``{"note": "subject is MCI"}`` is rejected as well.

Related repository contract: ``SangHyo/AGENTS.md`` section 1-1 (modality
contract) and section 2 (leakage contract).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Sequence

__all__ = [
    "LeakageError",
    "FORBIDDEN_TOKENS",
    "FORBIDDEN_SUBSTRINGS",
    "FORBIDDEN_PATTERNS",
    "CLASS_NAME_TOKENS",
    "CLASS_NAME_PATTERNS",
    "MMSE_TOKENS",
    "MMSE_PATTERNS",
    "assert_payload_is_label_free",
    "assert_payload_is_mmse_free",
    "assert_names_are_label_free",
    "assert_names_are_mmse_free",
    "assert_text_is_label_free",
    "assert_prompt_has_no_class_names",
    "assert_disjoint_subjects",
    "hash_subject_id",
    "find_label_like_names",
]


class LeakageError(RuntimeError):
    """Raised whenever forbidden information reaches a forbidden place."""


# Short/ambiguous names: compared token-exact, never as substrings.
FORBIDDEN_TOKENS = frozenset(
    {
        "cn",
        "mci",
        "dem",
        "ad",
        "y",
        "ytrue",
        "label",
        "labels",
        "target",
        "targets",
        "class",
        "classes",
        "cls",
        "diag",
        "dx",
        "outcome",
        "outcomes",
        "group",
        "groups",
        "normal",
        "patient",
        "disease",
        "risk",
        "severity",
        "stage",
        "impairment",
        "impaired",
    }
)

# Long, unambiguous words: substring match is safe here.
FORBIDDEN_SUBSTRINGS = (
    "diagnos",  # diagnosis, diagnostic, diagnosed
    "dementia",
    "alzheim",
    "cognitive_status",
    "cognitivestatus",
    "ground_truth",
    "groundtruth",
    "y_true",
    "ytrue",
    "class_probability",
    "classprobability",
    "risk_group",
    "riskgroup",
    "patient_group",
    "patientgroup",
    "mild_cognitive",
    "mildcognitive",
    "prognos",
    "clinical_label",
)

# Python's ``\b`` treats underscore as a word character, so ``\bmmse\b`` fails to
# match "mini_mental_score" or "cognitivefunction_block" - exactly the snake_case
# style used throughout this codebase.  These two custom boundaries require a
# non-alphanumeric (or start/end-of-string) neighbour instead, so underscores and
# hyphens count as separators the way they do everywhere else in this module.
_WORD_START = r"(?<![0-9A-Za-z])"
_WORD_END = r"(?![0-9A-Za-z])"

FORBIDDEN_PATTERNS = (
    re.compile(
        _WORD_START + r"(cn|mci|dem)[ _-]?(vs|versus|like|score|prob)" + _WORD_END, re.I
    ),
    re.compile(_WORD_START + r"(cn|mci|dem)" + _WORD_END, re.I),
    re.compile(_WORD_START + r"diag[_ ]?nm" + _WORD_END, re.I),
    re.compile(_WORD_START + r"cognitive\s+(status|impairment|decline)" + _WORD_END, re.I),
    re.compile(_WORD_START + r"class\s*(index|probability|label)" + _WORD_END, re.I),
)

# Prompts are *allowed* to say "do not diagnose" / "do not report risk", because
# <gemini_prompt_requirements> demands exactly those negative instructions.  What a
# prompt may never contain is a class name or a clinical-test name, so static
# prompt text is checked against this narrower set instead of FORBIDDEN_TOKENS.
CLASS_NAME_TOKENS = frozenset({"cn", "mci", "dem", "ad"})
CLASS_NAME_PATTERNS = (
    re.compile(_WORD_START + r"(cn|mci|dem|ad)" + _WORD_END, re.I),
    re.compile(r"dementia|alzheim|mild cognitive impairment", re.I),
    re.compile(
        _WORD_START
        + r"mmse"
        + _WORD_END
        + r"|"
        + _WORD_START
        + r"mini[ _-]?mental"
        + _WORD_END
        + r"|"
        + _WORD_START
        + r"moca"
        + _WORD_END
        + r"|"
        + _WORD_START
        + r"cdr"
        + _WORD_END,
        re.I,
    ),
    re.compile(r"cognitively\s+normal|normal\s+cognition", re.I),
)

MMSE_TOKENS = frozenset({"mmse", "kmmse", "mmsekc", "moca", "cdr", "gds"})
MMSE_PATTERNS = (
    re.compile(_WORD_START + r"mmse" + _WORD_END, re.I),
    re.compile(_WORD_START + r"mini[ _-]?mental" + _WORD_END, re.I),
    re.compile(r"^q\d{1,2}(_\d)?$", re.I),  # raw MMSE item column names (Q01, Q13_2)
    re.compile(_WORD_START + r"cognitivefunction" + _WORD_END, re.I),
    re.compile(_WORD_START + r"cognitive_?function" + _WORD_END, re.I),
)

_TOKEN_SPLIT = re.compile(r"[^0-9A-Za-z]+")
_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _tokens(text: str) -> list[str]:
    parts: list[str] = []
    for chunk in _TOKEN_SPLIT.split(str(text)):
        if not chunk:
            continue
        parts.extend(piece.lower() for piece in _CAMEL_SPLIT.split(chunk) if piece)
    return parts


def _violations_in_text(text: str, *, check_mmse: bool) -> list[str]:
    value = str(text)
    lowered = value.lower()
    hits: list[str] = []
    tokens = set(_tokens(value))
    for token in sorted(tokens & FORBIDDEN_TOKENS):
        hits.append(f"token:{token}")
    for needle in FORBIDDEN_SUBSTRINGS:
        if needle in lowered:
            hits.append(f"substring:{needle}")
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.search(value):
            hits.append(f"regex:{pattern.pattern}")
    if check_mmse:
        for token in sorted(tokens & MMSE_TOKENS):
            hits.append(f"mmse-token:{token}")
        for pattern in MMSE_PATTERNS:
            if pattern.search(value):
                hits.append(f"mmse-regex:{pattern.pattern}")
    return hits


def _mmse_violations_in_text(text: str) -> list[str]:
    value = str(text)
    hits: list[str] = []
    tokens = set(_tokens(value))
    for token in sorted(tokens & MMSE_TOKENS):
        hits.append(f"mmse-token:{token}")
    for pattern in MMSE_PATTERNS:
        if pattern.search(value):
            hits.append(f"mmse-regex:{pattern.pattern}")
    return hits


def _walk(node: Any, path: str, sink: list[str], *, check_mmse: bool, check_label: bool) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            key_path = f"{path}.{key}" if path else str(key)
            if check_label:
                for hit in _violations_in_text(key, check_mmse=False):
                    sink.append(f"{key_path} [key] -> {hit}")
            if check_mmse:
                for hit in _mmse_violations_in_text(key):
                    sink.append(f"{key_path} [key] -> {hit}")
            _walk(value, key_path, sink, check_mmse=check_mmse, check_label=check_label)
        return
    if isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _walk(
                value,
                f"{path}[{index}]",
                sink,
                check_mmse=check_mmse,
                check_label=check_label,
            )
        return
    if isinstance(node, str):
        if check_label:
            for hit in _violations_in_text(node, check_mmse=False):
                sink.append(f"{path} [value] -> {hit}")
        if check_mmse:
            for hit in _mmse_violations_in_text(node):
                sink.append(f"{path} [value] -> {hit}")


def assert_payload_is_label_free(payload: Any, *, context: str) -> None:
    """Fail before an API call if any target/diagnosis information is present."""

    findings: list[str] = []
    _walk(payload, "", findings, check_mmse=False, check_label=True)
    if findings:
        raise LeakageError(
            f"{context}: forbidden target/diagnosis information in payload: "
            + "; ".join(sorted(set(findings))[:20])
        )


def assert_payload_is_mmse_free(payload: Any, *, context: str) -> None:
    """Fail if MMSE/cognitive-test information is present (always true for Gemini)."""

    findings: list[str] = []
    _walk(payload, "", findings, check_mmse=True, check_label=False)
    if findings:
        raise LeakageError(
            f"{context}: forbidden MMSE information in payload: "
            + "; ".join(sorted(set(findings))[:20])
        )


def find_label_like_names(names: Iterable[str]) -> list[str]:
    bad: list[str] = []
    for name in names:
        if _violations_in_text(name, check_mmse=False):
            bad.append(str(name))
    return bad


def assert_names_are_label_free(names: Iterable[str], *, context: str) -> None:
    bad = find_label_like_names(names)
    if bad:
        raise LeakageError(f"{context}: label/diagnosis-like names: {sorted(set(bad))[:20]}")


def assert_names_are_mmse_free(names: Iterable[str], *, context: str) -> None:
    bad = [str(name) for name in names if _mmse_violations_in_text(name)]
    if bad:
        raise LeakageError(f"{context}: MMSE-derived names are forbidden here: {sorted(set(bad))[:20]}")


def assert_text_is_label_free(text: str, *, context: str) -> None:
    """Prompt/template guard: no diagnosis class name may appear in a prompt."""

    hits = _violations_in_text(text, check_mmse=False)
    if hits:
        raise LeakageError(f"{context}: prompt text contains forbidden terms: {sorted(set(hits))[:20]}")


def assert_prompt_has_no_class_names(text: str, *, context: str) -> None:
    """Static prompt guard: no diagnosis class name and no clinical-test name."""

    value = str(text)
    hits: list[str] = []
    tokens = set(_tokens(value))
    for token in sorted(tokens & CLASS_NAME_TOKENS):
        hits.append(f"class-token:{token}")
    for pattern in CLASS_NAME_PATTERNS:
        match = pattern.search(value)
        if match:
            hits.append(f"class-regex:{match.group(0)!r}")
    if hits:
        raise LeakageError(f"{context}: prompt contains class/test names: {sorted(set(hits))[:20]}")


def assert_disjoint_subjects(
    train_ids: Sequence[str], validation_ids: Sequence[str], *, context: str
) -> None:
    overlap = sorted(set(map(str, train_ids)) & set(map(str, validation_ids)))
    if overlap:
        raise LeakageError(f"{context}: {len(overlap)} subject(s) appear in both folds")


def hash_subject_id(subject_id: str, *, salt: str = "") -> str:
    """Stable pseudonymous ID.  Raw e-mail identifiers never reach outputs."""

    digest = hashlib.sha256(f"{salt}|{str(subject_id).strip()}".encode("utf-8"))
    return digest.hexdigest()[:16]
