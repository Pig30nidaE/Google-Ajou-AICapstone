"""검증 가능한 전역 wearable feature-program DSL.

Gemma는 실행 코드나 환자별 점수를 만들지 않는다. 대신 이 모듈이 허용하는 세
연산만 조합한 전역 프로그램을 JSON으로 제안한다. 실제 계산은 fold의 training
부분에서 이미 결측치 대치, winsorization, 표준화가 끝난 행렬에 대해 이 모듈이
결정론적으로 수행한다.

보안·누수 계약:

* 프로그램은 8--16개의 특징만 정의한다.
* 각 특징은 catalog의 wearable primitive 2--5개만 참조한다.
* 임의 코드, 임계값, 상수, 가중치는 DSL에 존재하지 않는다.
* 인지검사, 정답, 진단, 사람 식별자를 암시하는 primitive는 fail-closed로
  거부한다.
* ``apply_program``은 입력 z 값을 [-5, 5]로 자른 뒤 고정된 tanh 연산만
  수행한다. 프로그램의 자연어 rationale은 실행에 전혀 사용하지 않는다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import PROGRAM_VERSION

__all__ = [
    "PROGRAM_VERSION",
    "PrimitiveSpec",
    "apply_program",
    "canonical_json",
    "canonicalize_program",
    "catalog_hash",
    "catalog_payload",
    "program_feature_names",
    "program_hash",
    "response_schema",
    "schema_hash",
    "validate_program",
]

MIN_PROGRAM_FEATURES = 8
MAX_PROGRAM_FEATURES = 16
MIN_DEPENDENCIES = 2
MAX_DEPENDENCIES = 5
MISSING_POLICY = "fold_median_then_clip"
OPERATIONS: tuple[str, ...] = (
    "signed_mean",
    "signed_product",
    "absolute_gap",
)

_PRIMITIVE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{1,95}$")
_PROGRAM_NAME_RE = re.compile(r"^llmfp__[a-z][a-z0-9_]{2,55}$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Exact tokens are used instead of a raw ``"id" in name`` test: legitimate
# wearable names such as ``midsleep`` contain those letters.
_FORBIDDEN_TOKENS = frozenset(
    {
        "mmse",
        "label",
        "labels",
        "target",
        "targets",
        "outcome",
        "outcomes",
        "diagnosis",
        "diagnoses",
        "diagnostic",
        "diag",
        "class",
        "classes",
        "email",
        "subject",
        "subjects",
        "patient",
        "patients",
        "identifier",
        "identifiers",
        "userid",
        "sampleid",
        "id",
        "cn",
        "mci",
        "dem",
        "dementia",
        "cognitive",
        "cognition",
        "decline",
        "impairment",
        "impaired",
        "risk",
        "probability",
        "prediction",
        "predict",
        "clinical",
        "medical",
        "screening",
    }
)
_FORBIDDEN_CATALOG_TEXT_TOKENS = _FORBIDDEN_TOKENS - {
    # These words may legitimately occur in a measurement definition such as
    # "dispersion across the subject's hourly bins". Actual identifier
    # primitive *names* remain forbidden below.
    "subject",
    "subjects",
    "patient",
    "patients",
    "id",
}
_FORBIDDEN_RATIONALE_TOKENS = _FORBIDDEN_TOKENS - {
    # Generic prose such as "the subject's mean profile" is not an identifier.
    # Concrete identifier fields and tokens remain forbidden.
    "subject",
    "subjects",
    "patient",
    "patients",
}
_FORBIDDEN_PROGRAM_TEXT_TOKENS = frozenset(
    {
        "python",
        "lambda",
        "eval",
        "exec",
        "import",
        "script",
        "threshold",
        "thresholds",
        "cutoff",
        "cutoffs",
        "constant",
        "constants",
        "weight",
        "weights",
        "weighted",
        "coefficient",
        "coefficients",
    }
)


@dataclass(frozen=True)
class PrimitiveSpec:
    """하나의 label-neutral wearable primitive에 대한 prompt metadata.

    ``domain``은 예를 들어 ``activity``, ``sleep``, ``circadian``,
    ``quality``처럼 동작 영역만 나타내야 한다. 네 필드 어디에도 임상 방향,
    정답, 사람 식별 정보가 들어가면 안 된다.
    """

    name: str
    domain: str
    description: str
    unit: str


def _tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(value).lower()))


def _assert_safe_metadata(
    value: str,
    *,
    context: str,
    dependency_name: bool = False,
) -> str:
    cleaned = " ".join(str(value).strip().split())
    if not cleaned:
        raise ValueError(f"{context} must not be empty")
    forbidden_set = (
        _FORBIDDEN_TOKENS if dependency_name else _FORBIDDEN_CATALOG_TEXT_TOKENS
    )
    forbidden = sorted(_tokens(cleaned) & forbidden_set)
    if forbidden:
        raise ValueError(f"{context} contains forbidden metadata tokens: {forbidden}")
    return cleaned


def _normalize_catalog(catalog: Sequence[PrimitiveSpec]) -> tuple[PrimitiveSpec, ...]:
    if isinstance(catalog, (str, bytes)) or not isinstance(catalog, Sequence):
        raise TypeError("primitive catalog must be a sequence of PrimitiveSpec")
    normalized: list[PrimitiveSpec] = []
    seen_casefold: set[str] = set()
    for index, raw in enumerate(catalog):
        if not isinstance(raw, PrimitiveSpec):
            raise TypeError(
                f"catalog[{index}] must be PrimitiveSpec, got {type(raw).__name__}"
            )
        name = str(raw.name).strip()
        if not _PRIMITIVE_NAME_RE.fullmatch(name):
            raise ValueError(f"invalid primitive name at catalog[{index}]: {raw.name!r}")
        _assert_safe_metadata(
            name,
            context=f"catalog[{index}].name",
            dependency_name=True,
        )
        folded_name = name.casefold()
        if folded_name in seen_casefold:
            raise ValueError(f"duplicate primitive name: {name}")
        seen_casefold.add(folded_name)
        normalized.append(
            PrimitiveSpec(
                name=name,
                domain=_assert_safe_metadata(
                    raw.domain, context=f"catalog[{index}].domain"
                ),
                description=_assert_safe_metadata(
                    raw.description, context=f"catalog[{index}].description"
                ),
                unit=_assert_safe_metadata(raw.unit, context=f"catalog[{index}].unit"),
            )
        )
    if len(normalized) < MIN_DEPENDENCIES:
        raise ValueError(
            f"primitive catalog needs at least {MIN_DEPENDENCIES} safe entries"
        )
    # Catalog order is not semantic. Sorting makes prompts and hashes stable.
    return tuple(sorted(normalized, key=lambda item: item.name.casefold()))


def catalog_payload(catalog: Sequence[PrimitiveSpec]) -> list[dict[str, str]]:
    """Return the only four primitive fields that may be sent to Gemma."""

    return [asdict(item) for item in _normalize_catalog(catalog)]


def canonical_json(value: Any) -> str:
    """Stable UTF-8 JSON representation used by every contract hash."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def catalog_hash(catalog: Sequence[PrimitiveSpec]) -> str:
    return hashlib.sha256(canonical_json(catalog_payload(catalog)).encode("utf-8")).hexdigest()


def response_schema(catalog: Sequence[PrimitiveSpec]) -> dict[str, Any]:
    """Build the strict structured-output schema for one global program."""

    names = [item["name"] for item in catalog_payload(catalog)]
    feature_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {
                "type": "string",
                "description": "Stable output name beginning with llmfp__.",
            },
            "operation": {
                "type": "string",
                "enum": list(OPERATIONS),
            },
            "dependencies": {
                "type": "array",
                "items": {"type": "string", "enum": names},
                "minItems": MIN_DEPENDENCIES,
                "maxItems": MAX_DEPENDENCIES,
            },
            "directions": {
                "type": "array",
                "items": {"type": "integer", "enum": [-1, 1]},
                "minItems": MIN_DEPENDENCIES,
                "maxItems": MAX_DEPENDENCIES,
            },
            "missing_policy": {
                "type": "string",
                "enum": [MISSING_POLICY],
            },
            "rationale": {
                "type": "string",
                "description": (
                    "Short behavioural explanation based only on catalog metadata; "
                    "never a prediction or medical conclusion."
                ),
            },
        },
        "required": [
            "name",
            "operation",
            "dependencies",
            "directions",
            "missing_policy",
            "rationale",
        ],
        "propertyOrdering": [
            "name",
            "operation",
            "dependencies",
            "directions",
            "missing_policy",
            "rationale",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "program_version": {
                "type": "string",
                "enum": [PROGRAM_VERSION],
            },
            "features": {
                "type": "array",
                "items": feature_schema,
                "minItems": MIN_PROGRAM_FEATURES,
                "maxItems": MAX_PROGRAM_FEATURES,
            },
        },
        "required": ["program_version", "features"],
        "propertyOrdering": ["program_version", "features"],
    }


def schema_hash(catalog: Sequence[PrimitiveSpec]) -> str:
    return hashlib.sha256(
        canonical_json(response_schema(catalog)).encode("utf-8")
    ).hexdigest()


def _ensure_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    actual = {str(key) for key in value}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"{context} keys mismatch; missing={missing}, extra={extra}")


def _normalize_feature(
    raw: Mapping[str, Any],
    *,
    allowed_dependencies: set[str],
    index: int,
) -> dict[str, Any]:
    context = f"program.features[{index}]"
    if not isinstance(raw, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    expected = {
        "name",
        "operation",
        "dependencies",
        "directions",
        "missing_policy",
        "rationale",
    }
    _ensure_exact_keys(raw, expected, context=context)

    name = str(raw["name"]).strip().lower()
    if not _PROGRAM_NAME_RE.fullmatch(name):
        raise ValueError(
            f"{context}.name must match llmfp__[a-z][a-z0-9_]* and be <=64 chars"
        )
    _assert_safe_metadata(
        name,
        context=f"{context}.name",
        dependency_name=True,
    )

    operation = str(raw["operation"]).strip()
    if operation not in OPERATIONS:
        raise ValueError(f"{context}.operation is not allowed: {operation!r}")

    dependencies_raw = raw["dependencies"]
    directions_raw = raw["directions"]
    if (
        isinstance(dependencies_raw, (str, bytes))
        or not isinstance(dependencies_raw, Sequence)
    ):
        raise ValueError(f"{context}.dependencies must be an array")
    if isinstance(directions_raw, (str, bytes)) or not isinstance(
        directions_raw, Sequence
    ):
        raise ValueError(f"{context}.directions must be an array")
    if not MIN_DEPENDENCIES <= len(dependencies_raw) <= MAX_DEPENDENCIES:
        raise ValueError(
            f"{context}.dependencies must contain {MIN_DEPENDENCIES}--"
            f"{MAX_DEPENDENCIES} names"
        )
    if len(directions_raw) != len(dependencies_raw):
        raise ValueError(f"{context}.directions must align one-to-one with dependencies")

    pairs: list[tuple[str, int]] = []
    seen_dependencies: set[str] = set()
    for dep_index, (raw_dependency, raw_direction) in enumerate(
        zip(dependencies_raw, directions_raw)
    ):
        dependency = str(raw_dependency).strip()
        _assert_safe_metadata(
            dependency,
            context=f"{context}.dependencies[{dep_index}]",
            dependency_name=True,
        )
        if dependency not in allowed_dependencies:
            raise ValueError(
                f"{context}.dependencies[{dep_index}] is not in the frozen catalog: "
                f"{dependency!r}"
            )
        if dependency in seen_dependencies:
            raise ValueError(f"{context} repeats dependency {dependency!r}")
        seen_dependencies.add(dependency)
        if isinstance(raw_direction, bool) or not isinstance(raw_direction, int):
            raise ValueError(f"{context}.directions[{dep_index}] must be -1 or 1")
        direction = int(raw_direction)
        if direction not in (-1, 1):
            raise ValueError(f"{context}.directions[{dep_index}] must be -1 or 1")
        pairs.append((dependency, direction))

    missing_policy = str(raw["missing_policy"]).strip()
    if missing_policy != MISSING_POLICY:
        raise ValueError(
            f"{context}.missing_policy must be exactly {MISSING_POLICY!r}"
        )
    rationale = " ".join(str(raw["rationale"]).strip().split())
    if not 20 <= len(rationale) <= 240:
        raise ValueError(f"{context}.rationale must contain 20--240 characters")
    forbidden_rationale = sorted(
        _tokens(rationale)
        & (_FORBIDDEN_RATIONALE_TOKENS | _FORBIDDEN_PROGRAM_TEXT_TOKENS)
    )
    if forbidden_rationale:
        raise ValueError(
            f"{context}.rationale contains forbidden tokens: {forbidden_rationale}"
        )

    # All three operations are permutation invariant. Normalizing pairs removes
    # meaningless response-order differences from hashes and matrix columns.
    pairs.sort(key=lambda pair: pair[0].casefold())
    return {
        "name": name,
        "operation": operation,
        "dependencies": [pair[0] for pair in pairs],
        "directions": [pair[1] for pair in pairs],
        "missing_policy": missing_policy,
        "rationale": rationale,
    }


def _canonicalize_with_names(
    payload: Mapping[str, Any], allowed_dependencies: Iterable[str]
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"feature program must be a JSON object, got {type(payload).__name__}"
        )
    _ensure_exact_keys(
        payload, {"program_version", "features"}, context="feature program"
    )
    if str(payload["program_version"]).strip() != PROGRAM_VERSION:
        raise ValueError(
            f"program_version must be exactly {PROGRAM_VERSION!r}, "
            f"got {payload['program_version']!r}"
        )
    raw_features = payload["features"]
    if isinstance(raw_features, (str, bytes)) or not isinstance(
        raw_features, Sequence
    ):
        raise ValueError("program.features must be an array")
    if not MIN_PROGRAM_FEATURES <= len(raw_features) <= MAX_PROGRAM_FEATURES:
        raise ValueError(
            f"program.features must contain {MIN_PROGRAM_FEATURES}--"
            f"{MAX_PROGRAM_FEATURES} entries"
        )

    allowed = {str(name).strip() for name in allowed_dependencies}
    if len(allowed) < MIN_DEPENDENCIES:
        raise ValueError("not enough allowed primitive dependencies")
    features = [
        _normalize_feature(raw, allowed_dependencies=allowed, index=index)
        for index, raw in enumerate(raw_features)
    ]
    names = [feature["name"] for feature in features]
    if len(set(names)) != len(names):
        raise ValueError("program feature names must be unique")

    signatures: set[tuple[Any, ...]] = set()
    for feature in features:
        signature = (
            feature["operation"],
            tuple(feature["dependencies"]),
            tuple(feature["directions"]),
        )
        if signature in signatures:
            raise ValueError(
                "program contains duplicate executable definitions under "
                "different names"
            )
        signatures.add(signature)

    features.sort(key=lambda feature: feature["name"])
    return {"program_version": PROGRAM_VERSION, "features": features}


def canonicalize_program(
    payload: Mapping[str, Any], catalog: Sequence[PrimitiveSpec]
) -> dict[str, Any]:
    """Validate and normalize an API response into the executable DSL."""

    normalized_catalog = _normalize_catalog(catalog)
    return _canonicalize_with_names(payload, (item.name for item in normalized_catalog))


def validate_program(
    payload: Mapping[str, Any], catalog: Sequence[PrimitiveSpec]
) -> dict[str, Any]:
    """Public validation alias; returns the canonical program on success."""

    return canonicalize_program(payload, catalog)


def program_hash(
    payload: Mapping[str, Any], catalog: Sequence[PrimitiveSpec]
) -> str:
    canonical = canonicalize_program(payload, catalog)
    return hashlib.sha256(canonical_json(canonical).encode("utf-8")).hexdigest()


def program_feature_names(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return canonical output names without needing catalog metadata."""

    if not isinstance(payload, Mapping):
        raise ValueError("feature program must be a mapping")
    features = payload.get("features")
    if isinstance(features, (str, bytes)) or not isinstance(features, Sequence):
        raise ValueError("program.features must be an array")
    if any(not isinstance(item, Mapping) for item in features):
        raise ValueError("every program feature must be an object")
    names = tuple(str(item.get("name", "")).strip().lower() for item in features)
    if not names or any(not _PROGRAM_NAME_RE.fullmatch(name) for name in names):
        raise ValueError("program contains an invalid feature name")
    if len(set(names)) != len(names):
        raise ValueError("program feature names must be unique")
    return tuple(sorted(names))


def apply_program(
    X: Any,
    feature_names: Sequence[str],
    program: Mapping[str, Any],
) -> np.ndarray:
    """Apply a validated global program to a fold-standardized wearable matrix.

    Parameters
    ----------
    X:
        Two-dimensional, finite matrix. The caller must fit imputation,
        winsorization and scaling on the current outer-training data only.
    feature_names:
        Column names aligned with ``X``. They are also the complete dependency
        allowlist at execution time.
    program:
        A response previously validated against the same primitive catalog.

    Returns
    -------
    numpy.ndarray
        Matrix ordered lexicographically by ``llmfp__`` feature name. Values
        are finite and bounded to [-1, 1].
    """

    matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"X must be two-dimensional, got shape={matrix.shape}")
    if matrix.shape[0] < 1:
        raise ValueError("X must contain at least one row")
    names = tuple(str(name).strip() for name in feature_names)
    if matrix.shape[1] != len(names):
        raise ValueError(
            f"X has {matrix.shape[1]} columns but feature_names has {len(names)}"
        )
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("feature_names must be unique")
    for index, name in enumerate(names):
        if not _PRIMITIVE_NAME_RE.fullmatch(name):
            raise ValueError(f"feature_names[{index}] is invalid: {name!r}")
        _assert_safe_metadata(
            name,
            context=f"feature_names[{index}]",
            dependency_name=True,
        )
    if not np.isfinite(matrix).all():
        raise ValueError(
            "X must be finite after fold-local median imputation and scaling"
        )

    canonical = _canonicalize_with_names(program, names)
    name_to_index = {name: index for index, name in enumerate(names)}
    z = np.clip(matrix, -5.0, 5.0)
    columns: list[np.ndarray] = []
    for feature in canonical["features"]:
        indices = [name_to_index[name] for name in feature["dependencies"]]
        directions = np.asarray(feature["directions"], dtype=float)
        directed = z[:, indices] * directions[None, :]
        operation = feature["operation"]
        if operation == "signed_mean":
            values = np.tanh(np.mean(directed, axis=1))
        elif operation == "signed_product":
            values = np.prod(np.tanh(directed), axis=1)
        elif operation == "absolute_gap":
            values = np.tanh(
                np.max(directed, axis=1) - np.min(directed, axis=1)
            )
        else:  # pragma: no cover - canonical validation makes this unreachable
            raise AssertionError(f"unhandled feature-program operation: {operation}")
        if not np.isfinite(values).all():
            raise ValueError(
                f"program feature {feature['name']!r} produced non-finite values"
            )
        columns.append(np.clip(values, -1.0, 1.0))

    if not columns:
        # The validator already enforces 8--16; this is a defensive shape guard.
        return np.empty((matrix.shape[0], 0), dtype=float)
    result = np.column_stack(columns).astype(float, copy=False)
    if not np.isfinite(result).all() or not (
        np.max(result) <= 1.0 and np.min(result) >= -1.0
    ):
        raise ValueError("feature-program output escaped its finite [-1, 1] contract")
    return result
