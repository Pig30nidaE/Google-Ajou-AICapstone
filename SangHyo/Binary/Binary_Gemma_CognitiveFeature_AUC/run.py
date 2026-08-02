"""Single entrypoint for task-aware Gemma cognitive feature extraction and AUC.

The hosted Gemma model sees one anonymous MMSE score payload at a time.  The
payload is constructed only from the source-level MMSE allow-list used by
``Binary_Google_ROCAUC_Champion``.  Subject identifiers, diagnoses, labels,
administrative fields, dates, and collection-volume metadata never enter the
prompt or persistent feature cache.

The downstream experiment is intentionally non-nested: a predeclared LR/RBF
grid and rank blends are selected by mean repeated subject-level OOF ROC-AUC.
This is optimistic model selection, but it is not direct data leakage.  Every
preprocessing fit remains local to its current training fold.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "SangHyo.Binary_Gemma_CognitiveFeature_AUC"

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from SangHyo.Binary.Binary_Google_ROCAUC_Champion.data import (
    AccessAudit,
    MMSE_ALLOWED_SOURCE_COLUMNS,
    MMSE_DOMAINS,
    MMSE_FORBIDDEN_SOURCE_COLUMNS,
    MMSE_ITEMS,
    assert_disjoint_subjects,
    assert_subject_alignment,
    binary_target,
    load_diagnoses,
    load_mmse_allowed,
    resolve_data_root,
)
from SangHyo.Binary.Binary_Google_ROCAUC_Champion.features import build_mmse_features

from . import EXPERIMENT_NAME, PAYLOAD_VERSION, SCHEMA_VERSION


PACKAGE_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = PACKAGE_ROOT / "requirements_colab.in"
DEFAULT_MODEL = "gemma-4-31b-it"
DEFAULT_API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_SEED = 20260730
CLASS_MAPPING = {"CN": 0, "MCI_or_Dem": 1}
EXPECTED_TRAIN_COUNTS = {"CN": 85, "MCI": 47, "Dem": 9}
EXPECTED_VALIDATION_COUNTS = {"CN": 26, "MCI": 4, "Dem": 3}


FEATURE_SPECS: tuple[tuple[str, str], ...] = (
    (
        "memory_specific_deficit",
        "Evidence that delayed recall is selectively weak relative to otherwise preserved performance.",
    ),
    (
        "orientation_memory_gap",
        "Magnitude of discordance between orientation performance and delayed recall.",
    ),
    (
        "attention_recall_discordance",
        "Magnitude and direction-aware strength of preserved attention with weaker delayed recall.",
    ),
    (
        "ceiling_adjusted_subtle_error",
        "Evidence carried by one or a few errors when the total score remains near the ceiling.",
    ),
    (
        "temporal_orientation_weakness",
        "Evidence of weakness specifically within time orientation items.",
    ),
    (
        "multi_domain_error_burden",
        "Extent to which errors are distributed over multiple cognitive domains rather than focal.",
    ),
    (
        "preserved_function_with_focal_failure",
        "Evidence of broadly preserved items coexisting with a focal failure pattern.",
    ),
    (
        "mci_boundary_evidence",
        "Continuous evidence relevant to the difficult CN-versus-MCI boundary; not a diagnosis.",
    ),
    (
        "global_severity_evidence",
        "Continuous evidence of broad/global impairment rather than an isolated subtle deficit.",
    ),
    (
        "evidence_reliability",
        "Completeness and internal support of the available MMSE score/item evidence only.",
    ),
)
FEATURE_NAMES = tuple(name for name, _ in FEATURE_SPECS)


SYSTEM_PROMPT = """\
You are a cognitive-pattern feature engineer. The downstream research task is
binary ranking: cognitively normal (CN) is the negative class, while mild cognitive impairment (MCI)
or dementia is the positive class. The hard and most
important boundary is CN versus MCI, especially when an examinee has a near-
ceiling MMSE total but a small, structured error pattern.

Use clinical reasoning principles only as a fixed rubric:
- delayed-recall weakness can be more informative than the same number of
  errors scattered elsewhere;
- compare recall with attention and orientation to detect focal-versus-global
  deficits;
- distinguish temporal-orientation weakness from preserved orientation;
- near-ceiling totals do not erase subtle, domain-specific failures;
- dementia-like global severity requires broad multi-domain evidence, while an
  MCI-like boundary pattern may retain much function with focal failure.

You receive the score/item values of one anonymous examination. You never
receive that examinee's label, diagnosis, identifier, administrative metadata,
dates, or collection volume. Do not infer or output a diagnosis, class,
category, probability, threshold decision, or final prediction. Instead,
extract the exact continuous evidence features requested below. Apply the same
rubric to every examination. Use only supplied values; do not invent norms.
Every output must be a JSON number from 0.0 through 1.0. Return exactly one JSON
object, with no markdown, prose, code fence, or additional key.
"""

_FEATURE_INSTRUCTIONS = "\n".join(
    f"- `{name}`: {description} Scale 0.0=no evidence, 1.0=strong evidence."
    for name, description in FEATURE_SPECS
)
_EXACT_JSON_EXAMPLE = json.dumps(
    {name: 0.0 for name in FEATURE_NAMES},
    ensure_ascii=False,
    separators=(",", ":"),
)
USER_PROMPT_TEMPLATE = f"""\
Task-aware continuous feature extraction for one anonymous MMSE examination.

Required fields:
{_FEATURE_INSTRUCTIONS}

Reliability must depend only on completeness and consistency of the supplied
MMSE total/domain/item evidence. Do not use recording duration, observation
count, device coverage, or any administrative quantity.

The output contract has exactly these ten keys and numeric values in [0,1]:
{_EXACT_JSON_EXAMPLE}

Anonymous examination JSON:
{{payload_json}}

Return only the exact JSON object.
"""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def prompt_hash() -> str:
    return _sha256_text(SYSTEM_PROMPT + "\n@@USER@@\n" + USER_PROMPT_TEMPLATE)


def response_schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": list(FEATURE_NAMES),
        "properties": {
            name: {"type": "number", "minimum": 0.0, "maximum": 1.0}
            for name in FEATURE_NAMES
        },
    }


def schema_hash() -> str:
    return _sha256_text(canonical_json(response_schema()))


def validate_feature_response(value: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("Gemma response must be a JSON object")
    if set(value) != set(FEATURE_NAMES):
        missing = sorted(set(FEATURE_NAMES) - set(value))
        extra = sorted(set(value) - set(FEATURE_NAMES))
        raise ValueError(f"response keys mismatch; missing={missing}, extra={extra}")
    validated: dict[str, float] = {}
    for name in FEATURE_NAMES:
        raw = value[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{name} must be a JSON number")
        number = float(raw)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{name} must be finite and inside [0,1]")
        validated[name] = number
    return validated


_PAYLOAD_FORBIDDEN_TOKENS = frozenset(
    {
        "id",
        "identifier",
        "subject",
        "patient",
        "email",
        "sample",
        "label",
        "target",
        "class",
        "diagnosis",
        "diag",
        "doctor",
        "admin",
        "date",
        "timestamp",
        "days",
        "duration",
        "coverage",
        "count",
    }
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(value).lower()))


def assert_private_payload(payload: Mapping[str, Any]) -> None:
    """Fail closed if a payload key or textual value could carry forbidden data."""

    violations: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                token_hits = _tokens(str(key)) & _PAYLOAD_FORBIDDEN_TOKENS
                if token_hits:
                    violations.append(f"{path}.{key}:{sorted(token_hits)}")
                walk(child, f"{path}.{key}")
        elif isinstance(node, (list, tuple)):
            for index, child in enumerate(node):
                walk(child, f"{path}[{index}]")
        elif isinstance(node, str):
            token_hits = _tokens(node) & _PAYLOAD_FORBIDDEN_TOKENS
            if token_hits:
                violations.append(f"{path}:text:{sorted(token_hits)}")

    walk(payload, "$")
    if violations:
        raise ValueError("private payload contract failed: " + "; ".join(violations[:8]))


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_anonymous_payload(
    anchor_values: Sequence[float],
    anchor_names: Sequence[str],
) -> dict[str, Any]:
    """Build the only object that can be rendered into an API request."""

    names = tuple(map(str, anchor_names))
    values = np.asarray(anchor_values, dtype=np.float64).reshape(-1)
    if len(names) != 39 or values.size != 39:
        raise ValueError("MMSE MaxAUC anchor must contain exactly 39 values")
    lookup = {name: _finite_or_none(value) for name, value in zip(names, values)}
    expected = {
        "mmse__total",
        "mmse__failed_items",
        "mmse__recall_deficit",
        *(f"mmse__domain__{domain}_score" for domain in MMSE_DOMAINS),
        *(f"mmse__item__{item.lower()}_correct" for item in MMSE_ITEMS),
    }
    if set(lookup) != expected:
        raise ValueError("39-anchor feature-name contract changed")

    payload: dict[str, Any] = {
        "payload_version": PAYLOAD_VERSION,
        "mmse": {
            "total": {"score": lookup["mmse__total"], "maximum": 30.0},
            "domains": {
                domain: {
                    "score": lookup[f"mmse__domain__{domain}_score"],
                    "maximum": float(len(items)),
                }
                for domain, items in MMSE_DOMAINS.items()
            },
            "items": {
                item.lower(): {
                    "correct": lookup[f"mmse__item__{item.lower()}_correct"]
                }
                for item in MMSE_ITEMS
            },
            "derived": {
                "failed_items": lookup["mmse__failed_items"],
                "recall_deficit": lookup["mmse__recall_deficit"],
            },
        },
    }
    assert_private_payload(payload)
    return payload


def render_user_prompt(payload: Mapping[str, Any]) -> str:
    assert_private_payload(payload)
    # ``USER_PROMPT_TEMPLATE`` also embeds a literal example JSON object, so a
    # targeted replacement avoids treating that object's braces as format
    # placeholders.
    return USER_PROMPT_TEMPLATE.replace("{payload_json}", canonical_json(payload))


def _unwrap_json_fence(text: str) -> str:
    stripped = str(text).strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```" and lines[0].strip().lower() in {
        "```",
        "```json",
    }:
        return "\n".join(lines[1:-1]).strip()
    return stripped


@dataclass(frozen=True)
class ClientConfig:
    model: str = DEFAULT_MODEL
    api_key_env: str = DEFAULT_API_KEY_ENV
    temperature: float = 0.0
    max_output_tokens: int = 4096
    timeout_seconds: float = 180.0
    max_retries: int = 6
    initial_backoff_seconds: float = 2.0
    max_backoff_seconds: float = 90.0
    min_interval_seconds: float = 13.0
    offline: bool = False


@dataclass
class ExtractionSummary:
    requested: int = 0
    cached: int = 0
    fresh: int = 0
    api_attempts: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cache_misses: int = 0

    def to_dict(self, *, client: "GemmaFeatureClient") -> dict[str, Any]:
        return {
            "requested_examinations": self.requested,
            "served_from_cache": self.cached,
            "fresh_api_answers": self.fresh,
            "api_attempts": self.api_attempts,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "cache_misses": self.cache_misses,
            "model": client.config.model,
            "prompt_hash": prompt_hash(),
            "schema_hash": schema_hash(),
            "generation_hash": client.generation_hash(),
            "response_mode": "prompt_json_plus_strict_local_validation",
            "privacy_contract": {
                "subject_labels_sent": 0,
                "diagnoses_sent": 0,
                "identifiers_sent": 0,
                "administrative_fields_sent": 0,
                "absolute_dates_sent": 0,
                "collection_volume_sent": 0,
                "api_key_persisted": False,
                "cache_contains_payload": False,
                "cache_contains_identifier": False,
            },
        }


class _RateLimiter:
    def __init__(self, interval: float) -> None:
        self.interval = max(0.0, float(interval))
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if wait:
            time.sleep(wait)


def _status_code(error: BaseException) -> int | None:
    for candidate in (error, getattr(error, "response", None)):
        if candidate is None:
            continue
        for key in ("status_code", "status", "code"):
            raw = getattr(candidate, key, None)
            raw = getattr(raw, "value", raw)
            if isinstance(raw, int):
                return int(raw)
            if isinstance(raw, str):
                match = re.search(r"\b([45][0-9]{2})\b", raw)
                if match:
                    return int(match.group(1))
    match = re.search(r"\b([45][0-9]{2})\b", str(error))
    return int(match.group(1)) if match else None


def _retryable(error: BaseException) -> bool:
    status = _status_code(error)
    if status == 429 or (status is not None and 500 <= status <= 599):
        return True
    if isinstance(error, (json.JSONDecodeError, ValueError)):
        return True
    text = f"{type(error).__name__}: {error}".lower()
    return any(
        marker in text
        for marker in ("timeout", "connection", "temporarily", "unavailable", "rate limit")
    )


def _safe_error(error: BaseException, key: str) -> str:
    text = f"{type(error).__name__}: {error}"
    if len(key) >= 8:
        text = text.replace(key, "[REDACTED]")
    text = re.sub(
        r"(?i)(api[_-]?key|x-goog-api-key|[?&]key)(\s*[:=]\s*)[^\s,&]+",
        r"\1\2[REDACTED]",
        text,
    )
    return " ".join(text.split())[:500]


class GemmaFeatureClient:
    """Hosted Gemma client whose persistent cache is identifier-free."""

    CACHE_FORMAT = 1

    def __init__(self, config: ClientConfig, cache_root: str | Path) -> None:
        if float(config.temperature) != 0.0:
            raise ValueError("task-aware extraction is frozen at temperature=0")
        self.config = config
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.model)
        self.cache_dir = Path(cache_root) / "gemma_cognitive_features" / safe_model
        self._client: Any | None = None
        self._limiter = _RateLimiter(config.min_interval_seconds)

    def generation_config(self) -> dict[str, Any]:
        return {
            "temperature": 0.0,
            "max_output_tokens": int(self.config.max_output_tokens),
            "response_mode": "prompt_json_plus_strict_local_validation",
        }

    def generation_hash(self) -> str:
        return _sha256_text(canonical_json(self.generation_config()))

    def request_identity(self, payload: Mapping[str, Any]) -> dict[str, str]:
        assert_private_payload(payload)
        material = {
            "cache_format": self.CACHE_FORMAT,
            "payload_hash": _sha256_text(canonical_json(payload)),
            "prompt_hash": prompt_hash(),
            "schema_hash": schema_hash(),
            "model": self.config.model,
            "generation_hash": self.generation_hash(),
        }
        return {
            **{key: str(value) for key, value in material.items() if key != "cache_format"},
            "request_hash": _sha256_text(canonical_json(material)),
        }

    def cache_path(self, request_hash: str) -> Path:
        return self.cache_dir / f"{request_hash}.json"

    def _read_cache(
        self, identity: Mapping[str, str]
    ) -> dict[str, float] | None:
        path = self.cache_path(identity["request_hash"])
        if not path.is_file():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            expected_keys = {
                "cache_format",
                "request_hash",
                "payload_hash",
                "prompt_hash",
                "schema_hash",
                "generation_hash",
                "model",
                "features",
                "usage",
            }
            if not isinstance(record, Mapping) or set(record) != expected_keys:
                return None
            if int(record["cache_format"]) != self.CACHE_FORMAT:
                return None
            for key in (
                "request_hash",
                "payload_hash",
                "prompt_hash",
                "schema_hash",
                "generation_hash",
                "model",
            ):
                if str(record[key]) != str(identity[key]):
                    return None
            return validate_feature_response(record["features"])
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None

    def _write_cache(
        self,
        identity: Mapping[str, str],
        features: Mapping[str, float],
        usage: Mapping[str, int],
    ) -> None:
        record = {
            "cache_format": self.CACHE_FORMAT,
            **dict(identity),
            "features": validate_feature_response(features),
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "output_tokens": int(usage.get("output_tokens", 0)),
                "thinking_tokens": int(usage.get("thinking_tokens", 0)),
            },
        }
        serialized = canonical_json(record)
        lowered = serialized.lower()
        if any(
            marker in lowered
            for marker in ('"subject_', '"sample_', '"email"', '"diagnosis"', '"label"')
        ):
            raise AssertionError("identifier or label escaped into the persistent cache")
        path = self.cache_path(identity["request_hash"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        key = os.environ.get(self.config.api_key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"{self.config.api_key_env} is empty; configure the Colab secret "
                "or use --offline with a complete exact-request cache"
            )
        try:
            from google import genai  # type: ignore
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError("google-genai is required for live extraction") from error
        self._client = genai.Client(api_key=key)
        return self._client

    def _call(self, payload: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, int]]:
        from google.genai import types  # type: ignore

        self._limiter.acquire()
        response = self._get_client().models.generate_content(
            model=self.config.model,
            contents=render_user_prompt(payload),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                max_output_tokens=int(self.config.max_output_tokens),
                http_options=types.HttpOptions(
                    timeout=int(float(self.config.timeout_seconds) * 1000)
                ),
            ),
        )
        candidates = getattr(response, "candidates", None) or []
        finish_reason = (
            str(getattr(candidates[0], "finish_reason", "") or "") if candidates else ""
        )
        if "MAX_TOKENS" in finish_reason.upper():
            raise ValueError("Gemma response reached max_output_tokens")
        text = str(getattr(response, "text", "") or "")
        if not text.strip():
            raise ValueError("Gemma response body is empty")
        decoded = json.loads(_unwrap_json_fence(text))
        features = validate_feature_response(decoded)
        metadata = getattr(response, "usage_metadata", None)
        usage = {
            "prompt_tokens": int(getattr(metadata, "prompt_token_count", 0) or 0),
            "output_tokens": int(getattr(metadata, "candidates_token_count", 0) or 0),
            "thinking_tokens": int(getattr(metadata, "thoughts_token_count", 0) or 0),
        }
        return features, usage

    def extract_one(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[dict[str, float], str, int, dict[str, int]]:
        identity = self.request_identity(payload)
        cached = self._read_cache(identity)
        if cached is not None:
            return cached, "cached", 0, {
                "prompt_tokens": 0,
                "output_tokens": 0,
                "thinking_tokens": 0,
            }
        if self.config.offline:
            raise RuntimeError(
                "offline cache miss for exact anonymous MMSE payload "
                f"{identity['request_hash'][:12]}"
            )

        key = os.environ.get(self.config.api_key_env, "")
        backoff = max(0.0, float(self.config.initial_backoff_seconds))
        last_error: BaseException | None = None
        for attempt in range(1, int(self.config.max_retries) + 1):
            try:
                features, usage = self._call(payload)
                self._write_cache(identity, features, usage)
                return features, "fresh", attempt, usage
            except Exception as error:  # noqa: BLE001 - provider errors vary
                last_error = error
                if attempt >= int(self.config.max_retries) or not _retryable(error):
                    break
                if backoff:
                    time.sleep(min(backoff, float(self.config.max_backoff_seconds), 60.0))
                backoff = max(1.0, backoff * 2.0)
        assert last_error is not None
        raise RuntimeError(
            "Gemma extraction failed after bounded retries: "
            + _safe_error(last_error, key)
        ) from None


def anchor_table(
    mmse: pd.DataFrame,
    subject_ids: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    values, names, _core, indices = build_mmse_features(mmse, subject_ids)
    anchor = np.asarray(values[:, list(indices)], dtype=np.float64)
    anchor_names = tuple(names[index] for index in indices)
    if anchor.shape != (len(subject_ids), 39) or len(anchor_names) != 39:
        raise AssertionError("39-column MMSE anchor contract failed")
    return anchor, anchor_names


def build_payloads(
    anchor: np.ndarray,
    anchor_names: Sequence[str],
) -> list[dict[str, Any]]:
    matrix = np.asarray(anchor, dtype=np.float64)
    return [
        build_anonymous_payload(matrix[index], anchor_names)
        for index in range(matrix.shape[0])
    ]


def extract_payloads(
    payloads: Sequence[Mapping[str, Any]],
    client: GemmaFeatureClient,
) -> tuple[np.ndarray, ExtractionSummary]:
    summary = ExtractionSummary(requested=len(payloads))
    rows: list[list[float]] = []
    for payload in payloads:
        try:
            features, status, attempts, usage = client.extract_one(payload)
        except RuntimeError:
            summary.cache_misses += 1
            raise
        if status == "cached":
            summary.cached += 1
        else:
            summary.cache_misses += 1
            summary.fresh += 1
            summary.api_attempts += attempts
            summary.prompt_tokens += int(usage["prompt_tokens"])
            summary.output_tokens += int(usage["output_tokens"])
            summary.thinking_tokens += int(usage["thinking_tokens"])
        rows.append([features[name] for name in FEATURE_NAMES])
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (len(payloads), len(FEATURE_NAMES)):
        raise AssertionError("Gemma feature matrix shape contract failed")
    if not np.isfinite(matrix).all() or np.min(matrix) < 0 or np.max(matrix) > 1:
        raise ValueError("Gemma feature matrix escaped [0,1]")
    return matrix, summary


def hash_subject(value: str, *, salt: str) -> str:
    return hashlib.sha256(f"{salt}\0{value}".encode("utf-8")).hexdigest()[:20]


def _fingerprint_matrix(names: Sequence[str], values: np.ndarray) -> str:
    digest = hashlib.sha256("\n".join(map(str, names)).encode("utf-8"))
    matrix = np.asarray(values, dtype=np.float64)
    digest.update(np.nan_to_num(matrix, nan=-9999.0).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Profile:
    name: str
    folds: int
    repeats: int
    lr_c: tuple[float, ...]
    rbf: tuple[tuple[float, str | float], ...]
    blend_weights: tuple[float, ...]


PROFILES: Mapping[str, Profile] = {
    "smoke": Profile(
        "smoke",
        folds=2,
        repeats=1,
        lr_c=(0.1, 1.0),
        rbf=((1.0, "scale"), (3.0, "scale")),
        blend_weights=(0.25, 0.5, 0.75),
    ),
    "default": Profile(
        "default",
        folds=5,
        repeats=10,
        lr_c=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
        rbf=(
            (0.1, "scale"),
            (0.3, "scale"),
            (1.0, "scale"),
            (3.0, "scale"),
            (10.0, "scale"),
            (30.0, "scale"),
            (0.3, 0.01),
            (1.0, 0.01),
            (3.0, 0.01),
            (1.0, 0.03),
            (3.0, 0.03),
            (10.0, 0.03),
            (1.0, 0.1),
            (3.0, 0.1),
        ),
        blend_weights=tuple(float(value) for value in np.linspace(0.0, 1.0, 11)),
    ),
    "max": Profile(
        "max",
        folds=5,
        repeats=30,
        lr_c=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
        rbf=tuple(
            (float(c), gamma)
            for c in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
            for gamma in ("scale", 0.003, 0.01, 0.03, 0.1, 0.3)
        ),
        blend_weights=tuple(float(value) for value in np.linspace(0.0, 1.0, 21)),
    ),
}


@dataclass(frozen=True)
class SplitRecord:
    split_id: str
    repeat: int
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray


def build_subject_splits(
    y: Sequence[int],
    subject_ids: Sequence[str],
    *,
    folds: int,
    repeats: int,
    seed: int,
) -> tuple[SplitRecord, ...]:
    target = np.asarray(y, dtype=np.int64)
    subjects = np.asarray(subject_ids, dtype=str)
    if target.shape != subjects.shape or len(set(subjects)) != len(subjects):
        raise ValueError("one unique subject id is required per target")
    records: list[SplitRecord] = []
    for repeat in range(int(repeats)):
        splitter = StratifiedKFold(
            n_splits=int(folds),
            shuffle=True,
            random_state=int(seed) + repeat * 1009,
        )
        seen = np.zeros(len(target), dtype=np.int64)
        for fold, (train, test) in enumerate(
            splitter.split(np.zeros((len(target), 1)), target)
        ):
            assert_disjoint_subjects(subjects[train], subjects[test])
            seen[test] += 1
            records.append(
                SplitRecord(
                    split_id=f"r{repeat:02d}_f{fold:02d}",
                    repeat=repeat,
                    fold=fold,
                    train_indices=np.asarray(train, dtype=np.int64),
                    test_indices=np.asarray(test, dtype=np.int64),
                )
            )
        if not np.all(seen == 1):
            raise AssertionError("each subject must be OOF exactly once per repeat")
    return tuple(records)


def _model_key(family: str, params: Mapping[str, Any]) -> str:
    def token(value: Any) -> str:
        return str(value).replace(".", "p").replace("-", "m")

    if family == "lr":
        return f"lr__c_{token(params['C'])}"
    return f"rbf__c_{token(params['C'])}__g_{token(params['gamma'])}"


def _pipeline(family: str, params: Mapping[str, Any], *, seed: int) -> Pipeline:
    if family == "lr":
        classifier: Any = LogisticRegression(
            C=float(params["C"]),
            solver="liblinear",
            class_weight="balanced",
            max_iter=20_000,
            random_state=int(seed),
        )
    elif family == "rbf":
        classifier = SVC(
            C=float(params["C"]),
            gamma=params["gamma"],
            kernel="rbf",
            class_weight="balanced",
            cache_size=512,
            random_state=int(seed),
        )
    else:
        raise ValueError(f"unknown family: {family}")
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", classifier),
        ]
    )


def _training_cdf(reference: Sequence[float], values: Sequence[float]) -> np.ndarray:
    fitted = np.sort(np.asarray(reference, dtype=np.float64).reshape(-1))
    query = np.asarray(values, dtype=np.float64).reshape(-1)
    left = np.searchsorted(fitted, query, side="left")
    right = np.searchsorted(fitted, query, side="right")
    return (left + right).astype(np.float64) / (2.0 * float(len(fitted)))


def _candidate_parameters(profile: Profile) -> tuple[tuple[str, dict[str, Any]], ...]:
    candidates: list[tuple[str, dict[str, Any]]] = [
        ("lr", {"C": float(value)}) for value in profile.lr_c
    ]
    candidates.extend(
        ("rbf", {"C": float(c), "gamma": gamma}) for c, gamma in profile.rbf
    )
    return tuple(candidates)


def _auc_summary(y: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    repeat_auc = [
        float(roc_auc_score(y, scores[repeat]))
        for repeat in range(scores.shape[0])
    ]
    return {
        "metric": "ROC-AUC",
        "selection_value_mean_repeat_roc_auc": float(np.mean(repeat_auc)),
        "repeat_roc_auc_sd": (
            float(np.std(repeat_auc, ddof=1)) if len(repeat_auc) > 1 else 0.0
        ),
        "repeat_roc_auc": repeat_auc,
        "subject_mean_oof_roc_auc": float(roc_auc_score(y, scores.mean(axis=0))),
    }


@dataclass
class ViewEvaluation:
    view: str
    candidates: dict[str, dict[str, Any]]
    scores: dict[str, np.ndarray]
    best_lr: str
    best_rbf: str
    best_blend: str
    selected: str


def evaluate_view(
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[SplitRecord],
    profile: Profile,
    *,
    view: str,
    seed: int,
) -> ViewEvaluation:
    matrix = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.int64)
    predictions = {
        _model_key(family, params): np.full(
            (profile.repeats, len(target)), np.nan, dtype=np.float64
        )
        for family, params in _candidate_parameters(profile)
    }
    metadata: dict[str, dict[str, Any]] = {}
    for record in splits:
        train, test = record.train_indices, record.test_indices
        for candidate_index, (family, params) in enumerate(
            _candidate_parameters(profile)
        ):
            key = _model_key(family, params)
            fitted = _pipeline(
                family,
                params,
                seed=int(seed) + record.repeat * 10_007 + record.fold * 101 + candidate_index,
            )
            fitted.fit(matrix[train], target[train])
            training_margin = fitted.decision_function(matrix[train])
            test_margin = fitted.decision_function(matrix[test])
            predictions[key][record.repeat, test] = _training_cdf(
                training_margin, test_margin
            )
            metadata[key] = {
                "name": key,
                "view": view,
                "contains_gemma": view == "gemma_augmented",
                "family": family,
                "params": dict(params),
                "preprocessing": "fold-local median imputation + standard scaling",
                "score": "decision margin located in fitted training empirical CDF",
            }
    for key, score in predictions.items():
        if not np.isfinite(score).all():
            raise AssertionError(f"{view}/{key} left an OOF score missing")
        metadata[key]["auc"] = _auc_summary(target, score)

    def best_family(family: str) -> str:
        eligible = [
            key for key, item in metadata.items() if item["family"] == family
        ]
        return max(
            eligible,
            key=lambda key: (
                metadata[key]["auc"]["selection_value_mean_repeat_roc_auc"],
                key,
            ),
        )

    best_lr = best_family("lr")
    best_rbf = best_family("rbf")
    for weight in profile.blend_weights:
        key = f"rank_blend__lr_{weight:.2f}"
        score = float(weight) * predictions[best_lr] + (1.0 - float(weight)) * predictions[
            best_rbf
        ]
        predictions[key] = score
        metadata[key] = {
            "name": key,
            "view": view,
            "contains_gemma": view == "gemma_augmented",
            "family": "rank_blend",
            "params": {
                "lr_candidate": best_lr,
                "rbf_candidate": best_rbf,
                "lr_weight": float(weight),
            },
            "preprocessing": "component-specific fold-local preprocessing",
            "score": "convex blend of two training-CDF rank scores",
            "auc": _auc_summary(target, score),
        }
    blend_keys = [
        key for key, item in metadata.items() if item["family"] == "rank_blend"
    ]
    best_blend = max(
        blend_keys,
        key=lambda key: (
            metadata[key]["auc"]["selection_value_mean_repeat_roc_auc"],
            key,
        ),
    )
    finalists = (best_lr, best_rbf, best_blend)
    selected = max(
        finalists,
        key=lambda key: (
            metadata[key]["auc"]["selection_value_mean_repeat_roc_auc"],
            key,
        ),
    )
    return ViewEvaluation(
        view=view,
        candidates=metadata,
        scores=predictions,
        best_lr=best_lr,
        best_rbf=best_rbf,
        best_blend=best_blend,
        selected=selected,
    )


def _params_for_key(
    evaluation: ViewEvaluation, key: str
) -> tuple[str, dict[str, Any]]:
    item = evaluation.candidates[key]
    if item["family"] != "rank_blend":
        return str(item["family"]), dict(item["params"])
    raise ValueError("blend key does not map to one component")


def fit_full_component(
    X: np.ndarray,
    y: np.ndarray,
    *,
    family: str,
    params: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    fitted = _pipeline(family, params, seed=seed)
    fitted.fit(X, y)
    reference = np.sort(
        np.asarray(fitted.decision_function(X), dtype=np.float64).reshape(-1)
    )
    return {
        "family": family,
        "params": dict(params),
        "pipeline": fitted,
        "training_score_reference": reference,
    }


def predict_component(component: Mapping[str, Any], X: np.ndarray) -> np.ndarray:
    margin = component["pipeline"].decision_function(np.asarray(X, dtype=np.float64))
    return _training_cdf(component["training_score_reference"], margin)


def fit_deployment_bundle(
    evaluation: ViewEvaluation,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    model: str,
    seed: int,
) -> dict[str, Any]:
    selected = evaluation.candidates[evaluation.selected]
    bundle: dict[str, Any] = {
        "format": 1,
        "experiment": EXPERIMENT_NAME,
        "class_mapping": dict(CLASS_MAPPING),
        "selected_candidate": evaluation.selected,
        "feature_names": tuple(map(str, feature_names)),
        "gemma_model": model,
        "prompt_hash": prompt_hash(),
        "schema_hash": schema_hash(),
    }
    if selected["family"] == "rank_blend":
        lr_key = selected["params"]["lr_candidate"]
        rbf_key = selected["params"]["rbf_candidate"]
        lr_family, lr_params = _params_for_key(evaluation, lr_key)
        rbf_family, rbf_params = _params_for_key(evaluation, rbf_key)
        bundle["kind"] = "rank_blend"
        bundle["lr_weight"] = float(selected["params"]["lr_weight"])
        bundle["components"] = {
            "lr": fit_full_component(
                X, y, family=lr_family, params=lr_params, seed=seed + 11
            ),
            "rbf": fit_full_component(
                X, y, family=rbf_family, params=rbf_params, seed=seed + 23
            ),
        }
    else:
        family, params = _params_for_key(evaluation, evaluation.selected)
        bundle["kind"] = "single"
        bundle["components"] = {
            "single": fit_full_component(
                X, y, family=family, params=params, seed=seed + 37
            )
        }
    return bundle


def predict_bundle(bundle: Mapping[str, Any], X: np.ndarray) -> np.ndarray:
    if str(bundle["kind"]) == "single":
        return predict_component(bundle["components"]["single"], X)
    lr = predict_component(bundle["components"]["lr"], X)
    rbf = predict_component(bundle["components"]["rbf"], X)
    weight = float(bundle["lr_weight"])
    return weight * lr + (1.0 - weight) * rbf


def write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_registry(
    splits: Sequence[SplitRecord],
    subjects: Sequence[str],
    y: np.ndarray,
    *,
    salt: str,
    profile: Profile,
    seed: int,
) -> dict[str, Any]:
    ids = np.asarray(subjects, dtype=str)
    target = np.asarray(y, dtype=np.int64)
    return {
        "subject_level": True,
        "folds": profile.folds,
        "repeats": profile.repeats,
        "seed": seed,
        "records": [
            {
                "split_id": record.split_id,
                "repeat": record.repeat,
                "fold": record.fold,
                "train_hashes": [
                    hash_subject(value, salt=salt) for value in ids[record.train_indices]
                ],
                "test_hashes": [
                    hash_subject(value, salt=salt) for value in ids[record.test_indices]
                ],
                "test_class_counts": {
                    "CN": int(np.sum(target[record.test_indices] == 0)),
                    "MCI_or_Dem": int(np.sum(target[record.test_indices] == 1)),
                },
            }
            for record in splits
        ],
    }


def _write_oof(
    path: Path,
    subjects: Sequence[str],
    y: np.ndarray,
    baseline: ViewEvaluation,
    augmented: ViewEvaluation,
    *,
    salt: str,
) -> Path:
    rows: list[dict[str, Any]] = []
    for repeat in range(augmented.scores[augmented.selected].shape[0]):
        for index, subject in enumerate(subjects):
            rows.append(
                {
                    "subject_hash": hash_subject(str(subject), salt=salt),
                    "repeat": repeat,
                    "y_true": int(y[index]),
                    "baseline_score": float(
                        baseline.scores[baseline.selected][repeat, index]
                    ),
                    "gemma_champion_score": float(
                        augmented.scores[augmented.selected][repeat, index]
                    ),
                }
            )
    pd.DataFrame.from_records(rows).to_csv(path, index=False)
    return path


def resolve_paths(
    *,
    namespace: Mapping[str, Any],
    data_root: str | None,
    output_dir: str | None,
    cache_dir: str | None,
) -> tuple[Path, Path, Path]:
    data_candidates = [
        data_root,
        namespace.get("DATA_ROOT"),
        os.environ.get("BGCFA_DATA_ROOT"),
        str(PACKAGE_ROOT.parents[1] / "Data"),
    ]
    resolved_data: Path | None = None
    for candidate in data_candidates:
        if not candidate:
            continue
        try:
            resolved_data = resolve_data_root(Path(os.fspath(candidate)))
            break
        except FileNotFoundError:
            continue
    if resolved_data is None:
        raise FileNotFoundError("Data root with 1.Training and 2.Validation was not found")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
    if output_dir:
        output = Path(output_dir).expanduser().resolve()
    elif os.environ.get("BGCFA_OUTPUT_ROOT"):
        output = Path(os.environ["BGCFA_OUTPUT_ROOT"]).expanduser().resolve() / run_id
    elif Path("/content/drive/MyDrive").is_dir():
        output = (
            Path("/content/drive/MyDrive") / f"{EXPERIMENT_NAME}_result" / run_id
        )
    else:
        output = PACKAGE_ROOT / f"{EXPERIMENT_NAME}_result" / run_id

    if cache_dir:
        cache = Path(cache_dir).expanduser().resolve()
    elif os.environ.get("BGCFA_CACHE_ROOT"):
        cache = Path(os.environ["BGCFA_CACHE_ROOT"]).expanduser().resolve()
    elif Path("/content/drive/MyDrive").is_dir():
        cache = Path("/content/drive/MyDrive") / f"{EXPERIMENT_NAME}_cache"
    else:
        cache = PACKAGE_ROOT / f"{EXPERIMENT_NAME}_cache"
    return resolved_data, output, cache


def ensure_dependencies(*, include_api: bool, skip_install: bool) -> None:
    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "sklearn",
        "joblib": "joblib",
    }
    if include_api:
        required["google-genai"] = "google.genai"
    missing = [
        distribution
        for distribution, module in required.items()
        if importlib.util.find_spec(module) is None
    ]
    if not missing:
        return
    if skip_install:
        raise ModuleNotFoundError("Missing dependencies: " + ", ".join(missing))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            str(REQUIREMENTS_FILE),
        ],
        check=True,
    )
    importlib.invalidate_caches()


def inspect_data(data_root: Path, audit: AccessAudit) -> dict[str, Any]:
    mmse = load_mmse_allowed(data_root, "train", track="mmse", audit=audit)
    subjects = np.asarray(mmse.index, dtype=str)
    anchor, anchor_names = anchor_table(mmse, subjects)
    payloads = build_payloads(anchor, anchor_names)
    return {
        "stage": "inspect",
        "cohort": "1.Training label-free MMSE inspection",
        "n_examinations": int(len(subjects)),
        "mmse_source_allowlist": list(MMSE_ALLOWED_SOURCE_COLUMNS),
        "mmse_source_forbidden_not_read": sorted(MMSE_FORBIDDEN_SOURCE_COLUMNS),
        "anchor_features": list(anchor_names),
        "n_anchor_features": len(anchor_names),
        "anchor_fingerprint": _fingerprint_matrix(anchor_names, anchor),
        "payload_contract": {
            "payload_version": PAYLOAD_VERSION,
            "all_payloads_passed_privacy_guard": all(
                assert_private_payload(payload) is None for payload in payloads
            ),
            "identifiers_in_payload": False,
            "labels_in_payload": False,
            "diagnoses_in_payload": False,
            "administrative_fields_in_payload": False,
            "absolute_dates_in_payload": False,
            "collection_volume_in_payload": False,
        },
        "labels_opened": False,
        "historical_validation_opened": False,
        "source_access": audit.to_dict(),
    }


def _load_anchor_split(
    data_root: Path,
    split: str,
    audit: AccessAudit,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, tuple[str, ...], list[dict[str, Any]]]:
    mmse = load_mmse_allowed(data_root, split, track="mmse", audit=audit)
    subjects = np.asarray(mmse.index, dtype=str)
    anchor, anchor_names = anchor_table(mmse, subjects)
    payloads = build_payloads(anchor, anchor_names)
    return mmse, subjects, anchor, anchor_names, payloads


def _api_audit(
    summary: ExtractionSummary,
    client: GemmaFeatureClient,
    *,
    split: str,
) -> dict[str, Any]:
    return {"stage": "extract", "split": split, **summary.to_dict(client=client)}


def _features_csv(
    path: Path,
    subjects: Sequence[str],
    features: np.ndarray,
    *,
    salt: str,
) -> Path:
    frame = pd.DataFrame(features, columns=FEATURE_NAMES)
    frame.insert(
        0,
        "subject_hash",
        [hash_subject(str(value), salt=salt) for value in subjects],
    )
    frame.to_csv(path, index=False)
    return path


def train_experiment(
    *,
    data_root: Path,
    output: Path,
    audit: AccessAudit,
    subjects: np.ndarray,
    anchor: np.ndarray,
    anchor_names: Sequence[str],
    gemma_features: np.ndarray,
    profile: Profile,
    seed: int,
    model: str,
    salt: str,
) -> tuple[dict[str, Any], Mapping[str, Any], np.ndarray]:
    diagnoses = load_diagnoses(data_root, "train", audit=audit)
    assert_subject_alignment(subjects, diagnoses, role="training diagnosis")
    aligned = diagnoses.reindex(subjects)
    observed = aligned.value_counts().to_dict()
    if observed != EXPECTED_TRAIN_COUNTS:
        raise AssertionError(f"training diagnosis contract changed: {observed}")
    y = binary_target(aligned)
    if int(y.sum()) != 56:
        raise AssertionError("CN=0 versus MCI+Dem=1 target contract changed")

    augmented = np.column_stack((anchor, gemma_features))
    augmented_names = tuple((*map(str, anchor_names), *FEATURE_NAMES))
    splits = build_subject_splits(
        y,
        subjects,
        folds=profile.folds,
        repeats=profile.repeats,
        seed=seed,
    )
    write_json(
        output / "SPLIT_REGISTRY.json",
        _split_registry(
            splits, subjects, y, salt=salt, profile=profile, seed=seed
        ),
    )
    baseline_eval = evaluate_view(
        anchor, y, splits, profile, view="baseline_anchor39", seed=seed
    )
    augmented_eval = evaluate_view(
        augmented, y, splits, profile, view="gemma_augmented", seed=seed + 1_000_000
    )
    if not augmented_eval.candidates[augmented_eval.selected]["contains_gemma"]:
        raise AssertionError("champion selection escaped the Gemma-containing arm")

    _write_oof(
        output / "OOF_PREDICTIONS_HASHED.csv",
        subjects,
        y,
        baseline_eval,
        augmented_eval,
        salt=salt,
    )
    deployment = fit_deployment_bundle(
        augmented_eval,
        augmented,
        y,
        augmented_names,
        model=model,
        seed=seed + 2_000_000,
    )
    deployment_dir = output / "deployment"
    deployment_dir.mkdir(parents=True, exist_ok=True)
    model_path = deployment_dir / "model.joblib"
    joblib.dump(deployment, model_path)
    restored = joblib.load(model_path)
    before = predict_bundle(deployment, augmented)
    after = predict_bundle(restored, augmented)
    if not np.allclose(before, after, rtol=0.0, atol=1e-12):
        raise AssertionError("joblib roundtrip changed full-refit scores")
    write_json(
        deployment_dir / "deployment.json",
        {
            "experiment": EXPERIMENT_NAME,
            "model_path": str(model_path),
            "model_sha256": _file_sha256(model_path),
            "roundtrip_max_abs_difference": float(np.max(np.abs(before - after))),
            "selected_candidate": augmented_eval.selected,
            "n_features": len(augmented_names),
            "feature_names": list(augmented_names),
            "gemma_model": model,
            "prompt_hash": prompt_hash(),
            "schema_hash": schema_hash(),
        },
    )

    baseline_item = baseline_eval.candidates[baseline_eval.selected]
    champion_item = augmented_eval.candidates[augmented_eval.selected]
    report = {
        "experiment": EXPERIMENT_NAME,
        "task": {
            "negative": "CN",
            "positive": "MCI + Dem",
            "n_subjects": int(len(y)),
            "class_counts": {"CN": int((y == 0).sum()), "MCI_or_Dem": int(y.sum())},
        },
        "metric_contract": {
            "primary_metric": "mean repeated subject-level OOF ROC-AUC",
            "candidate_selection_metric": "mean repeated subject-level OOF ROC-AUC",
            "other_metrics_used_for_selection": [],
            "nested": False,
            "selection_note": (
                "Candidate and blend selection is deliberately non-nested and therefore "
                "optimistic; preprocessing remains fold-local and no test label enters a fit."
            ),
        },
        "cv": {
            "profile": profile.name,
            "folds": profile.folds,
            "repeats": profile.repeats,
            "subject_level": True,
            "same_splits_all_candidates": True,
        },
        "champion": champion_item,
        "baseline_ablation": baseline_item,
        "auc_difference_champion_minus_baseline": float(
            champion_item["auc"]["selection_value_mean_repeat_roc_auc"]
            - baseline_item["auc"]["selection_value_mean_repeat_roc_auc"]
        ),
        "gemma_candidate_finalists": {
            key: augmented_eval.candidates[key]
            for key in (
                augmented_eval.best_lr,
                augmented_eval.best_rbf,
                augmented_eval.best_blend,
            )
        },
        "baseline_candidate_finalists": {
            key: baseline_eval.candidates[key]
            for key in (
                baseline_eval.best_lr,
                baseline_eval.best_rbf,
                baseline_eval.best_blend,
            )
        },
        "all_gemma_candidates": sorted(
            augmented_eval.candidates.values(),
            key=lambda item: -item["auc"]["selection_value_mean_repeat_roc_auc"],
        ),
        "deployment": {
            "path": str(model_path),
            "full_refit": True,
            "joblib_roundtrip_verified": True,
        },
    }
    write_json(output / "FINAL_REPORT.json", report)
    return report, restored, y


def historical_validation(
    *,
    data_root: Path,
    output: Path,
    audit: AccessAudit,
    client: GemmaFeatureClient,
    bundle: Mapping[str, Any],
    training_subject_ids: Sequence[str],
    salt: str,
) -> dict[str, Any]:
    _mmse, subjects, anchor, anchor_names, payloads = _load_anchor_split(
        data_root, "val", audit
    )
    # Fail before any Validation API call or prediction if a trained subject
    # reappears in the historical cohort.
    assert_disjoint_subjects(training_subject_ids, subjects)
    gemma, summary = extract_payloads(payloads, client)
    augmented = np.column_stack((anchor, gemma))
    scores = predict_bundle(bundle, augmented)
    freeze_path = output / "HISTORICAL_VALIDATION_PREDICTIONS_FROZEN.csv"
    pd.DataFrame(
        {
            "subject_hash": [
                hash_subject(str(value), salt=salt) for value in subjects
            ],
            "score": scores,
        }
    ).to_csv(freeze_path, index=False)
    freeze_hash = _file_sha256(freeze_path)
    write_json(
        output / "HISTORICAL_VALIDATION_FREEZE.json",
        {
            "predictions_path": str(freeze_path),
            "sha256_before_label_open": freeze_hash,
            "n_predictions": len(scores),
            "labels_opened_at_freeze": False,
            "api": _api_audit(summary, client, split="val"),
        },
    )

    # The historical labels are intentionally opened only after the score file
    # and its digest have been persisted.
    diagnoses = load_diagnoses(data_root, "val", audit=audit)
    assert_subject_alignment(subjects, diagnoses, role="historical validation diagnosis")
    aligned = diagnoses.reindex(subjects)
    if aligned.value_counts().to_dict() != EXPECTED_VALIDATION_COUNTS:
        raise AssertionError("historical validation diagnosis contract changed")
    y = binary_target(aligned)
    report = {
        "role": "historical evaluation; not an untouched external test",
        "n_subjects": len(y),
        "roc_auc": float(roc_auc_score(y, scores)),
        "prediction_freeze_sha256": freeze_hash,
        "labels_opened_after_prediction_freeze": True,
    }
    write_json(output / "HISTORICAL_VALIDATION_REPORT.json", report)
    return report


def _status(output: Path, status: str, **extra: Any) -> None:
    write_json(
        output / "LAUNCHER_STATUS.json",
        {
            "status": status,
            "experiment": EXPERIMENT_NAME,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            **extra,
        },
    )


def run_pipeline(
    *,
    namespace: Mapping[str, Any] | None = None,
    stage: str = "all",
    profile_name: str = "default",
    data_root: str | None = None,
    output_dir: str | None = None,
    cache_dir: str | None = None,
    model: str = DEFAULT_MODEL,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    offline: bool = False,
    historical_eval: bool = False,
    min_interval_seconds: float = 13.0,
    seed: int = DEFAULT_SEED,
    skip_install: bool = False,
) -> dict[str, Any]:
    namespace = globals() if namespace is None else namespace
    profile = PROFILES[profile_name]
    resolved_data, output, cache = resolve_paths(
        namespace=namespace,
        data_root=data_root,
        output_dir=output_dir,
        cache_dir=cache_dir,
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    # This secret is intentionally never persisted. It makes subject tokens
    # consistent inside one run while preventing cross-run linkage or a simple
    # dictionary attack against a public fixed salt.
    artifact_salt = os.urandom(32).hex()
    config = {
        "experiment": EXPERIMENT_NAME,
        "stage": stage,
        "profile": asdict(profile),
        "data_root": str(resolved_data),
        "output_dir": str(output),
        "cache_dir": str(cache),
        "model": model,
        "api_key_env": api_key_env,
        "offline": offline,
        "historical_eval": historical_eval,
        "min_interval_seconds": min_interval_seconds,
        "seed": seed,
        "prompt_hash": prompt_hash(),
        "schema_hash": schema_hash(),
        "class_mapping": CLASS_MAPPING,
        "subject_tokenization": (
            "SHA-256 with a run-local random secret that is not persisted"
        ),
    }
    write_json(output / "RUN_CONFIG.json", config)
    _status(output, "starting", stage=stage, profile=profile_name)
    audit = AccessAudit()
    result: dict[str, Any] = {"output_dir": str(output)}
    try:
        include_api = stage in {"extract", "all"} and not offline
        ensure_dependencies(include_api=include_api, skip_install=skip_install)
        inspect_report = inspect_data(resolved_data, audit)
        write_json(output / "DATA_AUDIT.json", inspect_report)
        result["inspect"] = inspect_report
        if stage == "inspect":
            _status(
                output,
                "complete",
                stage=stage,
                elapsed_seconds=time.monotonic() - started,
            )
            return result

        _mmse, subjects, anchor, anchor_names, payloads = _load_anchor_split(
            resolved_data, "train", audit
        )
        client = GemmaFeatureClient(
            ClientConfig(
                model=model,
                api_key_env=api_key_env,
                min_interval_seconds=min_interval_seconds,
                offline=(offline or stage == "train"),
            ),
            cache,
        )
        gemma_features, summary = extract_payloads(payloads, client)
        api_report = _api_audit(summary, client, split="train")
        write_json(output / "API_AUDIT.json", api_report)
        _features_csv(
            output / "GEMMA_FEATURES_HASHED.csv",
            subjects,
            gemma_features,
            salt=artifact_salt,
        )
        write_json(
            output / "FEATURE_MANIFEST.json",
            {
                "anchor_feature_count": len(anchor_names),
                "anchor_feature_names": list(anchor_names),
                "gemma_feature_count": len(FEATURE_NAMES),
                "gemma_feature_names": list(FEATURE_NAMES),
                "augmented_feature_count": len(anchor_names) + len(FEATURE_NAMES),
                "gemma_model": model,
                "payload_version": PAYLOAD_VERSION,
                "prompt_hash": prompt_hash(),
                "schema_hash": schema_hash(),
                "identifier_or_label_features": False,
                "observation_or_collection_volume_features": False,
            },
        )
        result["api"] = api_report
        if stage == "extract":
            write_json(
                output / "DATA_AUDIT.json",
                {**inspect_report, "source_access": audit.to_dict()},
            )
            _status(
                output,
                "complete",
                stage=stage,
                elapsed_seconds=time.monotonic() - started,
            )
            return result

        final_report, bundle, _y = train_experiment(
            data_root=resolved_data,
            output=output,
            audit=audit,
            subjects=subjects,
            anchor=anchor,
            anchor_names=anchor_names,
            gemma_features=gemma_features,
            profile=profile,
            seed=seed,
            model=model,
            salt=artifact_salt,
        )
        result["final_report"] = final_report
        if historical_eval:
            validation_client = GemmaFeatureClient(
                ClientConfig(
                    model=model,
                    api_key_env=api_key_env,
                    min_interval_seconds=min_interval_seconds,
                    offline=(offline or stage == "train"),
                ),
                cache,
            )
            result["historical_validation"] = historical_validation(
                data_root=resolved_data,
                output=output,
                audit=audit,
                client=validation_client,
                bundle=bundle,
                training_subject_ids=subjects,
                salt=artifact_salt,
            )
        write_json(
            output / "DATA_AUDIT.json",
            {
                **inspect_report,
                "training_labels_opened_after_feature_extraction": True,
                "source_access": audit.to_dict(),
                "historical_validation_evaluated": bool(historical_eval),
            },
        )
        write_json(
            output / "LEAKAGE_AUDIT.json",
            {
                "direct_leakage_checks": {
                    "subject_level_oof": True,
                    "fold_train_test_subject_overlap": 0,
                    "preprocessing_fit_on_fold_training_only": True,
                    "subject_label_or_diagnosis_sent_to_api": False,
                    "subject_identifier_sent_to_api": False,
                    "diagnosis_admin_identifier_features": False,
                    "observation_or_collection_volume_features": False,
                    "historical_validation_subject_overlap_checked": bool(
                        historical_eval
                    ),
                    "historical_labels_opened_after_prediction_freeze": bool(
                        historical_eval
                    ),
                },
                "allowed_non_nested_optimism": {
                    "candidate_selected_on_same_repeated_oof": True,
                    "blend_selected_on_same_repeated_oof": True,
                    "reported_explicitly": True,
                },
                "source_access": audit.to_dict(),
            },
        )
        write_json(
            output / "TRAINING_COMPLETE.json",
            {
                "status": "complete",
                "experiment": EXPERIMENT_NAME,
                "profile": profile.name,
                "final_report": str(output / "FINAL_REPORT.json"),
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        _status(
            output,
            "complete",
            stage=stage,
            elapsed_seconds=time.monotonic() - started,
            final_report=str(output / "FINAL_REPORT.json"),
        )
        print("Complete:", output / "FINAL_REPORT.json")
        return result
    except Exception as error:
        _status(
            output,
            "failed",
            stage=stage,
            elapsed_seconds=time.monotonic() - started,
            error_type=type(error).__name__,
            error=str(error)[:1000],
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Task-aware hosted-Gemma cognitive features for CN vs MCI+Dem; "
            "candidate selection uses ROC-AUC only."
        )
    )
    parser.add_argument(
        "--stage", choices=("inspect", "extract", "train", "all"), default="all"
    )
    parser.add_argument(
        "--profile", choices=tuple(PROFILES), default="default"
    )
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--cache-dir")
    parser.add_argument("--model", default=os.environ.get("BGCFA_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--api-key-env",
        default=os.environ.get("BGCFA_API_KEY_ENV", DEFAULT_API_KEY_ENV),
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--historical-eval", action="store_true")
    parser.add_argument(
        "--min-interval-seconds",
        type=float,
        default=float(os.environ.get("BGCFA_MIN_INTERVAL_SECONDS", "13")),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-install", action="store_true")
    return parser


def _strip_jupyter_arguments(argv: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    index = 0
    values = list(map(str, argv))
    while index < len(values):
        token = values[index]
        if token in {"-f", "--f"} and index + 1 < len(values):
            path = Path(values[index + 1])
            if path.name.startswith("kernel-") and path.suffix == ".json":
                index += 2
                continue
        if token.startswith(("-f=", "--f=")):
            path = Path(token.split("=", 1)[1])
            if path.name.startswith("kernel-") and path.suffix == ".json":
                index += 1
                continue
        cleaned.append(token)
        index += 1
    return cleaned


def notebook_argv(environ: Mapping[str, str] | None = None) -> list[str]:
    environment = os.environ if environ is None else environ
    return shlex.split(
        str(environment.get("BGCFA_ARGS", "--stage all --profile default"))
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    namespace: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw = list(sys.argv[1:] if argv is None else argv)
    cleaned = _strip_jupyter_arguments(raw)
    if argv is None and not cleaned and os.environ.get("BGCFA_ARGS"):
        cleaned = notebook_argv()
    args = build_parser().parse_args(cleaned)
    return run_pipeline(
        namespace=namespace,
        stage=args.stage,
        profile_name=args.profile,
        data_root=args.data_root,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        model=args.model,
        api_key_env=args.api_key_env,
        offline=args.offline,
        historical_eval=args.historical_eval,
        min_interval_seconds=args.min_interval_seconds,
        seed=args.seed,
        skip_install=args.skip_install,
    )


if __name__ == "__main__":
    main()
