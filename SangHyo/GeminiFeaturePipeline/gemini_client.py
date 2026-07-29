"""Gemini structured-output client: cache, retries, validation, dry-run.

Nothing here calls the API unless the caller explicitly asks for it *and*
``dry_run`` is off *and* the request is not already cached.  The four execution
modes are:

``dry_run``   build and validate payloads, report size/cost estimates, no call.
``offline``   use cached answers only; a cache miss is an error, not a call.
``normal``    call the API for cache misses only.
``retry_failed`` additionally re-issue calls for subjects whose cached entry is
              a stored failure.

Cache key = SHA-256 over (payload, system prompt, user template, JSON schema,
model name, generation config).  Changing any of them produces a new key, so a
cached answer can never be silently mixed with a different contract
(<data_leakage_rules> item 15).  The file name is that hash and nothing else, so
the cache layout carries no semantic information about the subject.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import threading
import time
from typing import Any, Mapping, Sequence

from . import PAYLOAD_VERSION, SCHEMA_VERSION
from .config import GeminiConfig
from .guards import assert_payload_is_label_free, assert_payload_is_mmse_free
from .payload import canonical_json, payload_hash, payload_size_bytes
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, prompt_hash, render_user_prompt
from .schema import FEATURE_NAMES, response_schema, schema_hash, validate_feature_payload

__all__ = ["GeminiResult", "GeminiFeatureExtractor", "ExtractionSummary"]

CACHE_FORMAT = 1
_RETRYABLE_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "resource_exhausted",
    "resource exhausted",
    "unavailable",
    "deadline",
    "timeout",
    "timed out",
    "connection",
    "temporarily",
    "rate limit",
    "overloaded",
    "internal error",
)
# The free tier returns a message like "...Please retry in 55.451847999s." for a
# 429 RESOURCE_EXHAUSTED. A fixed exponential backoff (a few seconds, doubling)
# is almost always shorter than that, so every retry keeps hitting the same
# still-exhausted quota window until max_retries runs out. Honouring the
# server's own number fixes that; the multiplier and ceiling below are just a
# safety margin/cap in case the text is ever malformed or absurdly large.
_RETRY_AFTER_PATTERN = re.compile(r"retry\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*s", re.I)
_RETRY_AFTER_SAFETY_MARGIN = 1.05
_RETRY_AFTER_HARD_CEILING_SECONDS = 180.0


def _server_requested_retry_delay(error: BaseException) -> float | None:
    match = _RETRY_AFTER_PATTERN.search(f"{error}")
    if not match:
        return None
    try:
        seconds = float(match.group(1))
    except ValueError:
        return None
    return min(seconds * _RETRY_AFTER_SAFETY_MARGIN, _RETRY_AFTER_HARD_CEILING_SECONDS)


@dataclass(frozen=True)
class GeminiResult:
    subject_id: str
    subject_ref: str
    status: str  # cached | fresh | dry_run | failed | cache_miss
    features: dict[str, float] | None
    error: str | None
    record: Mapping[str, Any]


@dataclass
class ExtractionSummary:
    requested: int = 0
    cached: int = 0
    fresh: int = 0
    failed: int = 0
    cache_miss: int = 0
    dry_run: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    total_payload_bytes: int = 0
    estimated_cost_usd: float | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_subjects": self.requested,
            "served_from_cache": self.cached,
            "fresh_api_answers": self.fresh,
            "failed": self.failed,
            "cache_miss_offline": self.cache_miss,
            "dry_run_payloads": self.dry_run,
            "api_calls_executed": self.api_calls,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_payload_bytes": self.total_payload_bytes,
            "estimated_cost_usd": self.estimated_cost_usd,
            "errors": self.errors[:20],
        }


class _RateLimiter:
    """Minimum wall-clock spacing between API calls, shared by all workers."""

    def __init__(self, min_interval_seconds: float) -> None:
        self._interval = max(0.0, float(min_interval_seconds))
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self._interval
        if wait:
            time.sleep(wait)


def _is_retryable(error: BaseException) -> bool:
    text = f"{type(error).__name__}: {error}".lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


def _sdk_version() -> str:
    try:
        from importlib.metadata import version

        return str(version("google-genai"))
    except Exception:  # pragma: no cover - environment dependent
        return "unknown"


def _strip_code_fence(text: str) -> str:
    stripped = str(text).strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()
    return stripped


class GeminiFeatureExtractor:
    """Turns subject payloads into validated feature dictionaries."""

    def __init__(
        self,
        config: GeminiConfig,
        *,
        cache_root: str | Path,
        logger=print,
    ) -> None:
        self.config = config
        self.schema = response_schema()
        self.schema_hash = schema_hash()
        self.prompt_hash = prompt_hash()
        self.cache_dir = (
            Path(cache_root).expanduser()
            / "gemini"
            / str(config.model).replace("/", "_")
            / self.schema_hash[:12]
        )
        self._logger = logger
        self._client = None
        self._client_lock = threading.Lock()
        self._limiter = _RateLimiter(config.min_interval_seconds)

    # -- cache -------------------------------------------------------------- #
    def generation_config(self) -> dict[str, Any]:
        return {
            "temperature": float(self.config.temperature),
            "top_p": float(self.config.top_p),
            "max_output_tokens": int(self.config.max_output_tokens),
            "seed": self.config.response_seed,
            "response_mime_type": "application/json",
        }

    def request_fingerprint(self, payload: Mapping[str, Any]) -> str:
        material = {
            "cache_format": CACHE_FORMAT,
            "payload": payload,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt_template": USER_PROMPT_TEMPLATE,
            "schema": self.schema,
            "schema_version": SCHEMA_VERSION,
            "payload_version": PAYLOAD_VERSION,
            "model": self.config.model,
            "generation_config": self.generation_config(),
        }
        canonical = json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def cache_path(self, fingerprint: str) -> Path:
        return self.cache_dir / f"{fingerprint}.json"

    def read_cache(self, fingerprint: str) -> dict[str, Any] | None:
        path = self.cache_path(fingerprint)
        if not path.is_file():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if record.get("schema_hash") != self.schema_hash or record.get("prompt_hash") != self.prompt_hash:
            # Contract drifted: treat as a miss rather than reusing stale features.
            return None
        return record

    def write_cache(self, fingerprint: str, record: Mapping[str, Any]) -> None:
        path = self.cache_path(fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)

    # -- API ---------------------------------------------------------------- #
    def _get_client(self):
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            import os

            key = os.environ.get(self.config.api_key_env, "").strip()
            if not key:
                raise RuntimeError(
                    f"Environment variable {self.config.api_key_env} is empty. "
                    "Set it in Colab before running the gemini stage; the key is never "
                    "read from a file or hard-coded."
                )
            try:
                from google import genai  # type: ignore
            except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
                raise ModuleNotFoundError(
                    "google-genai is not installed. Install it with "
                    "`pip install -r requirements_colab.txt`."
                ) from error
            self._client = genai.Client(api_key=key)
            return self._client

    def list_available_models(self) -> dict[str, Any]:
        """Ask the API which models THIS key may actually call.

        Google no longer publishes per-model, per-project rate limits or
        availability (the docs redirect to the AI Studio console), and model
        IDs that are listed as "stable" in the public docs can still return
        404 "no longer available to new users" for a freshly created key.
        So the model list is discovered from the key itself rather than
        hard-coded from documentation.  ``models.list`` is a metadata call and
        does not consume the ``generate_content`` quota.
        """

        client = self._get_client()
        discovered: list[dict[str, Any]] = []
        for model in client.models.list():
            actions = [str(action) for action in (getattr(model, "supported_actions", None) or [])]
            name = str(getattr(model, "name", "") or "")
            discovered.append(
                {
                    "name": name,
                    "id": name.removeprefix("models/"),
                    "display_name": str(getattr(model, "display_name", "") or ""),
                    "supported_actions": actions,
                    "input_token_limit": getattr(model, "input_token_limit", None),
                    "output_token_limit": getattr(model, "output_token_limit", None),
                }
            )
        # Older SDK builds omit supported_actions; treat an empty list as "unknown"
        # rather than as "does not support generateContent".
        usable = [
            entry
            for entry in discovered
            if not entry["supported_actions"] or "generateContent" in entry["supported_actions"]
        ]
        configured = str(self.config.model)
        configured_ids = {entry["id"] for entry in usable}
        return {
            "configured_model": configured,
            "configured_model_is_available": configured in configured_ids,
            "n_models_visible": len(discovered),
            "generate_content_models": sorted(entry["id"] for entry in usable),
            "models": sorted(usable, key=lambda entry: entry["id"]),
        }

    def _call_api(self, user_prompt: str) -> tuple[str, dict[str, Any]]:
        from google.genai import types  # type: ignore

        client = self._get_client()
        generation = self.generation_config()
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=generation["temperature"],
            top_p=generation["top_p"],
            max_output_tokens=generation["max_output_tokens"],
            seed=generation["seed"],
            response_mime_type="application/json",
            response_schema=self.schema,
            http_options=types.HttpOptions(
                timeout=int(float(self.config.timeout_seconds) * 1000)
            ),
        )
        self._limiter.acquire()
        response = client.models.generate_content(
            model=self.config.model, contents=user_prompt, config=config
        )
        usage = getattr(response, "usage_metadata", None)
        usage_payload = {
            "prompt_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
            "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
            "total_tokens": int(getattr(usage, "total_token_count", 0) or 0),
        }
        text = getattr(response, "text", None)
        if not text:
            raise ValueError("Gemini returned an empty response body")
        return str(text), usage_payload

    # -- one subject -------------------------------------------------------- #
    def _extract_one(self, subject_id: str, payload: Mapping[str, Any]) -> GeminiResult:
        subject_ref = str(payload.get("subject_ref", ""))
        # Final gate: nothing leaves this process without passing both guards.
        assert_payload_is_label_free(payload, context=f"gemini request[{subject_ref}]")
        assert_payload_is_mmse_free(payload, context=f"gemini request[{subject_ref}]")

        fingerprint = self.request_fingerprint(payload)
        cached = self.read_cache(fingerprint)
        if cached is not None:
            if cached.get("status") == "ok":
                return GeminiResult(
                    subject_id, subject_ref, "cached", dict(cached["features"]), None, cached
                )
            if not self.config.retry_failed:
                return GeminiResult(
                    subject_id, subject_ref, "failed", None, str(cached.get("error")), cached
                )

        base_record: dict[str, Any] = {
            "cache_format": CACHE_FORMAT,
            "fingerprint": fingerprint,
            "subject_ref": subject_ref,
            "model": self.config.model,
            "sdk": "google-genai",
            "sdk_version": _sdk_version(),
            "schema_version": SCHEMA_VERSION,
            "schema_hash": self.schema_hash,
            "prompt_hash": self.prompt_hash,
            "payload_version": PAYLOAD_VERSION,
            "payload_hash": payload_hash(payload),
            "payload_size_bytes": payload_size_bytes(payload),
            "generation_config": self.generation_config(),
            "input_payload": payload,
        }

        if self.config.dry_run:
            record = {**base_record, "status": "dry_run", "features": None, "error": None}
            return GeminiResult(subject_id, subject_ref, "dry_run", None, None, record)
        if self.config.offline:
            record = {
                **base_record,
                "status": "cache_miss",
                "features": None,
                "error": "offline mode and no cached answer for this exact request",
            }
            return GeminiResult(
                subject_id, subject_ref, "cache_miss", None, str(record["error"]), record
            )

        user_prompt = render_user_prompt(payload)
        attempts: list[dict[str, Any]] = []
        started = datetime.now(timezone.utc)
        clock = time.monotonic()
        backoff = float(self.config.initial_backoff_seconds)
        for attempt in range(1, int(self.config.max_retries) + 1):
            try:
                text, usage = self._call_api(user_prompt)
                parsed = json.loads(_strip_code_fence(text))
                features = validate_feature_payload(parsed)
            except Exception as error:  # noqa: BLE001 - recorded, classified, retried
                server_delay = _server_requested_retry_delay(error)
                attempts.append(
                    {
                        "attempt": attempt,
                        "error": f"{type(error).__name__}: {error}"[:800],
                        "retryable": _is_retryable(error),
                        "server_requested_retry_delay_seconds": server_delay,
                    }
                )
                last = attempt >= int(self.config.max_retries)
                if last:
                    record = {
                        **base_record,
                        "status": "failed",
                        "features": None,
                        "error": attempts[-1]["error"],
                        "attempts": attempts,
                        "request_started_utc": started.isoformat(),
                        "duration_seconds": round(time.monotonic() - clock, 3),
                    }
                    self.write_cache(fingerprint, record)
                    return GeminiResult(
                        subject_id, subject_ref, "failed", None, attempts[-1]["error"], record
                    )
                exponential = min(
                    float(self.config.max_backoff_seconds),
                    backoff * (1.0 + random.random() * 0.25),  # jitter
                )
                # A quota error's own retry-after is authoritative: retrying sooner
                # just re-hits the same still-exhausted window, and this may
                # legitimately exceed max_backoff_seconds (rate-limit waits are
                # commonly 30-60s+, longer than a generic transient-error backoff).
                sleep_for = max(exponential, server_delay) if server_delay is not None else exponential
                time.sleep(sleep_for)
                backoff = min(
                    float(self.config.max_backoff_seconds),
                    backoff * float(self.config.backoff_multiplier),
                )
                continue

            record = {
                **base_record,
                "status": "ok",
                "features": features,
                "error": None,
                "attempts": attempts,
                "n_attempts": attempt,
                "raw_response_text": text,
                "response_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "usage": usage,
                "request_started_utc": started.isoformat(),
                "duration_seconds": round(time.monotonic() - clock, 3),
            }
            self.write_cache(fingerprint, record)
            return GeminiResult(subject_id, subject_ref, "fresh", features, None, record)
        raise AssertionError("unreachable retry loop exit")

    # -- many subjects ------------------------------------------------------ #
    def extract(
        self, payloads: Mapping[str, Mapping[str, Any]], *, subject_order: Sequence[str] | None = None
    ) -> tuple[dict[str, GeminiResult], ExtractionSummary]:
        subjects = list(subject_order or sorted(payloads))
        if self.config.limit_subjects is not None:
            subjects = subjects[: int(self.config.limit_subjects)]
        summary = ExtractionSummary(requested=len(subjects))

        workers = max(1, int(self.config.max_concurrency))
        results: dict[str, GeminiResult] = {}
        if workers == 1 or self.config.dry_run or self.config.offline:
            produced = [self._extract_one(subject, payloads[subject]) for subject in subjects]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                produced = list(
                    pool.map(lambda subject: self._extract_one(subject, payloads[subject]), subjects)
                )

        for result in produced:
            results[result.subject_id] = result
            summary.total_payload_bytes += int(result.record.get("payload_size_bytes", 0))
            if result.status == "cached":
                summary.cached += 1
            elif result.status == "fresh":
                summary.fresh += 1
                summary.api_calls += 1 + len(result.record.get("attempts", []))
                usage = result.record.get("usage") or {}
                summary.prompt_tokens += int(usage.get("prompt_tokens", 0))
                summary.output_tokens += int(usage.get("output_tokens", 0))
            elif result.status == "failed":
                summary.failed += 1
                summary.api_calls += len(result.record.get("attempts", []))
                summary.errors.append(f"{result.subject_ref}: {result.error}")
            elif result.status == "cache_miss":
                summary.cache_miss += 1
                summary.errors.append(f"{result.subject_ref}: {result.error}")
            elif result.status == "dry_run":
                summary.dry_run += 1

        prices = (
            self.config.price_per_million_input_tokens,
            self.config.price_per_million_output_tokens,
        )
        if all(price is not None for price in prices):
            summary.estimated_cost_usd = round(
                summary.prompt_tokens / 1e6 * float(prices[0])
                + summary.output_tokens / 1e6 * float(prices[1]),
                6,
            )
        return results, summary

    # -- reporting ---------------------------------------------------------- #
    def dry_run_report(
        self, payloads: Mapping[str, Mapping[str, Any]], *, subject_order: Sequence[str] | None = None
    ) -> dict[str, Any]:
        """No API call: only payload construction, guards, sizes and destinations."""

        subjects = list(subject_order or sorted(payloads))
        if self.config.limit_subjects is not None:
            subjects = subjects[: int(self.config.limit_subjects)]
        sizes: list[int] = []
        prompt_characters: list[int] = []
        would_call: list[str] = []
        already_cached = 0
        for subject in subjects:
            payload = payloads[subject]
            assert_payload_is_label_free(payload, context=f"dry-run[{subject}]")
            assert_payload_is_mmse_free(payload, context=f"dry-run[{subject}]")
            canonical_json(payload)  # serialization must not raise
            sizes.append(payload_size_bytes(payload))
            prompt_characters.append(len(render_user_prompt(payload)) + len(SYSTEM_PROMPT))
            fingerprint = self.request_fingerprint(payload)
            cached = self.read_cache(fingerprint)
            if cached is not None and cached.get("status") == "ok":
                already_cached += 1
            else:
                would_call.append(fingerprint)
        return {
            "mode": "dry_run",
            "api_calls_executed": 0,
            "subjects_considered": len(subjects),
            "already_cached": already_cached,
            "requests_that_would_be_sent": len(would_call),
            "payload_bytes_total": int(sum(sizes)),
            "payload_bytes_median": int(sorted(sizes)[len(sizes) // 2]) if sizes else 0,
            "payload_bytes_max": int(max(sizes)) if sizes else 0,
            "prompt_characters_median": (
                int(sorted(prompt_characters)[len(prompt_characters) // 2])
                if prompt_characters
                else 0
            ),
            "approximate_prompt_tokens_total": int(sum(prompt_characters) / 4),
            "cache_directory": str(self.cache_dir),
            "model": self.config.model,
            "schema_hash": self.schema_hash,
            "prompt_hash": self.prompt_hash,
            "feature_names": list(FEATURE_NAMES),
        }
