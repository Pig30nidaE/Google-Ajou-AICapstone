"""google-genai client for one cached, global Gemma feature program.

The client never receives subject rows. Its complete variable input is the
label-neutral ``PrimitiveSpec`` catalog rendered by :mod:`prompts`. A successful
response is validated as a non-executable DSL before it is cached.

Cache identity binds the static prompt, primitive catalog, response schema,
model, and generation settings. The cache stores only the canonical program and
a deliberately small manifest (token counts, successful attempt, model, and
hashes). It never stores the API key, environment contents, raw response,
provider error body, or patient data.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from .program_schema import (
    PrimitiveSpec,
    canonical_json,
    canonicalize_program,
    catalog_hash,
    program_hash,
    response_schema,
    schema_hash,
)
from .prompts import SYSTEM_PROMPT, prompt_hash, render_user_prompt

__all__ = [
    "FeatureProgramClient",
    "ProgramClient",
    "ProgramClientConfig",
    "ProgramResult",
    "generate_global_program",
]

_CACHE_FORMAT = 1
_RETRY_AFTER_RE = re.compile(
    r"(?:retry\s+(?:after|in)|retry-after[:=]?)\s*([0-9]+(?:\.[0-9]+)?)\s*s?",
    re.IGNORECASE,
)
_MANIFEST_KEYS = frozenset(
    {
        "model",
        "attempt",
        "prompt_tokens",
        "output_tokens",
        "thinking_tokens",
        "total_tokens",
        "prompt_hash",
        "catalog_hash",
        "schema_hash",
        "model_hash",
        "request_hash",
        "program_hash",
    }
)


@dataclass(frozen=True)
class ProgramClientConfig:
    """Minimal runtime configuration for the single global API request."""

    model: str = "gemma-4-31b-it"
    api_key_env: str = "GEMINI_API_KEY"
    temperature: float = 0.0
    max_output_tokens: int = 8192
    thinking_level: str | None = "minimal"
    thinking_budget: int | None = None
    timeout_seconds: float = 120.0
    max_retries: int = 5
    initial_backoff_seconds: float = 2.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 60.0
    allow_thinking_compatibility_fallback: bool = True
    offline: bool = False
    regenerate_program: bool = False


@dataclass(frozen=True)
class ProgramResult:
    """Canonical program plus privacy-minimal generation provenance."""

    program: dict[str, Any]
    manifest: dict[str, Any]
    from_cache: bool
    cache_path: Path


class _InvalidProgramResponse(RuntimeError):
    """The server answered, but not with a usable strict program."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _status_code(error: BaseException) -> int | None:
    for attribute in ("status_code", "status", "code"):
        raw = getattr(error, attribute, None)
        if callable(raw):
            try:
                raw = raw()
            except Exception:  # noqa: BLE001 - provider-specific accessor
                raw = None
        raw = getattr(raw, "value", raw)
        if isinstance(raw, int):
            return int(raw)
        if isinstance(raw, str):
            match = re.search(r"\b([45][0-9]{2})\b", raw)
            if match:
                return int(match.group(1))
    match = re.search(r"\b([45][0-9]{2})\b", f"{error}")
    return int(match.group(1)) if match else None


def _is_retryable(error: BaseException) -> bool:
    status = _status_code(error)
    if status == 429 or (status is not None and 500 <= status <= 599):
        return True
    if status is not None:
        return False
    text = f"{type(error).__name__}: {error}".lower()
    return any(
        marker in text
        for marker in (
            "timeout",
            "timed out",
            "connection",
            "temporarily unavailable",
            "resource exhausted",
            "rate limit",
        )
    )


def _is_thinking_capability_error(error: BaseException) -> bool:
    if _status_code(error) != 400:
        return False
    text = f"{type(error).__name__}: {error}".lower()
    thinking_marker = any(
        marker in text
        for marker in ("thinking", "thought", "thinking_config", "thinking level")
    )
    capability_marker = any(
        marker in text
        for marker in (
            "unsupported",
            "not supported",
            "invalid argument",
            "extra",
            "unknown field",
            "not available",
        )
    )
    return thinking_marker and capability_marker


def _server_retry_delay(error: BaseException) -> float | None:
    match = _RETRY_AFTER_RE.search(f"{error}")
    if not match:
        return None
    try:
        return max(0.0, min(float(match.group(1)) * 1.05, 60.0))
    except ValueError:
        return None


class FeatureProgramClient:
    """Generate at most one global program per exact request fingerprint."""

    def __init__(
        self,
        config: ProgramClientConfig,
        *,
        cache_root: str | Path,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.cache_root = Path(cache_root).expanduser()
        self._logger = logger
        self._client: Any | None = None
        self._client_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._validate_config()

    def _validate_config(self) -> None:
        if not str(self.config.model).strip():
            raise ValueError("program model must not be empty")
        if not str(self.config.api_key_env).strip():
            raise ValueError("api_key_env must not be empty")
        if float(self.config.temperature) != 0.0:
            raise ValueError(
                "global feature-program generation is frozen at temperature=0"
            )
        if int(self.config.max_output_tokens) < 1:
            raise ValueError("max_output_tokens must be positive")
        if int(self.config.max_retries) < 0:
            raise ValueError("max_retries must be non-negative")
        if float(self.config.timeout_seconds) <= 0:
            raise ValueError("timeout_seconds must be positive")
        if (
            self.config.thinking_level is not None
            and self.config.thinking_budget is not None
        ):
            raise ValueError(
                "Set only one of thinking_level and thinking_budget; null is omitted"
            )
        if self.config.thinking_level is not None:
            level = str(self.config.thinking_level).strip().lower()
            if level not in {"minimal", "low", "medium", "high"}:
                raise ValueError(
                    "thinking_level must be minimal|low|medium|high|null"
                )
        if (
            self.config.thinking_budget is not None
            and int(self.config.thinking_budget) < -1
        ):
            raise ValueError("thinking_budget must be >= -1 or null")
        if bool(getattr(self.config, "offline", False)) and bool(
            getattr(self.config, "regenerate_program", False)
        ):
            raise ValueError(
                "offline and regenerate_program cannot both be enabled"
            )

    def _log(self, message: str) -> None:
        # Messages are intentionally constructed locally and never include the
        # provider exception body or any environment value.
        if self._logger is not None:
            self._logger(message)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            environment_name = str(self.config.api_key_env).strip()
            api_key = os.environ.get(environment_name, "").strip()
            if not api_key:
                raise RuntimeError(
                    f"Environment variable {environment_name} is empty; set the "
                    "Colab secret before the program stage."
                )
            try:
                from google import genai  # type: ignore
            except ModuleNotFoundError as error:  # pragma: no cover - environment
                raise ModuleNotFoundError(
                    "google-genai is required for live program generation"
                ) from error
            # The key is passed directly to the SDK and is never stored as our
            # own attribute or included in a cache, manifest, prompt, or log.
            self._client = genai.Client(api_key=api_key)
            return self._client

    def _generation_material(self) -> dict[str, Any]:
        return {
            "temperature": 0.0,
            "max_output_tokens": int(self.config.max_output_tokens),
            "thinking_level": (
                None
                if self.config.thinking_level is None
                else str(self.config.thinking_level).strip().lower()
            ),
            "thinking_budget": self.config.thinking_budget,
        }

    def _request_identity(
        self, catalog: Sequence[PrimitiveSpec]
    ) -> dict[str, str]:
        hashes = {
            "prompt_hash": prompt_hash(),
            "catalog_hash": catalog_hash(catalog),
            "schema_hash": schema_hash(catalog),
            "model_hash": _sha256_text(str(self.config.model).strip()),
        }
        request_material = {
            "cache_format": _CACHE_FORMAT,
            **hashes,
            "generation": self._generation_material(),
        }
        hashes["request_hash"] = _sha256_text(canonical_json(request_material))
        return hashes

    def _cache_path(self, request_hash: str) -> Path:
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(self.config.model))
        return (
            self.cache_root
            / "gemma_feature_program"
            / safe_model
            / f"{request_hash}.json"
        )

    def _read_cache(
        self,
        path: Path,
        catalog: Sequence[PrimitiveSpec],
        identity: Mapping[str, str],
    ) -> ProgramResult | None:
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping) or set(payload) != {
                "program",
                "manifest",
            }:
                return None
            manifest = payload["manifest"]
            if not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_KEYS:
                return None
            if str(manifest["model"]) != str(self.config.model):
                return None
            for key in (
                "prompt_hash",
                "catalog_hash",
                "schema_hash",
                "model_hash",
                "request_hash",
            ):
                if str(manifest[key]) != identity[key]:
                    return None
            for key in (
                "attempt",
                "prompt_tokens",
                "output_tokens",
                "thinking_tokens",
                "total_tokens",
            ):
                value = manifest[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    return None
            if int(manifest["attempt"]) < 1:
                return None
            canonical = canonicalize_program(payload["program"], catalog)
            if program_hash(canonical, catalog) != str(manifest["program_hash"]):
                return None
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None
        return ProgramResult(
            program=canonical,
            manifest=dict(manifest),
            from_cache=True,
            cache_path=path,
        )

    @staticmethod
    def _write_cache(
        path: Path, program: Mapping[str, Any], manifest: Mapping[str, Any]
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {"program": program, "manifest": manifest},
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _thinking_config(self, types: Any) -> Any | None:
        level = self.config.thinking_level
        budget = self.config.thinking_budget
        if level is None and budget is None:
            return None
        thinking_class = getattr(types, "ThinkingConfig", None)
        if thinking_class is None:
            raise RuntimeError(
                "Installed google-genai cannot express thinking_config; upgrade "
                "google-genai or explicitly configure both thinking fields as null."
            )
        if level is not None:
            enum_class = getattr(types, "ThinkingLevel", None)
            enum_value = getattr(
                enum_class, str(level).strip().upper(), None
            )
            if enum_value is None:
                raise RuntimeError(
                    "Installed google-genai does not provide the configured "
                    "ThinkingLevel; upgrade google-genai."
                )
            return thinking_class(thinking_level=enum_value)
        return thinking_class(thinking_budget=int(budget))

    def _call_api(
        self,
        *,
        catalog: Sequence[PrimitiveSpec],
        include_thinking: bool,
    ) -> tuple[str, dict[str, int], str]:
        try:
            from google.genai import types  # type: ignore
        except ModuleNotFoundError as error:  # pragma: no cover - environment
            raise ModuleNotFoundError(
                "google-genai is required for live program generation"
            ) from error

        config_kwargs: dict[str, Any] = {
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0.0,
            "max_output_tokens": int(self.config.max_output_tokens),
            "response_mime_type": "application/json",
            "response_schema": response_schema(catalog),
            "http_options": types.HttpOptions(
                timeout=int(float(self.config.timeout_seconds) * 1000)
            ),
        }
        if include_thinking:
            thinking = self._thinking_config(types)
            if thinking is not None:
                config_kwargs["thinking_config"] = thinking
        response = self._get_client().models.generate_content(
            model=str(self.config.model),
            contents=render_user_prompt(catalog),
            config=types.GenerateContentConfig(**config_kwargs),
        )
        usage = getattr(response, "usage_metadata", None)
        token_counts = {
            "prompt_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
            "output_tokens": int(
                getattr(usage, "candidates_token_count", 0) or 0
            ),
            "thinking_tokens": int(
                getattr(usage, "thoughts_token_count", 0) or 0
            ),
            "total_tokens": int(getattr(usage, "total_token_count", 0) or 0),
        }
        candidates = getattr(response, "candidates", None) or []
        finish_reason = (
            str(getattr(candidates[0], "finish_reason", "") or "")
            if candidates
            else ""
        )
        text = getattr(response, "text", None)
        return ("" if text is None else str(text)), token_counts, finish_reason

    def generate_or_load(
        self, catalog: Sequence[PrimitiveSpec]
    ) -> ProgramResult:
        """Return the one cached program or issue one logical generation request.

        429 and 5xx failures use bounded exponential backoff. If a model returns
        HTTP 400 specifically because it cannot accept ``thinking_config``, one
        compatibility attempt is made without that field. No other 400 response
        is retried or silently changed.
        """

        identity = self._request_identity(catalog)
        path = self._cache_path(identity["request_hash"])
        with self._generation_lock:
            if not bool(getattr(self.config, "regenerate_program", False)):
                cached = self._read_cache(path, catalog, identity)
                if cached is not None:
                    return cached
            else:
                self._log(
                    "[program] explicit regenerate_program=true; generating one "
                    "replacement for the exact global-program cache key"
                )
            if bool(self.config.offline):
                raise RuntimeError(
                    "offline mode is enabled and this exact global program is not cached"
                )

            thinking_requested = (
                self.config.thinking_level is not None
                or self.config.thinking_budget is not None
            )
            include_thinking = thinking_requested
            thinking_fallback_used = False
            transient_retries = 0
            attempt = 0
            backoff = max(0.0, float(self.config.initial_backoff_seconds))
            while True:
                attempt += 1
                try:
                    text, tokens, finish_reason = self._call_api(
                        catalog=catalog,
                        include_thinking=include_thinking,
                    )
                    if "MAX_TOKENS" in finish_reason.upper():
                        raise _InvalidProgramResponse(
                            "global program response reached max_output_tokens"
                        )
                    if not text.strip():
                        raise _InvalidProgramResponse(
                            "global program response body was empty"
                        )
                    try:
                        decoded = json.loads(text)
                        canonical = canonicalize_program(decoded, catalog)
                    except (json.JSONDecodeError, TypeError, ValueError) as error:
                        raise _InvalidProgramResponse(
                            "global program response violated the strict JSON DSL"
                        ) from error
                    break
                except Exception as error:  # noqa: BLE001 - classify SDK errors
                    if (
                        include_thinking
                        and not thinking_fallback_used
                        and bool(
                            getattr(
                                self.config,
                                "allow_thinking_compatibility_fallback",
                                True,
                            )
                        )
                        and _is_thinking_capability_error(error)
                    ):
                        include_thinking = False
                        thinking_fallback_used = True
                        self._log(
                            "[program] model rejected thinking_config; using the "
                            "single compatibility attempt without thinking"
                        )
                        continue
                    if (
                        _is_retryable(error)
                        and transient_retries < int(self.config.max_retries)
                    ):
                        transient_retries += 1
                        status = _status_code(error)
                        delay = _server_retry_delay(error)
                        if delay is None:
                            delay = min(
                                backoff,
                                max(0.0, float(self.config.max_backoff_seconds)),
                                60.0,
                            )
                        self._log(
                            "[program] transient API failure "
                            f"(status={status or 'transport'}, "
                            f"retry={transient_retries}/"
                            f"{int(self.config.max_retries)})"
                        )
                        if delay > 0:
                            time.sleep(delay)
                        backoff = max(
                            backoff,
                            max(0.0, float(self.config.initial_backoff_seconds)),
                        ) * max(1.0, float(self.config.backoff_multiplier))
                        continue
                    status = _status_code(error)
                    if isinstance(error, _InvalidProgramResponse):
                        raise RuntimeError(
                            "Gemma returned an unusable global feature program; "
                            "no cache was written"
                        ) from None
                    raise RuntimeError(
                        "Global feature-program generation failed "
                        f"(status={status or 'transport'}, "
                        f"attempts={attempt}); no cache was written"
                    ) from None

            manifest: dict[str, Any] = {
                "model": str(self.config.model),
                "attempt": int(attempt),
                "prompt_tokens": int(tokens["prompt_tokens"]),
                "output_tokens": int(tokens["output_tokens"]),
                "thinking_tokens": int(tokens["thinking_tokens"]),
                "total_tokens": int(tokens["total_tokens"]),
                "prompt_hash": identity["prompt_hash"],
                "catalog_hash": identity["catalog_hash"],
                "schema_hash": identity["schema_hash"],
                "model_hash": identity["model_hash"],
                "request_hash": identity["request_hash"],
                "program_hash": program_hash(canonical, catalog),
            }
            self._write_cache(path, canonical, manifest)
            return ProgramResult(
                program=canonical,
                manifest=manifest,
                from_cache=False,
                cache_path=path,
            )


# Short alias for callers that do not need the more descriptive class name.
ProgramClient = FeatureProgramClient


def generate_global_program(
    config: ProgramClientConfig,
    *,
    cache_root: str | Path,
    catalog: Sequence[PrimitiveSpec],
    logger: Callable[[str], None] | None = None,
) -> ProgramResult:
    """Convenience wrapper around :class:`FeatureProgramClient`."""

    return FeatureProgramClient(
        config,
        cache_root=cache_root,
        logger=logger,
    ).generate_or_load(catalog)
