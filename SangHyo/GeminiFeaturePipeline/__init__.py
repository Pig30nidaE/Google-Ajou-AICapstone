"""Gemini-as-feature-extractor pipeline for wearable lifelog cognitive screening.

Gemini is used **only** as a diagnosis-neutral structured feature extractor.  It
never receives a target label, a diagnosis string, or MMSE, and it never returns
a class, probability, or risk.  All supervised learning happens afterwards in
``models.py`` / ``evaluation.py``.

Public entry point: ``run.py`` (single CLI + ``base.ipynb`` launcher).
"""

from __future__ import annotations

__all__ = ["EXPERIMENT_NAME", "PAYLOAD_VERSION", "SCHEMA_VERSION"]

EXPERIMENT_NAME = "GeminiFeaturePipeline"
PAYLOAD_VERSION = "gfp-payload-1"
SCHEMA_VERSION = "gfp-schema-1"
