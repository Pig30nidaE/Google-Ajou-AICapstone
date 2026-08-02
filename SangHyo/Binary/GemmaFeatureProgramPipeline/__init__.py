"""Gemma-generated, executable wearable feature-program pipeline.

The package keeps the repository's primary binary task fixed:
CN=0 versus MCI or Dem=1.  Gemma proposes a global, label-free feature
program once; Python validates and applies that program identically inside
each fold.
"""

from __future__ import annotations

EXPERIMENT_NAME = "GemmaFeatureProgramPipeline"
PACKAGE_VERSION = "1.0.0"
PROGRAM_VERSION = "llmfp-v1"
CATALOG_VERSION = "wearable-primitives-v1"

__all__ = [
    "CATALOG_VERSION",
    "EXPERIMENT_NAME",
    "PACKAGE_VERSION",
    "PROGRAM_VERSION",
]
