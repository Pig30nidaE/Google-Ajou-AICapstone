"""Leakage-safe dual-track experiment for maximizing subject ROC-AUC.

The package exposes two strictly separated tracks:

``mmse``
    MMSE score items are allowed.  Diagnosis and administrative fields in the
    MMSE source file are never read.

``wearable``
    Activity and Sleep only.  No CognitiveFunction path may be opened.
"""

EXPERIMENT_NAME = "Binary_Google_ROCAUC_Champion"

__all__ = ["EXPERIMENT_NAME"]
