"""Wearable-only (MMSE-free) CN vs MCI+DEM binary classification.

Reproduces the leakage-safe Conv1D+BiLSTM recipe that was the only prior binary
model with meaningful subject-level accuracy, and combines it with a robust
tabular ensemble.  See README_KO.md for the honest performance context.
"""
