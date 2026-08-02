"""Leakage-conscious dementia screening research pipeline.

The package predicts ``CN + MCI = 0`` versus ``Dem = 1`` at subject level.
Training is intentionally unavailable from import side effects; use
``run.py`` with the explicit training acknowledgement described in README_KO.md.
"""

from .config import ExperimentConfig, make_config

__all__ = ["ExperimentConfig", "make_config"]

