"""Paper-derived re-implementation of Cheon et al. (2025) with three validation designs.

R  reproduction          : paper-faithful record-level 5-fold CV
G  subject_independent   : same recipe, StratifiedGroupKFold by subject, learned steps train-fold only
N  nested_cv             : G + inner StratifiedGroupKFold for feature-count / hyperparameter selection
"""
__version__ = "0.1.0"
