"""
Final model (V4): 2-model hierarchy for CN / MCI / DEM classification.

  Model 1: DEM vs (CN+MCI)   -- 10 activity features, RBF Kernel Ridge
  Model 2: CN vs MCI         -- 10 sleep/HR features (fixed "compact" set), LDA
  Combined: CN = (1-d)(1-q), MCI = (1-d)*q, DEM = d
  Decision: argmax(log(score) + class offset), offset tuned on Training-only
            data via lifelog_v3.selection.decision_policy (see notes below).

Dependency: this script imports `lifelog_v2` and `lifelog_v3` from the
`my_lab` repository (github.com/Pig30nidaE/my_lab). It is NOT
self-contained -- place this file's parent directory on PYTHONPATH
together with a checkout of `my_lab`, e.g.:

    git clone https://github.com/Pig30nidaE/my_lab.git
    cd my_lab
    cp <this_file> ./final_model.py
    python final_model.py --data-root /path/to/aihub_data

See README.md in this folder for the full methodology, feature list,
performance numbers (both Selection-Validation and honest nested-CV), and
limitations. Do not report the Selection-Validation numbers below as
generalization performance without the nested-CV numbers alongside them.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from lifelog_v2 import feature_engineering as fe
from lifelog_v2.metrics import measure
from lifelog_v3.data import load_split
from lifelog_v3.models import Spec, fit_candidate
from lifelog_v3.selection import decision_policy

# The exact-duplicate legacy column removed before fitting (correlation 1.0,
# 100% identical values to v3__activity__state_code_1_longest_minutes__median).
DUPLICATE_COLUMN = 'activity__state_code_1_longest_minutes__median'

DEM_SPEC = Spec(
    task='dem', family='rbf_ridge', view='activity', top_k=10, selection='rfe',
    transform='quantile', correlation=1.0, pca=0, balance=0.0, augmentation='smote',
    seed=2026, params={'alpha': 0.0018005071512845635, 'gamma': 0.0013102849275700784},
)

CNMCI_SPEC = Spec(
    task='cn_mci', family='lda', view='compact', top_k=10, selection='anova',
    transform='signedlog', correlation=1.0, pca=0, balance=0.0, augmentation='smote',
    seed=2026, params={'shrinkage': 0.6},
)


class Pool:
    """Minimal container matching the (labels, representations) shape fit_candidate expects."""
    def __init__(self, labels, representations):
        self.labels = labels
        self.representations = representations


def _drop_duplicate_column(representation):
    from lifelog_v2.data import Representation
    keep = [c for c in representation.X.columns if c != DUPLICATE_COLUMN]
    return Representation(representation.X[keep], representation.groups, representation.meta.loc[keep])


def combine_hierarchy(d_score, q_score):
    """d = P(DEM), q = P(MCI | not DEM) -> normalized (CN, MCI, DEM) scores."""
    d = np.clip(d_score, 1e-9, 1 - 1e-9)
    q = np.clip(q_score, 1e-9, 1 - 1e-9)
    p = np.column_stack([(1 - d) * (1 - q), (1 - d) * q, d])
    return p / p.sum(axis=1, keepdims=True)


def fit_final_model(train_dataset):
    """Fit both branches on Training subjects only. Returns (dem_fitted, cnmci_fitted)."""
    reps = dict(train_dataset.representations)
    reps['subject'] = _drop_duplicate_column(reps['subject'])
    pool = Pool(train_dataset.labels, reps)
    dem_fitted = fit_candidate(pool, DEM_SPEC)
    cnmci_fitted = fit_candidate(pool, CNMCI_SPEC)
    return dem_fitted, cnmci_fitted, reps


def predict(dem_fitted, cnmci_fitted, representations, subjects):
    d = dem_fitted.predict(representations, subjects)[:, 1]
    q = cnmci_fitted.predict(representations, subjects)[:, 1]
    return combine_hierarchy(d, q)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=None)
    args = parser.parse_args()

    _, paths = fe.resolve_paths(args.data_root)
    train = load_split(paths['train'])
    val = load_split(paths['val'])
    assert not set(train.subjects).intersection(val.subjects)

    dem_fitted, cnmci_fitted, reps_train = fit_final_model(train)
    reps_val = dict(val.representations)
    reps_val['subject'] = _drop_duplicate_column(reps_val['subject'])

    print('DEM branch features:', dem_fitted.processor.columns)
    print('CN/MCI branch features:', cnmci_fitted.processor.columns)

    scores = predict(dem_fitted, cnmci_fitted, reps_val, val.subjects)

    # Decision offset: tuned against Validation labels here for illustration.
    # This makes the metrics printed below "Selection-Validation" performance,
    # NOT an independent generalization estimate -- see README section 6.
    offsets, metrics = decision_policy(val.y, scores)
    print('\nSelection-Validation metrics (offset tuned on Validation labels):')
    print(json.dumps(metrics, indent=2))
    print('\noffsets:', offsets.tolist())
    print('\nSee README.md for the honest nested-CV estimate to quote alongside this number.')


if __name__ == '__main__':
    main()
