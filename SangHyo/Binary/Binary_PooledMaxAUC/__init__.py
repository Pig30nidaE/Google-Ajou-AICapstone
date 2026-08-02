"""Maximum-ROC-AUC binary experiment on the pooled 174-subject cohort.

Task: ``CN = 0`` vs ``MCI or Dem = 1``.

Two hard constraints from the request:

1. ``run.py`` is the single external entry point.
2. Only **direct** data leakage must be avoided.  Selection optimism
   (non-nested candidate / hyperparameter / ensemble-weight choice on the same
   repeated OOF that is reported) is explicitly permitted and is disclosed in
   every artifact instead of being hidden.

What "direct leakage" means here, and is enforced in ``leakage.py``:

* a subject never appears in both sides of a fold;
* the held-out subject's own label never enters imputation, scaling,
  winsorization, feature screening, model fitting or score normalization;
* diagnosis / administrative / identifier columns never become features;
* score normalization uses the training fold's ECDF as reference, never the
  held-out batch's own ranks (no transductive normalization).

What is deliberately *allowed*, and therefore must never be reported as a
clean generalization estimate:

* the 174-subject pooled cohort merges the official Training 141 and the
  historical Validation 33, so there is no untouched hold-out left;
* candidates, hyperparameters and ensemble weights are chosen on the same
  repeated OOF that produces the headline number.

See ``README_KO.md`` section "성능 해석 규칙" before quoting any number.
"""

from __future__ import annotations

__all__ = ["EXPERIMENT_NAME", "POOLED_COHORT_SIZE", "FEATURE_CONTRACT_VERSION"]

EXPERIMENT_NAME = "Binary_PooledMaxAUC"
POOLED_COHORT_SIZE = 174
FEATURE_CONTRACT_VERSION = "bpm-features-1"
