"""Shared fixtures.

The synthetic cohort mimics the real one's shape -- subjects with different
numbers of days, deliberate calendar gaps, a minority positive class -- without
touching ``Data/``, so the contract tests run anywhere.  The gaps matter: a
fixture with perfectly continuous dates would let a broken sequence builder pass.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.data import schema  # noqa: E402
from src.data.loader import LifelogData  # noqa: E402

FEATURES = tuple(schema.PAPER_FEATURES)


def _make_daily(n_subjects: int = 24, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows, subject_rows = [], []

    for i in range(n_subjects):
        subject = f"subj{i:03d}"
        label = int(i % 3 == 0)                       # ~1/3 positive, like the real 36%
        diagnosis = "MCI" if label else "CN"
        n_days = int(rng.integers(20, 40))

        # Every third subject gets a genuine calendar gap in the middle.
        offsets = list(range(n_days))
        if i % 3 == 0:
            offsets = offsets[: n_days // 2] + [d + 4 for d in offsets[n_days // 2 :]]

        start = pd.Timestamp("2020-10-17")
        for offset in offsets:
            date = start + pd.Timedelta(days=offset)
            rows.append(
                {
                    schema.SUBJECT_ID: subject,
                    schema.DATE_COL: date,
                    schema.LABEL_COL: label,
                    schema.DIAGNOSIS_COL: diagnosis,
                    schema.SPLIT_ORIGIN: "train" if i < int(n_subjects * 0.8) else "validation",
                    **{f: float(rng.normal(label, 1.0)) for f in FEATURES},
                }
            )
        subject_rows.append(
            {
                schema.SUBJECT_ID: subject,
                schema.DIAGNOSIS_COL: diagnosis,
                schema.LABEL_COL: label,
                schema.SPLIT_ORIGIN: "train" if i < int(n_subjects * 0.8) else "validation",
            }
        )

    daily = (
        pd.DataFrame(rows)
        .sort_values([schema.SUBJECT_ID, schema.DATE_COL])
        .reset_index(drop=True)
    )
    daily[schema.RAW_ROW_ID] = np.arange(len(daily), dtype=np.int64)
    return daily, pd.DataFrame(subject_rows)


@pytest.fixture
def synthetic_data() -> LifelogData:
    daily, subjects = _make_daily()
    return LifelogData(
        daily=daily,
        subjects=subjects,
        feature_columns=FEATURES,
        notes={"synthetic": True},
    )


@pytest.fixture
def real_data_root() -> Path:
    """The repository's ``Data/`` directory, or skip if it is not present."""
    root = EXPERIMENT_ROOT.parents[2] / "Data"
    if not (root / "1.Training").is_dir():
        pytest.skip("Data/ is not available in this environment")
    return root


@pytest.fixture
def config_dir() -> Path:
    return EXPERIMENT_ROOT / "configs"
