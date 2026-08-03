"""Shared fixtures.

The synthetic cohort mimics the real one's shape (subjects with different numbers
of days, a minority positive class) without touching ``Data/``, so the contract
tests run anywhere.
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

FEATURES = ("activity_score", "activity_steps", "sleep_score", "sleep_restless")


def _make_daily(n_subjects: int = 24, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    subject_rows = []
    for i in range(n_subjects):
        subject = f"subj{i:03d}"
        label = int(i % 3 == 0)  # ~1/3 positive, like the real 36%
        n_days = int(rng.integers(8, 20))
        start = pd.Timestamp("2020-10-17")
        for day in range(n_days):
            rows.append(
                {
                    schema.SUBJECT_ID: subject,
                    schema.DATE_COL: start + pd.Timedelta(int(day), "D"),
                    schema.LABEL_COL: label,
                    "split_origin": "train" if i < int(n_subjects * 0.8) else "validation",
                    **{f: float(rng.normal(label, 1.0)) for f in FEATURES},
                }
            )
        subject_rows.append(
            {
                schema.SUBJECT_ID: subject,
                schema.DIAGNOSIS_COL: "MCI" if label else "CN",
                schema.LABEL_COL: label,
                "split_origin": "train" if i < int(n_subjects * 0.8) else "validation",
            }
        )

    daily = pd.DataFrame(rows).sort_values(
        [schema.SUBJECT_ID, schema.DATE_COL]
    ).reset_index(drop=True)
    daily["row_id"] = np.arange(len(daily), dtype=np.int64)
    return daily, pd.DataFrame(subject_rows)


@pytest.fixture
def synthetic_data() -> LifelogData:
    daily, subjects = _make_daily()
    return LifelogData(
        daily=daily,
        subjects=subjects,
        cognitive=pd.DataFrame({schema.SUBJECT_ID: subjects[schema.SUBJECT_ID], "TOTAL": 25}),
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
