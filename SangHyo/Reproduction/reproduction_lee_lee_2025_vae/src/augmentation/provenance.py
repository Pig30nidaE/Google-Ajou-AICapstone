"""합성행 provenance.

사용자 지시 15·16: 합성 row는 독립 피험자가 아니다.
그 사실을 데이터 구조 수준에서 못박기 위해 모든 합성행에 아래 메타데이터를 붙인다.
``subject`` 자리에는 실제 ID 대신 센티널이 들어간다 (``loader.append_synthetic``).
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

__all__ = ["SyntheticProvenance", "subject_set_hash", "build_provenance_frame"]


def subject_set_hash(subjects: Iterable, *, length: int = 16) -> str:
    """생성에 사용된 실제 피험자 집합의 해시.

    원본 ID를 노출하지 않으면서 "어떤 피험자 집합에서 생성되었는가"를 대조할 수 있게 한다.
    """
    payload = "|".join(sorted(str(s) for s in set(subjects))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


@dataclass(frozen=True)
class SyntheticProvenance:
    """합성행 1건의 출처 메타데이터 (synthetic_data_risk.md §4)."""

    is_synthetic: bool
    source_class: str
    source_outer_fold: str | None
    source_inner_fold: str | None
    generator: str
    generator_seed: int
    generator_config_hash: str
    source_subject_hash: str
    n_source_subjects: int
    n_source_rows: int
    created_at: str

    @classmethod
    def create(
        cls,
        *,
        source_class: str,
        generator: str,
        generator_seed: int,
        generator_config_hash: str,
        source_subjects: Iterable,
        n_source_rows: int,
        source_outer_fold: str | None = None,
        source_inner_fold: str | None = None,
    ) -> "SyntheticProvenance":
        subs = list(source_subjects)
        return cls(
            is_synthetic=True,
            source_class=source_class,
            source_outer_fold=source_outer_fold,
            source_inner_fold=source_inner_fold,
            generator=generator,
            generator_seed=generator_seed,
            generator_config_hash=generator_config_hash,
            source_subject_hash=subject_set_hash(subs),
            n_source_subjects=len({str(s) for s in subs}),
            n_source_rows=int(n_source_rows),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def to_frame(self, n_rows: int) -> pd.DataFrame:
        """동일 provenance를 ``n_rows``행에 복제한 프레임."""
        return pd.DataFrame([asdict(self)] * n_rows)


def build_provenance_frame(prov: SyntheticProvenance, n_rows: int) -> pd.DataFrame:
    return prov.to_frame(n_rows)
