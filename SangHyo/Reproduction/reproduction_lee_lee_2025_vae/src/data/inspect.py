"""``--inspect-data`` / ``--dry-run``이 출력하는 데이터 점검 리포트.

학습을 실행하지 않고 확인 가능한 모든 것을 검사한다.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .loader import LifelogData
from .paper_reference import SECTION51_AFTER_OUTLIER, TABLE3_COHORT
from .schema import (
    DATA_CONTRACT,
    INTEGER_VALUED_FEATURES,
    KNOWN_DUPLICATE_COLUMN_PAIRS,
    NON_NEGATIVE_FEATURES,
    PAPER_ACTIVITY_FEATURES,
    PAPER_FEATURES,
    PAPER_SLEEP_FEATURES,
)

log = logging.getLogger(__name__)

__all__ = ["inspect_data", "format_inspection", "percentile_retention_scan"]


def inspect_data(data: LifelogData) -> dict:
    """데이터 구조·계약·프로파일을 점검한다."""
    rec = data.class_counts(by="record")
    sub = data.class_counts(by="subject")
    X = data.X

    per_subject = pd.Series(data.subject).value_counts()
    dem_subjects = data.subject_labels()
    dem_ids = dem_subjects[dem_subjects == 2].index
    dem_counts = sorted(int(per_subject[s]) for s in dem_ids)

    dup_pairs = [
        {"left": a, "right": b, "identical": bool(np.array_equal(X[a].to_numpy(), X[b].to_numpy()))}
        for a, b in KNOWN_DUPLICATE_COLUMN_PAIRS
        if a in X.columns and b in X.columns
    ]

    report: dict = {
        "n_rows": data.n,
        "n_subjects": len(data.subjects()),
        "n_features": len(data.features),
        "feature_names": list(data.features),
        "n_activity_features": sum(c.startswith("activity_") for c in data.features),
        "n_sleep_features": sum(c.startswith("sleep_") for c in data.features),
        "records_per_class": rec,
        "subjects_per_class": sub,
        "paper_table3_records": TABLE3_COHORT["n_record"],
        "paper_table3_subjects": TABLE3_COHORT["n_person"],
        "matches_paper_table3": (
            rec == TABLE3_COHORT["n_record"] and sub == TABLE3_COHORT["n_person"]
        ),
        "contract": DATA_CONTRACT,
        "n_missing_cells": int(X.isna().sum().sum()),
        "n_duplicate_rows": int(X.duplicated().sum()),
        "records_per_subject": {
            "min": int(per_subject.min()),
            "max": int(per_subject.max()),
            "mean": float(per_subject.mean()),
            "median": float(per_subject.median()),
        },
        "dem_subject_record_counts": dem_counts,
        "duplicate_column_pairs": dup_pairs,
        "missing_paper_features": [c for c in PAPER_FEATURES if c not in X.columns],
        "n_negative_valued_features": int((X < 0).any().sum()),
        "negative_valued_features": [c for c in X.columns if (X[c] < 0).any()],
        "n_expected_nonnegative": len(NON_NEGATIVE_FEATURES),
        "n_expected_integer_valued": len(INTEGER_VALUED_FEATURES),
        "paper_after_outlier_counts": SECTION51_AFTER_OUTLIER,
        "paper_after_outlier_total": sum(SECTION51_AFTER_OUTLIER.values()),
        "paper_after_outlier_retention": sum(SECTION51_AFTER_OUTLIER.values()) / max(data.n, 1),
    }
    report["feature_profile"] = (
        pd.DataFrame(
            {
                "min": X.min(),
                "max": X.max(),
                "mean": X.mean(),
                "std": X.std(),
                "n_unique": X.nunique(),
            }
        )
        .round(4)
        .reset_index(names="feature")
        .to_dict(orient="records")
    )
    return report


def percentile_retention_scan(
    data: LifelogData, *, qs: tuple[float, ...] = (0.001, 0.003, 0.005, 0.01, 0.05, 0.10)
) -> pd.DataFrame:
    """분위수 절단이 논문 행 수를 재현하는지 검사한다 (학습 없이 산술만).

    report_inconsistencies.md I-1의 증거 B·C를 실행 시점에 재확인한다.
    """
    X, y = data.X, data.y
    target = SECTION51_AFTER_OUTLIER
    rows = []
    for q in qs:
        lo, hi = X.quantile(q), X.quantile(1 - q)
        keep = ((X >= lo) & (X <= hi)).all(axis=1).to_numpy()
        got = {cls: int(((y == code) & keep).sum()) for code, cls in enumerate(("CN", "MCI", "Dem"))}
        rows.append(
            {
                "q": q,
                "kept_rows": int(keep.sum()),
                "retention": round(float(keep.mean()), 4),
                **{f"kept_{k}": v for k, v in got.items()},
                "L1_distance_to_paper": sum(abs(got[k] - target[k]) for k in target),
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["paper_target"] = target
    df.attrs["note"] = (
        "논문 §5.1은 q=0.10을 보고했으나 그 설정의 잔존율은 논문이 보고한 90%와 크게 다르다. "
        "report_inconsistencies.md I-1 참조."
    )
    return df


def format_inspection(report: dict) -> str:
    """콘솔 출력용 텍스트."""
    L = []
    add = L.append
    add("=" * 78)
    add("데이터 점검 (--inspect-data)")
    add("=" * 78)
    add(f"행 수      : {report['n_rows']:,}")
    add(f"피험자 수  : {report['n_subjects']}")
    add(f"변수 수    : {report['n_features']} (활동 {report['n_activity_features']} + "
        f"수면 {report['n_sleep_features']})")
    add("")
    add("-- 논문 표 3 대조 --")
    add(f"{'클래스':<6} {'기록(실측)':>12} {'기록(논문)':>12} {'피험자(실측)':>14} {'피험자(논문)':>14}")
    for cls in ("CN", "MCI", "Dem"):
        add(
            f"{cls:<6} {report['records_per_class'].get(cls, 0):>12,} "
            f"{report['paper_table3_records'][cls]:>12,} "
            f"{report['subjects_per_class'].get(cls, 0):>14} "
            f"{report['paper_table3_subjects'][cls]:>14}"
        )
    add(f"→ 논문 표 3과 일치: {'예 ✅' if report['matches_paper_table3'] else '아니오 ❌'}")
    add("")
    add("-- 데이터 품질 --")
    add(f"결측 셀        : {report['n_missing_cells']}")
    add(f"완전 중복 행   : {report['n_duplicate_rows']}")
    add(f"누락된 논문 변수: {report['missing_paper_features'] or '없음'}")
    for d in report["duplicate_column_pairs"]:
        mark = "동일 ⚠️" if d["identical"] else "다름"
        add(f"중복 컬럼쌍   : {d['left']} vs {d['right']} → {mark}")
    add(f"음수 존재 변수 : {report['n_negative_valued_features']}개 "
        f"{report['negative_valued_features']}")
    add("")
    add("-- 피험자별 기록 수 --")
    rp = report["records_per_subject"]
    add(f"min {rp['min']} / median {rp['median']:.1f} / mean {rp['mean']:.1f} / max {rp['max']}")
    add(f"Dem 12명 기록 수: {report['dem_subject_record_counts']}")
    add("")
    add("-- 논문 §5.1 이상치 제거 후 --")
    add(f"논문 보고 합계 {report['paper_after_outlier_total']:,} / 전체 {report['n_rows']:,} "
        f"= 잔존율 {report['paper_after_outlier_retention']:.4f}")
    add("  (정확히 90.0%. IsolationForest(contamination=0.1)의 서명이다 — I-1 증거 A)")
    add("=" * 78)
    return "\n".join(L)
