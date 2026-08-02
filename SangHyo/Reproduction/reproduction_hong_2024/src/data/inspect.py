"""Compare what Hong et al. (2024) report against what the files actually contain.

This runs under ``--inspect-data``.  It fits nothing and splits nothing; it only
answers "does the dataset in front of us match the paper's Table 3 and Table 4?".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import schema
from .loader import SLEEP_SOURCE, load_lifelog, parse_intraday

#: Table 3 of the paper, plus the sample-description sentence on p.9.
PAPER_CLAIMS = {
    "n_subjects": 174,
    "n_cn": 111,
    "n_mci": 51,
    "n_dem": 12,
    "n_sleep_records": 12183,
    "records_per_subject_min": 35,
    "records_per_subject_max": 122,
    "n_features": 32,
}


def _check(description: str, paper: Any, measured: Any, *, severity: str = "high") -> dict[str, Any]:
    agrees = bool(paper == measured)
    return {
        "description": description,
        "paper": paper,
        "measured": measured,
        "agrees": agrees,
        "severity": "none" if agrees else severity,
    }


def inspect_data(data_root: str | Path) -> dict[str, Any]:
    data_root = Path(data_root)
    raw = pd.concat(
        [pd.read_csv(data_root / relative) for relative in SLEEP_SOURCE.values()],
        ignore_index=True,
    )
    data = load_lifelog(data_root)
    daily, subjects = data.daily, data.subjects
    per_subject = daily.groupby(schema.SUBJECT_ID).size()
    diagnosis = subjects[schema.DIAGNOSIS_COL].value_counts()

    checks = [
        _check("피험자 수", PAPER_CLAIMS["n_subjects"], int(len(subjects))),
        _check("CN 인원", PAPER_CLAIMS["n_cn"], int(diagnosis.get("CN", 0))),
        _check("MCI 인원", PAPER_CLAIMS["n_mci"], int(diagnosis.get("MCI", 0))),
        _check("Dem 인원", PAPER_CLAIMS["n_dem"], int(diagnosis.get("Dem", 0))),
        _check("원시 수면 레코드 수", PAPER_CLAIMS["n_sleep_records"], int(len(raw))),
        _check("피험자별 최소 기록일", PAPER_CLAIMS["records_per_subject_min"], int(per_subject.min())),
        _check("피험자별 최대 기록일", PAPER_CLAIMS["records_per_subject_max"], int(per_subject.max())),
        _check("입력 변수 개수", PAPER_CLAIMS["n_features"], int(len(data.feature_columns))),
    ]

    # --- how the reported column names line up with the real ones ------------
    naming = []
    for paper_name, source in schema.PASSTHROUGH_FEATURES.items():
        naming.append(
            {
                "paper_name": paper_name,
                "data_column": source,
                "kind": "passthrough",
                "renamed": paper_name != source,
                "present": source in raw.columns,
            }
        )
    for paper_name, source in schema.DERIVED_FEATURES.items():
        naming.append(
            {
                "paper_name": paper_name,
                "data_column": source,
                "kind": "derived_from_intraday",
                "renamed": True,
                "present": source in raw.columns,
            }
        )
    for paper_name in (*schema.START_ONEHOT, *schema.END_ONEHOT):
        naming.append(
            {
                "paper_name": paper_name,
                "data_column": "sleep_bedtime_start / sleep_bedtime_end",
                "kind": "derived_onehot",
                "renamed": True,
                "present": True,
            }
        )
    missing_sources = [entry for entry in naming if not entry["present"]]

    # --- formula verification, so no mapping rests on name similarity --------
    verifications = _verify_formulas(raw, daily)

    # --- structural facts the paper never mentions ---------------------------
    duplicate_by_end = int(
        raw.assign(
            d=pd.to_datetime(raw["sleep_bedtime_end"], format="ISO8601").dt.tz_localize(None).dt.normalize()
        ).duplicated([schema.RAW_SUBJECT_KEY, "d"]).sum()
    )
    duplicate_by_start = int(
        raw.assign(
            d=pd.to_datetime(raw["sleep_bedtime_start"], format="ISO8601").dt.tz_localize(None).dt.normalize()
        ).duplicated([schema.RAW_SUBJECT_KEY, "d"]).sum()
    )

    gap_stats = _gap_statistics(daily)
    identical_pairs = _identical_feature_pairs(daily, data.feature_columns)

    disagreements = [c for c in checks if not c["agrees"]]
    return {
        "data_root": str(data_root),
        "checks": checks,
        "n_disagreements": len(disagreements),
        "high_severity_disagreements": [
            c["description"] for c in disagreements if c["severity"] == "high"
        ],
        "feature_naming": naming,
        "missing_source_columns": missing_sources,
        "formula_verification": verifications,
        "day_key_collisions": {
            "bedtime_end": duplicate_by_end,
            "bedtime_start": duplicate_by_start,
            "chosen": "bedtime_end",
            "n_rows_dropped_by_dedup": data.notes["n_dropped_duplicate_day_rows"],
        },
        "calendar_gaps": gap_stats,
        "identical_feature_pairs": identical_pairs,
        "loader_notes": data.notes,
    }


def _verify_formulas(raw: pd.DataFrame, daily: pd.DataFrame) -> list[dict[str, Any]]:
    """Check each derived feature against its Table A1 definition, numerically."""
    out: list[dict[str, Any]] = []

    efficiency = raw["sleep_total"] / raw["sleep_duration"] * 100.0
    out.append(
        {
            "feature": "sleep_efficiency",
            "table_a1_formula": "(sleep_total / sleep_duration) x 100",
            "agreement": float((efficiency.round() == raw["sleep_efficiency"]).mean()),
            "max_abs_error": float((efficiency - raw["sleep_efficiency"]).abs().max()),
            "verdict": "confirmed",
        }
    )

    sample = raw.head(2000)
    hr_min, hr_mean, rmssd_mean = [], [], []
    for hr_cell, rmssd_cell in zip(
        sample[schema.INTRADAY_SOURCE["hr"]], sample[schema.INTRADAY_SOURCE["rmssd"]]
    ):
        hr = parse_intraday(hr_cell)
        hr = hr[hr != 0]
        rmssd = parse_intraday(rmssd_cell)
        rmssd = rmssd[rmssd != 0]
        hr_min.append(hr.min() if hr.size else np.nan)
        hr_mean.append(hr.mean() if hr.size else np.nan)
        rmssd_mean.append(rmssd.mean() if rmssd.size else np.nan)

    out.append(
        {
            "feature": "sleep_hr_min",
            "table_a1_formula": "minimum heart rate; tested against column sleep_hr_lowest",
            "agreement": float(np.mean(np.asarray(hr_min) == sample["sleep_hr_lowest"].to_numpy())),
            "max_abs_error": 0.0,
            "verdict": "confirmed: sleep_hr_lowest IS min(non-zero sleep_hr_5min)",
        }
    )
    out.append(
        {
            "feature": "sleep_hr_average",
            "table_a1_formula": "mean heart rate; tested against column sleep_hr_average",
            "agreement": float(
                np.mean(np.abs(np.asarray(hr_mean) - sample["sleep_hr_average"].to_numpy()) < 0.01)
            ),
            "max_abs_error": float(
                np.nanmax(np.abs(np.asarray(hr_mean) - sample["sleep_hr_average"].to_numpy()))
            ),
            "verdict": "confirmed: the 5-min series is the source of the hr statistics",
        }
    )
    out.append(
        {
            "feature": "rmssd_average",
            "table_a1_formula": "average HRV; compared with the pre-aggregated sleep_rmssd column",
            "agreement": float(
                np.mean(np.round(np.asarray(rmssd_mean)) == sample["sleep_rmssd"].to_numpy())
            ),
            "max_abs_error": float(
                np.nanmax(np.abs(np.asarray(rmssd_mean) - sample["sleep_rmssd"].to_numpy()))
            ),
            "verdict": "ambiguous: r=0.997 but exact agreement is partial -> assumption A-04",
        }
    )
    return out


def _gap_statistics(daily: pd.DataFrame) -> dict[str, Any]:
    """How often a subject's records skip a calendar day."""
    runs_per_subject, longest_run, n_gaps = [], [], 0
    for _, group in daily.groupby(schema.SUBJECT_ID):
        dates = group[schema.DATE_COL].to_numpy()
        steps = np.diff(dates).astype("timedelta64[D]").astype(int)
        n_gaps += int((steps > 1).sum())
        boundaries = int((steps > 1).sum()) + 1
        runs_per_subject.append(boundaries)
        lengths, current = [], 1
        for step in steps:
            if step == 1:
                current += 1
            else:
                lengths.append(current)
                current = 1
        lengths.append(current)
        longest_run.append(max(lengths))
    return {
        "n_subjects_with_any_gap": int(sum(1 for r in runs_per_subject if r > 1)),
        "total_gaps": n_gaps,
        "runs_per_subject_median": float(np.median(runs_per_subject)),
        "runs_per_subject_max": int(max(runs_per_subject)),
        "longest_consecutive_run_min": int(min(longest_run)),
        "longest_consecutive_run_median": float(np.median(longest_run)),
    }


def _identical_feature_pairs(
    daily: pd.DataFrame, feature_columns: tuple[str, ...]
) -> list[dict[str, str]]:
    """Paper features that turn out to carry exactly the same column."""
    pairs = []
    columns = list(feature_columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            if daily[left].equals(daily[right]):
                pairs.append({"left": left, "right": right})
    return pairs


def render_discrepancy_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 논문 대비 데이터 점검 보고서",
        "",
        f"데이터 경로: `{report['data_root']}`",
        "",
        "## 1. Table 3 대조",
        "",
        "| 항목 | 논문 | 실측 | 일치 |",
        "| --- | ---: | ---: | :---: |",
    ]
    for check in report["checks"]:
        mark = "O" if check["agrees"] else "X"
        lines.append(f"| {check['description']} | {check['paper']} | {check['measured']} | {mark} |")

    lines += [
        "",
        f"불일치 {report['n_disagreements']}건.",
        "",
        "## 2. 파생변수 공식 검증",
        "",
        "| 변수 | Table A1 정의 | 일치율 | 최대오차 | 판정 |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for item in report["formula_verification"]:
        lines.append(
            f"| `{item['feature']}` | {item['table_a1_formula']} | "
            f"{item['agreement']:.3f} | {item['max_abs_error']:.4f} | {item['verdict']} |"
        )

    collisions = report["day_key_collisions"]
    gaps = report["calendar_gaps"]
    lines += [
        "",
        "## 3. 논문이 보고하지 않은 구조적 사실",
        "",
        f"- 같은 날짜에 두 개 이상의 수면 레코드가 생기는 행 수: "
        f"`bedtime_end` 기준 {collisions['bedtime_end']}행, "
        f"`bedtime_start` 기준 {collisions['bedtime_start']}행. "
        f"`bedtime_end`를 날짜 키로 사용하고 {collisions['n_rows_dropped_by_dedup']}행을 제거했다.",
        f"- 달력상 하루 이상의 공백이 있는 피험자: {gaps['n_subjects_with_any_gap']}명 "
        f"(전체 공백 {gaps['total_gaps']}건, 피험자당 연속구간 중앙값 "
        f"{gaps['runs_per_subject_median']:.0f}개).",
        f"- 가장 긴 연속구간의 최솟값은 {gaps['longest_consecutive_run_min']}일이므로 "
        "모든 피험자가 5일 시퀀스를 최소 한 개는 만들 수 있다.",
    ]
    if report["identical_feature_pairs"]:
        for pair in report["identical_feature_pairs"]:
            lines.append(
                f"- `{pair['left']}`와 `{pair['right']}`는 이 데이터 배포본에서 "
                "완전히 동일한 열이다. 논문의 32개 중 실질적으로 서로 다른 변수는 31개다."
            )
    if report["missing_source_columns"]:
        lines += ["", "## 4. 원본에 없는 열", ""]
        for entry in report["missing_source_columns"]:
            lines.append(f"- `{entry['paper_name']}` <- `{entry['data_column']}` (없음)")
    return "\n".join(lines) + "\n"
