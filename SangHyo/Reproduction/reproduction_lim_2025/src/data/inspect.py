"""Paper-vs-data discrepancy checks (``run.py --inspect-data``).

Section 3 of the task brief lists the mismatches that must be checked
automatically.  Each check returns a row for ``discrepancy_report.md`` rather than
silently reconciling the paper to the data.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.io import hash_subject
from . import schema
from .loader import FILE_LAYOUT, _to_date

#: What the two papers state, so the report can show claim vs measurement.
PAPER_CLAIMS = {
    "n_subjects": 174,
    "n_daily_rows": 12184,
    "n_positive_subjects": 63,
    "n_features": 58,
    "split_ratio": "80:20",
}


@dataclass
class Check:
    key: str
    description: str
    paper: Any
    measured: Any
    agrees: bool
    severity: str  # "none" | "low" | "medium" | "high"
    note: str = ""


def _load_raw(root: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for split, files in FILE_LAYOUT.items():
        for key, rel in files.items():
            frames[f"{split}:{key}"] = pd.read_csv(root / rel, low_memory=False)
    return frames


def inspect_data(data_root: str | Path) -> dict[str, Any]:
    """Run every discrepancy check and return a structured report."""
    root = Path(data_root).expanduser().resolve()
    frames = _load_raw(root)
    checks: list[Check] = []

    ta, tsl = frames["train:activity"], frames["train:sleep"]
    va, vsl = frames["validation:activity"], frames["validation:sleep"]
    tm, vm = frames["train:mmse"], frames["validation:mmse"]

    # --- 1. subject count -----------------------------------------------------
    n_subjects = tm[schema.LABEL_PERSON_KEY].nunique() + vm[schema.LABEL_PERSON_KEY].nunique()
    checks.append(Check(
        "n_subjects", "총 피험자 수", PAPER_CLAIMS["n_subjects"], int(n_subjects),
        n_subjects == PAPER_CLAIMS["n_subjects"], "none",
        "Training 141 + Validation 33",
    ))

    # --- 2. daily row count (12,184 vs 12,183) --------------------------------
    n_rows_activity = len(ta) + len(va)
    n_rows_sleep = len(tsl) + len(vsl)
    checks.append(Check(
        "n_daily_rows", "일별 라이프로그 행 수 (activity)",
        PAPER_CLAIMS["n_daily_rows"], int(n_rows_activity),
        n_rows_activity == PAPER_CLAIMS["n_daily_rows"], "medium",
        f"activity {len(ta)}+{len(va)}, sleep {len(tsl)}+{len(vsl)}. "
        f"차이 {PAPER_CLAIMS['n_daily_rows'] - n_rows_activity:+d}행",
    ))
    checks.append(Check(
        "activity_sleep_row_parity", "activity/sleep 행 수 동일 여부",
        "언급 없음", n_rows_activity == n_rows_sleep, n_rows_activity == n_rows_sleep,
        "none", "두 modality의 전체 행 수가 같으면 1:1 정렬 가능성",
    ))

    # --- 3. class balance -----------------------------------------------------
    diagnoses = pd.concat([tm[schema.DIAGNOSIS_COL], vm[schema.DIAGNOSIS_COL]])
    counts = diagnoses.value_counts().to_dict()
    n_pos = int(sum(counts.get(d, 0) for d in schema.POSITIVE_DIAGNOSES))
    checks.append(Check(
        "n_positive_subjects", "양성(MCI+Dem) 피험자 수",
        PAPER_CLAIMS["n_positive_subjects"], n_pos,
        n_pos == PAPER_CLAIMS["n_positive_subjects"], "none",
        f"실측 분포 {counts}",
    ))

    # --- 4. duplicate rows ----------------------------------------------------
    for split, act, slp in (("train", ta, tsl), ("validation", va, vsl)):
        checks.append(Check(
            f"{split}_full_row_duplicates", f"[{split}] 완전 중복 행",
            "언급 없음", {"activity": int(act.duplicated().sum()),
                          "sleep": int(slp.duplicated().sum())},
            act.duplicated().sum() == 0 and slp.duplicated().sum() == 0, "none",
        ))

    # --- 5. same subject + same date -----------------------------------------
    for split, act, slp in (("train", ta, tsl), ("validation", va, vsl)):
        act_d = act.assign(
            sid=act[schema.SOURCE_PERSON_KEY].map(hash_subject),
            d=_to_date(act[schema.ACTIVITY_DATE_SOURCE]),
        )
        slp_d = slp.assign(
            sid=slp[schema.SOURCE_PERSON_KEY].map(hash_subject),
            d=_to_date(slp[schema.SLEEP_DATE_SOURCES["bedtime_end"]]),
        )
        act_dup = int(act_d.duplicated(["sid", "d"]).sum())
        slp_dup = int(slp_d.duplicated(["sid", "d"]).sum())
        checks.append(Check(
            f"{split}_subject_date_duplicates",
            f"[{split}] 동일 피험자·동일 날짜 중복",
            "언급 없음", {"activity": act_dup, "sleep": slp_dup},
            act_dup == 0 and slp_dup == 0, "medium" if slp_dup else "none",
            "sleep은 bedtime_end 날짜 기준. 중복이 있으면 date join에서 행이 손실된다",
        ))

    # --- 6. one label per subject --------------------------------------------
    multi = pd.concat([tm, vm]).groupby(schema.LABEL_PERSON_KEY)[schema.DIAGNOSIS_COL].nunique()
    checks.append(Check(
        "multi_label_subjects", "한 피험자에 여러 진단레이블",
        "언급 없음", int((multi > 1).sum()), bool((multi > 1).sum() == 0), "none",
    ))

    # --- 7. label copies agree -----------------------------------------------
    for split in ("train", "validation"):
        copies = [frames[f"{split}:{k}"] for k in ("label_gait", "label_sleep", "label_cog")]
        ref = copies[0].sort_values(schema.LABEL_PERSON_KEY).reset_index(drop=True)
        same = all(
            ref.equals(c.sort_values(schema.LABEL_PERSON_KEY).reset_index(drop=True))
            for c in copies[1:]
        )
        mmse = frames[f"{split}:mmse"]
        merged = ref.merge(
            mmse[[schema.LABEL_PERSON_KEY, schema.DIAGNOSIS_COL]],
            on=schema.LABEL_PERSON_KEY, how="outer", suffixes=("_label", "_mmse"),
        )
        mismatch = int(
            (merged[f"{schema.DIAGNOSIS_COL}_label"] != merged[f"{schema.DIAGNOSIS_COL}_mmse"]).sum()
        )
        checks.append(Check(
            f"{split}_label_consistency", f"[{split}] 라벨 사본 3개 + MMSE 일치",
            "언급 없음", {"copies_identical": same, "label_vs_mmse_mismatch": mismatch},
            bool(same and mismatch == 0), "none",
        ))

    # --- 8. train/val subject overlap ----------------------------------------
    overlap = set(tm[schema.LABEL_PERSON_KEY]) & set(vm[schema.LABEL_PERSON_KEY])
    checks.append(Check(
        "train_val_subject_overlap", "Training/Validation 피험자 중복",
        "언급 없음", len(overlap), len(overlap) == 0, "none",
    ))

    # --- 9. join key feasibility ---------------------------------------------
    has_date = "date" in ta.columns
    checks.append(Check(
        "date_column_exists", "`date` 컬럼 존재 여부",
        "표 1에 존재한다고 기재", has_date, has_date, "high",
        "부재 시 activity_day_start / sleep_bedtime_end에서 파생해야 하며, "
        "논문의 시계열 정렬 방법을 그대로 재현할 수 없다",
    ))

    parity = []
    for split, act, slp in (("train", ta, tsl), ("validation", va, vsl)):
        ac = act.groupby(schema.SOURCE_PERSON_KEY).size().sort_index()
        sc = slp.groupby(schema.SOURCE_PERSON_KEY).size().sort_index()
        same_order = bool(
            len(act) == len(slp)
            and (act[schema.SOURCE_PERSON_KEY].values == slp[schema.SOURCE_PERSON_KEY].values).all()
        )
        parity.append({
            "split": split,
            "per_subject_counts_equal": bool(ac.equals(sc)),
            "row_order_email_identical": same_order,
        })
    checks.append(Check(
        "activity_sleep_alignment", "activity/sleep 행 순서 1:1 정렬 가능성",
        "결합키 미보고", parity,
        all(p["per_subject_counts_equal"] and p["row_order_email_identical"] for p in parity),
        "high", "EMAIL 단독 결합은 피험자 집계에만 유효. 일별 결합은 EMAIL+날짜 필요",
    ))

    # --- 10. observation-day distribution and calendar gaps -------------------
    obs: dict[str, Any] = {}
    for split, act in (("train", ta), ("validation", va)):
        frame = act.assign(
            sid=act[schema.SOURCE_PERSON_KEY].map(hash_subject),
            d=_to_date(act[schema.ACTIVITY_DATE_SOURCE]),
        ).sort_values(["sid", "d"])
        per = frame.groupby("sid").size()
        gaps = frame.groupby("sid")["d"].apply(lambda s: s.diff().dt.days.dropna())
        values = np.asarray(gaps.values, dtype=float) if len(gaps) else np.array([])
        with_gap = int(
            frame.groupby("sid")["d"].apply(lambda s: bool((s.diff().dt.days.dropna() > 1).any())).sum()
        )
        obs[split] = {
            "days_per_subject": {
                "min": int(per.min()), "median": float(per.median()),
                "max": int(per.max()), "mean": round(float(per.mean()), 2),
            },
            "consecutive_fraction": round(float((values == 1).mean()), 4) if values.size else None,
            "max_gap_days": int(np.nanmax(values)) if values.size else None,
            "subjects_with_any_gap": with_gap,
            "n_subjects": int(per.size),
            "date_min": str(frame["d"].min().date()),
            "date_max": str(frame["d"].max().date()),
        }
    checks.append(Check(
        "observation_days", "피험자별 관찰일수 분포 / 비연속 날짜",
        "미보고", obs, True, "medium",
        "결측일이 흔하므로 시퀀스 구성 규칙(compress/calendar)이 결과에 영향을 준다",
    ))

    # --- 11. paper feature names vs real columns ------------------------------
    available = set(ta.columns) | set(tsl.columns)
    absent = {k: v for k, v in schema.PAPER_NAMES_ABSENT_FROM_DATA.items() if k not in available}
    non_numeric_present = [
        c for c in schema.PAPER_NAMES_DROPPED_AS_NON_NUMERIC
        if c in available and not pd.api.types.is_numeric_dtype(
            ta[c] if c in ta.columns else tsl[c]
        )
    ]
    checks.append(Check(
        "paper_feature_names", "논문 변수명 vs 실제 컬럼명",
        "56개 특징(표 9~11의 2~57번)", {
            "absent_from_data": absent,
            "dropped_as_non_numeric": non_numeric_present,
        },
        not absent and not non_numeric_present, "high",
        "numeric_only=True가 문자열 변수를 조용히 탈락시킨다",
    ))

    n_effective = len(schema.PAPER_CODE_FEATURES)
    checks.append(Check(
        "n_features", "실제 학습에 들어가는 특징 수",
        PAPER_CLAIMS["n_features"], n_effective,
        n_effective == PAPER_CLAIMS["n_features"], "high",
        "논문 코드를 그대로 실행하면 49개가 생성된다 "
        "(activity 22 + sleep 27). 58은 email과 DIAG_NIM을 포함한 표 행 수",
    ))

    # --- 12. zero-variance / administrative ----------------------------------
    zero_var = {}
    for col in schema.ZERO_VARIANCE_FEATURES:
        source = ta if col in ta.columns else tsl
        if col in source.columns:
            zero_var[col] = int(source[col].nunique())
    checks.append(Check(
        "zero_variance_features", "무분산 특징",
        "언급 없음", zero_var, all(v > 1 for v in zero_var.values()) if zero_var else True,
        "medium", "고유값 1이면 정보가 없다",
    ))

    # --- 13. missing values ---------------------------------------------------
    na = {
        "train_activity": int(ta.isna().sum().sum()),
        "train_sleep": int(tsl.isna().sum().sum()),
        "validation_activity": int(va.isna().sum().sum()),
        "validation_sleep": int(vsl.isna().sum().sum()),
    }
    checks.append(Check(
        "missing_values", "결측치", "미보고", na, all(v == 0 for v in na.values()), "none",
    ))

    # --- 14. MMSE coding ------------------------------------------------------
    scored_items = [
        c for c in schema.MMSE_ITEM_COLUMNS
        if c in tm.columns and c not in schema.MMSE_BROKEN_COLUMNS
    ]
    codes = sorted(
        {int(v) for df in (tm, vm) for c in scored_items for v in df[c].dropna().unique()}
    )
    checks.append(Check(
        "mmse_item_coding", "MMSE 문항 코딩",
        "정답/오답 = 2/1", codes,
        codes == [schema.MMSE_INCORRECT_CODE, schema.MMSE_CORRECT_CODE], "medium",
        "0/1이 아니라 1/2다. 0이 나타나면 척도 밖 값이며 미실시로 보아야 한다",
    ))

    sentinel_rows = []
    for split, df in (("train", tm), ("validation", vm)):
        mask = (df[scored_items] == schema.MMSE_MISSING_SENTINEL).all(axis=1)
        if mask.any():
            sentinel_rows.append({
                "split": split,
                "n_subjects": int(mask.sum()),
                "diagnoses": df.loc[mask, schema.DIAGNOSIS_COL].tolist(),
                "TOTAL": df.loc[mask, "TOTAL"].tolist(),
            })
    checks.append(Check(
        "mmse_all_zero_subjects", "전 문항이 0인 피험자 (미실시 추정)",
        "언급 없음", sentinel_rows or 0, not sentinel_rows, "medium",
        "MMSE 척도는 1/2이므로 전 문항 0은 점수가 아니라 결측 표식이다. "
        "secondary 분석에서 0점으로 합산하면 해당 피험자가 극단 이상치가 된다",
    ))

    broken = {}
    for col in schema.MMSE_BROKEN_COLUMNS:
        if col in tm.columns:
            values = sorted({int(v) for df in (tm, vm) for v in df[col].dropna().unique()})
            broken[col] = values
    checks.append(Check(
        "mmse_broken_columns", "상수 MMSE 컬럼",
        "표 7에 'Q12 total 점수'로 기재", broken,
        not any(len(v) == 1 for v in broken.values()), "medium",
        "Q12_TOTAL은 전 피험자 0이며 Q12_1~Q12_5의 합도 아니다. 사용 불가",
    ))

    total_ok = {}
    for split, df in (("train", tm), ("validation", vm)):
        n_correct = (df[scored_items] == schema.MMSE_CORRECT_CODE).sum(axis=1)
        total_ok[split] = round(float((n_correct == df["TOTAL"]).mean()), 4)
    checks.append(Check(
        "mmse_total_definition", "TOTAL = 정답(2) 문항 수 여부",
        "검사 답변의 총점", total_ok,
        all(v == 1.0 for v in total_ok.values()), "none",
        f"Q12_TOTAL을 제외한 {len(scored_items)}개 문항 기준",
    ))

    # --- 15. reconstructed evaluation set ------------------------------------
    val_pos = int(vm[schema.DIAGNOSIS_COL].isin(schema.POSITIVE_DIAGNOSES).sum())
    val_n = int(len(vm))
    checks.append(Check(
        "reconstructed_eval_set", "복원된 평가집단 (33명, 양성 7명)",
        "80:20 (단위 미보고)", {"validation_subjects": val_n, "validation_positives": val_pos},
        val_n == 33 and val_pos == 7, "high",
        "논문 Accuracy는 모두 k/33, Recall은 모두 k/7로 분해된다. "
        "reproduction_spec.md §2 참조",
    ))

    report = {
        "data_root": str(root),
        "paper_claims": PAPER_CLAIMS,
        "checks": [asdict(c) for c in checks],
        "n_disagreements": sum(1 for c in checks if not c.agrees),
        "high_severity_disagreements": [
            c.key for c in checks if not c.agrees and c.severity == "high"
        ],
    }
    return report


def render_discrepancy_markdown(report: dict[str, Any]) -> str:
    """Render the inspection report as ``discrepancy_report.md``."""
    lines = [
        "# discrepancy_report.md — 논문 기재값 vs 실측값",
        "",
        f"- 데이터 경로: `{report['data_root']}`",
        f"- 불일치 항목 수: **{report['n_disagreements']}**",
        f"- 심각도 high 불일치: {report['high_severity_disagreements'] or '없음'}",
        "",
        "| 항목 | 논문 | 실측 | 일치 | 심각도 | 비고 |",
        "| --- | --- | --- | :---: | :---: | --- |",
    ]
    for check in report["checks"]:
        mark = "✅" if check["agrees"] else "❌"
        measured = str(check["measured"]).replace("|", "\\|")
        paper = str(check["paper"]).replace("|", "\\|")
        note = str(check["note"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {check['description']} | {paper} | {measured} | {mark} "
            f"| {check['severity']} | {note} |"
        )
    lines += [
        "",
        "> 이 표는 실행할 때마다 실제 CSV에서 다시 계산된다. `paper_data_mapping.md` §7의",
        "> 값과 달라지면 데이터 판본이 바뀐 것이므로 재현을 진행하기 전에 확인해야 한다.",
        "",
    ]
    return "\n".join(lines)
