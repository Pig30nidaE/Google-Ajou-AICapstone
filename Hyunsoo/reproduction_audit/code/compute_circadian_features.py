"""
1분 단위 활동(MET) 시퀀스로부터 표준 비모수 서캐디안 리듬 지표(IS/IV/RA/M10/L5)를
환자 단위로 계산한다. activity_day_start가 매일 04:00 KST로 고정되어 있고
1441분(=24h+1분) 시퀀스가 제공되므로, 마지막 1분을 버리고 1440분(24시간)으로
정렬해 시각(시간대)별 평균 프로파일을 만들 수 있다.

공식 (Van Someren et al. 1999 표준):
  p = 하루를 나누는 구간 수 (여기선 24, 시간 단위)
  n = 전체 관측치 수 (환자의 전체 일수 * 24)
  x̄_h = 시간대 h(0~23)의 전체 일 평균 (프로파일)
  X̄   = 전체 대평균

  IS = [n * Σ_h (x̄_h - X̄)^2] / [p * Σ_i (x_i - X̄)^2]
  IV = [n * Σ_i (x_i - x_{i-1})^2] / [(n-1) * Σ_i (x_i - X̄)^2]   (날짜순 연속 시계열 기준)

  M10 = 평균 프로파일(x̄_h, 24개)에서 원형(circular) 10시간 연속 구간 중 최댓값
  L5  = 평균 프로파일에서 원형 5시간 연속 구간 중 최솟값
  RA  = (M10 - L5) / (M10 + L5)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path("/Users/sasaek/코딩/ML/aihub_data")
OUT_DIR = Path("/private/tmp/claude-501/-Users-sasaek----ML-aihub-data/75346f87-d882-44a3-87ed-52441e217762/scratchpad/nia219_experiment")

TRAIN_ACTIVITY = DATA_ROOT / "1.Training/원천데이터/1.걸음걸이/train_activity.csv"
VAL_ACTIVITY = DATA_ROOT / "2.Validation/원천데이터/1.걸음걸이/val_activity.csv"
TRAIN_LABEL = DATA_ROOT / "1.Training/라벨링데이터/3.인지기능/training_label.csv"
VAL_LABEL = DATA_ROOT / "2.Validation/라벨링데이터/3.인지기능/val_label.csv"

SEQ_COL = "CONVERT(activity_met_1min USING utf8)"


def read_csv_flexible(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Failed to read {path}")


def preprocess_label(label_df: pd.DataFrame) -> pd.DataFrame:
    label_df = label_df.copy()
    if "SAMPLE_EMAIL" in label_df.columns:
        label_df = label_df.rename(columns={"SAMPLE_EMAIL": "EMAIL"})
    original_label_map = {"CN": 0, "MCI": 1, "Dem": 2, "Dementia": 2}
    label_df["original_label"] = label_df["DIAG_NM"].map(original_label_map)
    return label_df[["EMAIL", "DIAG_NM", "original_label"]].drop_duplicates("EMAIL")


def parse_hourly_profile(seq_str: str) -> np.ndarray | None:
    """1441분 시퀀스 -> 마지막 1분 버리고 1440분 -> 24시간 평균 벡터."""
    if pd.isna(seq_str):
        return None
    tokens = seq_str.strip().split("/")
    vals = []
    for t in tokens:
        t = t.strip()
        if not t or t == "...":
            vals.append(np.nan)
            continue
        try:
            vals.append(float(t))
        except ValueError:
            vals.append(np.nan)
    arr = np.array(vals, dtype=float)
    if len(arr) < 1440:
        return None
    arr = arr[:1440]
    hourly = arr.reshape(24, 60)
    return np.nanmean(hourly, axis=1)  # 길이 24


def compute_is_iv(day_matrix: np.ndarray) -> tuple[float, float]:
    """day_matrix: (n_days, 24) 시간대별 평균 활동량."""
    n_days, p = day_matrix.shape
    n = n_days * p
    grand_mean = np.nanmean(day_matrix)

    hourly_profile = np.nanmean(day_matrix, axis=0)  # (24,)
    ss_between_hours = np.nansum((hourly_profile - grand_mean) ** 2)
    ss_total = np.nansum((day_matrix - grand_mean) ** 2)
    IS = (n * ss_between_hours) / (p * ss_total) if ss_total > 0 else np.nan

    flat = day_matrix.flatten()
    diffs = np.diff(flat)
    ss_successive = np.nansum(diffs ** 2)
    IV = (n * ss_successive) / ((n - 1) * ss_total) if ss_total > 0 else np.nan

    return float(IS), float(IV)


def compute_m10_l5_ra(hourly_profile: np.ndarray) -> tuple[float, float, float]:
    """24시간 평균 프로파일에서 원형(circular) 최대10h/최소5h 구간 평균."""
    extended = np.concatenate([hourly_profile, hourly_profile])  # 원형 처리를 위해 2배 연장

    m10_vals = [np.mean(extended[i:i + 10]) for i in range(24)]
    l5_vals = [np.mean(extended[i:i + 5]) for i in range(24)]

    M10 = float(np.max(m10_vals))
    L5 = float(np.min(l5_vals))
    RA = (M10 - L5) / (M10 + L5) if (M10 + L5) != 0 else np.nan
    return M10, L5, RA


def build_circadian_features(activity_df: pd.DataFrame) -> pd.DataFrame:
    activity_df = activity_df.copy()
    activity_df["day_start_dt"] = pd.to_datetime(activity_df["activity_day_start"], errors="coerce")

    rows = []
    for email, grp in activity_df.groupby("EMAIL"):
        grp = grp.sort_values("day_start_dt")
        profiles = []
        for _, r in grp.iterrows():
            prof = parse_hourly_profile(r[SEQ_COL])
            if prof is not None:
                profiles.append(prof)
        if len(profiles) < 3:
            continue
        day_matrix = np.vstack(profiles)  # (n_days, 24)

        IS, IV = compute_is_iv(day_matrix)
        mean_profile = np.nanmean(day_matrix, axis=0)
        M10, L5, RA = compute_m10_l5_ra(mean_profile)

        rows.append({
            "EMAIL": email, "n_days": len(profiles),
            "IS": IS, "IV": IV, "M10": M10, "L5": L5, "RA": RA,
        })

    return pd.DataFrame(rows)


def main():
    print("[1/3] Loading raw activity CSVs...")
    train_act = read_csv_flexible(TRAIN_ACTIVITY)
    val_act = read_csv_flexible(VAL_ACTIVITY)
    all_act = pd.concat([train_act, val_act], ignore_index=True)

    print("[2/3] Computing IS/IV/M10/L5/RA per patient...")
    feats = build_circadian_features(all_act)

    print("[3/3] Merging labels...")
    train_label = preprocess_label(read_csv_flexible(TRAIN_LABEL))
    val_label = preprocess_label(read_csv_flexible(VAL_LABEL))
    labels = pd.concat([train_label, val_label], ignore_index=True).drop_duplicates("EMAIL")

    merged = feats.merge(labels, on="EMAIL", how="inner")
    merged["dementia_label"] = (merged["original_label"] == 2).astype(int)

    out_path = OUT_DIR / "circadian_features.csv"
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\nDone. n_patients={len(merged)}")
    print(merged["DIAG_NM"].value_counts())
    print(f"\nDementia(1) vs CN+MCI(0): {merged['dementia_label'].sum()} vs {(merged['dementia_label']==0).sum()}")
    print("\n요약 통계 (그룹별 평균):")
    print(merged.groupby("dementia_label")[["IS", "IV", "M10", "L5", "RA", "n_days"]].mean())
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
