"""
YMJ(2026) 충실 재현 [1/2] : 원본 raw 수면 데이터 -> 일별 파생지표 -> 환자단위 5계열 피처

논문에 명시된 사항을 그대로 따른다:
  * 야간 데이터만 사용 (주간 활동지표 step/energy는 명시적으로 제외)
  * Supplementary Table 1의 raw 변수 15개에서 출발
  * Table 3 / Fig.2(SHAP)에 등장하는 파생 일별 지표를 재구성
      N3_ratio, REM_ratio, NREM_ratio, N1_plus_N2_ratio, NREM_proportion,
      SRI, TST, WASO, SE, HR_drop_ratio, HR_drop_per_hour_nrem,
      Daily_sleep_count, Sleep_bedtime_start_num, Sleep_bedtime_end_num,
      Sleep_midpoint_time, Sleep_duration, Sleep_breath_average,
      Sleep_hr_average, Sleep_hr_lowest, Sleep_RMSSD
  * 전처리: 결측 타임스탬프 레코드 제거, 무한값 보정, NaN 없음 확인
  * 최소 야간수 >= 35 (baseline configuration)
  * 변동성 윈도우 7일 (baseline configuration)
  * 환자당 5개 계열로 집계:
      M    : mean
      EM   : median, trimmed mean, mode
      Dist : min, max, MAD, kurtosis, range
      Disp : SD, CV, IQR
      TS   : short-term variability, time-bin variability, rolling CV,
             moving range, time-bin change rate   (Suppl. Table 2 정의)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA_ROOT = Path("/Users/sasaek/코딩/ML/aihub_data")
OUT_DIR = Path(__file__).parent

TRAIN_SLEEP = DATA_ROOT / "1.Training/원천데이터/2.수면/train_sleep.csv"
VAL_SLEEP = DATA_ROOT / "2.Validation/원천데이터/2.수면/val_sleep.csv"
TRAIN_LABEL = DATA_ROOT / "1.Training/라벨링데이터/3.인지기능/training_label.csv"
VAL_LABEL = DATA_ROOT / "2.Validation/라벨링데이터/3.인지기능/val_label.csv"

HYP_COL = "CONVERT(sleep_hypnogram_5min USING utf8)"
HR_COL = "CONVERT(sleep_hr_5min USING utf8)"

MIN_NIGHTS = 35   # baseline configuration
W = 7             # 7-day variability window (baseline configuration)
B = 4             # time-bin count (Suppl. Table 2 default)
EPS = 1e-8

# 일별 파생지표 (논문 Table 3 / Fig.2 기준)
DAILY_METRICS = [
    "TST", "Sleep_duration", "SE", "WASO",
    "N3_ratio", "REM_ratio", "N1_plus_N2_ratio", "NREM_ratio", "NREM_proportion",
    "Sleep_midpoint_time", "Sleep_bedtime_start_num", "Sleep_bedtime_end_num",
    "Sleep_hr_average", "Sleep_hr_lowest", "Sleep_RMSSD", "Sleep_breath_average",
    "HR_drop_ratio", "HR_drop_per_hour_nrem",
    "Daily_sleep_count", "SRI",
]


# ----------------------------------------------------------------------------- IO
def read_csv_flexible(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Failed to read {path}")


def preprocess_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "SAMPLE_EMAIL" in df.columns:
        df = df.rename(columns={"SAMPLE_EMAIL": "EMAIL"})
    df["original_label"] = df["DIAG_NM"].map({"CN": 0, "MCI": 1, "Dem": 2, "Dementia": 2})
    return df[["EMAIL", "DIAG_NM", "original_label"]].drop_duplicates("EMAIL")


def parse_seq(s, drop_zero=False) -> np.ndarray:
    if pd.isna(s):
        return np.array([], dtype=float)
    vals = []
    for t in str(s).split("/"):
        t = t.strip()
        if not t or t == "...":
            continue
        try:
            vals.append(float(t))
        except ValueError:
            continue
    a = np.asarray(vals, dtype=float)
    a = a[a != -1]
    if drop_zero:
        a = a[a != 0]
    return a


# ------------------------------------------------------------------ daily metrics
def compute_daily_row(row) -> dict:
    """한 수면 레코드(=1박)에서 논문식 일별 파생지표를 계산."""
    out = {}

    total = row["sleep_total"]
    deep, light, rem, awake = row["sleep_deep"], row["sleep_light"], row["sleep_rem"], row["sleep_awake"]

    out["TST"] = total
    out["Sleep_duration"] = row["sleep_duration"]
    out["SE"] = row["sleep_efficiency"]
    out["WASO"] = awake

    denom = total if (pd.notna(total) and total > 0) else np.nan
    out["N3_ratio"] = deep / denom
    out["REM_ratio"] = rem / denom
    out["N1_plus_N2_ratio"] = light / denom
    out["NREM_ratio"] = (deep + light) / denom
    nrem_rem = deep + light + rem
    out["NREM_proportion"] = (deep + light) / nrem_rem if (pd.notna(nrem_rem) and nrem_rem > 0) else np.nan

    out["Sleep_midpoint_time"] = row["sleep_midpoint_time"]
    out["Sleep_bedtime_start_num"] = row["_start_hour"]
    out["Sleep_bedtime_end_num"] = row["_end_hour"]

    out["Sleep_hr_average"] = row["sleep_hr_average"]
    out["Sleep_hr_lowest"] = row["sleep_hr_lowest"]
    out["Sleep_RMSSD"] = row["sleep_rmssd"]
    out["Sleep_breath_average"] = row["sleep_breath_average"]

    # --- HR drop: 수면 개시 기준선 대비 최저심박 하강폭 -----------------------
    hr = parse_seq(row[HYP_COL] if False else row[HR_COL], drop_zero=True)
    if len(hr) >= 4:
        baseline = float(np.mean(hr[:3]))
        lowest = float(np.min(hr))
        drop = baseline - lowest
        out["HR_drop_ratio"] = drop / baseline if baseline > 0 else np.nan
    else:
        baseline, drop = np.nan, np.nan
        out["HR_drop_ratio"] = np.nan

    # NREM(=deep+light) 시간당 하강폭
    hyp = parse_seq(row[HYP_COL], drop_zero=True)
    nrem_hours = float(np.sum((hyp == 1) | (hyp == 2))) * 5.0 / 60.0 if len(hyp) else np.nan
    out["HR_drop_per_hour_nrem"] = (drop / nrem_hours) if (pd.notna(drop) and nrem_hours and nrem_hours > 0) else np.nan

    return out


def build_sleep_wake_grid(grp: pd.DataFrame):
    """참가자의 전체 기록기간을 5분 격자로 만들고 수면구간을 1로 마킹.
    SRI(수면 규칙성 지수) 계산용."""
    starts = grp["_start_dt"].dropna()
    ends = grp["_end_dt"].dropna()
    if len(starts) == 0 or len(ends) == 0:
        return None, None

    t0 = starts.min().floor("D")
    t1 = ends.max().ceil("D")
    n_epochs = int((t1 - t0).total_seconds() // 300)
    if n_epochs <= 0:
        return None, None

    grid = np.zeros(n_epochs, dtype=np.int8)
    for s, e in zip(grp["_start_dt"], grp["_end_dt"]):
        if pd.isna(s) or pd.isna(e) or e <= s:
            continue
        i0 = int((s - t0).total_seconds() // 300)
        i1 = int((e - t0).total_seconds() // 300)
        i0, i1 = max(i0, 0), min(i1, n_epochs)
        if i1 > i0:
            grid[i0:i1] = 1
    return grid, t0


def daily_sri(grid: np.ndarray) -> np.ndarray:
    """일별 SRI: 인접한 두 날의 동일 시각 수면/각성 상태 일치도 (-100~100)."""
    per_day = 288  # 24h / 5min
    n_days = len(grid) // per_day
    if n_days < 2:
        return np.array([])
    mat = grid[: n_days * per_day].reshape(n_days, per_day)
    vals = []
    for i in range(n_days - 1):
        concord = float(np.mean(mat[i] == mat[i + 1]))
        vals.append(-100.0 + 200.0 * concord)
    return np.asarray(vals, dtype=float)


def build_daily_table(sleep: pd.DataFrame) -> pd.DataFrame:
    num_cols = ["sleep_total", "sleep_duration", "sleep_efficiency", "sleep_awake",
                "sleep_light", "sleep_deep", "sleep_rem", "sleep_midpoint_time",
                "sleep_hr_average", "sleep_hr_lowest", "sleep_rmssd",
                "sleep_breath_average", "sleep_period_id", "sleep_is_longest"]
    for c in num_cols:
        sleep[c] = pd.to_numeric(sleep[c], errors="coerce")

    sleep["_start_dt"] = pd.to_datetime(sleep["sleep_bedtime_start"], errors="coerce", utc=True)
    sleep["_end_dt"] = pd.to_datetime(sleep["sleep_bedtime_end"], errors="coerce", utc=True)

    # [전처리] 결측 타임스탬프 레코드 제거
    before = len(sleep)
    sleep = sleep.dropna(subset=["_start_dt", "_end_dt"]).copy()
    print(f"  결측 타임스탬프 레코드 제거: {before - len(sleep)}건 (남은 {len(sleep)}건)")

    sleep["_start_dt"] = sleep["_start_dt"].dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    sleep["_end_dt"] = sleep["_end_dt"].dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    sleep["_start_hour"] = (sleep["_start_dt"].dt.hour + sleep["_start_dt"].dt.minute / 60
                            + sleep["_start_dt"].dt.second / 3600)
    sleep["_end_hour"] = (sleep["_end_dt"].dt.hour + sleep["_end_dt"].dt.minute / 60
                          + sleep["_end_dt"].dt.second / 3600)
    # 취침시작 시각은 자정을 넘나들므로 연속화 (18시=18, 1시=25)
    sleep["_start_hour"] = np.where(sleep["_start_hour"] < 12,
                                    sleep["_start_hour"] + 24, sleep["_start_hour"])
    sleep["_date"] = sleep["_end_dt"].dt.date

    rows = []
    for email, grp in sleep.groupby("EMAIL"):
        grp = grp.sort_values("_start_dt")

        # 하루 수면 에피소드 개수
        counts = grp.groupby("_date").size()

        # 주 수면구간(가장 긴 것)만 대표 레코드로 사용
        grp2 = grp.copy()
        grp2["_dur"] = (grp2["_end_dt"] - grp2["_start_dt"]).dt.total_seconds()
        main = (grp2.sort_values(["_date", "_dur"], ascending=[True, False])
                    .drop_duplicates("_date", keep="first"))

        # SRI (일별)
        grid, t0 = build_sleep_wake_grid(grp)
        sri_vals = daily_sri(grid) if grid is not None else np.array([])
        sri_by_date = {}
        if len(sri_vals) and t0 is not None:
            for i, v in enumerate(sri_vals):
                sri_by_date[(t0 + pd.Timedelta(days=i)).date()] = v

        for _, r in main.iterrows():
            d = compute_daily_row(r)
            d["EMAIL"] = email
            d["date"] = r["_date"]
            d["Daily_sleep_count"] = float(counts.get(r["_date"], 1))
            d["SRI"] = sri_by_date.get(r["_date"], np.nan)
            rows.append(d)

    daily = pd.DataFrame(rows)
    # [전처리] 무한값 보정
    n_inf = int(np.isinf(daily[DAILY_METRICS].to_numpy(dtype=float)).sum())
    daily[DAILY_METRICS] = daily[DAILY_METRICS].replace([np.inf, -np.inf], np.nan)
    print(f"  무한값 보정: {n_inf}개")
    return daily


# ------------------------------------------------- Suppl. Table 2 variability defs
def short_term_variability(x, w=W):
    if len(x) < w:
        return np.nan
    return float(np.mean([np.std(x[i:i + w], ddof=1) for i in range(len(x) - w + 1)]))


def rolling_cv(x, w=W, eps=EPS):
    if len(x) < w:
        return np.nan
    out = []
    for i in range(len(x) - w + 1):
        seg = x[i:i + w]
        out.append(np.std(seg, ddof=1) / (abs(np.mean(seg)) + eps))
    return float(np.mean(out))


def moving_range(x, k=1):
    if len(x) < k + 1:
        return np.nan
    return float(np.mean(np.abs(np.diff(x, n=k))))


def _bin_means(x, b=B):
    n = len(x)
    if n < b:
        return None
    m = n // b
    if m == 0:
        return None
    return x[: m * b].reshape(b, m).mean(axis=1)


def time_bin_variability(x, b=B):
    bm = _bin_means(x, b)
    return float(np.var(bm, ddof=1)) if bm is not None and len(bm) > 1 else np.nan


def time_bin_change_rate(x, b=B):
    bm = _bin_means(x, b)
    return float(np.mean(np.abs(np.diff(bm)))) if bm is not None and len(bm) >= 2 else np.nan


def summarize(x: np.ndarray, prefix: str, w: int | None = None) -> dict:
    """w: 변동성 윈도우. None이면 모듈 전역 W를 그때그때 참조한다.
    (기본인자로 W를 묶으면 정의 시점 값이 고정되어 감도분석이 무력화되므로 주의)"""
    if w is None:
        w = W
    x = x[~np.isnan(x)]
    keys = ["mean", "median", "trimmed_mean", "mode", "min", "max", "mad", "kurtosis",
            "range", "sd", "cv", "iqr", "stv", "tbv", "rcv", "mr", "tbcr"]
    if len(x) < 3:
        return {f"{prefix}_{k}": np.nan for k in keys}

    mean_ = float(np.mean(x))
    sd_ = float(np.std(x, ddof=1))
    rounded = np.round(x, 1)
    vals, cnts = np.unique(rounded, return_counts=True)

    return {
        f"{prefix}_mean": mean_,
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_trimmed_mean": float(stats.trim_mean(x, 0.1)),
        f"{prefix}_mode": float(vals[np.argmax(cnts)]),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_mad": float(np.mean(np.abs(x - mean_))),
        f"{prefix}_kurtosis": float(stats.kurtosis(x)) if len(x) > 3 else np.nan,
        f"{prefix}_range": float(np.max(x) - np.min(x)),
        f"{prefix}_sd": sd_,
        f"{prefix}_cv": sd_ / (abs(mean_) + EPS),
        f"{prefix}_iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        f"{prefix}_stv": short_term_variability(x, w),
        f"{prefix}_tbv": time_bin_variability(x),
        f"{prefix}_rcv": rolling_cv(x, w),
        f"{prefix}_mr": moving_range(x),
        f"{prefix}_tbcr": time_bin_change_rate(x),
    }


def aggregate_patient(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for email, grp in daily.groupby("EMAIL"):
        grp = grp.sort_values("date")
        feats = {"EMAIL": email, "n_nights": len(grp)}
        for m in DAILY_METRICS:
            feats.update(summarize(grp[m].to_numpy(dtype=float), m))
        rows.append(feats)
    return pd.DataFrame(rows)


def main():
    print("[1/4] 원본 수면 CSV 로드...")
    sleep = pd.concat([read_csv_flexible(TRAIN_SLEEP), read_csv_flexible(VAL_SLEEP)],
                      ignore_index=True)
    print(f"  총 수면 레코드: {len(sleep)}건")

    print("[2/4] 일별 파생지표 계산 (N3_ratio, HR_drop, SRI, WASO, SE, TST ...)...")
    daily = build_daily_table(sleep)
    print(f"  일별 테이블: {daily.shape}, 참가자 {daily['EMAIL'].nunique()}명")
    daily.to_csv(OUT_DIR / "ymj_faithful_daily.csv", index=False, encoding="utf-8-sig")

    print(f"[3/4] 환자단위 집계 (5계열 x {len(DAILY_METRICS)}지표, 변동성 윈도우 {W}일)...")
    pat = aggregate_patient(daily)

    print("[4/4] 라벨 병합 + 최소 야간수 필터 + CN/MCI 한정...")
    labels = pd.concat([preprocess_label(read_csv_flexible(TRAIN_LABEL)),
                        preprocess_label(read_csv_flexible(VAL_LABEL))],
                       ignore_index=True).drop_duplicates("EMAIL")
    m = pat.merge(labels, on="EMAIL", how="inner")
    print(f"  라벨 병합 후: {len(m)}명")

    before = len(m)
    m = m[m["n_nights"] >= MIN_NIGHTS].copy()
    print(f"  최소 {MIN_NIGHTS}박 필터: {before} -> {len(m)}명")

    m = m[m["original_label"].isin([0, 1])].reset_index(drop=True)   # AD(치매) 제외
    m["mci_label"] = m["original_label"].astype(int)

    feat_cols = [c for c in m.columns
                 if c not in {"EMAIL", "n_nights", "DIAG_NM", "original_label", "mci_label"}]
    nan_rate = m[feat_cols].isna().mean()
    print(f"\n  후보 피처: {len(feat_cols)}개")
    print(f"  결측률 >20% 피처: {int((nan_rate > 0.2).sum())}개  (논문: 결측 높은 변수 제외)")
    drop_cols = nan_rate[nan_rate > 0.2].index.tolist()
    if drop_cols:
        print(f"    제외: {drop_cols[:8]}{' ...' if len(drop_cols) > 8 else ''}")
        m = m.drop(columns=drop_cols)
        feat_cols = [c for c in feat_cols if c not in drop_cols]

    out = OUT_DIR / "ymj_faithful_features.csv"
    m.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n=== 최종 ===")
    print(f"  n = {len(m)}  (CN={int((m['mci_label']==0).sum())}, MCI={int((m['mci_label']==1).sum())})")
    print(f"  논문 보고치: n=162 (CN=111, MCI=51)")
    print(f"  야간수 평균 {m['n_nights'].mean():.1f}일 (논문: 평균 70일), 범위 {m['n_nights'].min()}~{m['n_nights'].max()}")
    print(f"  최종 후보 피처 {len(feat_cols)}개")
    print(f"  저장: {out}")


if __name__ == "__main__":
    main()
