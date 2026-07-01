"""
============================================================================
치매 라이프로그 전처리 (논문 재현)

Kim, J. & Lee, J. (2024) "Dementia Prediction using LSTM Model based on
Life-Log Data", Journal of KIIT 22(9), 123-132.
============================================================================

sample_activity.csv, sample_sleep.csv 를 입력으로
논문 3.2~3.3절 전처리를 그대로 재현한다.


  1) 날짜 결합 : 수면 종료일(date(sleep_bedtime_end)) 기준으로
                 활동(date(activity_day_start))과 하루치 결합
  2) BLOB 시계열 파싱 : '/' split -> float, 빈값/'...' 무시
                 (activity_class_5min 의 0 = Not worn 은 유효값이므로 보존)
  3) activity_met_1min(1분, 1440개) -> 5개씩 평균 -> 288개 activity_met_5min
  4) 나머지 5분 시계열 -> 288개로 패딩(부족 시 -1) 또는 truncate
  5) 연속형(288,5) / 비연속형(일단위 수치) 분리
       연속형 5채널 : activity_met_5min, activity_class_5min,
                      sleep_hr_5min, sleep_rmssd_5min, sleep_hypnogram_5min
  6) 7일 슬라이딩 윈도우 (step=1)
  7) 유저 단위 train/test 70:30 분할 (동일 유저가 양쪽에 안 걸치도록)
  8) MinMaxScaler : train 에서만 fit, -1 패딩값은 스케일링 제외 후 -1 유지
  9) 연속형 + 비연속형 통합 -> 단일 LSTM 입력
       timestep = 7일, 일자 feature = 연속형 flatten(288*5=1440) + 비연속형
       => X_lstm (n_window, 7, 1440 + n_disc)

     (RF/LGBM 용으로 비연속형만 7일 평균한 agg 도 별도 저장)


[주의] 이 CSV 에는 라벨(DIAG_NM/class)이 없으므로 X 만 저장하고
       모델 학습은 하지 않는다. 라벨 파일이 생기면 윈도우 meta 의 EMAIL 에
       매핑해 y 를 붙이고 모델 단계만 추가하면 된다 (논문은 이진: 정상/치매).
============================================================================
"""

from pathlib import Path
import glob
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 설정 (필요 시 경로만 수정)
# ---------------------------------------------------------------------------
INPUT_DIR = Path(".")                    # CSV 가 있는 폴더
OUT_DIR = Path("./paper_repro_output")   # 결과 저장 폴더

PAD = -1.0
DAY_LEN = 288          # 24h * 60 / 5
MIN1_LEN = 1440        # 24h * 60
WINDOW = 7
TEST_RATIO = 0.30
RANDOM_STATE = 42

CONT_ORDER = [
    "activity_met_5min", "activity_class_5min",
    "sleep_hr_5min", "sleep_rmssd_5min", "sleep_hypnogram_5min",
]
BLOB_BASE = ["activity_class_5min", "activity_met_1min",
             "sleep_hr_5min", "sleep_hypnogram_5min", "sleep_rmssd_5min"]
EXCLUDE_FROM_DISCRETE = set(
    ["EMAIL", "date",
     "activity_day_start", "activity_day_end",
     "sleep_bedtime_start", "sleep_bedtime_end"]
    + BLOB_BASE + [f"CONVERT({b} USING utf8)" for b in BLOB_BASE]
)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def find_csv(keyword: str) -> Path:
    """INPUT_DIR 에서 키워드가 포함된 csv 자동 탐색."""
    cands = sorted(INPUT_DIR.glob(f"*{keyword}*.csv"))
    if not cands:
        raise FileNotFoundError(f"'{keyword}' 가 포함된 csv 를 {INPUT_DIR} 에서 찾지 못함")
    return cands[0]


def read_csv_flexible(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"인코딩 실패: {path}")


# ---------------------------------------------------------------------------
# 시계열 파싱 / 표준화
# ---------------------------------------------------------------------------
def parse_slash_sequence(x):
    if pd.isna(x):
        return []
    out = []
    for p in str(x).strip().split("/"):
        p = p.strip()
        if p in ("", "..."):
            continue
        try:
            out.append(float(p))
        except ValueError:
            continue
    return out


def to_fixed_length(vals, length=DAY_LEN, pad=PAD):
    arr = list(vals)
    if len(arr) >= length:
        return np.asarray(arr[:length], dtype=float)
    return np.asarray(arr + [pad] * (length - len(arr)), dtype=float)


def met_1min_to_5min(vals, length=DAY_LEN, pad=PAD):
    """1분 met(1440) -> 5개씩 평균(-1 제외) -> 288."""
    arr = to_fixed_length(vals, length=MIN1_LEN, pad=pad).reshape(length, 5)
    out = np.full(length, pad, dtype=float)
    valid = arr != pad
    cnt = valid.sum(axis=1)
    s = np.where(valid, arr, 0.0).sum(axis=1)
    nz = cnt > 0
    out[nz] = s[nz] / cnt[nz]
    return out


def build_continuous(row):
    ch = {
        "activity_met_5min": met_1min_to_5min(
            parse_slash_sequence(row["CONVERT(activity_met_1min USING utf8)"])),
        "activity_class_5min": to_fixed_length(
            parse_slash_sequence(row["CONVERT(activity_class_5min USING utf8)"])),
        "sleep_hr_5min": to_fixed_length(
            parse_slash_sequence(row["CONVERT(sleep_hr_5min USING utf8)"])),
        "sleep_rmssd_5min": to_fixed_length(
            parse_slash_sequence(row["CONVERT(sleep_rmssd_5min USING utf8)"])),
        "sleep_hypnogram_5min": to_fixed_length(
            parse_slash_sequence(row["CONVERT(sleep_hypnogram_5min USING utf8)"])),
    }
    return np.stack([ch[c] for c in CONT_ORDER], axis=1)  # (288, 5)


# ---------------------------------------------------------------------------
# 날짜 결합
# ---------------------------------------------------------------------------
def make_daily_table(activity, sleep):
    activity = activity.copy()
    sleep = sleep.copy()
    activity["date"] = pd.to_datetime(activity["activity_day_start"], errors="coerce").dt.date
    sleep["_e"] = pd.to_datetime(sleep["sleep_bedtime_end"], errors="coerce")
    sleep["_s"] = pd.to_datetime(sleep["sleep_bedtime_start"], errors="coerce")
    sleep["date"] = sleep["_e"].dt.date
    sleep["_dur"] = (sleep["_e"] - sleep["_s"]).dt.total_seconds()

    # 같은 EMAIL+date 수면 여러 건 -> 가장 긴 수면(main sleep) 1개
    sleep = (sleep.sort_values(["EMAIL", "date", "_dur"], ascending=[True, True, False])
                  .drop_duplicates(["EMAIL", "date"], keep="first")
                  .reset_index(drop=True))

    daily = (activity.merge(sleep, on=["EMAIL", "date"], how="inner")
                     .sort_values(["EMAIL", "date"]).reset_index(drop=True))
    return daily


def select_discrete_columns(daily):
    cols = []
    for c in daily.columns:
        if c in EXCLUDE_FROM_DISCRETE or c.endswith("_dt") or c.startswith("_"):
            continue
        if pd.to_numeric(daily[c], errors="coerce").notna().mean() >= 0.5:
            cols.append(c)
    return cols


# ---------------------------------------------------------------------------
# 7일 윈도우 / 분할 / 스케일
# ---------------------------------------------------------------------------
def make_windows(daily, cont_arrays, discrete_cols, window=WINDOW):
    X_cont, X_disc, meta = [], [], []
    disc_mat = daily[discrete_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    for email, idx in daily.groupby("EMAIL").groups.items():
        idx = list(idx)
        if len(idx) < window:
            continue
        for s in range(len(idx) - window + 1):
            win = idx[s:s + window]
            X_cont.append(np.stack([cont_arrays[i] for i in win], axis=0))  # (7,288,5)
            X_disc.append(disc_mat[win])                                    # (7,n_disc)
            meta.append({"EMAIL": email,
                         "start_date": daily.loc[win[0], "date"],
                         "end_date": daily.loc[win[-1], "date"]})
    return (np.asarray(X_cont, dtype=float),
            np.asarray(X_disc, dtype=float),
            pd.DataFrame(meta))


def user_split(meta, test_ratio=TEST_RATIO, seed=RANDOM_STATE):
    users = np.array(sorted(meta["EMAIL"].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(users)
    n_test = max(1, int(round(len(users) * test_ratio)))
    test_users = set(users[:n_test])
    is_test = meta["EMAIL"].isin(test_users).to_numpy()
    return ~is_test, is_test, sorted(set(users) - test_users), sorted(test_users)


def fit_minmax_cont(X, pad=PAD):
    flat = X.reshape(-1, X.shape[-1])
    mins, maxs = np.zeros(X.shape[-1]), np.ones(X.shape[-1])
    for c in range(X.shape[-1]):
        v = flat[:, c]; v = v[v != pad]
        if v.size:
            mins[c], maxs[c] = v.min(), v.max()
    return mins, maxs


def apply_minmax_cont(X, mins, maxs, pad=PAD):
    Xs = X.copy()
    for c in range(X.shape[-1]):
        rng = (maxs[c] - mins[c]) or 1.0
        sl = Xs[..., c]; mask = sl != pad
        Xs[..., c] = np.where(mask, (sl - mins[c]) / rng, pad)
    return Xs


def fit_minmax_disc(X):
    flat = X.reshape(-1, X.shape[-1])
    return np.nanmin(flat, axis=0), np.nanmax(flat, axis=0)


def apply_minmax_disc(X, mins, maxs):
    rng = maxs - mins; rng[rng == 0] = 1.0
    return np.nan_to_num((X - mins) / rng, nan=0.0)


# ===========================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1] load")
    activity = read_csv_flexible(find_csv("activity"))
    sleep = read_csv_flexible(find_csv("sleep"))
    n_subj = activity["EMAIL"].nunique()
    print(f"    activity {activity.shape} | sleep {sleep.shape} | subjects {n_subj}")

    print("[2] merge by sleep_bedtime_end date")
    daily = make_daily_table(activity, sleep)
    loss = len(activity) - len(daily)
    print(f"    daily rows {len(daily)} | subjects {daily['EMAIL'].nunique()} | merge loss {loss}")

    print("[3-5] build continuous (288,5) + select discrete")
    cont_arrays = [build_continuous(daily.iloc[i]) for i in range(len(daily))]
    discrete_cols = select_discrete_columns(daily)
    print(f"    n_discrete {len(discrete_cols)}")

    print("[6] 7-day sliding windows")
    X_cont, X_disc, meta = make_windows(daily, cont_arrays, discrete_cols)
    print(f"    X_cont {X_cont.shape} | windows {len(meta)}")

    print("[7] user-level 70/30 split")
    tr, te, tr_u, te_u = user_split(meta)
    assert set(tr_u).isdisjoint(set(te_u)), "user leakage!"
    print(f"    train {tr.sum()} ({len(tr_u)} users) | test {te.sum()} ({len(te_u)} users)")

    print("[8] MinMaxScaler (fit on train, keep -1)")
    cmin, cmax = fit_minmax_cont(X_cont[tr])
    Xc_tr, Xc_te = apply_minmax_cont(X_cont[tr], cmin, cmax), apply_minmax_cont(X_cont[te], cmin, cmax)
    dmin, dmax = fit_minmax_disc(X_disc[tr])
    Xd_tr, Xd_te = apply_minmax_disc(X_disc[tr], dmin, dmax), apply_minmax_disc(X_disc[te], dmin, dmax)

    print("[9] integrate continuous + discrete -> single LSTM input")
    def integrate(Xc, Xd):
        n, w, t, ch = Xc.shape
        return np.concatenate([Xc.reshape(n, w, t * ch), Xd], axis=2)
    X_lstm_tr, X_lstm_te = integrate(Xc_tr, Xd_tr), integrate(Xc_te, Xd_te)
    Xd_tr_agg, Xd_te_agg = Xd_tr.mean(axis=1), Xd_te.mean(axis=1)  # RF/LGBM 용

    print("[10] save (no labels -> X only)")
    np.save(OUT_DIR / "X_lstm_train.npy", X_lstm_tr)
    np.save(OUT_DIR / "X_lstm_test.npy", X_lstm_te)
    np.save(OUT_DIR / "X_discrete_agg_train.npy", Xd_tr_agg)
    np.save(OUT_DIR / "X_discrete_agg_test.npy", Xd_te_agg)
    meta.loc[tr].reset_index(drop=True).to_csv(OUT_DIR / "meta_train.csv", index=False, encoding="utf-8-sig")
    meta.loc[te].reset_index(drop=True).to_csv(OUT_DIR / "meta_test.csv", index=False, encoding="utf-8-sig")

    n_disc = len(discrete_cols)
    with open(OUT_DIR / "feature_reference.txt", "w", encoding="utf-8") as f:
        f.write(f"X_lstm shape (n,7,{DAY_LEN*5+n_disc}) | timestep=7일\n")
        f.write(f"  [0:{DAY_LEN*5}] 연속형 288slot x 5ch flatten (slot s, ch c -> s*5+c)\n")
        f.write(f"  [{DAY_LEN*5}:{DAY_LEN*5+n_disc}] 비연속형 {n_disc}개\n\n")
        f.write("연속형 채널 순서: " + ", ".join(CONT_ORDER) + "\n\n")
        f.write("비연속형 컬럼:\n")
        for i, c in enumerate(discrete_cols):
            f.write(f"  [{DAY_LEN*5+i}] {c}\n")

    # ---- 검증 리포트 ----
    print("\n=== 검증 리포트 ===")
    print(f"인원: {n_subj}  (논문 300명)")
    print(f"병합 행: {len(daily)}  (논문 9705)  | 병합 손실: {loss}")
    print(f"X_lstm_train: {X_lstm_tr.shape}  (feature {DAY_LEN*5+n_disc} = 288*5 + {n_disc})")
    cp = X_lstm_tr[..., :DAY_LEN*5]
    print(f"연속형 -1 패딩 보존: {bool((cp==PAD).any())}"
          f" | 유효값 범위 [{cp[cp!=PAD].min():.2f}, {cp[cp!=PAD].max():.2f}]")
    print(f"비연속형 범위 [{X_lstm_tr[...,DAY_LEN*5:].min():.2f},"
          f" {X_lstm_tr[...,DAY_LEN*5:].max():.2f}]")
    print(f"\n저장 위치: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
