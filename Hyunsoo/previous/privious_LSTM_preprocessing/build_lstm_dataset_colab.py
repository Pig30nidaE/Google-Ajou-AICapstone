# Auto-generated Python script converted from a Jupyter notebook.
# Source notebook: Hyunsoo/previous/privious_LSTM_preprocessing/build_lstm_dataset_colab.ipynb
# Do not edit this generated file if you need exact notebook parity; edit the source notebook or copy this file first.

from __future__ import annotations

# Notebook compatibility helpers. Generated to keep notebook shell/magic cells runnable as Python.
import os as _NOTEBOOK_OS
import subprocess as _NOTEBOOK_SUBPROCESS
from pathlib import Path as _NOTEBOOK_PATH


def _NOTEBOOK_RUN_SHELL(command: str) -> None:
    _NOTEBOOK_SUBPROCESS.run(command, shell=True, check=True)


def _NOTEBOOK_RUN_BASH(script: str) -> None:
    _NOTEBOOK_SUBPROCESS.run(script, shell=True, executable="/bin/bash", check=True)


def _NOTEBOOK_CD(path: str) -> None:
    _NOTEBOOK_OS.chdir(_NOTEBOOK_OS.path.expanduser(path))
    print(_NOTEBOOK_PATH.cwd())


# %% [markdown] cell 1
# # LSTM 최종 데이터셋 생성 Colab 노트북
#
# `LSTM/build_lstm_dataset.py`를 Colab 공유드라이브 환경에서 실행할 수 있게 변환한 노트북입니다.
#
# 이 스크립트는 최종 LSTM 학습 입력인 `lstm_dataset.pkl`, `lstm_window_index.csv`, `lstm_discrete_daily.csv`, `lstm_dataset_guide.md`를 생성합니다. 기존 guide 기준으로 `X_integrated_seq`, `X_continuous_seq`, `X_discrete_seq`, `y`, 환자/날짜 메타, scaler 기준까지 포함하므로 LSTM 모델팀에 넘기는 최종 데이터셋 생성 파일로 보아도 됩니다.

# %% cell 2
# Colab 기본 환경에는 대부분 포함되어 있지만, 버전 차이를 줄이기 위해 명시적으로 설치합니다.
_NOTEBOOK_RUN_SHELL('pip -q install pandas numpy')

# %% cell 3
from google.colab import drive

drive.mount('/content/drive')

# %% [markdown] cell 4
# ## 경로 설정
#
# 공유드라이브 이름을 알고 있으면 `SHARED_DRIVE_NAME`에 입력하세요. 비워두면 MyDrive와 모든 공유드라이브에서 `ML/data` 폴더를 자동 탐색합니다.

# %% cell 5
from pathlib import Path
import sys

# 사용자 요청에 따라 DATA_ROOT를 직접 지정합니다.
DATA_ROOT = Path('/content/drive/MyDrive/GoogleAI_contest/aihub_original_data')

# DATA_ROOT의 부모 디렉토리를 프로젝트 루트로 설정하여 일관성을 유지합니다.
PROJECT_ROOT = DATA_ROOT.parent

# 출력 디렉토리는 PROJECT_ROOT 아래에 'LSTM' 폴더로 설정합니다.
OUTPUT_DIR = PROJECT_ROOT / 'LSTM'

print('PROJECT_ROOT =', PROJECT_ROOT)
print('DATA_ROOT exists =', DATA_ROOT.exists(), DATA_ROOT)
print('OUTPUT_DIR =', OUTPUT_DIR)

# %% [markdown] cell 6
# ## 원본 전처리 코드
#
# 노트북에서 `DATA_ROOT`, `OUTPUT_DIR`를 먼저 지정할 수 있도록 경로 상수만 Colab 친화적으로 감쌌고, 나머지 전처리/검증 로직은 원본 `.py` 기준입니다.

# %% cell 7
"""
LSTM 학습용 전처리 데이터 생성 스크립트.

전체 흐름:
1. train/val 라벨 파일을 읽어 진단명을 0/1 이진 라벨로 바꿉니다.
2. 활동(activity)과 수면(sleep) 원천 데이터를 각각 날짜 단위로 정리합니다.
3. 5분 단위로 반복 측정된 continuous 데이터와 하루 단위 discrete 데이터를 따로 만듭니다.
4. 결측치 처리와 MinMax 정규화를 train 기준으로만 계산하고 val에는 같은 기준을 적용합니다.
5. 환자별로 연속 7일씩 묶어 LSTM에 넣을 수 있는 window 데이터를 만듭니다.
6. continuous와 discrete window가 모두 있는 날짜만 맞춰 최종 pickle/CSV/guide 파일로 저장합니다.

용어 정리:
- continuous: 하루 안에서 5분 간격으로 이어지는 값입니다. 예: 심박, 수면 단계, 활동량.
- discrete: 하루를 대표하는 하나의 값입니다. 예: 총 수면시간, 평균 점수, 총 활동량.
- window: LSTM이 한 번에 보는 연속 7일 묶음입니다.
"""

# Moved to file top: from __future__ import annotations

import argparse
import pickle
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ===== 기본 설정 =====
# 데이터 폴더 위치입니다. 폴더 구조가 달라지면 이 값만 바꾸면 됩니다.
try:
    DATA_ROOT
except NameError:
    DATA_ROOT = Path("data")

try:
    OUTPUT_DIR
except NameError:
    OUTPUT_DIR = Path(".")

# LSTM은 한 샘플을 "연속 7일"로 봅니다.
SEQ_DAYS = 7

# 하루 24시간을 5분 단위로 나누면 24 * 60 / 5 = 288칸입니다.
DAILY_STEPS = 288

# 실제 값이 없거나 하루 길이가 288칸보다 짧을 때 채워 넣는 표시값입니다.
PADDING_VALUE = -1.0

# train 데이터에서 결측률이 50% 이상인 discrete feature는 제거합니다.
MISSING_THRESHOLD = 0.50
TIMEZONE = "Asia/Seoul"

# 모델 입력값이 아니라 샘플을 설명하는 기본 정보 컬럼입니다.
META_COLS = ["patient_id", "sample_date", "split", "binary_class"]

# 진단명을 모델이 이해할 수 있는 숫자 라벨로 바꿉니다. 0=정상, 1=인지저하/치매 계열.
LABEL_MAP = {"CN": 0, "NORMAL": 0, "MCI": 1, "DEMENTIA": 1, "DEM": 1, "AD": 1}

# 원본 파일마다 ID/라벨 컬럼명이 조금씩 다를 수 있어서 후보 이름을 넉넉히 둡니다.
ID_CANDIDATES = ["user_id", "subject_id", "participant_id", "patient_id", "sample_email", "email", "SAMPLE_EMAIL", "EMAIL"]
LABEL_CANDIDATES = ["DIAG_NM", "diagnosis", "diag_nm", "label", "class", "target"]

# discrete feature를 고를 때 제외할 단어들입니다. 날짜, ID, 라벨, 긴 시계열 컬럼은 모델 입력에서 뺍니다.
EXCLUDE_KEYWORDS = [
    "5min", "1min", "timestamp", "datetime", "date", "time", "start", "end",
    "convert(", "using utf8", "email", "id", "period", "diag", "doctor",
    "sample", "split", "class", "label",
]

# LSTM에 넣을 5분 단위 continuous feature 목록입니다.
CONTINUOUS_FEATURE_NAMES = [
    "activity_class_5min",
    "activity_met_5min",
    "sleep_hr_5min",
    "sleep_hypnogram_5min",
    "sleep_rmssd_5min",
]
ACTIVITY_CONT_FEATURES = CONTINUOUS_FEATURE_NAMES[:2]
SLEEP_CONT_FEATURES = CONTINUOUS_FEATURE_NAMES[2:]


def read_csv_any(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """CSV 파일의 문자 인코딩이 제각각일 수 있어 여러 방식으로 읽어봅니다."""
    last_error = None
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc, nrows=nrows, low_memory=False)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"CSV 읽기 실패: {path}") from last_error


def csv_in(folder: Path, stem: str) -> Path:
    """폴더 안에서 이름에 stem이 들어간 CSV 파일 하나를 찾습니다."""
    files = sorted({p for pat in ("*.csv", "*.csv.part*", "*.CSV", "*.CSV.part*") for p in folder.glob(pat)})
    files = [p for p in files if stem.lower() in p.name.lower()] or files
    if not files:
        raise FileNotFoundError(f"CSV 파일을 찾지 못했습니다: {folder}")
    return files[0]


def pick_col(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    """데이터셋마다 다른 컬럼명 중에서 목적에 맞는 컬럼을 자동으로 고릅니다."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for cand in candidates:
        hit = next((c for c in df.columns if cand.lower() in c.lower()), None)
        if hit:
            return hit
    raise KeyError(f"{purpose} 컬럼 자동 탐지 실패: {list(df.columns)}")


def roots(split: str) -> Path:
    """train/val 이름을 실제 폴더명으로 바꿉니다."""
    return DATA_ROOT / ("1.Training" if split == "train" else "2.Validation")


def local_dt(s: pd.Series) -> pd.Series:
    """원본 시각을 한국 시간 기준 날짜/시간으로 맞춥니다."""
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(TIMEZONE)


def numeric(s: pd.Series) -> pd.Series:
    """문자처럼 들어온 숫자를 실제 숫자로 바꾸고, 바꿀 수 없는 값은 결측치로 둡니다."""
    return pd.to_numeric(s, errors="coerce")


def load_labels() -> dict[str, pd.DataFrame]:
    """진단 라벨을 읽어 환자별 0/1 정답값을 만들고 train/val ID 중복을 검사합니다."""
    labels = {}
    for split in ("train", "val"):
        df = read_csv_any(csv_in(roots(split) / "라벨링데이터" / "3.인지기능", "label"))
        df = df[[pick_col(df, ID_CANDIDATES, "환자 ID"), pick_col(df, LABEL_CANDIDATES, "진단 라벨")]]
        df.columns = ["patient_id", "diagnosis"]
        df["patient_id"] = df["patient_id"].astype(str).str.strip()
        df["diagnosis"] = df["diagnosis"].astype(str).str.strip()
        df["binary_class"] = df["diagnosis"].str.upper().map(LABEL_MAP)
        unknown = sorted(df.loc[df["binary_class"].isna(), "diagnosis"].unique())
        assert not unknown, f"알 수 없는 진단 라벨: {unknown}"
        assert (df.groupby("patient_id")["binary_class"].nunique() <= 1).all(), f"{split} 라벨 충돌"
        labels[split] = df.sort_values(["patient_id", "diagnosis"]).drop_duplicates("patient_id").reset_index(drop=True)

    overlap = set(labels["train"]["patient_id"]) & set(labels["val"]["patient_id"])
    assert not overlap, f"train/val patient_id 중복: {sorted(overlap)[:10]}"
    return labels


def parse_seq(value: object) -> list[float]:
    """CSV 한 칸에 '1/2/3'처럼 들어있는 시계열 문자열을 숫자 목록으로 풉니다."""
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text in {"", "...", "nan", "None"}:
        return []
    vals = []
    for token in re.split(r"[/,]", text):
        token = token.strip()
        if not token:
            continue
        try:
            vals.append(float(token))
        except ValueError:
            pass
    return vals


def mode(values: list[float]) -> float:
    """범주형 값 여러 개 중 가장 자주 나온 값을 고릅니다."""
    clean = [v for v in values if not pd.isna(v)]
    return Counter(clean).most_common(1)[0][0] if clean else np.nan


def one_min_to_5min(values: list[float], rule: str) -> list[float]:
    """1분 단위 값 5개를 묶어 5분 단위 값 1개로 줄입니다."""
    out = []
    for i in range(0, len(values), 5):
        chunk = values[i:i + 5]
        out.append(float(np.nanmean(chunk)) if rule == "mean" and chunk else mode(chunk))
    return out


def to_288(values: list[float]) -> np.ndarray:
    """하루 시계열 길이를 항상 288칸으로 맞춥니다."""
    arr = np.full(DAILY_STEPS, PADDING_VALUE, dtype=np.float32)
    if values:
        vals = np.asarray(values[:DAILY_STEPS], dtype=np.float32)
        vals[np.isnan(vals)] = PADDING_VALUE
        arr[:len(vals)] = vals
    return arr


def convert_col(df: pd.DataFrame, base: str) -> str:
    """일부 CSV에 남아 있는 CONVERT(...) 형태의 컬럼명을 원래 feature명과 맞춰 찾습니다."""
    candidates = [f"CONVERT({base} USING utf8)", base]
    return next((c for c in candidates if c in df.columns), base)


def build_activity_continuous(split: str) -> dict[tuple[str, str], np.ndarray]:
    """활동 데이터를 환자-날짜별 5분 단위 행렬로 정리합니다."""
    df = read_csv_any(csv_in(roots(split) / "원천데이터" / "1.걸음걸이", "activity"))
    df = df.rename(columns={pick_col(df, ID_CANDIDATES, "환자 ID"): "patient_id"})
    ts = local_dt(df[pick_col(df, ["activity_day_start", "activity_start_time", "activity_day_end", "timestamp", "datetime", "time"], "활동 타임스탬프")])
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["sample_date"] = ts.dt.date

    # 새벽 0~3시는 전날 활동의 연장으로 보고 날짜를 전날로 붙입니다.
    df.loc[ts.dt.hour < 4, "sample_date"] = (ts[ts.dt.hour < 4] - pd.Timedelta(days=1)).dt.date
    df["sample_date"] = df["sample_date"].astype(str)

    class_col = convert_col(df, "activity_class_5min")
    met_1min_col = convert_col(df, "activity_met_1min")
    daily = {}
    for row in df.itertuples(index=False):
        # activity_class는 이미 5분 단위이고, activity_met은 1분 단위를 5분 평균으로 변환합니다.
        cls = to_288(parse_seq(row[df.columns.get_loc(class_col)]))
        met = to_288(one_min_to_5min(parse_seq(row[df.columns.get_loc(met_1min_col)]), "mean"))
        daily[(str(row[df.columns.get_loc("patient_id")]), str(row[df.columns.get_loc("sample_date")]))] = np.stack([cls, met], axis=1)
    return daily


def build_sleep_continuous(split: str) -> dict[tuple[str, str], np.ndarray]:
    """수면 데이터를 환자-수면종료일별 5분 단위 행렬로 정리합니다."""
    df = read_csv_any(csv_in(roots(split) / "원천데이터" / "2.수면", "sleep"))
    df = df.rename(columns={pick_col(df, ID_CANDIDATES, "환자 ID"): "patient_id"})
    end_col = pick_col(df, ["sleep_bedtime_end", "sleep_end_time", "bedtime_end", "end_time", "end"], "수면 종료 시각")
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["sample_date"] = local_dt(df[end_col]).dt.date.astype(str)

    cols = [convert_col(df, c) for c in ("sleep_hr_5min", "sleep_hypnogram_5min", "sleep_rmssd_5min")]
    duration = numeric(df["sleep_duration"]) if "sleep_duration" in df.columns else pd.Series([0] * len(df))
    df["_duration"] = duration

    # 0초 이하이거나 24시간을 넘는 수면 기록은 비정상 기록으로 보고 제외합니다.
    df = df[(df["_duration"] > 0) & (df["_duration"] <= 24 * 60 * 60)].copy()

    # 같은 날짜에 여러 수면 기록이 있으면 가장 긴 기록을 대표값으로 사용합니다.
    df = df.sort_values("_duration", ascending=False).drop_duplicates(["patient_id", "sample_date"])

    daily = {}
    for row in df.itertuples(index=False):
        seqs = [to_288(parse_seq(row[df.columns.get_loc(c)])) for c in cols]
        daily[(str(row[df.columns.get_loc("patient_id")]), str(row[df.columns.get_loc("sample_date")]))] = np.stack(seqs, axis=1)
    return daily


def build_daily_continuous(labels: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, int]]:
    """활동과 수면이 둘 다 있는 날짜만 남겨 continuous 일 단위 데이터를 만듭니다."""
    rows = []
    stats = {"activity_only_days": 0, "sleep_only_days": 0}
    for split in ("train", "val"):
        activity = build_activity_continuous(split)
        sleep = build_sleep_continuous(split)
        label_map = labels[split].set_index("patient_id")["binary_class"].to_dict()
        activity_keys, sleep_keys = set(activity), set(sleep)
        stats["activity_only_days"] += len(activity_keys - sleep_keys)
        stats["sleep_only_days"] += len(sleep_keys - activity_keys)

        # 두 데이터가 모두 있는 환자-날짜만 LSTM 입력 후보로 유지합니다.
        for key in sorted(activity_keys & sleep_keys):
            patient_id, sample_date = key
            if patient_id not in label_map:
                continue
            rows.append({
                "patient_id": patient_id,
                "sample_date": sample_date,
                "split": split,
                "binary_class": int(label_map[patient_id]),
                "matrix": np.concatenate([activity[key], sleep[key]], axis=1).astype(np.float32),
            })
    out = pd.DataFrame(rows).sort_values(["split", "patient_id", "sample_date"]).reset_index(drop=True)
    assert out.duplicated(["patient_id", "sample_date"]).sum() == 0, "continuous daily 중복"
    return out, stats


def is_sequence_like(s: pd.Series) -> bool:
    """값 안에 /, [], {} 등이 자주 보이면 하루 요약값이 아니라 시계열 컬럼으로 판단합니다."""
    sample = s.dropna().astype(str).head(200)
    return False if sample.empty else sample.str.contains(r"[/,\[\]\{\}]", regex=True).mean() > 0.05


def discrete_features(df: pd.DataFrame) -> list[str]:
    """모델에 넣을 하루 단위 숫자 feature만 골라냅니다."""
    cols = []
    for col in df.columns:
        name = col.lower()
        if name.startswith("_") or name in {"patient_id", "sample_date"}:
            continue
        if any(k in name for k in EXCLUDE_KEYWORDS) or is_sequence_like(df[col]):
            continue
        if numeric(df[col]).notna().any():
            cols.append(col)
    return cols


def build_activity_discrete(split: str) -> pd.DataFrame:
    """활동 데이터에서 하루 단위 요약 feature를 만들고 환자-날짜별로 합칩니다."""
    df = read_csv_any(csv_in(roots(split) / "원천데이터" / "1.걸음걸이", "activity"))
    df = df.rename(columns={pick_col(df, ID_CANDIDATES, "환자 ID"): "patient_id"})
    ts = local_dt(df[pick_col(df, ["activity_day_start", "activity_start_time", "activity_day_end", "timestamp", "datetime", "time"], "활동 타임스탬프")])
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["sample_date"] = ts.dt.date

    # 새벽 활동은 직전 날짜의 활동으로 계산해 하루 경계가 어긋나는 문제를 줄입니다.
    df.loc[ts.dt.hour < 4, "sample_date"] = (ts[ts.dt.hour < 4] - pd.Timedelta(days=1)).dt.date
    df["sample_date"] = df["sample_date"].astype(str)
    feats = discrete_features(df)
    df[feats] = df[feats].apply(numeric)

    # 평균 성격의 지표는 mean, 시간/횟수/총량 성격의 지표는 sum으로 하루 값을 만듭니다.
    agg = {c: ("mean" if any(k in c.lower() for k in ("score", "average", "efficiency", "met")) else "sum") for c in feats}
    return df.dropna(subset=["patient_id", "sample_date"]).groupby(["patient_id", "sample_date"], as_index=False).agg(agg)


def build_sleep_discrete(split: str) -> pd.DataFrame:
    """수면 데이터에서 하루 단위 요약 feature를 만들고 대표 수면 기록만 남깁니다."""
    df = read_csv_any(csv_in(roots(split) / "원천데이터" / "2.수면", "sleep"))
    df = df.rename(columns={pick_col(df, ID_CANDIDATES, "환자 ID"): "patient_id"})
    end_col = pick_col(df, ["sleep_bedtime_end", "sleep_end_time", "bedtime_end", "end_time", "end"], "수면 종료 시각")
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["sample_date"] = local_dt(df[end_col]).dt.date.astype(str)
    df["_duration"] = numeric(df["sleep_duration"]) if "sleep_duration" in df.columns else np.nan

    # 수면 시간이 비정상인 기록은 제거합니다.
    df = df[(df["_duration"] > 0) & (df["_duration"] <= 24 * 60 * 60)].copy()
    feats = discrete_features(df)
    df[feats] = df[feats].apply(numeric)

    # 같은 날짜에 기록이 여러 개면 가장 긴 수면 기록 하나만 대표로 사용합니다.
    out = df.sort_values("_duration", ascending=False).drop_duplicates(["patient_id", "sample_date"])
    return out[["patient_id", "sample_date", *feats]].reset_index(drop=True)


def build_daily_discrete(labels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """활동 요약값과 수면 요약값을 날짜 기준으로 붙이고 라벨을 연결합니다."""
    parts = []
    for split in ("train", "val"):
        daily = build_activity_discrete(split).merge(build_sleep_discrete(split), on=["patient_id", "sample_date"], how="inner")
        daily = daily.merge(labels[split][["patient_id", "binary_class"]], on="patient_id", how="inner")
        daily["split"] = split
        parts.append(daily)
    out = pd.concat(parts, ignore_index=True)
    feats = [c for c in out.columns if c not in META_COLS]
    out[feats] = out[feats].apply(numeric)
    assert out.duplicated(["patient_id", "sample_date"]).sum() == 0, "discrete daily 중복"
    return out[[*feats, *META_COLS]].sort_values(["split", "patient_id", "sample_date"]).reset_index(drop=True)


def fit_transform_minmax(values: np.ndarray, train_mask: np.ndarray, feature_names: list[str], keep_padding: bool) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """train 데이터의 최소/최대값을 기준으로 모든 값을 0~1 범위로 바꿉니다."""
    out = values.astype(np.float32, copy=True)
    params = {}
    for i, name in enumerate(feature_names):
        if out.ndim in (3, 4):
            train_vals = out[train_mask, ..., i].reshape(-1)
            target = out[..., i]
        else:
            train_vals = out[train_mask, i]
            target = out[:, i]

        # padding 값(-1)은 실제 측정값이 아니므로 정규화 기준 계산에서 제외합니다.
        valid = train_vals[train_vals != PADDING_VALUE] if keep_padding else train_vals
        valid = valid[~np.isnan(valid)]
        mn = float(valid.min()) if len(valid) else 0.0
        mx = float(valid.max()) if len(valid) else mn
        denom = mx - mn
        mask = (target != PADDING_VALUE) & ~np.isnan(target) if keep_padding else ~np.isnan(target)

        # val 값이 train 범위를 벗어나도 모델 입력 범위가 0~1을 넘지 않도록 자릅니다.
        target[mask] = 0.0 if denom == 0 else np.clip((target[mask] - mn) / denom, 0.0, 1.0)
        if keep_padding:
            target[~mask] = PADDING_VALUE
        params[name] = {"min": mn, "max": mx}
    return out, params


def impute_scale_discrete(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, object]]:
    """discrete feature의 결측치를 채우고 0~1로 정규화합니다."""
    feats = [c for c in df.columns if c not in META_COLS]
    train = df["split"] == "train"
    missing_ratio = df.loc[train, feats].isna().mean()

    # train에서 너무 많이 비어 있는 feature는 신뢰하기 어려워 제거합니다.
    dropped = missing_ratio[missing_ratio >= MISSING_THRESHOLD].index.tolist()
    feats = [c for c in feats if c not in dropped]

    # 결측치 대체값도 train 데이터에서만 계산합니다. val 정보가 train에 새어 들어가지 않게 하기 위함입니다.
    med = df.loc[train, feats].median(numeric_only=True)
    assert not med.isna().any(), f"median 계산 불가: {med[med.isna()].index.tolist()}"
    out = df[[*feats, *META_COLS]].copy()
    out[feats] = out[feats].fillna(med)
    arr, scaler = fit_transform_minmax(out[feats].to_numpy(np.float32), train.to_numpy(), feats, keep_padding=False)
    out[feats] = arr
    return out, scaler, {"dropped_discrete_features": dropped, "discrete_missing_before": int(df[[c for c in df.columns if c not in META_COLS]].isna().sum().sum())}


def make_7day_windows(df: pd.DataFrame, value_cols: list[str] | None, matrix_col: str | None) -> tuple[np.ndarray, pd.DataFrame, int]:
    """환자별 날짜 데이터를 연속 7일 묶음으로 잘라 LSTM 샘플을 만듭니다."""
    windows, meta, skipped = [], [], 0
    for patient_id, g in df.assign(_date=pd.to_datetime(df["sample_date"])).sort_values(["patient_id", "_date"]).groupby("patient_id"):
        rows = g.reset_index(drop=True)
        for i in range(0, len(rows) - SEQ_DAYS + 1):
            chunk = rows.iloc[i:i + SEQ_DAYS]
            dates = chunk["_date"].to_list()
            if any((dates[j + 1] - dates[j]).days != 1 for j in range(SEQ_DAYS - 1)):
                # 중간 날짜가 빠진 7일 묶음은 시간 흐름이 끊기므로 사용하지 않습니다.
                skipped += 1
                continue
            assert chunk["split"].nunique() == 1, "하나의 window에 train/val이 섞였습니다."
            assert chunk["binary_class"].nunique() == 1, "하나의 window에 라벨이 섞였습니다."
            if matrix_col:
                windows.append(np.stack(chunk[matrix_col].to_list()).astype(np.float32))
            else:
                windows.append(chunk[value_cols].to_numpy(np.float32))
            meta.append({
                "patient_id": patient_id,
                "window_start_date": dates[0].date().isoformat(),
                "window_end_date": dates[-1].date().isoformat(),
                "split": chunk["split"].iloc[0],
                "binary_class": int(chunk["binary_class"].iloc[0]),
            })
    assert windows, "7일 연속 window가 없습니다."
    meta_df = pd.DataFrame(meta)
    assert meta_df.duplicated(["patient_id", "window_start_date", "window_end_date"]).sum() == 0, "window key 중복"
    return np.stack(windows).astype(np.float32), meta_df, skipped


def build_integrated_inputs(X_continuous_seq: np.ndarray, X_discrete_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """5분 단위 continuous 값을 하루 단위로 펼친 뒤 discrete 값과 나란히 붙입니다."""
    n_windows, _, _, n_cont = X_continuous_seq.shape
    X_continuous_flat_seq = X_continuous_seq.transpose(0, 1, 3, 2).reshape(n_windows, SEQ_DAYS, n_cont * DAILY_STEPS)
    return X_continuous_flat_seq.astype(np.float32), np.concatenate([X_continuous_flat_seq, X_discrete_seq], axis=2).astype(np.float32)


def align_windows(
    X_cont: np.ndarray,
    cont_meta: pd.DataFrame,
    X_disc: np.ndarray,
    disc_meta: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, int]]:
    """continuous와 discrete가 같은 환자-시작일-종료일을 가진 window만 맞춰 남깁니다."""
    key_cols = ["patient_id", "window_start_date", "window_end_date"]
    cont_keys = [tuple(x) for x in cont_meta[key_cols].to_numpy()]
    disc_pos = {tuple(x): i for i, x in enumerate(disc_meta[key_cols].to_numpy())}
    keep_cont, keep_disc, keys = [], [], []
    for i, key in enumerate(cont_keys):
        if key in disc_pos:
            keep_cont.append(i)
            keep_disc.append(disc_pos[key])
            keys.append(key)
    common = pd.DataFrame(keys, columns=key_cols).merge(cont_meta, on=key_cols, how="left")
    return X_cont[keep_cont], X_disc[keep_disc], common, {
        "continuous_only_windows": len(set(cont_keys) - set(disc_pos)),
        "discrete_only_windows": len(set(disc_pos) - set(cont_keys)),
    }


def save_outputs(
    output_dir: Path,
    data: dict[str, object],
    discrete_daily: pd.DataFrame,
    meta: pd.DataFrame,
) -> tuple[Path, Path, Path]:
    """최종 산출물을 저장하고 다시 읽어 shape가 유지되는지 확인합니다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = output_dir / "lstm_dataset.pkl"
    index_path = output_dir / "lstm_window_index.csv"
    daily_path = output_dir / "lstm_discrete_daily.csv"
    with pkl_path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta.reset_index(names="window_index").to_csv(index_path, index=False, encoding="utf-8-sig")
    discrete_daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    with pkl_path.open("rb") as f:
        loaded = pickle.load(f)
    assert loaded["X_integrated_seq"].shape == data["X_integrated_seq"].shape, "pickle 재로드 shape 불일치"
    return pkl_path, index_path, daily_path


def validate_final(data: dict[str, object], window_index: pd.DataFrame) -> None:
    """최종 데이터가 LSTM 입력으로 쓰기에 안전한 형태인지 마지막으로 검사합니다."""
    Xc, Xf, Xd, Xi, y = (data[k] for k in ("X_continuous_seq", "X_continuous_flat_seq", "X_discrete_seq", "X_integrated_seq", "y"))
    assert Xc.ndim == 4 and Xc.shape[1] == SEQ_DAYS and Xc.shape[2] == DAILY_STEPS
    assert Xf.ndim == 3 and Xf.shape[1] == SEQ_DAYS
    assert Xd.ndim == 3 and Xd.shape[1] == SEQ_DAYS
    assert Xi.ndim == 3 and Xi.shape[1] == SEQ_DAYS
    assert Xc.shape[0] == Xf.shape[0] == Xd.shape[0] == Xi.shape[0] == len(y)
    assert len(data["patient_id"]) == len(data["window_start_date"]) == len(data["window_end_date"]) == len(data["split"]) == len(y)
    assert set(np.unique(y)).issubset({0, 1})
    assert window_index.duplicated(["patient_id", "window_start_date", "window_end_date"]).sum() == 0
    assert not (set(window_index.loc[window_index["split"] == "train", "patient_id"]) & set(window_index.loc[window_index["split"] == "val", "patient_id"]))
    assert len(data["continuous_feature_names"]) == Xc.shape[3]
    assert len(data["continuous_flat_feature_names"]) == Xf.shape[2]
    assert len(data["discrete_feature_names"]) == Xd.shape[2]
    assert data["integrated_feature_names"] == data["continuous_flat_feature_names"] + data["discrete_feature_names"]
    assert len(data["integrated_feature_names"]) == Xi.shape[2]
    cont_valid = Xc[Xc != PADDING_VALUE]
    assert len(cont_valid) == 0 or (cont_valid.min() >= -1e-6 and cont_valid.max() <= 1 + 1e-6)
    assert np.all((Xc == PADDING_VALUE) | ((Xc >= -1e-6) & (Xc <= 1 + 1e-6)))
    assert Xd.min() >= -1e-6 and Xd.max() <= 1 + 1e-6


def md_table(df: pd.DataFrame) -> str:
    """guide 문서에 넣을 간단한 Markdown 표를 만듭니다."""
    rows = [[str(c) for c in df.columns], ["---"] * len(df.columns)]
    rows += df.astype(str).values.tolist()
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def write_guide(output_dir: Path, data: dict[str, object], stats: dict[str, object], pkl_path: Path, index_path: Path, daily_path: Path) -> Path:
    """모델팀이 산출물 구조와 전처리 기준을 확인할 수 있는 설명 문서를 씁니다."""
    idx = pd.DataFrame({
        "split": data["split"],
        "binary_class": data["y"],
        "patient_id": data["patient_id"],
    })
    lines = [
        "# LSTM Dataset Guide",
        "",
        f"- 생성일: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "- 설명: 논문 LSTM 전처리 설명에 맞춘 재현형 전처리 데이터",
        "",
        "## 산출 파일",
        "",
        f"- `{pkl_path.name}`: 3차원/4차원 모델 입력을 담은 메인 pickle",
        f"- `{index_path.name}`: pickle window 순서 확인용 CSV",
        f"- `{daily_path.name}`: sliding window 전 discrete daily feature 확인용 CSV",
        "",
        "## Pickle Keys / Shapes",
        "",
        f"- `X_continuous_seq`: 5분 단위 continuous 4D 입력, {data['X_continuous_seq'].shape}",
        f"- `X_continuous_flat_seq`: day-level LSTM용 continuous flatten 입력, {data['X_continuous_flat_seq'].shape}",
        f"- `X_discrete_seq`: 일 단위 discrete 3D 입력, {data['X_discrete_seq'].shape}",
        f"- `X_integrated_seq`: continuous flatten + discrete 통합 입력, {data['X_integrated_seq'].shape}",
        f"- `y`: window label, {data['y'].shape}",
        "- `patient_id`, `window_start_date`, `window_end_date`, `split`: window별 메타 배열",
        "- `continuous_scaler_params`, `discrete_scaler_params`: train 기준 MinMax scaler 통계",
        f"- continuous feature 수: {len(data['continuous_feature_names'])}",
        f"- continuous flat feature 수: {len(data['continuous_flat_feature_names'])}",
        f"- discrete feature 수: {len(data['discrete_feature_names'])}",
        f"- integrated feature 수: {len(data['integrated_feature_names'])}",
        "",
        "## Window / Split",
        "",
        f"- 전체 window 수: {len(data['y'])}",
        f"- 전체 patient 수: {idx['patient_id'].nunique()}",
        f"- 원본 label 기준 patient 수: {stats['label_patient_count']}",
        f"- strict 7일 window 기준 최종 patient 수: {idx['patient_id'].nunique()}",
        f"- strict 7일 window 기준 제외 patient 수: {stats['excluded_patient_count']}",
        f"- 제외 patient_id: {', '.join(stats['excluded_patient_ids'])}",
        f"- sequence days: {SEQ_DAYS}",
        f"- daily sequence length: {DAILY_STEPS}",
        f"- padding value: {PADDING_VALUE}",
        "- window 기준: `patient_id + window_start_date + window_end_date`",
        "- `lstm_window_index.csv`의 `window_index`는 pickle 배열의 첫 번째 축 index와 일치합니다.",
        "- `split` 컬럼은 원본 `1.Training`/`2.Validation` 출처 표시용 참고 컬럼입니다.",
        "- 모델팀 검증 방식: 전체 pickle을 사용하여 `patient_id` 기준 5-fold GroupKFold를 직접 수행합니다.",
        "- 일반 KFold 및 row 단위 random split은 사용하지 않습니다.",
        "",
        "### Split별 Window 수",
        "",
        md_table(idx["split"].value_counts().sort_index().rename_axis("split").reset_index(name="windows")),
        "",
        "### Split별 Patient 수",
        "",
        md_table(idx.groupby("split")["patient_id"].nunique().reset_index(name="patients")),
        "",
        "### Split별 Label 분포",
        "",
        md_table(pd.crosstab(idx["split"], idx["binary_class"]).reset_index()),
        "",
        "## MinMax / 결측 / Padding",
        "",
        "- 논문 명시: LSTM 학습 전 MinMax 정규화와 일주일 시퀀스 작업을 수행합니다.",
        "- 구현상 선택: continuous와 discrete 모두 train 기준으로 scaler를 fit하고 val에는 transform만 적용합니다.",
        "- 구현상 선택: continuous padding/missing placeholder `-1.0`은 scaler fit에서 제외하고 정규화 후에도 `-1.0`으로 유지합니다.",
        "- 구현상 선택: discrete 결측치는 train median으로 대체한 뒤 train 기준 MinMax를 적용합니다.",
        "- 구현상 선택: val 값이 train min/max 범위를 벗어나면 0~1 범위로 clipping합니다.",
        "- 논문 미기재, 구현상 선택: 결측률 50% 이상 feature 제거, activity 일 단위 집계, 연속 7일 window만 사용, timestep 결측과 padding을 `-1.0`으로 처리, 1min to 5min 변환.",
        "- activity와 sleep이 모두 존재하는 `patient_id + sample_date`만 유지합니다. 한쪽 modality가 없는 날을 `-1.0`으로 채우는 방식은 사용하지 않습니다.",
        "- 1min to 5min 변환: `activity_met_1min`은 5개씩 묶어 평균을 사용했습니다. 1분 단위 범주형 시계열이 추가될 경우 5개 단위 최빈값을 사용하도록 함수가 준비되어 있습니다.",
        "",
        "## Integrated Input",
        "",
        "- 통합 방식: day-level LSTM 입력",
        "- timestep = 7일",
        "- feature = 하루 288개 5분 단위 continuous flatten + 일 단위 discrete feature",
        "- 5min-level LSTM(timestep=2016)도 가능하지만, 시퀀스 길이 증가와 일 단위 discrete 결합 문제 때문에 본 구현에서는 day-level 통합 방식을 사용합니다.",
        "- 이 통합 방식은 논문 미기재, 구현상 선택입니다.",
        "",
        "## Feature Names",
        "",
        "### Continuous",
        "",
        "\n".join(f"- `{x}`" for x in data["continuous_feature_names"]),
        "",
        "### Discrete",
        "",
        "\n".join(f"- `{x}`" for x in data["discrete_feature_names"]),
        "",
        "## 전처리 통계",
        "",
        f"- 날짜 누락으로 제외된 continuous window 수: {stats['continuous_skipped_windows']}",
        f"- 날짜 누락으로 제외된 discrete window 수: {stats['discrete_skipped_windows']}",
        f"- continuous에만 있는 window 수: {stats['continuous_only_windows']}",
        f"- discrete에만 있는 window 수: {stats['discrete_only_windows']}",
        f"- activity만 있어 제외한 daily sample 수: {stats['activity_only_days']}",
        f"- sleep만 있어 제외한 daily sample 수: {stats['sleep_only_days']}",
        f"- dropped discrete features: {stats['dropped_discrete_features']}",
        "",
        "## 모델팀 사용 예시",
        "",
        "```python",
        "import pickle",
        "",
        "with open('lstm_dataset.pkl', 'rb') as f:",
        "    data = pickle.load(f)",
        "",
        "X_cont_4d = data['X_continuous_seq']",
        "X_cont_flat = data['X_continuous_flat_seq']",
        "X_disc = data['X_discrete_seq']",
        "X_int = data['X_integrated_seq']",
        "y = data['y']",
        "groups = data['patient_id']",
        "```",
    ]
    path = output_dir / "lstm_dataset_guide.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    """전처리 전체 순서를 실제로 실행하는 진입점입니다."""
    parser = argparse.ArgumentParser(description="논문 LSTM 전처리 설명에 맞춘 재현형 전처리 데이터 생성")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    a = parser.parse_args()
    output_dir = a.output_dir

    # 1. 라벨을 먼저 정리합니다. 이후 모든 데이터는 이 환자 라벨과 연결됩니다.
    labels = load_labels()

    # 2. 활동/수면의 5분 단위 continuous 데이터를 하루 단위로 만들고 합칩니다.
    continuous_daily, cont_stats = build_daily_continuous(labels)

    # 3. 활동/수면의 하루 요약 discrete 데이터를 만들고 라벨을 붙입니다.
    discrete_daily = build_daily_discrete(labels)

    # 4. discrete 데이터의 결측치를 채우고 train 기준으로 정규화합니다.
    discrete_daily, discrete_scaler_params, discrete_stats = impute_scale_discrete(discrete_daily)

    # 5. continuous 데이터도 train 기준으로 MinMax 정규화합니다. padding(-1)은 그대로 둡니다.
    cont_train_mask = (continuous_daily["split"] == "train").to_numpy()
    cont_values = np.stack(continuous_daily["matrix"].to_list()).astype(np.float32)
    cont_values, continuous_scaler_params = fit_transform_minmax(cont_values, cont_train_mask, CONTINUOUS_FEATURE_NAMES, keep_padding=True)
    continuous_daily = continuous_daily.drop(columns="matrix")
    continuous_daily["matrix"] = list(cont_values)

    # 6. 환자별로 날짜가 끊기지 않는 연속 7일 window를 만듭니다.
    X_cont, cont_meta, cont_skipped = make_7day_windows(continuous_daily, None, "matrix")
    discrete_feature_names = [c for c in discrete_daily.columns if c not in META_COLS]
    X_disc, disc_meta, disc_skipped = make_7day_windows(discrete_daily, discrete_feature_names, None)

    # 7. continuous와 discrete가 모두 존재하는 같은 7일 window만 최종 입력으로 사용합니다.
    X_cont, X_disc, window_meta, align_stats = align_windows(X_cont, cont_meta, X_disc, disc_meta)

    # 8. continuous 5분 단위 값을 하루 단위 feature로 펼치고 discrete feature와 붙입니다.
    X_cont_flat, X_integrated = build_integrated_inputs(X_cont, X_disc)

    # 9. 모델팀이 각 배열의 열 의미를 추적할 수 있도록 feature 이름을 저장합니다.
    continuous_flat_feature_names = [f"{feat}_t{step:03d}" for feat in CONTINUOUS_FEATURE_NAMES for step in range(DAILY_STEPS)]
    integrated_feature_names = continuous_flat_feature_names + discrete_feature_names
    y = window_meta["binary_class"].to_numpy(np.int64)

    # 10. pickle 하나에 모델 입력, 정답, 환자/날짜 정보, 정규화 기준을 모두 담습니다.
    data = {
        "X_continuous_seq": X_cont,
        "X_continuous_flat_seq": X_cont_flat,
        "X_discrete_seq": X_disc,
        "X_integrated_seq": X_integrated,
        "y": y,
        "patient_id": window_meta["patient_id"].astype(str).to_numpy(),
        "window_start_date": window_meta["window_start_date"].astype(str).to_numpy(),
        "window_end_date": window_meta["window_end_date"].astype(str).to_numpy(),
        "split": window_meta["split"].astype(str).to_numpy(),
        "continuous_feature_names": CONTINUOUS_FEATURE_NAMES,
        "continuous_flat_feature_names": continuous_flat_feature_names,
        "discrete_feature_names": discrete_feature_names,
        "integrated_feature_names": integrated_feature_names,
        "continuous_scaler_params": continuous_scaler_params,
        "discrete_scaler_params": discrete_scaler_params,
        "meta": {
            "sequence_days": SEQ_DAYS,
            "daily_sequence_length": DAILY_STEPS,
            "padding_value": PADDING_VALUE,
            "sample_key": ["patient_id", "window_start_date", "window_end_date"],
            "label_map": LABEL_MAP,
            "note": "논문 LSTM 전처리 설명에 맞춘 재현형 전처리 데이터",
        },
    }

    # 11. 저장 전에 모양, 라벨, 중복, 값 범위가 기대와 맞는지 검증합니다.
    validate_final(data, window_meta)

    # 12. pickle, window index CSV, discrete daily CSV를 저장합니다.
    pkl_path, index_path, daily_path = save_outputs(output_dir, data, discrete_daily, window_meta)

    # 13. 몇 명/몇 개 window가 최종 데이터에 남았는지 안내 문서와 로그에 남깁니다.
    label_patients = set(pd.concat(labels.values())["patient_id"])
    final_patients = set(window_meta["patient_id"])
    excluded_patient_ids = sorted(label_patients - final_patients)
    stats = {
        **cont_stats,
        **discrete_stats,
        **align_stats,
        "continuous_skipped_windows": cont_skipped,
        "discrete_skipped_windows": disc_skipped,
        "label_patient_count": len(label_patients),
        "excluded_patient_count": len(excluded_patient_ids),
        "excluded_patient_ids": excluded_patient_ids,
    }
    guide_path = write_guide(output_dir, data, stats, pkl_path, index_path, daily_path)

    idx = pd.DataFrame({"split": data["split"], "binary_class": data["y"], "patient_id": data["patient_id"]})
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"continuous feature 수: {len(CONTINUOUS_FEATURE_NAMES)}")
    print(f"discrete feature 수: {len(discrete_feature_names)}")
    print(f"X_continuous_seq.shape: {data['X_continuous_seq'].shape}")
    print(f"X_continuous_flat_seq.shape: {data['X_continuous_flat_seq'].shape}")
    print(f"X_discrete_seq.shape: {data['X_discrete_seq'].shape}")
    print(f"X_integrated_seq.shape: {data['X_integrated_seq'].shape}")
    print(f"y.shape: {data['y'].shape}")
    print(f"전체 window 수: {len(data['y'])}")
    print(f"전체 patient 수: {idx['patient_id'].nunique()}")
    print("split별 patient 수:")
    print(idx.groupby("split")["patient_id"].nunique().to_string())
    print("split별 label 분포:")
    print(pd.crosstab(idx["split"], idx["binary_class"]).to_string())
    print(f"날짜 누락으로 제외된 window 수: continuous={cont_skipped}, discrete={disc_skipped}")
    print(f"continuous/discrete 정렬 후 제외된 window 수: continuous_only={align_stats['continuous_only_windows']}, discrete_only={align_stats['discrete_only_windows']}")
    print(f"원본 label 기준 patient 수: {len(label_patients)}")
    print(f"strict 7일 window 기준 제외 patient 수: {len(excluded_patient_ids)}")
    print(f"제외 patient_id: {excluded_patient_ids}")
    print(f"activity만 있어 제외한 daily sample 수: {cont_stats['activity_only_days']}")
    print(f"sleep만 있어 제외한 daily sample 수: {cont_stats['sleep_only_days']}")
    print("MinMax scaler fit 기준: train only, val transform")
    print(f"저장된 lstm_dataset.pkl 경로: {pkl_path}")
    print(f"저장된 lstm_window_index.csv 경로: {index_path}")
    print(f"저장된 lstm_discrete_daily.csv 경로: {daily_path}")
    print(f"저장된 guide 경로: {guide_path}")

# %% [markdown] cell 8
# ## 실행
#
# 아래 셀을 실행하면 최종 LSTM 데이터셋 파일들이 `OUTPUT_DIR`에 저장됩니다.

# %% cell 9
sys.argv = [
    'build_lstm_dataset_colab.ipynb',
    '--output-dir', str(OUTPUT_DIR),
]
main()
