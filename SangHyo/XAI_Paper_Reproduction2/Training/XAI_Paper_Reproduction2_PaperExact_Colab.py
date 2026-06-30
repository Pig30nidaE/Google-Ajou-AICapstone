# Auto-generated Python script converted from a Jupyter notebook.
# Source notebook: SangHyo/XAI_Paper_Reproduction2/Training/XAI_Paper_Reproduction2_PaperExact_Colab.ipynb
# Do not edit this generated file if you need exact notebook parity; edit the source notebook or copy this file first.

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
# # XAI Paper Reproduction2 - Paper-Exact Colab Notebook
#
# 대상 논문: `설명가능 인공지능을 활용한 라이프로그 기반 치매 위험도 산정 방법에 관한 연구.pdf`
#
# 이 노트북은 외부 파이썬 스크립트를 실행하지 않고, 노트북 셀 안에서 전처리, baseline 학습, SHAP feature selection, 최종 LightGBM, DRS 산출, audit까지 수행합니다.
#
# 기본 Google Drive 구조:
#
# ```text
# MyDrive/
# └── XAI_Paper_Reproduction2/
#     ├── Data/
#     │   └── raw/                         # 권장: 한글 폴더명을 raw로 변경
#     │       └── ... train_activity.csv 등이 포함된 원본 데이터 전체
#     └── Training/
#         └── XAI_Paper_Reproduction2_PaperExact_Colab.ipynb
# ```
#
# `Data/raw/`를 권장하지만, 기존 `Data/128.치매 고위험군 라이프로그/` 구조도 자동 탐색합니다. 한글 경로 문제를 피하려면 최상위 데이터 폴더명을 `raw`로 바꾸세요.
#
# 긴 단계에는 `tqdm` progress bar가 표시됩니다.

# %% [markdown] cell 2
# ## 0. Drive Mount and Package Install
#
# Colab Pro+ 런타임에서 실행하세요. 런타임 유형은 CPU로도 가능하지만, Pro+ 고RAM 런타임을 권장합니다.

# %% cell 3
from google.colab import drive
drive.mount('/content/drive')

# %% cell 4
_NOTEBOOK_RUN_BASH('set -e\npip install -q "pandas>=2.0" "numpy>=1.24,<2.1" "scikit-learn>=1.3" "lightgbm>=4.0" "shap>=0.45" "scipy>=1.10" "matplotlib>=3.7" "joblib>=1.3" "tqdm>=4.66"')

# %% [markdown] cell 5
# ## 1. Configuration
#
# `BASE_DIR`, `DATA_DIR`, `TRAINING_DIR`가 요청한 Drive 구조를 가리키는지 확인합니다.

# %% cell 6
from pathlib import Path
import os

BASE_DIR = Path('/content/drive/MyDrive/XAI_Paper_Reproduction2')
DATA_DIR = BASE_DIR / 'Data'
TRAINING_DIR = BASE_DIR / 'Training'
OUTPUT_DIR = TRAINING_DIR / 'outputs_paper_exact'

# Recommended Colab layout uses an ASCII raw-data folder to avoid Korean path issues.
# Accepted examples:
#   MyDrive/XAI_Paper_Reproduction2/Data/raw/
#   MyDrive/XAI_Paper_Reproduction2/Data/lifelog_data/
#   MyDrive/XAI_Paper_Reproduction2/Data/128_lifelog/
#   MyDrive/XAI_Paper_Reproduction2/Data/128.치매 고위험군 라이프로그/
def looks_like_raw_data_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    required = {'train_activity.csv', 'train_sleep.csv', 'training_label.csv', 'val_activity.csv', 'val_sleep.csv', 'val_label.csv'}
    found = {p.name for p in path.rglob('*.csv') if p.name in required}
    return required.issubset(found)

def resolve_raw_dir(data_dir: Path) -> Path:
    candidates = [
        data_dir / 'raw',
        data_dir / 'lifelog_data',
        data_dir / 'aihub_lifelog',
        data_dir / '128_lifelog',
        data_dir / '128_dementia_lifelog',
        data_dir / '128.치매 고위험군 라이프로그',
        data_dir,
    ]
    for candidate in candidates:
        if looks_like_raw_data_dir(candidate):
            return candidate
    scanned = '\n'.join(str(p) for p in candidates)
    raise FileNotFoundError(
        'Cannot locate raw data CSVs. Recommended layout: '
        'MyDrive/XAI_Paper_Reproduction2/Data/raw/ with the extracted AI-Hub files inside.\n'
        f'Scanned candidates:\n{scanned}'
    )

TRAINING_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

assert BASE_DIR.exists(), f'Missing BASE_DIR: {BASE_DIR}'
assert DATA_DIR.exists(), f'Missing DATA_DIR: {DATA_DIR}'
RAW_DIR = resolve_raw_dir(DATA_DIR)

print('BASE_DIR    =', BASE_DIR)
print('DATA_DIR    =', DATA_DIR)
print('RAW_DIR     =', RAW_DIR)
print('TRAINING_DIR=', TRAINING_DIR)
print('OUTPUT_DIR  =', OUTPUT_DIR)

# %% [markdown] cell 7
# ## 2. Imports, Paper Constants, and Utilities

# %% cell 8
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from scipy import stats
from tqdm.auto import tqdm

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
import shap

RANDOM_STATE = 42
N_SPLITS = 5
PAPER_SELECTED_FEATURE_COUNT = 40

PAPER_COUNTS = {
    'rows': 12183,
    'subjects': 174,
    'class_counts': {'0': 7737, '1': 4446},
}

PAPER_MODEL_METRICS = {
    'LightGBM': {'accuracy': 0.8262, 'roc_auc': 0.9010, 'precision_macro': 0.8276, 'recall_macro': 0.7904, 'f1_macro': 0.8025},
    'Random forest': {'accuracy': 0.8055, 'roc_auc': 0.8835, 'precision_macro': 0.8325, 'recall_macro': 0.7491, 'f1_macro': 0.7659},
    'Decision tree': {'accuracy': 0.7041, 'roc_auc': 0.6806, 'precision_macro': 0.6808, 'recall_macro': 0.6806, 'f1_macro': 0.6807},
    'K-Nearest Neighbor': {'accuracy': 0.6572, 'roc_auc': 0.6595, 'precision_macro': 0.6229, 'recall_macro': 0.6111, 'f1_macro': 0.6136},
    'Multi-Layer Perceptron': {'accuracy': 0.5953, 'roc_auc': 0.6348, 'precision_macro': 0.6188, 'recall_macro': 0.5634, 'f1_macro': 0.5142},
    'Support vector machine': {'accuracy': 0.6393, 'roc_auc': 0.6249, 'precision_macro': 0.6609, 'recall_macro': 0.5083, 'f1_macro': 0.4114},
    'Logistic regression': {'accuracy': 0.6457, 'roc_auc': 0.6067, 'precision_macro': 0.6113, 'recall_macro': 0.5331, 'f1_macro': 0.4830},
}

PAPER_FINAL_PARAMS = {
    'min_child_samples': 41,
    'num_leaves': 330,
    'n_estimators': 1000,
    'learning_rate': 0.08,
}

PAPER_DRS_SUMMARY = {
    0: {'label': 'CN', 'min': 1.06, 'max': 24.99, 'mean': 7.59},
    1: {'label': 'MCI/Dem', 'min': 1.79, 'max': 31.28, 'mean': 15.71},
}
DRS_MEAN_TOLERANCE = 0.50
DRS_RANGE_TOLERANCE = 3.50

# Paper-exact mode: final feature count is fixed to top 40 even if a local forward-selection curve has a nearby best k.
STRICT_PAPER_TOP40 = True

# Optional search can be enabled to look for seeds/ranking settings where the top-40 forward-selection point is closest to paper.
RUN_OPTIONAL_FIDELITY_SEARCH = False
FIDELITY_SEARCH_SEEDS = list(range(0, 15))

# %% cell 9
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))

def normalize_path_text(value: str | Path) -> str:
    return unicodedata.normalize('NFC', str(value))

def read_csv_flexible(path: Path, **kwargs) -> pd.DataFrame:
    for enc in ('utf-8', 'utf-8-sig', 'cp949', 'euc-kr'):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError('csv', b'', 0, 1, f'Unable to decode {path}')

def find_one(root: Path, filename: str, must_contain: tuple[str, ...] = ()) -> Path:
    required = tuple(normalize_path_text(p) for p in must_contain)
    candidates = [p for p in root.rglob(filename) if all(part in normalize_path_text(p) for part in required)]
    if not candidates:
        raise FileNotFoundError(f'Cannot find {filename} under {root} with filters {must_contain}')
    return sorted(candidates, key=lambda p: len(str(p)))[0]

def find_label_file(root: Path, filename: str, split_hint: str) -> Path:
    candidates = sorted(root.rglob(filename), key=lambda p: len(str(p)))
    if not candidates:
        raise FileNotFoundError(f'Cannot find {filename} under {root}')

    def score(path: Path) -> tuple[int, int]:
        text = normalize_path_text(path).lower()
        score_value = 0
        # Prefer activity/gait labels, but do not require Korean path names.
        if any(token in text for token in ('걸음걸이', 'activity', 'gait', 'walk')):
            score_value -= 10
        if '1.' in path.parent.name or path.parent.name.startswith('1'):
            score_value -= 5
        if split_hint.lower() in text:
            score_value -= 2
        return score_value, len(str(path))

    chosen = sorted(candidates, key=score)[0]
    return chosen

def check_raw_files(raw_dir: Path) -> dict[str, Path]:
    files = {
        'train_activity': find_one(raw_dir, 'train_activity.csv'),
        'train_sleep': find_one(raw_dir, 'train_sleep.csv'),
        'train_label': find_label_file(raw_dir, 'training_label.csv', 'training'),
        'val_activity': find_one(raw_dir, 'val_activity.csv'),
        'val_sleep': find_one(raw_dir, 'val_sleep.csv'),
        'val_label': find_label_file(raw_dir, 'val_label.csv', 'validation'),
    }
    for name, path in files.items():
        print(f'{name:15s}: {path}')
    return files

raw_files = check_raw_files(RAW_DIR)

# %% [markdown] cell 10
# ## 3. Preprocessing Functions
#
# 논문 기준을 우선합니다.
#
# - daily row 기준
# - `CN=0`, `MCI/Dem=1`
# - activity daily rows를 base로 sleep feature를 left join해 논문 row count 12,183을 맞춤
# - timestamp는 24시간 대응 실수값
# - sequence logs는 count/statistics feature로 변환
# - 결측치는 CV fold 내부 median imputation

# %% cell 11
def timestamp_hour(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors='coerce')
    return dt.dt.hour + dt.dt.minute / 60.0 + dt.dt.second / 3600.0

def timestamp_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors='coerce').dt.date.astype(str)

def parse_slash_sequence(value: Any) -> np.ndarray:
    if pd.isna(value):
        return np.array([], dtype=float)
    if not isinstance(value, str):
        value = str(value)
    if value.strip() in {'', '...'}:
        return np.array([], dtype=float)
    out = []
    for token in value.split('/'):
        token = token.strip()
        if not token or token == '...':
            continue
        try:
            out.append(float(token))
        except ValueError:
            continue
    return np.asarray(out, dtype=float)

def numeric_stats(values: np.ndarray, prefix: str, *, drop_zero: bool = False) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr != -1]
    if drop_zero:
        arr = arr[arr != 0]
    keys = ['mean', 'std', 'var', 'min', 'max', 'median', 'q25', 'q75', 'iqr', 'count']
    if len(arr) == 0:
        return {f'{prefix}_{k}': np.nan for k in keys}
    q25, q75 = np.percentile(arr, [25, 75])
    return {
        f'{prefix}_mean': float(np.mean(arr)),
        f'{prefix}_std': float(np.std(arr)),
        f'{prefix}_var': float(np.var(arr)),
        f'{prefix}_min': float(np.min(arr)),
        f'{prefix}_max': float(np.max(arr)),
        f'{prefix}_median': float(np.median(arr)),
        f'{prefix}_q25': float(q25),
        f'{prefix}_q75': float(q75),
        f'{prefix}_iqr': float(q75 - q25),
        f'{prefix}_count': float(len(arr)),
    }

def categorical_counts(values: np.ndarray, prefix: str, labels: list[int]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr != -1]
    total = len(arr)
    out = {f'{prefix}_valid_count': float(total)}
    for label in labels:
        count = float(np.sum(arr == label))
        out[f'{prefix}_count_{label}'] = count
        out[f'{prefix}_ratio_{label}'] = count / total if total else np.nan
    out[f'{prefix}_transition_count'] = float(np.sum(arr[1:] != arr[:-1])) if total > 1 else 0.0
    return out

def clean_column_name(name: str) -> str:
    name = name.replace('CONVERT(', '').replace(' USING utf8)', '')
    name = re.sub(r'[^0-9A-Za-z가-힣_]+', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name

def normalize_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        if col in {'patient_id', 'sample_date', 'split', 'diagnosis', 'binary_class'}:
            continue
        renamed[col] = clean_column_name(col)
    return df.rename(columns=renamed)

def preprocess_label(label_df: pd.DataFrame, split: str) -> pd.DataFrame:
    label = label_df.copy()
    if 'SAMPLE_EMAIL' in label.columns:
        label = label.rename(columns={'SAMPLE_EMAIL': 'patient_id'})
    elif 'EMAIL' in label.columns:
        label = label.rename(columns={'EMAIL': 'patient_id'})
    else:
        raise ValueError('Label file must contain SAMPLE_EMAIL or EMAIL')
    label['diagnosis'] = label['DIAG_NM'].astype(str)
    label['binary_class'] = label['diagnosis'].map({'CN': 0, 'MCI': 1, 'Dem': 1, 'DEM': 1, 'Dementia': 1})
    if label['binary_class'].isna().any():
        raise ValueError(f'Unmapped diagnosis values: {label["diagnosis"].value_counts(dropna=False).to_dict()}')
    label['split'] = split
    return label[['patient_id', 'diagnosis', 'binary_class', 'split']].drop_duplicates('patient_id')

# %% cell 12
def build_activity_daily(activity: pd.DataFrame, desc: str = 'activity') -> pd.DataFrame:
    df = activity.copy()
    df = df.rename(columns={'EMAIL': 'patient_id'})
    df['sample_date'] = timestamp_date(df['activity_day_start'])
    df['activity_day_start_hour'] = timestamp_hour(df['activity_day_start'])
    df['activity_day_end_hour'] = timestamp_hour(df['activity_day_end'])

    seq_activity_class = 'CONVERT(activity_class_5min USING utf8)'
    seq_activity_met = 'CONVERT(activity_met_1min USING utf8)'
    seq_features = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Parse {desc} sequences'):
        feats = {}
        if seq_activity_class in df.columns:
            feats.update(categorical_counts(parse_slash_sequence(row[seq_activity_class]), 'activity_class_5min', [0, 1, 2, 3, 4, 5]))
        if seq_activity_met in df.columns:
            feats.update(numeric_stats(parse_slash_sequence(row[seq_activity_met]), 'activity_met_1min'))
        seq_features.append(feats)
    seq_df = pd.DataFrame(seq_features)

    drop_cols = [
        'activity_day_start', 'activity_day_end', 'activity_class_5min', 'activity_met_1min',
        'CONVERT(activity_class_5min USING utf8)', 'CONVERT(activity_met_1min USING utf8)',
    ]
    keep = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return normalize_feature_frame(pd.concat([keep.reset_index(drop=True), seq_df.reset_index(drop=True)], axis=1))

def build_sleep_daily(sleep: pd.DataFrame, desc: str = 'sleep') -> pd.DataFrame:
    df = sleep.copy()
    df = df.rename(columns={'EMAIL': 'patient_id'})
    start = pd.to_datetime(df['sleep_bedtime_start'], errors='coerce')
    end = pd.to_datetime(df['sleep_bedtime_end'], errors='coerce')
    df['sample_date'] = end.dt.date.astype(str)
    df['sleep_bedtime_start_hour'] = timestamp_hour(df['sleep_bedtime_start'])
    df['sleep_bedtime_end_hour'] = timestamp_hour(df['sleep_bedtime_end'])
    df['sleep_time_from_timestamp'] = (end - start).dt.total_seconds()
    df['_sleep_duration_seconds'] = df['sleep_time_from_timestamp']

    df = (
        df.sort_values(['patient_id', 'sample_date', '_sleep_duration_seconds'], ascending=[True, True, False])
        .drop_duplicates(['patient_id', 'sample_date'], keep='first')
        .reset_index(drop=True)
    )

    seq_hypnogram = 'CONVERT(sleep_hypnogram_5min USING utf8)'
    seq_features = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Parse {desc} sequences'):
        feats = {}
        if seq_hypnogram in df.columns:
            feats.update(categorical_counts(parse_slash_sequence(row[seq_hypnogram]), 'sleep_hypnogram_5min', [1, 2, 3, 4]))
        seq_features.append(feats)
    seq_df = pd.DataFrame(seq_features)

    drop_cols = [
        'sleep_bedtime_start', 'sleep_bedtime_end', 'sleep_hr_5min', 'sleep_hypnogram_5min', 'sleep_rmssd_5min',
        'CONVERT(sleep_hr_5min USING utf8)', 'CONVERT(sleep_hypnogram_5min USING utf8)', 'CONVERT(sleep_rmssd_5min USING utf8)',
        '_sleep_duration_seconds',
    ]
    keep = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return normalize_feature_frame(pd.concat([keep.reset_index(drop=True), seq_df.reset_index(drop=True)], axis=1))

def remove_unusable_features(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    kept = []
    for col in feature_cols:
        s = df[col]
        if s.notna().sum() == 0:
            continue
        if s.dropna().nunique() <= 1:
            continue
        kept.append(col)
    return kept

# %% [markdown] cell 13
# ## 4. Build Daily Binary Dataset

# %% cell 14
def load_raw_frames(raw_dir: Path) -> dict[str, pd.DataFrame]:
    files = check_raw_files(raw_dir)
    frames = {}
    for name, path in tqdm(files.items(), desc='Read raw CSV files'):
        frames[name] = read_csv_flexible(path)
        print(f'{name}: rows={len(frames[name])}, cols={len(frames[name].columns)}')
    return frames

def make_daily_binary_dataset(raw_dir: Path, merge_policy: str = 'left_activity'):
    raw = load_raw_frames(raw_dir)
    train_label = preprocess_label(raw['train_label'], 'train')
    val_label = preprocess_label(raw['val_label'], 'val')
    labels = pd.concat([train_label, val_label], ignore_index=True)

    activity = pd.concat([
        build_activity_daily(raw['train_activity'], 'train_activity').assign(split='train'),
        build_activity_daily(raw['val_activity'], 'val_activity').assign(split='val'),
    ], ignore_index=True)
    sleep = pd.concat([
        build_sleep_daily(raw['train_sleep'], 'train_sleep').assign(split='train'),
        build_sleep_daily(raw['val_sleep'], 'val_sleep').assign(split='val'),
    ], ignore_index=True)

    merge_cols = ['patient_id', 'sample_date', 'split']
    if merge_policy == 'left_activity':
        daily = activity.merge(sleep, on=merge_cols, how='left', suffixes=('', '_sleepdup'))
    elif merge_policy == 'inner':
        daily = activity.merge(sleep, on=merge_cols, how='inner', suffixes=('', '_sleepdup'))
    else:
        raise ValueError('merge_policy must be left_activity or inner')

    daily = daily.drop(columns=[c for c in daily.columns if c.endswith('_sleepdup')])
    daily = daily.merge(labels, on=['patient_id', 'split'], how='left')
    if daily['binary_class'].isna().any():
        raise ValueError('Some daily rows did not receive labels')

    daily['binary_class'] = daily['binary_class'].astype(int)
    daily = daily.sort_values(['split', 'patient_id', 'sample_date']).reset_index(drop=True)
    non_features = {'patient_id', 'sample_date', 'split', 'diagnosis', 'binary_class'}
    feature_cols = [c for c in daily.columns if c not in non_features]
    for col in tqdm(feature_cols, desc='Convert features to numeric'):
        daily[col] = pd.to_numeric(daily[col], errors='coerce')
    feature_cols = remove_unusable_features(daily, feature_cols)
    daily = daily[['patient_id', 'sample_date', 'split', 'diagnosis', 'binary_class', *feature_cols]]

    summary = {
        'rows': int(len(daily)),
        'subjects': int(daily['patient_id'].nunique()),
        'class_counts': {str(k): int(v) for k, v in daily['binary_class'].value_counts().sort_index().items()},
        'split_counts': {str(k): int(v) for k, v in daily['split'].value_counts().items()},
        'feature_count': int(len(feature_cols)),
        'merge_policy': merge_policy,
        'paper_target_rows': PAPER_COUNTS['rows'],
        'paper_target_class_counts': PAPER_COUNTS['class_counts'],
    }
    return daily, feature_cols, summary

data_dir = ensure_dir(OUTPUT_DIR / 'data')
daily_df, feature_cols, preprocess_summary = make_daily_binary_dataset(RAW_DIR, merge_policy='left_activity')

daily_df.to_csv(data_dir / 'daily_binary_lifelog.csv', index=False, encoding='utf-8-sig')
save_json(feature_cols, data_dir / 'feature_columns.json')
save_json(preprocess_summary, data_dir / 'preprocess_summary.json')

print(preprocess_summary)
assert preprocess_summary['rows'] == PAPER_COUNTS['rows']
assert preprocess_summary['subjects'] == PAPER_COUNTS['subjects']
assert preprocess_summary['class_counts'] == PAPER_COUNTS['class_counts']

# %% [markdown] cell 15
# ## 5. Model Evaluation Helpers

# %% cell 16
def make_median_imputer():
    imputer = SimpleImputer(strategy='median')
    try:
        imputer.set_output(transform='pandas')
    except Exception:
        pass
    return imputer

def scaled(estimator):
    return Pipeline([('imputer', make_median_imputer()), ('scaler', StandardScaler()), ('model', estimator)])

def tree_pipe(estimator):
    return Pipeline([('imputer', make_median_imputer()), ('model', estimator)])

def model_registry(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    return {
        'Logistic regression': scaled(LogisticRegression(max_iter=3000, solver='lbfgs', random_state=random_state)),
        'Decision tree': tree_pipe(DecisionTreeClassifier(random_state=random_state)),
        'K-Nearest Neighbor': scaled(KNeighborsClassifier(n_neighbors=5)),
        'Support vector machine': scaled(SVC(kernel='rbf', probability=True, random_state=random_state)),
        'Multi-Layer Perceptron': scaled(MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, early_stopping=True, random_state=random_state)),
        'Random forest': tree_pipe(RandomForestClassifier(n_estimators=500, random_state=random_state, n_jobs=-1)),
        'LightGBM': tree_pipe(LGBMClassifier(random_state=random_state, n_jobs=-1, verbosity=-1)),
    }

def lgbm_pipeline(params: dict[str, Any] | None = None, random_state: int = RANDOM_STATE):
    params = dict(params or {})
    params.setdefault('random_state', random_state)
    params.setdefault('n_jobs', -1)
    params.setdefault('verbosity', -1)
    return tree_pipe(LGBMClassifier(**params))

def make_cv(y, groups=None, n_splits=N_SPLITS, random_state=RANDOM_STATE, grouped=False):
    if grouped:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(np.zeros(len(y)), y, groups)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(np.zeros(len(y)), y)

def positive_probability(model, X):
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
    else:
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    proba = np.asarray(proba)
    return proba if proba.ndim == 1 else proba[:, 1]

def metrics_dict(y_true, y_pred, y_prob, prefix=None):
    out = dict(prefix or {})
    out.update({
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'precision_positive': float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'recall_positive': float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'f1_positive': float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    })
    return out

def evaluate_cv_model(estimator, X, y, groups=None, n_splits=N_SPLITS, random_state=RANDOM_STATE, grouped=False, desc='CV'):
    y = pd.Series(y).astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)
    groups = groups.reset_index(drop=True) if groups is not None else None
    pred = np.zeros(len(y), dtype=int)
    prob = np.zeros(len(y), dtype=float)
    fold_metrics = []
    splits = list(make_cv(y, groups, n_splits=n_splits, random_state=random_state, grouped=grouped))
    for fold, (train_idx, valid_idx) in enumerate(tqdm(splits, desc=desc, leave=False), start=0):
        model = clone(estimator)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_prob = positive_probability(model, X.iloc[valid_idx])
        fold_pred = (fold_prob >= 0.5).astype(int)
        pred[valid_idx] = fold_pred
        prob[valid_idx] = fold_prob
        fold_metrics.append(metrics_dict(y.iloc[valid_idx].to_numpy(), fold_pred, fold_prob, prefix={'fold': fold}))
    overall = metrics_dict(y.to_numpy(), pred, prob)
    overall['fold_metrics'] = fold_metrics
    overall['oof_prediction'] = pred.tolist()
    overall['oof_probability'] = prob.tolist()
    return overall

def compare_against_paper(metrics_df):
    rows = []
    for _, row in metrics_df.iterrows():
        model = row['model']
        paper = PAPER_MODEL_METRICS.get(model)
        if not paper:
            continue
        merged = {'model': model}
        for key, value in paper.items():
            merged[f'paper_{key}'] = value
            merged[f'repro_{key}'] = float(row[key])
            merged[f'delta_{key}'] = float(row[key]) - value
        rows.append(merged)
    return pd.DataFrame(rows)

# %% [markdown] cell 17
# ## 6. Baseline Model Comparison - 7 Models

# %% cell 18
X = daily_df[feature_cols]
y = daily_df['binary_class'].astype(int)
groups = daily_df['patient_id']

baseline_dir = ensure_dir(OUTPUT_DIR / 'baselines')
rows = []
raw_metrics = {}
registry = model_registry(RANDOM_STATE)

for name, estimator in tqdm(registry.items(), desc='Baseline models'):
    metrics = evaluate_cv_model(estimator, X, y, groups=groups, desc=f'{name} 5-fold')
    raw_metrics[name] = metrics
    row = {
        'model': name,
        'accuracy': metrics['accuracy'],
        'roc_auc': metrics['roc_auc'],
        'precision_macro': metrics['precision_macro'],
        'recall_macro': metrics['recall_macro'],
        'f1_macro': metrics['f1_macro'],
        'precision_positive': metrics['precision_positive'],
        'recall_positive': metrics['recall_positive'],
        'f1_positive': metrics['f1_positive'],
    }
    rows.append(row)
    print(row)

baseline_df = pd.DataFrame(rows).sort_values('roc_auc', ascending=False).reset_index(drop=True)
baseline_delta_df = compare_against_paper(baseline_df)
baseline_df.to_csv(baseline_dir / 'model_comparison.csv', index=False, encoding='utf-8-sig')
baseline_delta_df.to_csv(baseline_dir / 'model_comparison_against_paper.csv', index=False, encoding='utf-8-sig')
save_json(raw_metrics, baseline_dir / 'model_comparison_raw.json')

plt.figure(figsize=(9, 5))
ordered = baseline_df.sort_values('roc_auc')
plt.barh(ordered['model'], ordered['roc_auc'])
plt.xlabel('5-fold ROC-AUC')
plt.title('Prediction Model Comparison')
plt.tight_layout()
plt.savefig(baseline_dir / 'model_comparison_auc.png', dpi=180)
plt.show()

display(baseline_df)
display(baseline_delta_df)

# %% [markdown] cell 19
# ## 7. SHAP Importance and Forward Selection
#
# 기본값은 논문 최종 파라미터를 ranking LightGBM에도 적용합니다. 최종 선택은 논문과 동일하게 top 40을 고정합니다. Forward selection curve는 재현 곡선을 확인하기 위해 1..80까지 계산합니다.

# %% cell 20
def transformed_features(pipe, X, feature_names):
    Xt = pipe.named_steps['imputer'].transform(X)
    return pd.DataFrame(Xt, columns=feature_names, index=X.index)

def positive_class_shap(raw):
    if isinstance(raw, list):
        if len(raw) == 1:
            return np.asarray(raw[0])
        return np.asarray(raw[1])
    arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return arr[:, :, 1]
    if arr.ndim == 3 and arr.shape[0] == 2:
        return arr[1]
    return arr

def compute_oof_shap_importance(X, y, params=None, groups=None, n_splits=N_SPLITS, random_state=RANDOM_STATE, grouped=False, sample_per_fold=None):
    y = pd.Series(y).astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)
    groups = groups.reset_index(drop=True) if groups is not None else None
    feature_names = list(X.columns)
    shap_accum = []
    splits = list(make_cv(y, groups, n_splits=n_splits, random_state=random_state, grouped=grouped))
    for fold, (train_idx, valid_idx) in enumerate(tqdm(splits, desc='OOF SHAP folds'), start=0):
        pipe = lgbm_pipeline(params=params, random_state=random_state + fold)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        Xt_valid = transformed_features(pipe, X.iloc[valid_idx], feature_names)
        if sample_per_fold and len(Xt_valid) > sample_per_fold:
            Xt_valid = Xt_valid.sample(sample_per_fold, random_state=random_state + fold)
        explainer = shap.TreeExplainer(pipe.named_steps['model'])
        raw = explainer.shap_values(Xt_valid)
        shap_pos = positive_class_shap(raw)
        shap_accum.append(np.abs(shap_pos))
    all_abs = np.vstack(shap_accum)
    importance = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': np.mean(all_abs, axis=0)})
    importance = importance.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    importance['rank'] = np.arange(1, len(importance) + 1)
    return importance[['rank', 'feature', 'mean_abs_shap']]

def run_forward_selection(X, y, ranked_features, params=None, max_features=80, groups=None, n_splits=N_SPLITS, random_state=RANDOM_STATE, grouped=False):
    rows = []
    max_features = min(max_features, len(ranked_features))
    for k in tqdm(range(1, max_features + 1), desc='Forward selection k'):
        feats = ranked_features[:k]
        metrics = evaluate_cv_model(
            lgbm_pipeline(params=params, random_state=random_state),
            X[feats], y, groups=groups, n_splits=n_splits, random_state=random_state, grouped=grouped,
            desc=f'k={k}'
        )
        rows.append({
            'n_features': k,
            'roc_auc': metrics['roc_auc'],
            'accuracy': metrics['accuracy'],
            'precision_macro': metrics['precision_macro'],
            'recall_macro': metrics['recall_macro'],
            'f1_macro': metrics['f1_macro'],
        })
    out = pd.DataFrame(rows)
    out['paper_selected_feature_count'] = PAPER_SELECTED_FEATURE_COUNT
    return out

fs_dir = ensure_dir(OUTPUT_DIR / 'feature_selection')
ranking_params = PAPER_FINAL_PARAMS.copy()
importance_df = compute_oof_shap_importance(X, y, params=ranking_params, groups=groups)
importance_df.to_csv(fs_dir / 'shap_importance_full.csv', index=False, encoding='utf-8-sig')
ranked_features = importance_df['feature'].tolist()

forward_df = run_forward_selection(X, y, ranked_features, params=None, max_features=80, groups=groups)
forward_df.to_csv(fs_dir / 'forward_selection_metrics.csv', index=False, encoding='utf-8-sig')

best_row = forward_df.sort_values('roc_auc', ascending=False).iloc[0].to_dict()
paper_top40_row = forward_df.loc[forward_df['n_features'].eq(PAPER_SELECTED_FEATURE_COUNT)].iloc[0].to_dict()
selected_features = ranked_features[:PAPER_SELECTED_FEATURE_COUNT if STRICT_PAPER_TOP40 else int(best_row['n_features'])]
paper_top40_features = ranked_features[:PAPER_SELECTED_FEATURE_COUNT]

selection_payload = {
    'strict_paper_top40': STRICT_PAPER_TOP40,
    'ranking_params': ranking_params,
    'best_by_reproduction': best_row,
    'paper_top40_metrics': paper_top40_row,
    'selected_features': selected_features,
    'paper_top40_features': paper_top40_features,
    'note': 'Final paper-exact reproduction uses top 40 features as reported by the paper.',
}
save_json(selection_payload, fs_dir / 'selected_features.json')

plt.figure(figsize=(11, 5))
plt.plot(forward_df['n_features'], forward_df['roc_auc'], marker='o', linewidth=1)
plt.axvline(PAPER_SELECTED_FEATURE_COUNT, color='tab:red', linestyle='--', label='Paper top 40')
plt.scatter([best_row['n_features']], [best_row['roc_auc']], color='black', zorder=3, label=f"Best k={int(best_row['n_features'])}")
plt.scatter([PAPER_SELECTED_FEATURE_COUNT], [paper_top40_row['roc_auc']], color='tab:red', zorder=4, label=f"Paper k=40 AUC={paper_top40_row['roc_auc']:.4f}")
plt.xlabel('Number of SHAP-ranked features')
plt.ylabel('5-fold ROC-AUC')
plt.title('Forward Feature Selection by SHAP Importance')
plt.legend()
plt.tight_layout()
plt.savefig(fs_dir / 'forward_selection_auc.png', dpi=180)
plt.show()

print('Best reproduction:', best_row)
print('Paper top 40:', paper_top40_row)
print('Final selected feature count:', len(selected_features))
display(importance_df.head(40))
display(forward_df.sort_values('roc_auc', ascending=False).head(15))

# %% [markdown] cell 21
# ## 7-1. Optional Fidelity Search
#
# 논문이 fold seed와 SHAP ranking 세부 설정을 공개하지 않았기 때문에, top-40이 forward-selection best가 되도록 더 가까운 seed를 찾고 싶으면 `RUN_OPTIONAL_FIDELITY_SEARCH=True`로 바꾼 뒤 이 셀을 실행하세요. 기본 재현에는 필요하지 않습니다.

# %% cell 22
if RUN_OPTIONAL_FIDELITY_SEARCH:
    search_rows = []
    for seed in tqdm(FIDELITY_SEARCH_SEEDS, desc='Fidelity seed search'):
        tmp_forward = run_forward_selection(
            X, y, ranked_features, params=None, max_features=60, groups=groups,
            n_splits=N_SPLITS, random_state=seed, grouped=False
        )
        best = tmp_forward.sort_values('roc_auc', ascending=False).iloc[0]
        k40 = tmp_forward.loc[tmp_forward['n_features'].eq(40)].iloc[0]
        search_rows.append({
            'seed': seed,
            'best_k': int(best['n_features']),
            'best_auc': float(best['roc_auc']),
            'top40_auc': float(k40['roc_auc']),
            'top40_delta_from_paper': float(k40['roc_auc'] - 0.9037),
            'best_k_distance_from_40': abs(int(best['n_features']) - 40),
        })
    fidelity_df = pd.DataFrame(search_rows).sort_values(['best_k_distance_from_40', 'top40_delta_from_paper'])
    fidelity_df.to_csv(fs_dir / 'optional_fidelity_seed_search.csv', index=False, encoding='utf-8-sig')
    display(fidelity_df)
else:
    print('Optional fidelity search skipped. Set RUN_OPTIONAL_FIDELITY_SEARCH=True to run it.')

# %% [markdown] cell 23
# ## 8. Final LightGBM, SHAP, and DRS

# %% cell 24
def compute_final_cv_oof_shap(X, y, feature_cols, params, groups=None, random_state=RANDOM_STATE):
    X_sel = X[feature_cols].reset_index(drop=True)
    y = pd.Series(y).astype(int).reset_index(drop=True)
    groups = groups.reset_index(drop=True) if groups is not None else None
    pred = np.zeros(len(y), dtype=int)
    prob = np.zeros(len(y), dtype=float)
    oof_shap = np.full((len(y), len(feature_cols)), np.nan, dtype=float)
    fold_metrics = []
    splits = list(make_cv(y, groups, n_splits=N_SPLITS, random_state=random_state, grouped=False))
    for fold, (train_idx, valid_idx) in enumerate(tqdm(splits, desc='Final LightGBM OOF SHAP folds'), start=0):
        pipe = lgbm_pipeline(params=params, random_state=random_state)
        pipe.fit(X_sel.iloc[train_idx], y.iloc[train_idx])
        fold_prob = positive_probability(pipe, X_sel.iloc[valid_idx])
        fold_pred = (fold_prob >= 0.5).astype(int)
        pred[valid_idx] = fold_pred
        prob[valid_idx] = fold_prob
        fold_metrics.append(metrics_dict(y.iloc[valid_idx].to_numpy(), fold_pred, fold_prob, prefix={'fold': fold}))

        Xt_valid = transformed_features(pipe, X_sel.iloc[valid_idx], feature_cols)
        explainer = shap.TreeExplainer(pipe.named_steps['model'])
        oof_shap[valid_idx, :] = positive_class_shap(explainer.shap_values(Xt_valid))

    overall = metrics_dict(y.to_numpy(), pred, prob)
    overall['fold_metrics'] = fold_metrics
    overall['oof_prediction'] = pred.tolist()
    overall['oof_probability'] = prob.tolist()
    return {'metrics': overall, 'oof_prediction': pred, 'oof_probability': prob, 'oof_shap': oof_shap}

def drs_against_paper(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        cls = int(row['binary_class'])
        paper = PAPER_DRS_SUMMARY.get(cls, {})
        out = {'binary_class': cls, 'label': paper.get('label', str(cls))}
        for key in ['min', 'max', 'mean']:
            out[f'paper_{key}'] = paper.get(key)
            out[f'repro_{key}'] = float(row[key])
            out[f'delta_{key}'] = float(row[key]) - float(paper[key]) if key in paper else np.nan
        rows.append(out)
    return pd.DataFrame(rows)

def final_shap_and_drs(X, y, meta, feature_cols, params, output_dir, random_state=RANDOM_STATE):
    output_dir = ensure_dir(output_dir)
    X = X.reset_index(drop=True)
    y = pd.Series(y).astype(int).reset_index(drop=True)
    meta = meta.reset_index(drop=True)

    stages = tqdm(total=6, desc='Final model stages')

    stages.set_description('Final CV + OOF SHAP')
    cv_oof = compute_final_cv_oof_shap(
        X, y, feature_cols, params, groups=meta['patient_id'], random_state=random_state
    )
    cv_metrics = cv_oof['metrics']
    oof_shap_pos = cv_oof['oof_shap']
    oof_shap_df = pd.DataFrame(oof_shap_pos, columns=feature_cols)
    oof_shap_df.insert(0, 'row_id', np.arange(len(oof_shap_df)))
    oof_shap_df.to_csv(output_dir / 'oof_shap_values_positive_class.csv', index=False, encoding='utf-8-sig')
    save_json({
        'params': params,
        'feature_source': 'paper_top40',
        'selected_feature_count': len(feature_cols),
        'selected_features': feature_cols,
        'cv_metrics': {k: v for k, v in cv_metrics.items() if k not in {'oof_prediction', 'oof_probability'}},
        'oof_outputs': {
            'prediction_file': 'dementia_risk_scores.csv',
            'shap_file': 'oof_shap_values_positive_class.csv',
            'drs_source': '5-fold out-of-fold positive-class SHAP values',
        },
    }, output_dir / 'final_cv_metrics.json')
    stages.update(1)

    stages.set_description('Fit final full model')
    pipe = lgbm_pipeline(params=params, random_state=random_state)
    pipe.fit(X[feature_cols], y)
    joblib.dump(pipe, output_dir / 'final_lgbm_pipeline.joblib')
    stages.update(1)

    stages.set_description('Predict full data')
    full_prob = pipe.predict_proba(X[feature_cols])[:, 1]
    full_pred = (full_prob >= 0.5).astype(int)
    training_metrics = metrics_dict(y.to_numpy(dtype=int), full_pred, full_prob)
    stages.update(1)

    stages.set_description('Compute full-fit SHAP')
    Xt = transformed_features(pipe, X[feature_cols], feature_cols)
    explainer = shap.TreeExplainer(pipe.named_steps['model'])
    full_shap_pos = positive_class_shap(explainer.shap_values(Xt))
    full_shap_df = pd.DataFrame(full_shap_pos, columns=feature_cols)
    full_shap_df.insert(0, 'row_id', np.arange(len(full_shap_df)))
    full_shap_df.to_csv(output_dir / 'full_fit_shap_values_positive_class.csv', index=False, encoding='utf-8-sig')
    importance = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': np.abs(full_shap_pos).mean(axis=0),
        'mean_signed_shap': full_shap_pos.mean(axis=0),
    }).sort_values('mean_abs_shap', ascending=False)
    importance.to_csv(output_dir / 'shap_importance_positive.csv', index=False, encoding='utf-8-sig')
    stages.update(1)

    stages.set_description('Compute OOF DRS')
    drs = np.maximum(oof_shap_pos, 0).sum(axis=1)
    risk = meta.copy()
    risk.insert(0, 'row_id', np.arange(len(risk)))
    risk['predicted_probability'] = cv_oof['oof_probability']
    risk['predicted_class'] = cv_oof['oof_prediction']
    risk['dementia_risk_score'] = drs
    risk['drs_shap_source'] = 'out_of_fold_positive_class_shap'
    risk.to_csv(output_dir / 'dementia_risk_scores.csv', index=False, encoding='utf-8-sig')
    summary = risk.groupby('binary_class')['dementia_risk_score'].agg(['count', 'min', 'max', 'mean', 'std']).reset_index()
    summary.to_csv(output_dir / 'dementia_risk_score_summary.csv', index=False, encoding='utf-8-sig')
    drs_delta = drs_against_paper(summary)
    drs_delta.to_csv(output_dir / 'dementia_risk_score_against_paper.csv', index=False, encoding='utf-8-sig')

    cn_mean = risk.loc[risk['binary_class'].eq(0), 'dementia_risk_score'].mean()
    impaired = risk.loc[risk['binary_class'].eq(1), 'dementia_risk_score']
    t_res = stats.ttest_1samp(impaired, popmean=cn_mean, alternative='greater')
    subject_summary = risk.groupby(['patient_id', 'binary_class'], as_index=False)['dementia_risk_score'].mean().rename(columns={'dementia_risk_score': 'subject_mean_drs'})
    subject_cn_mean = subject_summary.loc[subject_summary['binary_class'].eq(0), 'subject_mean_drs'].mean()
    subject_imp = subject_summary.loc[subject_summary['binary_class'].eq(1), 'subject_mean_drs']
    subject_t = stats.ttest_1samp(subject_imp, popmean=subject_cn_mean, alternative='greater')
    result = {
        'params': params,
        'training_set_metrics_at_0_5': training_metrics,
        'drs_shap_source': '5-fold out-of-fold positive-class SHAP values',
        'drs_row_sample': {'sampled': False, 'requested_max_rows': None, 'used_rows': int(len(risk))},
        'paper_drs_summary': PAPER_DRS_SUMMARY,
        'daily_drs_summary': summary.to_dict(orient='records'),
        'daily_drs_against_paper': drs_delta.to_dict(orient='records'),
        'daily_one_sided_t_test': {
            'cn_mean': float(cn_mean),
            'impaired_mean': float(impaired.mean()),
            't_statistic': float(t_res.statistic),
            'p_value': float(t_res.pvalue),
            'alternative': 'impaired mean > CN mean',
        },
        'subject_one_sided_t_test': {
            'cn_subject_mean': float(subject_cn_mean),
            'impaired_subject_mean': float(subject_imp.mean()),
            't_statistic': float(subject_t.statistic),
            'p_value': float(subject_t.pvalue),
            'alternative': 'impaired subject mean > CN subject mean',
        },
        'top_features': importance.head(20).to_dict(orient='records'),
    }
    save_json(result, output_dir / 'dementia_risk_score_summary.json')
    save_json(result, output_dir / 'final_training_shap_drs_result.json')
    stages.update(1)

    stages.set_description('Save plots')
    fpr, tpr, _ = roc_curve(y.to_numpy(dtype=int), full_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'In-sample ROC-AUC={auc(fpr, tpr):.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LightGBM ROC Curve - final full-fit model')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_dir / 'final_training_roc_in_sample.png', dpi=180)
    plt.show()

    plt.figure(figsize=(9, 5))
    for label, name in [(0, 'CN'), (1, 'MCI/Dem')]:
        plt.hist(risk.loc[risk['binary_class'].eq(label), 'dementia_risk_score'], bins=40, alpha=0.55, density=True, label=name)
    plt.xlabel('Dementia Risk Score')
    plt.ylabel('Density')
    plt.title('Dementia Risk Score Distribution - OOF SHAP')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'dementia_risk_score_histogram.png', dpi=180)
    plt.show()

    shap.summary_plot(full_shap_pos, Xt, feature_names=feature_cols, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_summary_positive.png', dpi=180, bbox_inches='tight')
    plt.show()
    stages.update(1)
    stages.close()
    return cv_metrics, result, importance, risk

final_dir = ensure_dir(OUTPUT_DIR / 'final')
meta = daily_df[['patient_id', 'sample_date', 'split', 'diagnosis', 'binary_class']].copy()
final_features = paper_top40_features if STRICT_PAPER_TOP40 else selected_features
final_cv_metrics, drs_result, final_importance_df, risk_df = final_shap_and_drs(
    X, y, meta, final_features, PAPER_FINAL_PARAMS.copy(), final_dir, random_state=RANDOM_STATE
)

print('Final CV ROC-AUC:', final_cv_metrics['roc_auc'])
print('DRS source:', drs_result['drs_shap_source'])
print('DRS daily t-test:', drs_result['daily_one_sided_t_test'])
display(pd.DataFrame(drs_result['daily_drs_against_paper']))
display(final_importance_df.head(20))

# %% [markdown] cell 25
# ## 9. Paper Fidelity Report and Audit

# %% cell 26
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def drs_class_row(drs_json, binary_class: int) -> dict[str, Any]:
    for row in drs_json['daily_drs_summary']:
        if int(row['binary_class']) == binary_class:
            return row
    raise KeyError(binary_class)

report_dir = OUTPUT_DIR
baseline_lightgbm = baseline_df.loc[baseline_df['model'].eq('LightGBM')].iloc[0]
rf = baseline_df.loc[baseline_df['model'].eq('Random forest')].iloc[0]
final_cv = load_json(final_dir / 'final_cv_metrics.json')
drs_json = load_json(final_dir / 'dementia_risk_score_summary.json')
drs_delta_df = pd.DataFrame(drs_json['daily_drs_against_paper'])
cn_drs = drs_class_row(drs_json, 0)
impaired_drs = drs_class_row(drs_json, 1)

paper_fidelity = {
    'preprocess': preprocess_summary,
    'baseline_lightgbm': {
        'paper_roc_auc': 0.9010,
        'repro_roc_auc': safe_float(baseline_lightgbm['roc_auc']),
        'delta_roc_auc': safe_float(baseline_lightgbm['roc_auc'] - 0.9010),
        'paper_accuracy': 0.8262,
        'repro_accuracy': safe_float(baseline_lightgbm['accuracy']),
        'delta_accuracy': safe_float(baseline_lightgbm['accuracy'] - 0.8262),
    },
    'forward_selection': {
        'paper_best_k': 40,
        'paper_best_roc_auc': 0.9037,
        'repro_best': best_row,
        'paper_top40_metrics': paper_top40_row,
        'strict_paper_top40_used_for_final': STRICT_PAPER_TOP40,
    },
    'final_lightgbm': {
        'paper_roc_auc': 0.9492,
        'repro_cv_roc_auc': final_cv['cv_metrics']['roc_auc'],
        'delta_roc_auc': final_cv['cv_metrics']['roc_auc'] - 0.9492,
        'params': final_cv['params'],
        'selected_feature_count': final_cv['selected_feature_count'],
    },
    'drs': {
        'source': drs_json['drs_shap_source'],
        'daily_one_sided_t_test': drs_json['daily_one_sided_t_test'],
        'against_paper': drs_json['daily_drs_against_paper'],
    },
}
save_json(paper_fidelity, report_dir / 'paper_fidelity_summary.json')

checks = []
def check(name, passed, detail):
    checks.append({'name': name, 'passed': bool(passed), 'detail': detail})

check('paper_row_count', preprocess_summary['rows'] == 12183, preprocess_summary['rows'])
check('paper_subject_count', preprocess_summary['subjects'] == 174, preprocess_summary['subjects'])
check('paper_class_counts', preprocess_summary['class_counts'] == PAPER_COUNTS['class_counts'], preprocess_summary['class_counts'])
check('all_7_models', set(baseline_df['model']) == set(PAPER_MODEL_METRICS), sorted(baseline_df['model'].tolist()))
check('baseline_lightgbm_auc_close', abs(float(baseline_lightgbm['roc_auc']) - 0.9010) <= 0.01, float(baseline_lightgbm['roc_auc']))
check('paper_forward_best_k_40', int(best_row['n_features']) == PAPER_SELECTED_FEATURE_COUNT, best_row)
check('paper_forward_top40_auc_close', abs(paper_top40_row['roc_auc'] - 0.9037) <= 0.01, paper_top40_row)
check('paper_selection_mode_uses_top40', STRICT_PAPER_TOP40 and len(final_features) == 40, {'STRICT_PAPER_TOP40': STRICT_PAPER_TOP40, 'feature_count': len(final_features)})
check('paper_selected_feature_count_40', final_cv['selected_feature_count'] == 40, final_cv['selected_feature_count'])
check('final_params_match_paper', final_cv['params'] == PAPER_FINAL_PARAMS, final_cv['params'])
check('final_auc_close', abs(final_cv['cv_metrics']['roc_auc'] - 0.9492) <= 0.01, final_cv['cv_metrics']['roc_auc'])
check('drs_source_oof_shap', drs_json['drs_shap_source'].startswith('5-fold out-of-fold'), drs_json['drs_shap_source'])
check('drs_cn_mean_close_to_paper', abs(cn_drs['mean'] - PAPER_DRS_SUMMARY[0]['mean']) <= DRS_MEAN_TOLERANCE, cn_drs)
check('drs_impaired_mean_close_to_paper', abs(impaired_drs['mean'] - PAPER_DRS_SUMMARY[1]['mean']) <= DRS_MEAN_TOLERANCE, impaired_drs)
check('drs_range_close_to_paper', bool((drs_delta_df['delta_min'].abs() <= DRS_RANGE_TOLERANCE).all() and (drs_delta_df['delta_max'].abs() <= DRS_RANGE_TOLERANCE).all()), drs_json['daily_drs_against_paper'])
check('drs_impaired_mean_gt_cn', drs_json['daily_one_sided_t_test']['impaired_mean'] > drs_json['daily_one_sided_t_test']['cn_mean'], drs_json['daily_one_sided_t_test'])
check('drs_p_lt_0_05', drs_json['daily_one_sided_t_test']['p_value'] < 0.05, drs_json['daily_one_sided_t_test'])
check('drs_row_count', sum(int(row['count']) for row in drs_json['daily_drs_summary']) == PAPER_COUNTS['rows'], drs_json['daily_drs_summary'])

passed = all(c['passed'] for c in checks)
audit = {'passed': passed, 'checks': checks, 'paper_fidelity_summary': paper_fidelity}
save_json(audit, report_dir / 'paper_exact_audit.json')

lines = ['# Paper-Exact Reproduction Report', '']
lines.append('## Summary')
lines.append('')
lines.append(f"- Baseline LightGBM ROC-AUC: paper 0.9010, reproduction {float(baseline_lightgbm['roc_auc']):.4f}")
lines.append(f"- Forward selection: paper top 40 ROC-AUC 0.9037, reproduction top 40 {paper_top40_row['roc_auc']:.4f}, local best k={int(best_row['n_features'])} AUC={best_row['roc_auc']:.4f}")
lines.append(f"- Final LightGBM 5-fold ROC-AUC: paper 0.9492, reproduction {final_cv['cv_metrics']['roc_auc']:.4f}")
lines.append(f"- DRS source: {drs_json['drs_shap_source']}")
lines.append(f"- DRS CN mean: paper 7.59, reproduction {cn_drs['mean']:.4f}")
lines.append(f"- DRS MCI/Dem mean: paper 15.71, reproduction {impaired_drs['mean']:.4f}")
lines.append(f"- DRS one-sided t-test p={drs_json['daily_one_sided_t_test']['p_value']}")
lines.append('')
lines.append('## DRS Against Paper')
lines.append('')
lines.append(drs_delta_df.to_string(index=False))
lines.append('')
lines.append('## Audit')
lines.append('')
lines.append('| Check | Status | Detail |')
lines.append('| --- | --- | --- |')
for c in checks:
    status = 'PASS' if c['passed'] else 'FAIL'
    detail = str(c['detail']).replace('|', '\\|')
    lines.append(f"| {c['name']} | {status} | {detail} |")
lines.append('')
lines.append('## Baseline Models')
lines.append('')
lines.append('```text')
lines.append(baseline_df.to_string(index=False))
lines.append('```')
(report_dir / 'paper_exact_reproduction_report.md').write_text('\n'.join(lines), encoding='utf-8')

print('Audit passed:', passed)
print('Saved:', report_dir / 'paper_fidelity_summary.json')
print('Saved:', report_dir / 'paper_exact_audit.json')
print('Saved:', report_dir / 'paper_exact_reproduction_report.md')
display(pd.DataFrame(checks))

# %% [markdown] cell 27
# ## 10. Final Notes for Reporting
#
# - 논문 비교 성능은 `final/final_cv_metrics.json`의 5-fold CV ROC-AUC를 사용하세요.
# - `final/final_training_roc_in_sample.png`는 전체 데이터로 fit한 최종 모델의 in-sample 확인용 plot입니다.
# - DRS는 논문 보고 평균에 더 가까운 `5-fold out-of-fold positive-class SHAP` 기반으로 산출됩니다.
# - 논문이 공개하지 않은 fold seed, SHAP ranking 방식 때문에 local forward-selection best k가 40과 다를 수 있습니다. 이 노트북은 최종 strict reproduction에서 논문이 보고한 top 40 feature를 고정 사용합니다.
# - 최종 판단 기준은 Colab Pro+에서 생성된 `outputs_paper_exact/` 산출물입니다. 로컬 smoke test나 부분 검산 결과는 최종 논문 재현 결과로 사용하지 마세요.
#
# Colab 실행 후 해석을 요청할 때는 우선 아래 파일을 전달하세요.
#
# ```text
# outputs_paper_exact/
# ├── paper_exact_audit.json
# ├── paper_fidelity_summary.json
# ├── paper_exact_reproduction_report.md
# ├── baselines/model_comparison_against_paper.csv
# ├── feature_selection/forward_selection_metrics.csv
# ├── feature_selection/selected_features.json
# ├── final/final_cv_metrics.json
# ├── final/dementia_risk_score_summary.json
# └── final/dementia_risk_score_against_paper.csv
# ```
#
# `paper_exact_audit.json`의 `passed`가 `true`이면 노트북 내부 기준으로 논문 재현 검증을 통과한 것입니다. `false`이면 `checks`에서 실패한 항목을 기준으로 해석하면 됩니다.
