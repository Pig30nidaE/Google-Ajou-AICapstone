# Auto-generated Python script converted from a Jupyter notebook.
# Source notebook: Jaehwang/코드 3가지/Experiment2.ipynb의 사본
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
# # 05. Regularized Hierarchical LSTM + Data Sanity Check
#
# **목적**: 04 결과(CV 0.65 → Test 0.44 붕괴, train epoch 2에 즉시 외움)의 원인을 동시에 두 갈래로 검증·해결.
#
# **Part A — Sanity Check (Step 1)**
# 1. 라벨 매핑 일관성 (CN=0, MCI+Dem=1)
# 2. Train/Test Subject 누수
# 3. Train/Test 패딩(-1) 비율
# 4. Train/Test 분포 시프트 (KS test)
# 5. Multi-seed CV 안정성 (AUC 분산)
#
# **Part B — Regularization 강화 모델 학습 (Step 2-B)**
# 1. **모델 다이어트** (50만 → 약 1.5만 파라미터)
# 2. **L2 weight decay** 모든 Conv/LSTM/Dense에 적용
# 3. **Feature selection**: train RF importance top-20만 사용 (Lee et al. 2024 후진제거법 동기)
# 4. **Per-subject z-score 정규화** (사람마다 baseline 심박/활동량이 다름)
# 5. **Subject-level AUC monitor** (val sequence-level AUC는 노이즈 큼)
# 6. `class_weight='balanced'` 강제 + threshold tuning
#
# **판단 기준 (마지막 셀)**
# - Subject-level AUC ≥ 0.65 → 구조·정규화 효과 확정 → 06 attention/focal 진행
# - Sanity에서 분포 시프트 확정 + AUC 여전히 낮음 → 06 repeated-CV (train+test pooled)
# - 둘 다 부정 → 데이터 자체 한계, 결측치 임계 완화 0602 pkl 필요

# %% [markdown] cell 2
# ## 0. 환경 설정

# %% cell 3
from google.colab import drive
drive.mount('/content/drive')

# %% cell 4
import os
from pathlib import Path

PKL_PATH = "/content/drive/MyDrive/ML_preprocessing/O_0531/LSTM/lstm_dataset.pkl"
ARTIFACT_DIR = "/content/drive/MyDrive/DataSanity/"

# 폴더가 없으면 생성
os.makedirs(ARTIFACT_DIR, exist_ok=True)

assert os.path.exists(PKL_PATH), f'pkl 파일이 없습니다: {PKL_PATH}'
print(f'PKL  : {PKL_PATH}')
print(f'ART  : {ARTIFACT_DIR}')

# %% cell 5
import numpy as np
import pandas as pd
import pickle, random, json, warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, Activation,
                                     LSTM, Bidirectional, GlobalAveragePooling1D,
                                     Dense, Dropout)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import mixed_precision

from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             balanced_accuracy_score, average_precision_score,
                             classification_report, confusion_matrix)
from scipy.stats import ks_2samp

import matplotlib.pyplot as plt
import seaborn as sns

# AUTO-INJECTED: Korean font setup for matplotlib
import os as _os
import matplotlib.font_manager as _fm
import matplotlib.pyplot as _plt
if not any('NanumGothic' in f.name for f in _fm.fontManager.ttflist):
    for _font in ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                  '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf']:
        if _os.path.exists(_font):
            _fm.fontManager.addfont(_font)
_plt.rcParams.update({'font.family': 'NanumGothic', 'axes.unicode_minus': False})
del _os, _fm, _plt
# END AUTO-INJECTED Korean font setup


SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        try: tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError: pass
    mixed_precision.set_global_policy('mixed_float16')
    print(f'✅ GPU: {[g.name for g in gpus]}  | precision: {mixed_precision.global_policy().name}')
else:
    print('⚠️ CPU 실행')
print(f'TF: {tf.__version__}')

# %% [markdown] cell 6
# ## 1. 데이터 로드

# %% cell 7
import gc


def _numpy_frombuffer_compat(buf, dtype, shape, order, axis_order=None):
    """Load NumPy 2.4 pickles on Colab runtimes with an older private helper."""
    array = np.frombuffer(buf, dtype=dtype)
    if order == 'K' and axis_order is not None:
        return array.reshape(shape, order='C').transpose(axis_order)
    return array.reshape(shape, order=order)


class CompatibleNumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'numpy._core.numeric' and name == '_frombuffer':
            return _numpy_frombuffer_compat
        return super().find_class(module, name)


def load_pickle_compat(path):
    with open(path, 'rb') as f:
        return CompatibleNumpyUnpickler(f).load()


data = load_pickle_compat(PKL_PATH)
legacy_keys = {'X_train_raw', 'y_train', 'groups_train', 'X_test_raw',
               'y_test', 'groups_test', 'feature_names', 'label_names'}
current_keys = {'X_integrated_seq', 'y', 'patient_id', 'split', 'integrated_feature_names'}

if legacy_keys.issubset(data):
    print('schema: legacy pre-split dataset')
    X_train_raw = np.asarray(data['X_train_raw'], dtype=np.float32)
    y_train = np.asarray(data['y_train'], dtype=np.int64)
    g_train = np.asarray(data['groups_train']).astype(str)
    X_test_raw = np.asarray(data['X_test_raw'], dtype=np.float32)
    y_test = np.asarray(data['y_test'], dtype=np.int64)
    g_test = np.asarray(data['groups_test']).astype(str)
    feature_names = list(data['feature_names'])
    label_names = data['label_names']
elif current_keys.issubset(data):
    print('schema: 0531 integrated dataset')
    X_all = np.asarray(data['X_integrated_seq'], dtype=np.float32)
    y_all = np.asarray(data['y'], dtype=np.int64)
    g_all = np.asarray(data['patient_id']).astype(str)
    split_all = np.asarray(data['split']).astype(str)
    train_mask = split_all == 'train'
    test_mask = split_all == 'val'
    X_train_raw, X_test_raw = X_all[train_mask], X_all[test_mask]
    y_train, y_test = y_all[train_mask], y_all[test_mask]
    g_train, g_test = g_all[train_mask], g_all[test_mask]
    feature_names = list(data['integrated_feature_names'])
    label_names = {0: 'CN', 1: 'MCI+Dem'}
    del X_all, y_all, g_all, split_all
else:
    raise KeyError(f'지원하지 않는 pickle schema입니다. available keys={sorted(data)}')

del data
gc.collect()
assert X_train_raw.ndim == X_test_raw.ndim == 3
assert X_train_raw.shape[1:] == X_test_raw.shape[1:]
assert X_train_raw.shape[-1] == len(feature_names)
assert not (set(g_train) & set(g_test))
assert np.isfinite(X_train_raw).all() and np.isfinite(X_test_raw).all()

print(f'X_train: {X_train_raw.shape}, X_test: {X_test_raw.shape}')
print(f'features: {len(feature_names)}')
print(f'label_names: {label_names}')
print(f'\nTrain CN={(y_train==0).sum()}, MCI+Dem={(y_train==1).sum()}')
print(f'Test  CN={(y_test==0).sum()}, MCI+Dem={(y_test==1).sum()}')
print(f'Train subj={len(set(g_train))}, Test subj={len(set(g_test))}')

# %% [markdown] cell 8
# # Part A — Sanity Check
#
# ## A1. 라벨 매핑 & Subject 누수

# %% cell 9
sanity = {}

# 라벨 매핑
assert 0 in label_names and 1 in label_names
sanity['label_0'] = str(label_names[0])
sanity['label_1'] = str(label_names[1])
print(f'label_names[0] = {label_names[0]}  (must be CN)')
print(f'label_names[1] = {label_names[1]}  (must be MCI+Dem)')

# subject별 라벨 일관성: 한 사람의 모든 sequence 라벨이 동일해야 함
def per_subject_label_consistency(y, g, name):
    df = pd.DataFrame({'g': g, 'y': y})
    counts = df.groupby('g')['y'].nunique()
    inconsistent = counts[counts > 1]
    print(f'  {name}: subject별 라벨 unique 개수 분포 → {counts.value_counts().to_dict()}')
    if len(inconsistent) > 0:
        print(f'  ⚠️ 한 subject가 여러 라벨을 가짐: {len(inconsistent)}명')
    return len(inconsistent)

sanity['train_inconsistent_subjects'] = per_subject_label_consistency(y_train, g_train, 'Train')
sanity['test_inconsistent_subjects']  = per_subject_label_consistency(y_test,  g_test,  'Test')

# Subject 누수 (정확/접두사)
overlap_exact = set(g_train) & set(g_test)
print(f'\nExact subject overlap: {len(overlap_exact)}  (must be 0)')
sanity['subject_overlap_exact'] = len(overlap_exact)

def email_prefix(s): return str(s).split('@')[0].lower().strip().replace('+','').replace('_','')
tr_pref = {email_prefix(s) for s in set(g_train)}
te_pref = {email_prefix(s) for s in set(g_test)}
overlap_pref = tr_pref & te_pref
print(f'Prefix overlap: {len(overlap_pref)}  (의심 케이스: {sorted(overlap_pref)[:5]})')
sanity['subject_overlap_prefix'] = len(overlap_pref)

# %% [markdown] cell 10
# ## A2. Padding(-1) 비율 — Train vs Test

# %% cell 11
def pad_ratio(X):
    return float((X == -1).mean())

pad_tr = pad_ratio(X_train_raw)
pad_te = pad_ratio(X_test_raw)
print(f'Train -1 비율: {pad_tr:.4f}  ({pad_tr*100:.2f}%)')
print(f'Test  -1 비율: {pad_te:.4f}  ({pad_te*100:.2f}%)')
print(f'차이         : {abs(pad_tr-pad_te):.4f}')

# 피처별 padding 비율 차이
pad_feat_tr = (X_train_raw == -1).reshape(-1, X_train_raw.shape[-1]).mean(axis=0)
pad_feat_te = (X_test_raw  == -1).reshape(-1, X_test_raw.shape[-1]).mean(axis=0)
pad_diff = pd.DataFrame({
    'feature': feature_names,
    'pad_train': pad_feat_tr,
    'pad_test' : pad_feat_te,
    'diff': pad_feat_te - pad_feat_tr,
}).sort_values('diff', key=abs, ascending=False)
print('\n=== Top 10 padding 차이 큰 feature ===')
print(pad_diff.head(10).to_string(index=False))

sanity['pad_ratio_train'] = pad_tr
sanity['pad_ratio_test']  = pad_te
sanity['pad_ratio_diff']  = abs(pad_tr - pad_te)
pad_diff.to_csv(os.path.join(ARTIFACT_DIR, 'padding_diff.csv'), index=False)

# %% [markdown] cell 12
# ## A3. 분포 시프트 — Kolmogorov-Smirnov 검정

# %% cell 13
# -1 padding 제외하고 feature별 KS test
def feature_values_no_pad(X, fi):
    v = X[..., fi].ravel()
    return v[v != -1]

ks_rows = []
for fi, fn in enumerate(feature_names):
    vt = feature_values_no_pad(X_train_raw, fi)
    ve = feature_values_no_pad(X_test_raw,  fi)
    if len(vt) < 30 or len(ve) < 30:
        ks_rows.append({'feature': fn, 'ks_stat': np.nan, 'p_value': np.nan,
                        'mean_train': np.nan, 'mean_test': np.nan})
        continue
    ks, pv = ks_2samp(vt, ve)
    ks_rows.append({'feature': fn, 'ks_stat': float(ks), 'p_value': float(pv),
                    'mean_train': float(vt.mean()), 'mean_test': float(ve.mean())})

ks_df = pd.DataFrame(ks_rows).sort_values('ks_stat', ascending=False)
n_shifted = int((ks_df['p_value'] < 0.05).sum())
n_shifted_strict = int((ks_df['p_value'] < 0.001).sum())
print(f'p<0.05  feature: {n_shifted} / {len(feature_names)}')
print(f'p<0.001 feature: {n_shifted_strict} / {len(feature_names)}')
print('\n=== Top 15 분포 차이 큰 feature ===')
print(ks_df.head(15).to_string(index=False))

ks_df.to_csv(os.path.join(ARTIFACT_DIR, 'ks_test.csv'), index=False)
sanity['ks_p05_count']   = n_shifted
sanity['ks_p001_count']  = n_shifted_strict
sanity['ks_top1_stat']   = float(ks_df['ks_stat'].iloc[0])

# %% [markdown] cell 14
# ## A4. Multi-seed CV 안정성
#
# 3개 seed로 5-fold CV를 가볍게 돌려 AUC 분산만 확인. 베이스라인은 04의 작은 hierarchical 모델.

# %% cell 15
from sklearn.preprocessing import MinMaxScaler

def normalize_with(X, sc):
    return np.asarray([sc.transform(s) for s in X], dtype=np.float32)

def tiny_baseline(input_shape):
    inp = Input(shape=input_shape)
    x = Conv1D(16, 3, padding='same')(inp)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Bidirectional(LSTM(8, return_sequences=True))(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(8, activation='relu')(x); x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid', dtype='float32')(x)
    m = Model(inp, out)
    m.compile(optimizer=Adam(5e-4), loss='binary_crossentropy',
              metrics=[tf.keras.metrics.AUC(name='auc')])
    return m

def cv_one_seed(seed, X, y, g):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for tr, va in sgkf.split(X, y, groups=g):
        sc = MinMaxScaler(); sc.fit(X[tr].reshape(-1, X.shape[-1]))
        Xt = normalize_with(X[tr], sc); Xv = normalize_with(X[va], sc)
        cw = compute_class_weight('balanced', classes=np.unique(y[tr]), y=y[tr])
        cw = dict(zip(np.unique(y[tr]).astype(int), cw))
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed); np.random.seed(seed)
        m = tiny_baseline(Xt.shape[1:])
        m.fit(Xt, y[tr], validation_data=(Xv, y[va]),
              epochs=30, batch_size=32, class_weight=cw,
              callbacks=[EarlyStopping(monitor='val_auc', mode='max',
                                       patience=6, restore_best_weights=True)],
              verbose=0)
        p = m.predict(Xv, verbose=0).ravel()
        if len(np.unique(y[va])) > 1:
            aucs.append(roc_auc_score(y[va], p))
    return aucs

multi_seed_results = {}
for s in [42, 7, 2024]:
    aucs = cv_one_seed(s, X_train_raw, y_train, g_train)
    multi_seed_results[s] = aucs
    print(f'Seed {s}: AUCs={[f"{a:.3f}" for a in aucs]}  mean={np.mean(aucs):.3f}  std={np.std(aucs):.3f}')

all_aucs = [a for v in multi_seed_results.values() for a in v]
sanity['multi_seed_auc_mean']  = float(np.mean(all_aucs))
sanity['multi_seed_auc_std']   = float(np.std(all_aucs))
sanity['multi_seed_auc_min']   = float(np.min(all_aucs))
sanity['multi_seed_auc_max']   = float(np.max(all_aucs))
print(f'\n=== 종합: {len(all_aucs)} folds, mean={np.mean(all_aucs):.3f} ± {np.std(all_aucs):.3f} '
      f'[{np.min(all_aucs):.3f}, {np.max(all_aucs):.3f}]')

# %% [markdown] cell 16
# ## A5. Sanity 종합 진단

# %% cell 17
def diagnose(sanity, n_features):
    flags = []
    if sanity['subject_overlap_exact'] > 0 or sanity['subject_overlap_prefix'] > 0:
        flags.append('🔴 CRITICAL: subject leakage 의심 (overlap > 0)')
    if sanity['train_inconsistent_subjects'] > 0 or sanity['test_inconsistent_subjects'] > 0:
        flags.append('🔴 CRITICAL: subject별 라벨이 일관되지 않음 (라벨링 오류)')
    if sanity['pad_ratio_diff'] > 0.10:
        flags.append(f'🟠 WARN: Train/Test padding 비율 차이 {sanity["pad_ratio_diff"]:.3f} (>0.10)')
    if sanity['ks_p001_count'] / n_features > 0.5:
        flags.append(f'🟠 WARN: KS p<0.001 feature {sanity["ks_p001_count"]}/{n_features} (>50% 분포 시프트)')
    elif sanity['ks_p05_count'] / n_features > 0.5:
        flags.append(f'🟡 NOTE: KS p<0.05 feature {sanity["ks_p05_count"]}/{n_features} (>50% 약한 시프트)')
    if sanity['multi_seed_auc_std'] > 0.10:
        flags.append(f'🟠 WARN: CV AUC seed간 std={sanity["multi_seed_auc_std"]:.3f} (>0.10, 불안정)')
    if sanity['multi_seed_auc_mean'] < 0.55:
        flags.append(f'🔴 CRITICAL: baseline CV AUC mean={sanity["multi_seed_auc_mean"]:.3f} (<0.55)')
    return flags

flags = diagnose(sanity, len(feature_names))
print('=== SANITY 종합 진단 ===\n')
print(json.dumps(sanity, indent=2))
print('\n--- 플래그 ---')
if flags:
    for f in flags: print(' ', f)
else:
    print('  ✅ 큰 문제 없음')

with open(os.path.join(ARTIFACT_DIR, 'sanity_report.json'), 'w') as f:
    json.dump({'sanity': sanity, 'flags': flags}, f, indent=2)

# %% [markdown] cell 18
# # Part B — Regularized Hierarchical LSTM
#
# ## B1. Feature Selection (Random Forest Importance Top-20)
#
# 전체 Train (시퀀스를 펴서 day-level row 만든 뒤) RF로 importance 계산, top-20만 선별.
# 회의록의 Lee et al. 2024 후진제거법과 동일한 동기.

# %% cell 19
# (N, 7, F) → (N*7, F) day-level rows, -1 padding 제거
def flatten_day_level(X, y, g):
    n, t, f = X.shape
    Xf = X.reshape(-1, f)
    yf = np.repeat(y, t)
    gf = np.repeat(g, t)
    # -1 padding row 제외
    mask = ~(Xf == -1).all(axis=1)
    return Xf[mask], yf[mask], gf[mask]

Xd, yd, gd = flatten_day_level(X_train_raw, y_train, g_train)
print(f'Day-level rows: {Xd.shape}  (CN={(yd==0).sum()}, MCI+Dem={(yd==1).sum()})')

# 결측 마커(-1)을 0으로 일시 치환 (RF는 imputation 필요)
Xd_imp = np.where(Xd == -1, 0.0, Xd)

rf = RandomForestClassifier(n_estimators=300, max_depth=8,
                            class_weight='balanced',
                            n_jobs=-1, random_state=SEED)
rf.fit(Xd_imp, yd)
imp = pd.DataFrame({'feature': feature_names,
                    'importance': rf.feature_importances_}
                  ).sort_values('importance', ascending=False).reset_index(drop=True)
print('\n=== Top 20 features ===')
print(imp.head(20).to_string(index=False))
imp.to_csv(os.path.join(ARTIFACT_DIR, 'rf_feature_importance.csv'), index=False)

TOP_K = 20
top_idx = imp.index[:TOP_K].tolist()
# imp는 sorted라 index가 원본 순서와 다름. feature 이름으로 다시 매핑
top_features = imp['feature'].iloc[:TOP_K].tolist()
top_indices = [feature_names.index(fn) for fn in top_features]
print(f'\n선택된 feature indices: {top_indices}')

# %% cell 20
# 선택된 feature만 추출
X_train_sel = X_train_raw[:, :, top_indices]
X_test_sel  = X_test_raw[:,  :, top_indices]
selected_features = top_features
print(f'X_train_sel: {X_train_sel.shape}')
print(f'X_test_sel : {X_test_sel.shape}')

# %% [markdown] cell 21
# ## B2. Per-subject Z-score 정규화
#
# 각 subject의 모든 sequence를 합쳐 -1 제외 mean/std로 z-score. Padding 위치는 0.

# %% cell 22
def per_subject_zscore(X, groups):
    X = X.copy().astype(np.float32)
    for s in np.unique(groups):
        idx = (groups == s)
        block = X[idx]                     # (n_seq, T, F)
        valid = (block != -1)
        for fi in range(X.shape[-1]):
            col  = block[..., fi]
            mask = valid[..., fi]
            if mask.sum() < 2:
                col_new = np.zeros_like(col)
            else:
                m, s_ = col[mask].mean(), col[mask].std() + 1e-8
                col_new = (col - m) / s_
                col_new[~mask] = 0.0
            block[..., fi] = col_new
        X[idx] = block
    return X

X_train_norm = per_subject_zscore(X_train_sel, g_train)
X_test_norm  = per_subject_zscore(X_test_sel,  g_test)
print(f'X_train_norm: {X_train_norm.shape}  mean={X_train_norm.mean():.4f}, std={X_train_norm.std():.4f}')
print(f'X_test_norm : {X_test_norm.shape}   mean={X_test_norm.mean():.4f},  std={X_test_norm.std():.4f}')

# %% [markdown] cell 23
# ## B3. 정규화 강화 모델 + Subject-AUC Callback

# %% cell 24
L2 = 1e-3

def build_reg_model(input_shape, conv_filters=16, lstm_units=8,
                    dense_units=8, dropout=0.4, lr=5e-4, l2_coef=L2):
    inp = Input(shape=input_shape)
    x = Conv1D(conv_filters, 3, padding='same',
               kernel_regularizer=l2(l2_coef))(inp)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True,
                           kernel_regularizer=l2(l2_coef),
                           recurrent_regularizer=l2(l2_coef)))(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(dense_units, activation='relu',
              kernel_regularizer=l2(l2_coef))(x)
    x = Dropout(dropout)(x)
    out = Dense(1, activation='sigmoid', dtype='float32',
                kernel_regularizer=l2(l2_coef))(x)
    m = Model(inp, out)
    m.compile(optimizer=Adam(lr), loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return m

class SubjectAUCMonitor(Callback):
    """매 epoch end에서 val을 subject-level로 집계한 AUC를 logs에 기록 + best 추적."""
    def __init__(self, X_val, y_val, g_val, patience=10, restore=True):
        super().__init__()
        self.X, self.y, self.g = X_val, y_val, g_val
        self.patience, self.restore = patience, restore
        self.best, self.wait = -np.inf, 0
        self.best_weights, self.best_epoch = None, 0
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        p = self.model.predict(self.X, verbose=0).ravel()
        df = pd.DataFrame({'g': self.g, 'p': p, 'y': self.y})
        agg = df.groupby('g').agg(p=('p','mean'), y=('y','first'))
        auc = roc_auc_score(agg['y'], agg['p']) if agg['y'].nunique()>1 else np.nan
        logs['val_subj_auc'] = float(auc)
        self.history.append(auc)
        if not np.isnan(auc) and auc > self.best:
            self.best, self.wait = auc, 0
            self.best_epoch = epoch + 1
            if self.restore:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.restore and self.best_weights is not None:
            self.model.set_weights(self.best_weights)

print('Reg model demo:')
demo = build_reg_model((7, TOP_K))
demo.summary()
print(f'Total params: {demo.count_params():,}')

# %% [markdown] cell 25
# ## B4. CV 학습 (Subject-level AUC monitor)

# %% cell 26
def train_one_fold_reg(hp, X_tr, y_tr, g_tr, X_va, y_va, g_va, verbose=0):
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED); np.random.seed(SEED)
    m = build_reg_model(input_shape=X_tr.shape[1:],
                        conv_filters=hp['conv_filters'],
                        lstm_units=hp['lstm_units'],
                        dense_units=hp['dense_units'],
                        dropout=hp['dropout'], lr=hp['lr'],
                        l2_coef=hp.get('l2', L2))
    cw = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    cw = dict(zip(np.unique(y_tr).astype(int), cw))
    monitor = SubjectAUCMonitor(X_va, y_va, g_va, patience=10)
    rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5)
    m.fit(X_tr, y_tr, validation_data=(X_va, y_va),
          epochs=hp['epochs'], batch_size=hp['batch_size'],
          class_weight=cw, callbacks=[monitor, rl], verbose=verbose)
    p_va = m.predict(X_va, verbose=0).ravel()
    df = pd.DataFrame({'g': g_va, 'p': p_va, 'y': y_va})
    agg = df.groupby('g').agg(p=('p','mean'), y=('y','first')).reset_index()
    # threshold tuning on subject-level
    grid = np.arange(0.10, 0.91, 0.02)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(agg['y'], (agg['p']>=t).astype(int), average='macro')
        if f1 > best_f1: best_t, best_f1 = float(t), f1
    return {
        'subj_auc': monitor.best,
        'subj_f1' : best_f1,
        'subj_thr': best_t,
        'best_epoch': monitor.best_epoch,
    }

def run_cv_reg(hp, X, y, g, n_splits=5, verbose=0):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rows = []
    for fi, (tr, va) in enumerate(sgkf.split(X, y, groups=g)):
        r = train_one_fold_reg(hp,
                               X[tr], y[tr], g[tr],
                               X[va], y[va], g[va],
                               verbose=verbose)
        rows.append({'fold': fi+1, **r})
    return pd.DataFrame(rows)

baseline_hp = dict(conv_filters=16, lstm_units=8, dense_units=8,
                   dropout=0.4, lr=5e-4, l2=1e-3,
                   batch_size=32, epochs=80)

print('=== Regularized Baseline CV ===')
base_df = run_cv_reg(baseline_hp, X_train_norm, y_train, g_train, verbose=0)
print(base_df.to_string(index=False))
print(f"\nMean subj AUC: {base_df['subj_auc'].mean():.4f} ± {base_df['subj_auc'].std():.4f}")
print(f"Mean subj F1 : {base_df['subj_f1'].mean():.4f} ± {base_df['subj_f1'].std():.4f}")

# %% [markdown] cell 27
# ## B5. Random Search (8 trials, subject AUC 기준)
#
# 정규화 강화 모델은 학습이 빨라 trial 수를 줄여도 충분히 탐색됩니다.

# %% cell 28
HP_SPACE = {
    'conv_filters': [8, 16, 32],
    'lstm_units' : [4, 8, 16],
    'dense_units': [4, 8, 16],
    'dropout'    : [0.3, 0.4, 0.5, 0.6],
    'lr'         : [1e-3, 5e-4, 3e-4, 1e-4],
    'l2'         : [1e-4, 1e-3, 5e-3, 1e-2],
    'batch_size' : [16, 32],
}

def sample(rng):
    return {k: v[rng.randint(len(v))] for k, v in HP_SPACE.items()}

N_TRIALS = 8
rng = np.random.RandomState(SEED)

rs_rows = []
for t in range(N_TRIALS):
    hp = sample(rng); hp['epochs'] = 60
    df = run_cv_reg(hp, X_train_norm, y_train, g_train, verbose=0)
    rs_rows.append({'trial': t+1, **hp,
                    'cv_subj_auc': df['subj_auc'].mean(),
                    'cv_subj_f1' : df['subj_f1'].mean(),
                    'cv_subj_auc_std': df['subj_auc'].std()})
    print(f"Trial {t+1}/{N_TRIALS}: subj_AUC={rs_rows[-1]['cv_subj_auc']:.4f} ± "
          f"{rs_rows[-1]['cv_subj_auc_std']:.4f}  | subj_F1={rs_rows[-1]['cv_subj_f1']:.4f}")
    print(f'  HP: {hp}')

rs_df = pd.DataFrame(rs_rows).sort_values(['cv_subj_auc','cv_subj_f1'], ascending=False).reset_index(drop=True)
rs_df.to_csv(os.path.join(ARTIFACT_DIR, 'random_search.csv'), index=False)
print('\n=== Top 5 ===')
print(rs_df.head(5)[['trial','conv_filters','lstm_units','dense_units',
                     'dropout','lr','l2','batch_size',
                     'cv_subj_auc','cv_subj_f1']].to_string(index=False))

# %% cell 29
best = rs_df.iloc[0]
best_hp = dict(
    conv_filters=int(best['conv_filters']),
    lstm_units=int(best['lstm_units']),
    dense_units=int(best['dense_units']),
    dropout=float(best['dropout']),
    lr=float(best['lr']),
    l2=float(best['l2']),
    batch_size=int(best['batch_size']),
    epochs=100,
)
print('=== Best HP ===')
print(json.dumps(best_hp, indent=2))
print(f"\nCV subj AUC: {best['cv_subj_auc']:.4f}")
print(f"CV subj F1 : {best['cv_subj_f1']:.4f}")
with open(os.path.join(ARTIFACT_DIR, 'best_hp.json'), 'w') as f:
    json.dump(best_hp, f, indent=2)

# %% [markdown] cell 30
# ## B6. 최종 모델 + Threshold (subject-level) + Test 평가

# %% cell 31
# 내부 val 분리 (subject-level)
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
tr_idx, va_idx = next(gss.split(X_train_norm, y_train, groups=g_train))
print(f'Train: {len(tr_idx)} seq ({len(set(g_train[tr_idx]))}명)')
print(f'Val  : {len(va_idx)} seq ({len(set(g_train[va_idx]))}명)')

tf.keras.backend.clear_session()
tf.random.set_seed(SEED); np.random.seed(SEED)
final_model = build_reg_model(
    input_shape=(7, TOP_K),
    conv_filters=best_hp['conv_filters'],
    lstm_units=best_hp['lstm_units'],
    dense_units=best_hp['dense_units'],
    dropout=best_hp['dropout'], lr=best_hp['lr'],
    l2_coef=best_hp['l2'],
)
cw = compute_class_weight('balanced', classes=np.unique(y_train[tr_idx]), y=y_train[tr_idx])
cw = dict(zip(np.unique(y_train[tr_idx]).astype(int), cw))
print(f'class_weight: {cw}')

monitor = SubjectAUCMonitor(X_train_norm[va_idx], y_train[va_idx],
                            g_train[va_idx], patience=15)
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
hist = final_model.fit(
    X_train_norm[tr_idx], y_train[tr_idx],
    validation_data=(X_train_norm[va_idx], y_train[va_idx]),
    epochs=best_hp['epochs'], batch_size=best_hp['batch_size'],
    class_weight=cw, callbacks=[monitor, rl], verbose=1)

final_model.save(os.path.join(ARTIFACT_DIR, 'final_model.keras'))
print(f'\nBest subj AUC on val: {monitor.best:.4f} (epoch {monitor.best_epoch})')

# Val에서 subject-level threshold 튜닝
p_val = final_model.predict(X_train_norm[va_idx], verbose=0).ravel()
val_subj = pd.DataFrame({'g': g_train[va_idx], 'p': p_val, 'y': y_train[va_idx]}
                       ).groupby('g').agg(p=('p','mean'), y=('y','first')).reset_index()
grid = np.arange(0.10, 0.91, 0.02)
t_star, f1_star = 0.5, -1.0
for t in grid:
    f1 = f1_score(val_subj['y'], (val_subj['p']>=t).astype(int), average='macro')
    if f1 > f1_star: t_star, f1_star = float(t), f1
print(f'Val subject threshold (F1-max): t={t_star:.3f}  →  F1={f1_star:.4f}')

# %% cell 32
# 학습 곡선
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(hist.history['loss'], label='train')
ax[0].plot(hist.history['val_loss'], label='val')
ax[0].set_title('Loss'); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(hist.history['auc'], label='train')
ax[1].plot(hist.history['val_auc'], label='val_seq')
ax[1].set_title('Sequence AUC'); ax[1].legend(); ax[1].grid(alpha=0.3)
ax[2].plot(monitor.history, label='val_subj_auc', color='tab:green')
ax[2].axvline(monitor.best_epoch-1, color='r', ls='--', label=f'best @{monitor.best_epoch}')
ax[2].set_title('Subject AUC (val)'); ax[2].legend(); ax[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACT_DIR, 'training_curve.png'), dpi=120, bbox_inches='tight')
plt.show()

# %% cell 33
# Test 평가
p_test = final_model.predict(X_test_norm, verbose=0).ravel()

# AUC sanity
auc_seq_raw  = roc_auc_score(y_test, p_test)     if len(np.unique(y_test))>1 else np.nan
auc_seq_flip = roc_auc_score(y_test, 1-p_test)   if len(np.unique(y_test))>1 else np.nan
print(f'[Sanity] Seq AUC raw={auc_seq_raw:.4f}  flipped={auc_seq_flip:.4f}')

# Subject aggregation
subj_test = pd.DataFrame({'subj': g_test, 'p': p_test, 'y': y_test}
                        ).groupby('subj').agg(p=('p','mean'), y=('y','first')).reset_index()
auc_subj_raw  = roc_auc_score(subj_test['y'], subj_test['p'])    if subj_test['y'].nunique()>1 else np.nan
auc_subj_flip = roc_auc_score(subj_test['y'], 1-subj_test['p'])  if subj_test['y'].nunique()>1 else np.nan
print(f'[Sanity] Subj AUC raw={auc_subj_raw:.4f}  flipped={auc_subj_flip:.4f}')

def metrics_at(y, p, t):
    pred = (p >= t).astype(int)
    return {
        'acc': accuracy_score(y, pred),
        'bal_acc': balanced_accuracy_score(y, pred),
        'f1_macro': f1_score(y, pred, average='macro'),
        'auc': roc_auc_score(y, p) if len(np.unique(y))>1 else np.nan,
        'pr_auc': average_precision_score(y, p) if len(np.unique(y))>1 else np.nan,
        'threshold': float(t),
    }

m_seq_05  = metrics_at(y_test, p_test, 0.5)
m_seq_t   = metrics_at(y_test, p_test, t_star)
m_subj_05 = metrics_at(subj_test['y'].values, subj_test['p'].values, 0.5)
m_subj_t  = metrics_at(subj_test['y'].values, subj_test['p'].values, t_star)

print('\n' + '='*60)
print(f'  SEQUENCE LEVEL  ({len(y_test)} seq)')
print('='*60)
for nm, m in [('@0.5', m_seq_05), (f'@tuned={t_star:.2f}', m_seq_t)]:
    print(f'  {nm:18s} acc={m["acc"]:.4f}  bal={m["bal_acc"]:.4f}  '
          f'F1={m["f1_macro"]:.4f}  AUC={m["auc"]:.4f}  PR-AUC={m["pr_auc"]:.4f}')

print('\n' + '='*60)
print(f'  SUBJECT LEVEL  ({len(subj_test)} subj, '
      f'CN={(subj_test.y==0).sum()}, MCI+Dem={(subj_test.y==1).sum()})')
print('='*60)
for nm, m in [('@0.5', m_subj_05), (f'@tuned={t_star:.2f}', m_subj_t)]:
    print(f'  {nm:18s} acc={m["acc"]:.4f}  bal={m["bal_acc"]:.4f}  '
          f'F1={m["f1_macro"]:.4f}  AUC={m["auc"]:.4f}  PR-AUC={m["pr_auc"]:.4f}')

print('\n[Subject classification report — tuned]')
print(classification_report(subj_test['y'], (subj_test['p']>=t_star).astype(int),
                            target_names=['CN','MCI+Dem']))

# confusion matrix
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
cm_seq = confusion_matrix(y_test, (p_test>=t_star).astype(int))
sns.heatmap(cm_seq, annot=True, fmt='d', cmap='Blues',
            xticklabels=['CN','MCI+Dem'], yticklabels=['CN','MCI+Dem'], ax=ax[0])
ax[0].set_title(f'Sequence (t={t_star:.2f})  AUC={m_seq_t["auc"]:.3f}')
cm_subj = confusion_matrix(subj_test['y'], (subj_test['p']>=t_star).astype(int))
sns.heatmap(cm_subj, annot=True, fmt='d', cmap='Greens',
            xticklabels=['CN','MCI+Dem'], yticklabels=['CN','MCI+Dem'], ax=ax[1])
ax[1].set_title(f'Subject (t={t_star:.2f})  AUC={m_subj_t["auc"]:.3f}')
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACT_DIR, 'confusion_matrices.png'), dpi=120, bbox_inches='tight')
plt.show()

subj_test.to_csv(os.path.join(ARTIFACT_DIR, 'subject_predictions.csv'), index=False)

# %% [markdown] cell 34
# ## B7. 결과 정리 + 다음 단계 자동 판단

# %% cell 35
final_metrics = {
    'best_hp': best_hp,
    'tuned_threshold': float(t_star),
    'cv_subj_auc_mean': float(best['cv_subj_auc']),
    'cv_subj_f1_mean' : float(best['cv_subj_f1']),
    'test_seq_at_0.5'   : m_seq_05,
    'test_seq_at_tuned' : m_seq_t,
    'test_subj_at_0.5'  : m_subj_05,
    'test_subj_at_tuned': m_subj_t,
    'auc_sanity': {'seq_raw': float(auc_seq_raw), 'seq_flip': float(auc_seq_flip),
                   'subj_raw': float(auc_subj_raw), 'subj_flip': float(auc_subj_flip)},
    'selected_features': selected_features,
    'sanity_flags': flags,
}
with open(os.path.join(ARTIFACT_DIR, 'final_metrics.json'), 'w') as f:
    json.dump(final_metrics, f, indent=2)
print(json.dumps({k: v for k, v in final_metrics.items()
                  if k not in ('selected_features',)}, indent=2))

# 다음 단계 자동 판단
print('\n' + '='*60)
print('  NEXT STEP RECOMMENDATION')
print('='*60)
subj_auc = m_subj_t['auc']
critical_flags = [f for f in flags if f.startswith('🔴')]

if critical_flags:
    print('🛑 SANITY 단계에서 CRITICAL 플래그 감지:')
    for f in critical_flags: print(f'   {f}')
    print('   → 모델 변경 전 데이터 자체 점검 필요. 전처리 파이프라인 재확인 권장.')
elif subj_auc >= 0.70:
    print(f'✅ Subject AUC = {subj_auc:.3f} (≥0.70)')
    print('   → 정규화 강화가 효과적. 다음: 06 SHAP 해석 + 추가 미세조정')
elif subj_auc >= 0.60:
    print(f'🟡 Subject AUC = {subj_auc:.3f} (0.60~0.70)')
    print('   → 부분 개선. 다음 후보:')
    print('     (a) 06_focal_loss.ipynb — focal γ=2.0 + label smoothing')
    print('     (b) 06_attention.ipynb  — GlobalAvgPool → Attention pooling')
    print('     (c) 06_ensemble.ipynb   — best top-3 HP 모델 확률 평균')
elif subj_auc >= 0.55:
    print(f'🟠 Subject AUC = {subj_auc:.3f} (0.55~0.60)')
    print('   → 한계점 근접. 회의록 우선순위에 따라:')
    if sanity['ks_p001_count']/len(feature_names) > 0.3:
        print('     ▶ 분포 시프트 있음 → 06_repeated_cv_pooled.ipynb (train+test 합쳐 25-fold)')
    print('     ▶ 결측치 임계 완화 0602 pkl 재전처리 권장')
else:
    print(f'🔴 Subject AUC = {subj_auc:.3f} (<0.55)')
    print('   → 데이터 자체 한계 가능성. 다음:')
    print('     1) 0602 pkl (결측치 임계 완화) 재학습')
    print('     2) 회의록 합의: 6일 시퀀스도 포함하여 데이터 4명 부활')
    print('     3) AI-Hub Training/Validation 분할 분포 시프트 확인 (sanity A3 결과 참조)')

print('\n💾 ARTIFACT_DIR:')
for f in sorted(os.listdir(ARTIFACT_DIR)):
    print(f'  - {f}')
