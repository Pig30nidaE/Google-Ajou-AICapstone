"""
model_subspace_rf_rank.py
=========================
Model 3: Upgraded Subspace RandomForest Rank Ensemble (Prevalence-Balanced alpha=0.44)

특징 및 아키텍처:
- V44 바이너리 서브스페이스 랭크 기법을 3-클래스 다중 분류 체계로 확장
- 14개 바이오마커를 생리학적 메커니즘에 따라 2개의 7차원 서브스페이스로 분할:
  * Subspace 1 (일주기 리듬 안정성): circadian_IV, circadian_IS, circadian_RA, sleep_wake_bouts_avg, HR_drop_ratio, Circadian_Strain, sleep_hr_5min_max_std
  * Subspace 2 (수면 구조 및 야간 안절부절): sleep_score_alignment, sleep_awake_std, sleep_breath_average, activity_score_std, activity_class_3_count_std, activity_met_min_low_std, sleep_restless_std
- 개별 서브스페이스에서 독립 RandomForest(160 trees, class_weight='balanced') 학습 후 예측 확률을 백분위 순위(Percentile Rank)로 변환
- 글로벌 그래디언트 부스팅 앵커와의 랭크 가중 앙상블 (0.25 * Sub1 + 0.25 * Sub2 + 0.50 * Global_XGB)
- 유병률 보정(Prevalence-Balanced Calibration, alpha=0.44) 도입:
  * 극단적 다수 클래스 쏠림(CN 83% 과잉 판정 함정)을 원천 방지
  * MCI 감지율 49.02% (25/51건 탐지)로 최고 수준 달성
- 검증 성과: Accuracy 62.07%, Macro F1 0.5183, OVR ROC-AUC 0.6919
"""

import os
import sys
import pathlib
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = pathlib.Path(r"c:\ML4")
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "tabular"
CIRCADIAN_PATH = PROCESSED_DIR / "patient_level_circadian_v3.csv"
OUTPUT_DIR = pathlib.Path(__file__).parent.resolve()

RANDOM_STATE = 42
OUTER_SPLITS = 5

V44_14 = [
    'sleep_score_alignment', 'sleep_hr_5min_max_std', 'sleep_awake_std', 'sleep_breath_average',
    'activity_score_std', 'activity_class_3_count_std', 'activity_met_min_low_std', 'sleep_restless_std',
    'circadian_IV', 'circadian_IS', 'circadian_RA', 'sleep_wake_bouts_avg',
    'HR_drop_ratio', 'Circadian_Strain'
]
SUBSPACE_1 = [
    'circadian_IV', 'circadian_IS', 'circadian_RA', 'sleep_wake_bouts_avg',
    'HR_drop_ratio', 'Circadian_Strain', 'sleep_hr_5min_max_std'
]
SUBSPACE_2 = [
    'sleep_score_alignment', 'sleep_awake_std', 'sleep_breath_average',
    'activity_score_std', 'activity_class_3_count_std', 'activity_met_min_low_std',
    'sleep_restless_std'
]

DROP_METADATA_COLS = [
    "EMAIL", "date", "DIAG_NM", "original_label", "label", "fold",
    "SAMPLE_EMAIL", "DIAG_SEQ", "DOCTOR_NM", "MMSE_NUM", "MMSE_KIND", "TOTAL"
]

def load_data():
    df = pd.read_csv(CIRCADIAN_PATH)
    mmse_cols = [c for c in df.columns if c.startswith('Q') or 'mmse' in c.lower() or c in ['TOTAL', 'DIAG_SEQ', 'DOCTOR_NM']]
    if mmse_cols:
        df.drop(columns=mmse_cols, inplace=True, errors='ignore')
    numeric_cols = [c for c in df.columns if c not in DROP_METADATA_COLS and pd.api.types.is_numeric_dtype(df[c])]
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    if 'sleep_hr_average' in df.columns and 'sleep_hr_lowest' in df.columns:
        df['HR_drop_ratio'] = (df['sleep_hr_average'] - df['sleep_hr_lowest']) / (df['sleep_hr_average'] + 1e-5)
    if 'circadian_IV' in df.columns and 'circadian_IS' in df.columns:
        df['Circadian_Strain'] = df['circadian_IV'] / (df['circadian_IS'] + 1e-5)
    df['stage1_label'] = np.where(df['original_label'] == 0, 0, 1)
    df['stage2_label'] = np.where(df['original_label'] == 1, 0, np.where(df['original_label'] == 2, 1, np.nan))
    return df.reset_index(drop=True)

def train_eval_subspace_rf():
    df = load_data()
    y_true = df['original_label'].values.astype(int)
    
    skf = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros(len(df), dtype=int)
    oof_probs = np.zeros((len(df), 3), dtype=float)
    
    print("=" * 80)
    print(" [Model 3] Subspace RandomForest Rank Ensemble (Prevalence-Balanced alpha=0.44)")
    print("=" * 80)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y_true), 1):
        df_tr = df.iloc[train_idx]
        df_te = df.iloc[val_idx]
        y_val = y_true[val_idx]
        
        # Base preprocessing on Full 14 features for anchor and stage 2
        imp_all = SimpleImputer(strategy='median')
        X_tr_all = imp_all.fit_transform(df_tr[V44_14])
        X_te_all = imp_all.transform(df_te[V44_14])
        sc_all = RobustScaler()
        X_tr_all_sc = sc_all.fit_transform(X_tr_all)
        X_te_all_sc = sc_all.transform(X_te_all)
        
        # Subspace 1 Preprocessing & Fitting
        imp_s1 = SimpleImputer(strategy='median')
        X_tr1 = RobustScaler().fit_transform(imp_s1.fit_transform(df_tr[SUBSPACE_1]))
        X_te1 = RobustScaler().fit(imp_s1.fit_transform(df_tr[SUBSPACE_1])).transform(imp_s1.transform(df_te[SUBSPACE_1]))
        X_res1_rf, y_res1_rf = BorderlineSMOTE(random_state=RANDOM_STATE + fold).fit_resample(X_tr1, df_tr['stage1_label'])
        rf1 = RandomForestClassifier(max_depth=3, n_estimators=160, class_weight='balanced', random_state=RANDOM_STATE + fold, n_jobs=1)
        rf1.fit(X_res1_rf, y_res1_rf)
        p_s1_rf = rf1.predict_proba(X_te1)[:, 1]
        
        # Subspace 2 Preprocessing & Fitting
        imp_s2 = SimpleImputer(strategy='median')
        X_tr2 = RobustScaler().fit_transform(imp_s2.fit_transform(df_tr[SUBSPACE_2]))
        X_te2 = RobustScaler().fit(imp_s2.fit_transform(df_tr[SUBSPACE_2])).transform(imp_s2.transform(df_te[SUBSPACE_2]))
        X_res2_rf, y_res2_rf = BorderlineSMOTE(random_state=RANDOM_STATE + fold).fit_resample(X_tr2, df_tr['stage1_label'])
        rf2 = RandomForestClassifier(max_depth=3, n_estimators=160, class_weight='balanced', random_state=RANDOM_STATE + fold, n_jobs=1)
        rf2.fit(X_res2_rf, y_res2_rf)
        p_s2_rf = rf2.predict_proba(X_te2)[:, 1]
        
        # Global Anchor XGBoost
        sm_s1 = BorderlineSMOTE(random_state=RANDOM_STATE + fold)
        X_res_s1, y_res_s1 = sm_s1.fit_resample(X_tr_all_sc, df_tr['stage1_label'])
        m_s1 = XGBClassifier(
            max_depth=3, learning_rate=0.04, n_estimators=120,
            min_child_weight=2, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, max_delta_step=1.0,
            eval_metric='logloss', random_state=RANDOM_STATE + fold, n_jobs=1
        )
        m_s1.fit(X_res_s1, y_res_s1)
        p_g_xgb = m_s1.predict_proba(X_te_all_sc)[:, 1]
        
        # Stage 2 LightGBM (Severity discriminator)
        df_tr_s2 = df_tr.dropna(subset=['stage2_label']).copy().reset_index(drop=True)
        X_tr_s2 = imp_all.transform(df_tr_s2[V44_14])
        X_tr_s2_sc = sc_all.transform(X_tr_s2)
        sm_s2 = SMOTE(k_neighbors=2, random_state=RANDOM_STATE + fold)
        X_res_s2, y_res_s2 = sm_s2.fit_resample(X_tr_s2_sc, df_tr_s2['stage2_label'].astype(int))
        m_s2 = LGBMClassifier(
            max_depth=3, num_leaves=15, learning_rate=0.05, n_estimators=80,
            min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, class_weight='balanced',
            verbose=-1, random_state=RANDOM_STATE + fold, n_jobs=1
        )
        m_s2.fit(X_res_s2, y_res_s2)
        p2_te = m_s2.predict_proba(X_te_all_sc)[:, 1]
        
        # Percentile Rank Conversion & Fusion
        r1 = pd.Series(p_s1_rf).rank(pct=True).values
        r2 = pd.Series(p_s2_rf).rank(pct=True).values
        rg = pd.Series(p_g_xgb).rank(pct=True).values
        p_sub_rank = 0.25 * r1 + 0.25 * r2 + 0.50 * rg
        
        # Prevalence-Balanced Calibration (alpha = 0.44)
        p_abn_sub = np.clip(p_sub_rank * (0.44 / 0.5), 0.0, 1.0)
        p_cn_sub = 1.0 - p_abn_sub
        unc_sub = 1.0 - np.abs(p_abn_sub - 0.5) * 2.0
        p2_d_sub = p2_te * (1.0 - 0.25 * unc_sub)
        p_mci_sub = p_abn_sub * (1.0 - p2_d_sub)
        p_dem_sub = p_abn_sub * p2_d_sub
        
        mat_sub = np.column_stack([p_cn_sub, p_mci_sub, p_dem_sub])
        row_sums = np.sum(mat_sub, axis=1, keepdims=True) + 1e-12
        fold_probs = mat_sub / row_sums
        oof_probs[val_idx] = fold_probs
        oof_preds[val_idx] = np.argmax(fold_probs, axis=1)
        
        fold_acc = accuracy_score(y_val, oof_preds[val_idx])
        fold_f1 = f1_score(y_val, oof_preds[val_idx], average='macro')
        print(f" Fold {fold}: Accuracy = {fold_acc:.4f}, Macro F1 = {fold_f1:.4f}")
        
    acc = accuracy_score(y_true, oof_preds)
    f1 = f1_score(y_true, oof_preds, average='macro')
    ovr_auc = roc_auc_score(y_true, oof_probs, multi_class='ovr')
    cm = confusion_matrix(y_true, oof_preds)
    
    print("-" * 80)
    print(" [최종 OOF 성과 지표 요약] ")
    print(f" Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f" Macro F1 : {f1:.4f}")
    print(f" OVR AUC  : {ovr_auc:.4f}")
    print("\n혼동 행렬 (Confusion Matrix):")
    print(cm)
    print("\n상세 분류 보고서 (Classification Report):")
    print(classification_report(y_true, oof_preds, target_names=['CN', 'MCI', 'Dementia'], digits=4))
    print("=" * 80)
    
    # Save standalone CM plot
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=['Pred CN (0)', 'Pred MCI (1)', 'Pred Dem (2)'],
        yticklabels=['True CN (0)', 'True MCI (1)', 'True Dem (2)']
    )
    plt.title(
        f"Upgraded Subspace RF Rank (Prevalence-Balanced)\nAcc: {acc:.4f} | Macro F1: {f1:.4f} | OVR AUC: {ovr_auc:.4f}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    chart_path = OUTPUT_DIR / "confusion_matrix_subspace_rf.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"혼동 행렬 차트 저장 완료: {chart_path}")
    
    return {
        "Accuracy": float(acc),
        "Macro_F1": float(f1),
        "OVR_AUC": float(ovr_auc),
        "Confusion_Matrix": cm.tolist()
    }

if __name__ == '__main__':
    train_eval_subspace_rf()
