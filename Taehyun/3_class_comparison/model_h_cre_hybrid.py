"""
model_h_cre_hybrid.py
=====================
Model 2: Upgraded H-CRE Flagship Hybrid (Stage 1 XGBoost -> Stage 2 LightGBM)

특징 및 아키텍처:
- 임상적 선별 파이프라인을 모사한 2단계 계층적 구조:
  * Stage 1 (정상 선별): XGBoost 기반 CN vs Impaired(MCI+Dementia)
  * Stage 2 (중증도 감별): LightGBM 기반 MCI vs Dementia (불확실성 감쇠 패널티 적용)
- 치매 과잉 진단(False Alarm)을 억제하여 임상적 신뢰도(Precision 42.86%, 치매 오경보 단 4건) 극대화
- Fold별 Train 세트 완전 격리 전처리 (SimpleImputer + RobustScaler + BorderlineSMOTE)
- Strict 5-Fold Stratified K-Fold 완전 격리 교차검증 (Zero Data Leakage)
- 검증 성과: Accuracy 60.92%, Macro F1 0.4949, OVR ROC-AUC 0.6782, 치매 오진(FP) 단 4건
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

def train_eval_h_cre_hybrid():
    df = load_data()
    y_true = df['original_label'].values.astype(int)
    
    skf = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros(len(df), dtype=int)
    oof_probs = np.zeros((len(df), 3), dtype=float)
    
    print("=" * 80)
    print(" [Model 2] Upgraded H-CRE Flagship Hybrid (Stage 1 XGB -> Stage 2 LGBM)")
    print("=" * 80)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y_true), 1):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        y_val = y_true[val_idx]
        
        # -------------------------------------------------------------
        # Stage 1: CN (0) vs Impaired (1) with XGBoost
        # -------------------------------------------------------------
        s1_imputer = SimpleImputer(strategy='median')
        X_train_raw = s1_imputer.fit_transform(train_df[V44_14])
        X_val_raw = s1_imputer.transform(val_df[V44_14])
        
        s1_scaler = RobustScaler()
        X_train_s1 = s1_scaler.fit_transform(X_train_raw)
        X_val_s1 = s1_scaler.transform(X_val_raw)
        
        sm_s1 = BorderlineSMOTE(random_state=RANDOM_STATE + fold)
        X_res_s1, y_res_s1 = sm_s1.fit_resample(X_train_s1, train_df['stage1_label'])
        
        m_s1 = XGBClassifier(
            max_depth=3, learning_rate=0.04, n_estimators=120,
            min_child_weight=2, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, max_delta_step=1.0,
            eval_metric='logloss', random_state=RANDOM_STATE + fold, n_jobs=1
        )
        m_s1.fit(X_res_s1, y_res_s1)
        p1_val = m_s1.predict_proba(X_val_s1)[:, 1]
        
        # -------------------------------------------------------------
        # Stage 2: MCI (0) vs Dementia (1) with LightGBM
        # -------------------------------------------------------------
        train_s2 = train_df.dropna(subset=['stage2_label']).copy().reset_index(drop=True)
        X_train_s2_raw = s1_imputer.transform(train_s2[V44_14])
        X_train_s2 = s1_scaler.transform(X_train_s2_raw)
        
        sm_s2 = SMOTE(k_neighbors=2, random_state=RANDOM_STATE + fold)
        X_res_s2, y_res_s2 = sm_s2.fit_resample(X_train_s2, train_s2['stage2_label'].astype(int))
        
        m_s2 = LGBMClassifier(
            max_depth=3, num_leaves=15, learning_rate=0.05, n_estimators=80,
            min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, class_weight='balanced',
            verbose=-1, random_state=RANDOM_STATE + fold, n_jobs=1
        )
        m_s2.fit(X_res_s2, y_res_s2)
        p2_val = m_s2.predict_proba(X_val_s1)[:, 1]
        
        # -------------------------------------------------------------
        # Hierarchical Uncertainty Attenuation Probability Synthesis
        # -------------------------------------------------------------
        p_cn = 1.0 - p1_val
        p_abn = p1_val
        unc = 1.0 - np.abs(p_abn - 0.5) * 2.0
        p2_d = p2_val * (1.0 - 0.30 * unc)
        p_mci = p_abn * (1.0 - p2_d)
        p_dem = p_abn * p2_d
        
        fold_probs = np.column_stack([p_cn, p_mci, p_dem])
        row_sums = fold_probs.sum(axis=1, keepdims=True) + 1e-12
        fold_probs = fold_probs / row_sums
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
        f"Upgraded H-CRE Flagship Hybrid (XGB -> LGBM)\nAcc: {acc:.4f} | Macro F1: {f1:.4f} | OVR AUC: {ovr_auc:.4f}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    chart_path = OUTPUT_DIR / "confusion_matrix_h_cre_hybrid.png"
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
    train_eval_h_cre_hybrid()
