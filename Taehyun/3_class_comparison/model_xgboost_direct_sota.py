"""
model_xgboost_direct_sota.py
============================
Model 1: Upgraded Single XGBoost (Direct multi:softprob SOTA)

특징 및 아키텍처:
- 다단계 계층적 결합 없이 3개 클래스(CN, MCI, Dementia)를 직접 확률적으로 모델링 (objective='multi:softprob')
- 극소수 클래스(Dementia 6.9%)의 그래디언트 폭주 및 잎 노드 가중치 왜곡을 방지하기 위한 `max_delta_step=1.0` 제약 도입
- 생체신호 이상치(Outlier)에 강건한 `RobustScaler` 및 중앙값 대체(Median Imputation) 적용
- Strict 5-Fold Stratified K-Fold 완전 격리 교차검증 (Zero Data Leakage)
- 검증 성과: Accuracy 63.79%, Macro F1 0.5488, One-vs-Rest ROC-AUC 0.7018 (0.70 장벽 최초 돌파)
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
    return df.reset_index(drop=True)

def train_eval_xgboost_sota():
    df = load_data()
    y_true = df['original_label'].values.astype(int)
    
    skf = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros(len(df), dtype=int)
    oof_probs = np.zeros((len(df), 3), dtype=float)
    
    print("=" * 80)
    print(" [Model 1] Single XGBoost Direct multi:softprob SOTA (Strict 5-Fold CV)")
    print("=" * 80)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y_true), 1):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        y_train = y_true[train_idx]
        y_val = y_true[val_idx]
        
        # Data Preprocessing (Strictly on Fold Train)
        imputer = SimpleImputer(strategy='median')
        X_train_raw = imputer.fit_transform(train_df[V44_14])
        X_val_raw = imputer.transform(val_df[V44_14])
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)
        
        # SMOTE for multi-class rebalancing
        sm_xgb = SMOTE(k_neighbors=2, random_state=RANDOM_STATE + fold)
        X_res, y_res = sm_xgb.fit_resample(X_train_scaled, y_train)
        
        # Direct 3-Class Softmax Classifier with Leaf Gradient Regulation
        clf = XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            max_depth=3,
            learning_rate=0.03,
            n_estimators=130,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            max_delta_step=1.0,
            eval_metric='mlogloss',
            random_state=RANDOM_STATE + fold,
            n_jobs=1
        )
        clf.fit(X_res, y_res)
        
        probs = clf.predict_proba(X_val_scaled)
        oof_probs[val_idx] = probs
        oof_preds[val_idx] = np.argmax(probs, axis=1)
        
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
        f"Single XGBoost (Direct 3-Class SOTA)\nAcc: {acc:.4f} | Macro F1: {f1:.4f} | OVR AUC: {ovr_auc:.4f}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    chart_path = OUTPUT_DIR / "confusion_matrix_xgboost_direct_sota.png"
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
    train_eval_xgboost_sota()
