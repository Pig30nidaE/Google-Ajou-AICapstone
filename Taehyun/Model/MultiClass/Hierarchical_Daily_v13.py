# =========================================================
# 1. 라이브러리 임포트
# =========================================================
import os
import sys
import pathlib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# 2. 글로벌 설정 및 변수
# =========================================================
BASE_DIR = pathlib.Path(r"c:\ML4")
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "tabular"
TRAIN_PATH = PROCESSED_DIR / "train_tabular_base.csv"
VAL_PATH = PROCESSED_DIR / "val_tabular_base.csv"
PLOT_DIR = BASE_DIR / "report" / "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

TARGET_COL = "original_label"
DROP_COLS = ["EMAIL", "date", "DIAG_NM", "label", TARGET_COL, "fold"]
RANDOM_STATE = 42
N_SPLITS = 5

def load_data():
    print("\n[데이터 로드] 일일(Daily) 단위 원형 데이터를 불러옵니다...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)
    df = pd.concat([df_train, df_val], ignore_index=True)
    
    print(f"  -> Total Daily Rows: {len(df)}")
    print(f"  -> Total Unique Patients: {df['EMAIL'].nunique()}")
    
    # 계층적 라벨 생성
    df['stage1_label'] = np.where(df[TARGET_COL] == 0, 0, 1)
    df['stage2_label'] = np.where(df[TARGET_COL] == 1, 0, np.where(df[TARGET_COL] == 2, 1, np.nan))
    
    return df

def run_v13():
    df = load_data()
    all_feats = [c for c in df.columns if c not in DROP_COLS and c not in ['stage1_label', 'stage2_label']]
    
    # 결측치는 전체 중앙값으로 사전 처리 (일부 환자의 첫날 데이터가 없을 수 있으므로)
    X = df[all_feats].fillna(df[all_feats].median())
    
    # 🌟 환자별(Group) 분할: 같은 환자의 날짜 데이터가 Train/Val에 섞이지 않도록 보장
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    y_true_patient = []
    y_pred_patient = []
    
    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X, df[TARGET_COL], groups=df['EMAIL'])):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        df_tr = df.iloc[tr_idx]
        df_va = df.iloc[va_idx]
        
        # Train Stage 1
        X_tr_s1 = df_tr[all_feats].fillna(df_tr[all_feats].median())
        y_tr_s1 = df_tr['stage1_label']
        
        # Train Stage 2 (Abnormal 환자만)
        df_tr_s2 = df_tr[df_tr['stage1_label'] == 1]
        X_tr_s2 = df_tr_s2[all_feats].fillna(df_tr_s2[all_feats].median())
        y_tr_s2 = df_tr_s2['stage2_label']
        
        # Val Data
        X_va = df_va[all_feats].fillna(df_tr[all_feats].median())
        
        # 🌟 모델 선언: 12,000개의 데이터이므로 Overfitting 방어를 위해 깊이(max_depth) 적당히 부여, 
        # 그리고 class_weight='balanced'로 불균형 해소
        rf_s1 = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=5, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
        rf_s2 = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_split=5, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
        
        print(f"  > Train Stage 1 (n={len(X_tr_s1)})...")
        rf_s1.fit(X_tr_s1, y_tr_s1)
        
        print(f"  > Train Stage 2 (n={len(X_tr_s2)})...")
        rf_s2.fit(X_tr_s2, y_tr_s2)
        
        # 일일 단위 예측 확률 계산
        prob_s1 = rf_s1.predict_proba(X_va)[:, 1]
        prob_s2 = rf_s2.predict_proba(X_va)[:, 1]
        
        p_CN = 1.0 - prob_s1
        p_MCI = prob_s1 * (1.0 - prob_s2)
        p_Dem = prob_s1 * prob_s2
        
        df_va_copy = df_va.copy()
        df_va_copy['p_CN'] = p_CN
        df_va_copy['p_MCI'] = p_MCI
        df_va_copy['p_Dem'] = p_Dem
        
        # 🌟 환자 단위 묶기 (Aggregation)
        patient_grouped = df_va_copy.groupby('EMAIL')
        for email, group in patient_grouped:
            # 해당 환자의 모든 날짜(일반적으로 60일)의 확률을 평균 냄
            mean_probs = group[['p_CN', 'p_MCI', 'p_Dem']].mean().values
            
            # 최종 환자 단위 예측
            pred_class = np.argmax(mean_probs)
            true_class = group[TARGET_COL].iloc[0] # 환자의 Ground Truth
            
            y_true_patient.append(true_class)
            y_pred_patient.append(pred_class)
            
    # 전체 OOF (Out-of-fold) 환자 단위 평가
    acc = accuracy_score(y_true_patient, y_pred_patient)
    prec = precision_score(y_true_patient, y_pred_patient, average='macro', zero_division=0)
    rec = recall_score(y_true_patient, y_pred_patient, average='macro', zero_division=0)
    f1 = f1_score(y_true_patient, y_pred_patient, average='macro', zero_division=0)
    
    print("\n" + "="*80)
    print("FINAL PATIENT-LEVEL RESULTS (V13: Daily-to-Patient Aggregation)")
    print("="*80)
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | Macro F1: {f1:.4f}")
    
    final_cm = confusion_matrix(y_true_patient, y_pred_patient)
    
    # Save Confusion Matrix
    class_names = ['CN(0)', 'MCI(1)', 'Dem(2)']
    plt.figure(figsize=(6, 5))
    sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 14})
    plt.title("V13 Patient-Level Aggregation", fontsize=14, pad=10)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix_v13.png", dpi=150)

if __name__ == "__main__":
    start_time = datetime.now()
    run_v13()
    end_time = datetime.now()
    print(f"\nElapsed: {end_time - start_time}")
