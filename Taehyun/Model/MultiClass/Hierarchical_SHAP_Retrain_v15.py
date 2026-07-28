# =========================================================
# 1. 라이브러리 임포트
# =========================================================
import os
import pathlib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# 2. 글로벌 설정 및 변수
# =========================================================
BASE_DIR = pathlib.Path(r"c:\ML4")
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "tabular"
PATIENT_PATH = PROCESSED_DIR / "patient_level_all_v2.csv"
PLOT_DIR = BASE_DIR / "report" / "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

TARGET_COL = "original_label"
RANDOM_STATE = 42
N_SPLITS = 5

# SHAP 분석을 통해 입증된 최정예 20개 핵심 변수 (Hardcoded)
STAGE1_TOP20 = [
    'sleep_light_ratio_5min', 'sleep_breath_average', 'sleep_score_alignment_std', 
    'activity_met_min_low_std', 'sleep_hr_5min_max_std', 'sleep_period_id_std', 
    'activity_high_std', 'sleep_light_std', 'activity_met_min_high_std', 
    'sleep_score', 'sleep_period_id', 'activity_class_3_count_std', 
    'sleep_rmssd_5min_max', 'activity_score_recovery_time_std', 'activity_inactivity_alerts', 
    'activity_class_4_ratio_std', 'sleep_score_deep_std', 'sleep_rmssd_5min_var_std', 
    'sleep_light_count_5min_std', 'activity_met_1min_max_std'
]

STAGE2_TOP20 = [
    'activity_met_min_inactive', 'activity_active_ratio_std', 'sleep_rem_count_5min_std', 
    'activity_low_std', 'sleep_score_latency_std', 'activity_total_std', 
    'sleep_hr_lowest_std', 'sleep_deep_std', 'sleep_awake_ratio_5min_std', 
    'sleep_awake_count_5min_std', 'sleep_temperature_delta', 'activity_met_min_low_std', 
    'activity_met_1min_q25', 'activity_met_1min_max_std', 'sleep_score_disturbances', 
    'activity_class_3_ratio_std', 'sleep_score_alignment', 'sleep_rem', 
    'sleep_onset_latency_std', 'sleep_light_count_5min_std'
]

def load_data():
    df = pd.read_csv(PATIENT_PATH)
    df['stage1_label'] = np.where(df[TARGET_COL] == 0, 0, 1)
    df['stage2_label'] = np.where(df[TARGET_COL] == 1, 0, np.where(df[TARGET_COL] == 2, 1, np.nan))
    
    # Fill NA for selected features
    for f in set(STAGE1_TOP20 + STAGE2_TOP20):
        df[f].fillna(df[f].median(), inplace=True)
        
    return df

def run_v15():
    df = load_data()
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    acc_list, prec_list, rec_list, f1_list, auc_list = [], [], [], [], []
    final_cm = np.zeros((3, 3), dtype=int)
    
    # V7 RF Params
    rf_params_s1 = {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 300}
    rf_params_s2 = {'max_depth': 5, 'min_samples_split': 5, 'n_estimators': 300}
    
    print("\n[최종 평가] V15 SHAP 핵심 20개 변수 전용 (Hard Cut-off) 5-Fold 평가 시작...")
    
    for fold, (tr_idx, te_idx) in enumerate(skf.split(df, df[TARGET_COL])):
        print(f"--- Fold {fold+1}/{N_SPLITS} ---")
        df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
        y_te = df_te[TARGET_COL].astype(int)
        
        # 🌟 나머지 변수 모두 삭제하고 Top 20만 주입 🌟
        X_tr_stg1 = df_tr[STAGE1_TOP20]
        y_tr_stg1 = df_tr['stage1_label'].astype(int)
        
        df_tr_stg2 = df_tr[df_tr['stage1_label'] == 1]
        X_tr_stg2 = df_tr_stg2[STAGE2_TOP20]
        y_tr_stg2 = df_tr_stg2['stage2_label'].astype(int)
        
        smotetomek = SMOTETomek(random_state=RANDOM_STATE)
        X_tr_stg1_res, y_tr_stg1_res = smotetomek.fit_resample(X_tr_stg1, y_tr_stg1)
        X_tr_stg2_res, y_tr_stg2_res = smotetomek.fit_resample(X_tr_stg2, y_tr_stg2)
        
        model_s1 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **rf_params_s1)
        model_s2 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **rf_params_s2)
        
        model_s1.fit(X_tr_stg1_res, y_tr_stg1_res)
        model_s2.fit(X_tr_stg2_res, y_tr_stg2_res)
        
        prob1 = model_s1.predict_proba(df_te[STAGE1_TOP20])[:, 1]
        prob2 = model_s2.predict_proba(df_te[STAGE2_TOP20])[:, 1]
        
        p_CN = 1.0 - prob1
        p_MCI = prob1 * (1.0 - prob2)
        p_Dem = prob1 * prob2
        
        prob_matrix = np.vstack([p_CN, p_MCI, p_Dem]).T
        fold_preds = np.argmax(prob_matrix, axis=1)
        
        acc_list.append(accuracy_score(y_te, fold_preds))
        prec_list.append(precision_score(y_te, fold_preds, average='macro', zero_division=0))
        rec_list.append(recall_score(y_te, fold_preds, average='macro', zero_division=0))
        f1_list.append(f1_score(y_te, fold_preds, average='macro', zero_division=0))
        auc_list.append(roc_auc_score(y_te, prob_matrix, multi_class='ovr'))
        final_cm += confusion_matrix(y_te, fold_preds)
        
    print("\n" + "="*80)
    print("FINAL RESULTS (V15: SHAP Top 20 Hard Cut-off)")
    print("="*80)
    print(f"Accuracy: {np.mean(acc_list):.4f} | Precision: {np.mean(prec_list):.4f} | Recall: {np.mean(rec_list):.4f} | Macro F1: {np.mean(f1_list):.4f} | OVR AUC: {np.mean(auc_list):.4f}")
    
    class_names = ['CN(0)', 'MCI(1)', 'Dem(2)']
    plt.figure(figsize=(6, 5))
    sns.heatmap(final_cm, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 14})
    plt.title("V15 SHAP Top 20 Hard Cut-off", fontsize=14, pad=10)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix_v15.png", dpi=150)

if __name__ == "__main__":
    start_time = datetime.now()
    run_v15()
    end_time = datetime.now()
    print(f"\nElapsed: {end_time - start_time}")
