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
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
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

def run_v16():
    df = load_data()
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    models_list = ["LightGBM", "CatBoost", "XGBoost", "RandomForest", "Ensemble"]
    final_metrics = {m: {'acc':[], 'prec':[], 'rec':[], 'f1':[], 'auc':[], 'cm':[]} for m in models_list}
    
    print("\n[최종 평가] V16 (V6 SMOTE 앙상블 + SHAP 20 변수 결합) 5-Fold 평가 시작...")
    
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
        
        # V6 아키텍처 (SMOTE)
        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr_stg1_res, y_tr_stg1_res = smote.fit_resample(X_tr_stg1, y_tr_stg1)
        X_tr_stg2_res, y_tr_stg2_res = smote.fit_resample(X_tr_stg2, y_tr_stg2)
        
        fold_probs = {}
        
        for model_name in ["LightGBM", "CatBoost", "XGBoost", "RandomForest"]:
            if model_name == "LightGBM":
                model_s1 = LGBMClassifier(objective="binary", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, class_weight='balanced')
                model_s2 = LGBMClassifier(objective="binary", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, class_weight='balanced')
            elif model_name == "CatBoost":
                model_s1 = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, auto_class_weights='Balanced')
                model_s2 = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, auto_class_weights='Balanced')
            elif model_name == "XGBoost":
                model_s1 = XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss')
                model_s2 = XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss')
            elif model_name == "RandomForest":
                model_s1 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')
                model_s2 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')
                
            model_s1.fit(X_tr_stg1_res, y_tr_stg1_res)
            model_s2.fit(X_tr_stg2_res, y_tr_stg2_res)
            
            prob1 = model_s1.predict_proba(df_te[STAGE1_TOP20])[:, 1]
            prob2 = model_s2.predict_proba(df_te[STAGE2_TOP20])[:, 1]
            
            p_CN = 1.0 - prob1
            p_MCI = prob1 * (1.0 - prob2)
            p_Dem = prob1 * prob2
            
            prob_matrix = np.vstack([p_CN, p_MCI, p_Dem]).T
            fold_probs[model_name] = prob_matrix
            
            fold_preds = np.argmax(prob_matrix, axis=1)
            
            final_metrics[model_name]['acc'].append(accuracy_score(y_te, fold_preds))
            final_metrics[model_name]['prec'].append(precision_score(y_te, fold_preds, average='macro', zero_division=0))
            final_metrics[model_name]['rec'].append(recall_score(y_te, fold_preds, average='macro', zero_division=0))
            final_metrics[model_name]['f1'].append(f1_score(y_te, fold_preds, average='macro', zero_division=0))
            final_metrics[model_name]['auc'].append(roc_auc_score(y_te, prob_matrix, multi_class='ovr'))
            
        # 앙상블 (소프트 보팅)
        ensemble_prob = (fold_probs["LightGBM"] + fold_probs["CatBoost"] + fold_probs["XGBoost"] + fold_probs["RandomForest"]) / 4.0
        ensemble_preds = np.argmax(ensemble_prob, axis=1)
        
        final_metrics["Ensemble"]['acc'].append(accuracy_score(y_te, ensemble_preds))
        final_metrics["Ensemble"]['prec'].append(precision_score(y_te, ensemble_preds, average='macro', zero_division=0))
        final_metrics["Ensemble"]['rec'].append(recall_score(y_te, ensemble_preds, average='macro', zero_division=0))
        final_metrics["Ensemble"]['f1'].append(f1_score(y_te, ensemble_preds, average='macro', zero_division=0))
        final_metrics["Ensemble"]['auc'].append(roc_auc_score(y_te, ensemble_prob, multi_class='ovr'))
        
    print("\n" + "="*80)
    print("FINAL RESULTS (V16: V6 Ensemble + SHAP Top 20 Hard Cut-off)")
    print("="*80)
    for model_name in models_list:
        mean_acc = np.mean(final_metrics[model_name]['acc'])
        mean_prec = np.mean(final_metrics[model_name]['prec'])
        mean_rec = np.mean(final_metrics[model_name]['rec'])
        mean_f1 = np.mean(final_metrics[model_name]['f1'])
        mean_auc = np.mean(final_metrics[model_name]['auc'])
        print(f"[{model_name}] Acc: {mean_acc:.4f} | Prec: {mean_prec:.4f} | Rec: {mean_rec:.4f} | F1: {mean_f1:.4f} | AUC: {mean_auc:.4f}")
        
if __name__ == "__main__":
    start_time = datetime.now()
    run_v16()
    end_time = datetime.now()
    print(f"\nElapsed: {end_time - start_time}")
