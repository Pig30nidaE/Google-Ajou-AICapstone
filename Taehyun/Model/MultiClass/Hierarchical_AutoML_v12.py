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

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.combine import SMOTETomek
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from flaml import AutoML

# =========================================================
# 2. 글로벌 설정 및 변수
# =========================================================
BASE_DIR = pathlib.Path(r"c:\ML4")
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "tabular"
PATIENT_PATH = PROCESSED_DIR / "patient_level_all_v2.csv"
PLOT_DIR = BASE_DIR / "report" / "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

TARGET_COL = "original_label"
DROP_COLS = ["EMAIL", "date", "DIAG_NM", "label", TARGET_COL, "fold"]

RANDOM_STATE = 42
N_SPLITS = 5
TIME_BUDGET = 20 # 각 Fold, 각 Stage 당 AutoML에 주어지는 탐색 시간(초)

def load_data():
    df = pd.read_csv(PATIENT_PATH)
    df['stage1_label'] = np.where(df[TARGET_COL] == 0, 0, 1)
    df['stage2_label'] = np.where(df[TARGET_COL] == 1, 0, np.where(df[TARGET_COL] == 2, 1, np.nan))
    all_feats = [c for c in df.columns if c not in DROP_COLS and c not in ['stage1_label', 'stage2_label']]
    return df, all_feats

def add_unsupervised_features(df, features):
    X = df[features].copy()
    X.fillna(X.median(), inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    new_features_df = pd.DataFrame(index=df.index)
    
    pca = PCA(n_components=min(5, X_scaled.shape[1]), random_state=RANDOM_STATE)
    pca_feats = pca.fit_transform(X_scaled)
    for i in range(pca_feats.shape[1]):
        new_features_df[f'pca_{i+1}'] = pca_feats[:, i]
        
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    kmeans_dist = kmeans.fit_transform(X_scaled)
    for i in range(3):
        new_features_df[f'kmeans_dist_{i}'] = kmeans_dist[:, i]
    
    gmm = GaussianMixture(n_components=3, random_state=RANDOM_STATE)
    gmm.fit(X_scaled)
    gmm_probs = gmm.predict_proba(X_scaled)
    for i in range(3):
        new_features_df[f'gmm_prob_{i}'] = gmm_probs[:, i]
        
    agg = AgglomerativeClustering(n_clusters=3)
    new_features_df['hierarchical_cluster_label'] = agg.fit_predict(X_scaled)
    
    df_new = pd.concat([df, new_features_df], axis=1)
    new_cols = new_features_df.columns.tolist()
    return df_new, features + new_cols

def hierarchical_automl_evaluation(df, all_feats):
    print(f"\n[최종 평가] V12 MS FLAML AutoML 기반 최적 알고리즘 탐색 (Fold당 {TIME_BUDGET}초 제한)...")
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    acc_list, prec_list, rec_list, f1_list, auc_list = [], [], [], [], []
    final_cm = np.zeros((3, 3), dtype=int)
    
    for fold, (tr_idx, te_idx) in enumerate(skf.split(df, df[TARGET_COL])):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
        y_te = df_te[TARGET_COL].astype(int)
        
        # Stage 1 Train Data
        X_tr_stg1 = df_tr[all_feats]
        y_tr_stg1 = df_tr['stage1_label'].astype(int)
        
        # Stage 2 Train Data
        df_tr_stg2 = df_tr[df_tr['stage1_label'] == 1]
        X_tr_stg2 = df_tr_stg2[all_feats]
        y_tr_stg2 = df_tr_stg2['stage2_label'].astype(int)
        
        smotetomek = SMOTETomek(random_state=RANDOM_STATE)
        X_tr_stg1_res, y_tr_stg1_res = smotetomek.fit_resample(X_tr_stg1, y_tr_stg1)
        X_tr_stg2_res, y_tr_stg2_res = smotetomek.fit_resample(X_tr_stg2, y_tr_stg2)
        
        # Stage 1 AutoML
        automl1 = AutoML()
        automl_settings_1 = {
            "time_budget": TIME_BUDGET,
            "metric": 'f1',
            "task": 'classification',
            "log_file_name": "",
            "verbose": 0,
            "eval_method": "cv",
            "n_splits": 3,
            "seed": RANDOM_STATE
        }
        print(f"  > Stage 1 AutoML 탐색 중...")
        automl1.fit(X_tr_stg1_res, y_tr_stg1_res, **automl_settings_1)
        print(f"    - Best Model: {automl1.best_estimator} | CV Score: {1 - automl1.best_loss:.4f}")
        
        # Stage 2 AutoML
        automl2 = AutoML()
        automl_settings_2 = {
            "time_budget": TIME_BUDGET,
            "metric": 'f1',
            "task": 'classification',
            "log_file_name": "",
            "verbose": 0,
            "eval_method": "cv",
            "n_splits": 3,
            "seed": RANDOM_STATE
        }
        print(f"  > Stage 2 AutoML 탐색 중...")
        automl2.fit(X_tr_stg2_res, y_tr_stg2_res, **automl_settings_2)
        print(f"    - Best Model: {automl2.best_estimator} | CV Score: {1 - automl2.best_loss:.4f}")
        
        # Predict Probabilities
        prob1 = automl1.predict_proba(df_te[all_feats])[:, 1]
        prob2 = automl2.predict_proba(df_te[all_feats])[:, 1]
        
        p_CN = 1.0 - prob1
        p_MCI = prob1 * (1.0 - prob2)
        p_Dem = prob1 * prob2
        
        prob_matrix = np.vstack([p_CN, p_MCI, p_Dem]).T
        fold_preds = np.argmax(prob_matrix, axis=1)
        
        acc = accuracy_score(y_te, fold_preds)
        f1 = f1_score(y_te, fold_preds, average='macro', zero_division=0)
        print(f"  >> Fold {fold+1} Test Accuracy: {acc:.4f} | F1: {f1:.4f}")
        
        acc_list.append(acc)
        prec_list.append(precision_score(y_te, fold_preds, average='macro', zero_division=0))
        rec_list.append(recall_score(y_te, fold_preds, average='macro', zero_division=0))
        f1_list.append(f1)
        auc_list.append(roc_auc_score(y_te, prob_matrix, multi_class='ovr'))
        final_cm += confusion_matrix(y_te, fold_preds)
        
    print("\n" + "="*80)
    print("FINAL RESULTS (V12: FLAML AutoML)")
    print("="*80)
    print(f"Accuracy: {np.mean(acc_list):.4f} | Precision: {np.mean(prec_list):.4f} | Recall: {np.mean(rec_list):.4f} | Macro F1: {np.mean(f1_list):.4f} | OVR AUC: {np.mean(auc_list):.4f}")
    
    return final_cm

if __name__ == "__main__":
    start_time = datetime.now()
    
    df, all_feats = load_data()
    df, all_feats = add_unsupervised_features(df, all_feats)
    
    final_cm = hierarchical_automl_evaluation(df, all_feats)
    
    end_time = datetime.now()
    print(f"\nElapsed: {end_time - start_time}")
    
    # Save Confusion Matrix
    print("\n[시각화] V12 혼동 행렬 (Confusion Matrix)")
    class_names = ['CN(0)', 'MCI(1)', 'Dem(2)']
    plt.figure(figsize=(6, 5))
    
    sns.heatmap(final_cm, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 14})
    plt.title("V12 FLAML AutoML", fontsize=14, pad=10)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix_v12.png", dpi=150)
