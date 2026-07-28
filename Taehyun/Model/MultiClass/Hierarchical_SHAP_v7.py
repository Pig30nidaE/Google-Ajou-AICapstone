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

from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

import shap

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
TOP_K = 20

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

def run_shap_analysis():
    df, all_feats = load_data()
    df, all_feats = add_unsupervised_features(df, all_feats)
    
    # 1. Feature Selection using LGBM (V7 logic)
    print("\n[피처 선택] LGBM 기반 최적 피처 추출 중...")
    smote = SMOTETomek(random_state=RANDOM_STATE)
    
    # Stage 1
    X_s1 = df[all_feats]
    y_s1 = df['stage1_label']
    X_s1_res, y_s1_res = smote.fit_resample(X_s1, y_s1)
    
    lgbm1 = LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
    lgbm1.fit(X_s1_res, y_s1_res)
    imp1 = lgbm1.feature_importances_
    stg1_feats = [feat for _, feat in sorted(zip(imp1, all_feats), reverse=True)][:TOP_K]
    
    # Stage 2
    df_s2 = df[df['stage1_label'] == 1].copy()
    X_s2 = df_s2[all_feats]
    y_s2 = df_s2['stage2_label']
    X_s2_res, y_s2_res = smote.fit_resample(X_s2, y_s2)
    
    lgbm2 = LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
    lgbm2.fit(X_s2_res, y_s2_res)
    imp2 = lgbm2.feature_importances_
    stg2_feats = [feat for _, feat in sorted(zip(imp2, all_feats), reverse=True)][:TOP_K]
    
    # 2. Train V7 RandomForest Models
    print("\n[학습] V7 Random Forest 글로벌 모델 학습 중...")
    rf_params_s1 = {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 300}
    rf_params_s2 = {'max_depth': 5, 'min_samples_split': 5, 'n_estimators': 300}
    
    X_s1_top = df[stg1_feats]
    X_s1_top_res, y_s1_top_res = smote.fit_resample(X_s1_top, y_s1)
    
    model_s1 = RandomForestClassifier(random_state=RANDOM_STATE, **rf_params_s1)
    model_s1.fit(X_s1_top_res, y_s1_top_res)
    
    X_s2_top = df_s2[stg2_feats]
    X_s2_top_res, y_s2_top_res = smote.fit_resample(X_s2_top, y_s2)
    
    model_s2 = RandomForestClassifier(random_state=RANDOM_STATE, **rf_params_s2)
    model_s2.fit(X_s2_top_res, y_s2_top_res)
    
    # 3. SHAP Analysis
    print("\n[SHAP] Stage 1 (정상 vs 인지장애) 설명 시각화 생성 중...")
    explainer_s1 = shap.TreeExplainer(model_s1)
    # Extract SHAP values on ORIGINAL data
    shap_values_s1 = explainer_s1.shap_values(X_s1_top)
    if isinstance(shap_values_s1, list):
        shap_values_s1 = shap_values_s1[1]
    elif len(shap_values_s1.shape) == 3:
        shap_values_s1 = shap_values_s1[:, :, 1]
        
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_s1, X_s1_top, show=False)
    plt.title("V7 Stage 1 (Normal vs Abnormal) SHAP Summary", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "shap_summary_stage1_v7.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n[SHAP] Stage 2 (MCI vs Dem) 설명 시각화 생성 중...")
    explainer_s2 = shap.TreeExplainer(model_s2)
    shap_values_s2 = explainer_s2.shap_values(X_s2_top)
    if isinstance(shap_values_s2, list):
        shap_values_s2 = shap_values_s2[1]
    elif len(shap_values_s2.shape) == 3:
        shap_values_s2 = shap_values_s2[:, :, 1]
        
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_s2, X_s2_top, show=False)
    plt.title("V7 Stage 2 (MCI vs Dem) SHAP Summary", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "shap_summary_stage2_v7.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✅ SHAP analysis complete!")

if __name__ == "__main__":
    start_time = datetime.now()
    run_shap_analysis()
    end_time = datetime.now()
    print(f"\nElapsed: {end_time - start_time}")
