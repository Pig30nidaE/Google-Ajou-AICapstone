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
from scipy.optimize import minimize

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.combine import SMOTETomek

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

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
SEED_ENSEMBLE_COUNT = 30 # 시간 관계상 30개 시드로 합의

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

def optimize_thresholds(y_true, prob_matrix):
    # F1 Score를 극대화하기 위해 각 클래스 확률에 곱해줄 가중치(w0, w1, w2)를 찾음
    def loss_func(weights):
        w_prob = prob_matrix * weights
        preds = np.argmax(w_prob, axis=1)
        # return negative Macro F1 to minimize
        return -f1_score(y_true, preds, average='macro')
    
    initial_weights = [1.0, 1.0, 1.0]
    bounds = [(0.1, 5.0), (0.1, 5.0), (0.1, 5.0)]
    result = minimize(loss_func, initial_weights, bounds=bounds, method='L-BFGS-B')
    return result.x

def hierarchical_v11_evaluation(df, all_feats):
    print(f"\n[최종 평가] V11 대규모 시드 앙상블({SEED_ENSEMBLE_COUNT} seeds) + 확률 임계값 최적화 평가 시작...")
    
    # V7에서 가장 좋았던 파라미터 (CatBoost, RF 혼합)
    rf_params_s1 = {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 300}
    rf_params_s2 = {'max_depth': 5, 'min_samples_split': 5, 'n_estimators': 300}
    cb_params_s1 = {'depth': 4, 'iterations': 200, 'learning_rate': 0.05, 'verbose': 0}
    cb_params_s2 = {'depth': 4, 'iterations': 200, 'learning_rate': 0.05, 'verbose': 0}
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    acc_list, prec_list, rec_list, f1_list, auc_list = [], [], [], [], []
    final_cm = np.zeros((3, 3), dtype=int)
    
    seeds = np.random.RandomState(RANDOM_STATE).randint(0, 10000, size=SEED_ENSEMBLE_COUNT)
    
    for fold, (tr_idx, te_idx) in enumerate(skf.split(df, df[TARGET_COL])):
        df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
        y_te = df_te[TARGET_COL].astype(int)
        
        # 피처 중요도 기반 Top 50 추출 (폴드 내에서)
        smote_fs = SMOTETomek(random_state=RANDOM_STATE)
        X_tr_fs, y_tr_fs = smote_fs.fit_resample(df_tr[all_feats], df_tr['stage1_label'])
        fs_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        fs_model.fit(X_tr_fs, y_tr_fs)
        imp = fs_model.feature_importances_
        stg1_feats = [feat for _, feat in sorted(zip(imp, all_feats), reverse=True)][:50]
        
        X_tr_fs2, y_tr_fs2 = smote_fs.fit_resample(df_tr[df_tr['stage1_label']==1][all_feats], df_tr[df_tr['stage1_label']==1]['stage2_label'])
        fs_model.fit(X_tr_fs2, y_tr_fs2)
        imp2 = fs_model.feature_importances_
        stg2_feats = [feat for _, feat in sorted(zip(imp2, all_feats), reverse=True)][:50]
        
        # Stage 1 Train Data
        X_tr_stg1 = df_tr[stg1_feats]
        y_tr_stg1 = df_tr['stage1_label'].astype(int)
        
        # Stage 2 Train Data
        df_tr_stg2 = df_tr[df_tr['stage1_label'] == 1]
        X_tr_stg2 = df_tr_stg2[stg2_feats]
        y_tr_stg2 = df_tr_stg2['stage2_label'].astype(int)
        
        smotetomek = SMOTETomek(random_state=RANDOM_STATE)
        X_tr_stg1_res, y_tr_stg1_res = smotetomek.fit_resample(X_tr_stg1, y_tr_stg1)
        X_tr_stg2_res, y_tr_stg2_res = smotetomek.fit_resample(X_tr_stg2, y_tr_stg2)
        
        prob1_fold = np.zeros(len(y_te))
        prob2_fold = np.zeros(len(y_te))
        
        # Train ensemble models
        for s in seeds:
            # Stage 1 RF & CB
            m1_rf = RandomForestClassifier(random_state=s, n_jobs=1, **rf_params_s1)
            m1_cb = CatBoostClassifier(random_state=s, **cb_params_s1)
            m1_rf.fit(X_tr_stg1_res, y_tr_stg1_res)
            m1_cb.fit(X_tr_stg1_res, y_tr_stg1_res)
            
            prob1_fold += (m1_rf.predict_proba(df_te[stg1_feats])[:, 1] + m1_cb.predict_proba(df_te[stg1_feats])[:, 1]) / 2.0
            
            # Stage 2 RF & CB
            m2_rf = RandomForestClassifier(random_state=s, n_jobs=1, **rf_params_s2)
            m2_cb = CatBoostClassifier(random_state=s, **cb_params_s2)
            m2_rf.fit(X_tr_stg2_res, y_tr_stg2_res)
            m2_cb.fit(X_tr_stg2_res, y_tr_stg2_res)
            
            prob2_fold += (m2_rf.predict_proba(df_te[stg2_feats])[:, 1] + m2_cb.predict_proba(df_te[stg2_feats])[:, 1]) / 2.0
            
        prob1_fold /= len(seeds)
        prob2_fold /= len(seeds)
        
        p_CN = 1.0 - prob1_fold
        p_MCI = prob1_fold * (1.0 - prob2_fold)
        p_Dem = prob1_fold * prob2_fold
        
        prob_matrix = np.vstack([p_CN, p_MCI, p_Dem]).T
        
        # 🌟 핵심 기법: 확률 임계값 최적화 (Threshold Tuning)
        # 테스트 세트의 성능을 시뮬레이션하기 위해, 각 클래스의 사전 확률이나 F1을 극대화하는 가중치 적용
        # OOF(Out-of-fold)를 구하기엔 시간이 걸리므로, 단순한 휴리스틱으로 클래스 불균형에 대한 패널티를 곱해줍니다.
        # 원래 데이터의 비율에 반비례하도록 확률을 증폭 (Inverse Class Frequency)
        counts = df_tr[TARGET_COL].value_counts().sort_index()
        weights = len(df_tr) / (3.0 * counts.values) 
        
        # 가중치 곱한 뒤 다시 정규화
        w_prob_matrix = prob_matrix * weights
        fold_preds = np.argmax(w_prob_matrix, axis=1)
        
        acc_list.append(accuracy_score(y_te, fold_preds))
        prec_list.append(precision_score(y_te, fold_preds, average='macro', zero_division=0))
        rec_list.append(recall_score(y_te, fold_preds, average='macro', zero_division=0))
        f1_list.append(f1_score(y_te, fold_preds, average='macro', zero_division=0))
        auc_list.append(roc_auc_score(y_te, prob_matrix, multi_class='ovr')) # AUC는 원본 확률 사용
        final_cm += confusion_matrix(y_te, fold_preds)
        
    print("\n" + "="*80)
    print("FINAL RESULTS (V11: The Law of Large Numbers + Threshold Tuning)")
    print("="*80)
    print(f"Accuracy: {np.mean(acc_list):.4f} | Precision: {np.mean(prec_list):.4f} | Recall: {np.mean(rec_list):.4f} | Macro F1: {np.mean(f1_list):.4f} | OVR AUC: {np.mean(auc_list):.4f}")
    
    return final_cm

if __name__ == "__main__":
    start_time = datetime.now()
    
    df, all_feats = load_data()
    df, all_feats = add_unsupervised_features(df, all_feats)
    
    final_cm = hierarchical_v11_evaluation(df, all_feats)
    
    end_time = datetime.now()
    print(f"\nElapsed: {end_time - start_time}")
    
    # Save Confusion Matrix
    print("\n[시각화] V11 혼동 행렬 (Confusion Matrix)")
    class_names = ['CN(0)', 'MCI(1)', 'Dem(2)']
    plt.figure(figsize=(6, 5))
    
    sns.heatmap(final_cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 14})
    plt.title("V11 LLN & Threshold Tuning", fontsize=14, pad=10)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix_v11.png", dpi=150)
