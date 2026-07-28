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
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

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

def load_data():
    print(f"\n[데이터 로드] 환자 단위(Patient-level) 데이터를 불러옵니다...")
    df = pd.read_csv(PATIENT_PATH)
    print(f"  Total Data: {df.shape} | subjects: {df['EMAIL'].nunique()}")
    
    df['stage1_label'] = np.where(df[TARGET_COL] == 0, 0, 1)
    df['stage2_label'] = np.where(df[TARGET_COL] == 1, 0, np.where(df[TARGET_COL] == 2, 1, np.nan))
    
    all_feats = [c for c in df.columns if c not in DROP_COLS and c not in ['stage1_label', 'stage2_label']]
    return df, all_feats

def preprocess_and_add_features(df, features):
    print("\n[전처리 및 피처 엔지니어링] KNN Imputer, Robust Scaler, 비지도 학습 피처 생성...")
    X = df[features].copy()
    
    # KNN Imputation
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features, index=X.index)
    
    # Robust Scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    new_features_df = pd.DataFrame(index=df.index)
    
    pca = PCA(n_components=min(5, X_scaled.shape[1]), random_state=RANDOM_STATE)
    pca_feats = pca.fit_transform(X_scaled)
    for i in range(pca_feats.shape[1]):
        new_features_df[f'pca_{i+1}'] = pca_feats[:, i]
        
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    kmeans_dist = kmeans.fit_transform(X_scaled)
    for i in range(3):
        new_features_df[f'kmeans_dist_{i}'] = kmeans_dist[:, i]
    new_features_df['kmeans_label'] = kmeans.labels_
    
    gmm = GaussianMixture(n_components=3, random_state=RANDOM_STATE)
    gmm.fit(X_scaled)
    gmm_probs = gmm.predict_proba(X_scaled)
    for i in range(3):
        new_features_df[f'gmm_prob_{i}'] = gmm_probs[:, i]
        
    agg = AgglomerativeClustering(n_clusters=3)
    new_features_df['hierarchical_cluster_label'] = agg.fit_predict(X_scaled)
    
    df_new = pd.concat([X_imputed, new_features_df, df[['stage1_label', 'stage2_label', TARGET_COL]]], axis=1)
    new_cols = new_features_df.columns.tolist()
    return df_new, features + new_cols

def select_features_l1(df, all_feats, stage_target, stage_name):
    print(f"\n[피처 선택] {stage_name} - L1(Lasso) 정규화 기반 무자비한 피처 삭제 진행 중...")
    df_stage = df.dropna(subset=[stage_target]).copy()
    X = df_stage[all_feats]
    y = df_stage[stage_target].astype(int)
    
    # Scale first since L1 is highly sensitive to magnitude
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    smotetomek = SMOTETomek(random_state=RANDOM_STATE)
    X_res, y_res = smotetomek.fit_resample(X_scaled, y)
    
    # C=0.5 applies strong regularization to force more coefficients to exactly 0
    selector = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=0.2, random_state=RANDOM_STATE))
    selector.fit(X_res, y_res)
    
    selected_mask = selector.get_support()
    selected_features = [all_feats[i] for i in range(len(all_feats)) if selected_mask[i]]
    
    # If C=0.2 is too strict and removes all, fallback to a softer penalty
    if len(selected_features) < 3:
        selector = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=RANDOM_STATE))
        selector.fit(X_res, y_res)
        selected_mask = selector.get_support()
        selected_features = [all_feats[i] for i in range(len(all_feats)) if selected_mask[i]]
        
    print(f"-> {stage_name} 최적 피처 선택 완료 (남은 피처: {len(selected_features)}개)")
    return selected_features

def hierarchical_minimal_evaluation(df, stg1_feats, stg2_feats):
    print(f"\n[최종 평가] V10 미니멀리즘 (SVM + Logistic + Restricted RF) 5-Fold 평가 시작...")
    
    models = ["SVC_RBF", "SVC_Linear", "LogisticReg", "Restricted_RF", "SoftVoting"]
    final_metrics = {m: {'acc':[], 'prec':[], 'rec':[], 'f1':[], 'auc':[], 'cm':[]} for m in models}
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    for tr_idx, te_idx in skf.split(df, df[TARGET_COL]):
        df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
        
        # Stage 1 Train Data
        X_tr_stg1 = df_tr[stg1_feats]
        y_tr_stg1 = df_tr['stage1_label'].astype(int)
        
        # Stage 2 Train Data
        df_tr_stg2 = df_tr[df_tr['stage1_label'] == 1]
        X_tr_stg2 = df_tr_stg2[stg2_feats]
        y_tr_stg2 = df_tr_stg2['stage2_label'].astype(int)
        
        y_te = df_te[TARGET_COL].astype(int)
        
        # Scaling inside fold
        scaler1 = RobustScaler()
        X_tr_stg1_sc = scaler1.fit_transform(X_tr_stg1)
        X_te_stg1_sc = scaler1.transform(df_te[stg1_feats])
        
        scaler2 = RobustScaler()
        X_tr_stg2_sc = scaler2.fit_transform(X_tr_stg2)
        X_te_stg2_sc = scaler2.transform(df_te[stg2_feats])
        
        smotetomek = SMOTETomek(random_state=RANDOM_STATE)
        X_tr_stg1_res, y_tr_stg1_res = smotetomek.fit_resample(X_tr_stg1_sc, y_tr_stg1)
        X_tr_stg2_res, y_tr_stg2_res = smotetomek.fit_resample(X_tr_stg2_sc, y_tr_stg2)
        
        base_models_dict = {
            "SVC_RBF": SVC(probability=True, kernel='rbf', C=1.0, random_state=RANDOM_STATE),
            "SVC_Linear": SVC(probability=True, kernel='linear', C=1.0, random_state=RANDOM_STATE),
            "LogisticReg": LogisticRegression(penalty='l2', C=1.0, random_state=RANDOM_STATE),
            "Restricted_RF": RandomForestClassifier(n_estimators=100, max_depth=4, max_features='log2', random_state=RANDOM_STATE, n_jobs=1)
        }
        
        fold_probs = {}
        
        for model_name, base_model in base_models_dict.items():
            model_s1 = type(base_model)(**base_model.get_params())
            model_s2 = type(base_model)(**base_model.get_params())
            
            model_s1.fit(X_tr_stg1_res, y_tr_stg1_res)
            model_s2.fit(X_tr_stg2_res, y_tr_stg2_res)
            
            prob1 = model_s1.predict_proba(X_te_stg1_sc)[:, 1]
            prob2 = model_s2.predict_proba(X_te_stg2_sc)[:, 1]
            
            p_CN = 1.0 - prob1
            p_MCI = prob1 * (1.0 - prob2)
            p_Dem = prob1 * prob2
            
            prob_matrix = np.vstack([p_CN, p_MCI, p_Dem]).T
            fold_probs[model_name] = prob_matrix
            final_preds = np.argmax(prob_matrix, axis=1)
            
            final_metrics[model_name]['acc'].append(accuracy_score(y_te, final_preds))
            final_metrics[model_name]['prec'].append(precision_score(y_te, final_preds, average='macro', zero_division=0))
            final_metrics[model_name]['rec'].append(recall_score(y_te, final_preds, average='macro', zero_division=0))
            final_metrics[model_name]['f1'].append(f1_score(y_te, final_preds, average='macro', zero_division=0))
            final_metrics[model_name]['auc'].append(roc_auc_score(y_te, prob_matrix, multi_class='ovr'))
            final_metrics[model_name]['cm'].append(confusion_matrix(y_te, final_preds))
            
        # Soft Voting
        ensemble_prob = sum(fold_probs[m] for m in base_models_dict.keys()) / len(base_models_dict)
        ensemble_preds = np.argmax(ensemble_prob, axis=1)
        
        final_metrics["SoftVoting"]['acc'].append(accuracy_score(y_te, ensemble_preds))
        final_metrics["SoftVoting"]['prec'].append(precision_score(y_te, ensemble_preds, average='macro', zero_division=0))
        final_metrics["SoftVoting"]['rec'].append(recall_score(y_te, ensemble_preds, average='macro', zero_division=0))
        final_metrics["SoftVoting"]['f1'].append(f1_score(y_te, ensemble_preds, average='macro', zero_division=0))
        final_metrics["SoftVoting"]['auc'].append(roc_auc_score(y_te, ensemble_prob, multi_class='ovr'))
        final_metrics["SoftVoting"]['cm'].append(confusion_matrix(y_te, ensemble_preds))
            
    results = {}
    for m in models:
        results[m] = {
            'accuracy': np.mean(final_metrics[m]['acc']),
            'precision': np.mean(final_metrics[m]['prec']),
            'recall': np.mean(final_metrics[m]['rec']),
            'f1': np.mean(final_metrics[m]['f1']),
            'auc': np.mean(final_metrics[m]['auc']),
            'cm': np.sum(final_metrics[m]['cm'], axis=0)
        }
    return results

if __name__ == "__main__":
    start_time = datetime.now()
    
    df, all_feats = load_data()
    df, all_feats = preprocess_and_add_features(df, all_feats)
    
    opt_feats_s1 = select_features_l1(df, all_feats, "stage1_label", "Stage1")
    opt_feats_s2 = select_features_l1(df, all_feats, "stage2_label", "Stage2")
    
    results = hierarchical_minimal_evaluation(df, opt_feats_s1, opt_feats_s2)
    
    print("\n" + "="*80)
    print("FINAL RESULTS (V10 Hierarchical Minimalist | L1 + SVM + Logistic + Rest. RF)")
    print("="*80)
    for model_name, r in results.items():
        print(f"\n[{model_name}]")
        print(f"Accuracy: {r['accuracy']:.4f} | Precision: {r['precision']:.4f} | Recall: {r['recall']:.4f} | Macro F1: {r['f1']:.4f} | OVR AUC: {r['auc']:.4f}")
        
    end_time = datetime.now()
    print(f"\nElapsed: {end_time - start_time}")
    
    # Save Confusion Matrix
    print("\n[시각화] V10 혼동 행렬 (Confusion Matrix)")
    class_names = ['CN(0)', 'MCI(1)', 'Dem(2)']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    sns.heatmap(results['SVC_RBF']['cm'], annot=True, fmt='d', cmap='Purples', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 13}, ax=axes[0])
    axes[0].set_title("V10 SVC (RBF)", fontsize=12, pad=10)
    axes[0].set_xlabel('Predicted Label', fontsize=10)
    axes[0].set_ylabel('True Label', fontsize=10)
    
    sns.heatmap(results['SoftVoting']['cm'], annot=True, fmt='d', cmap='Purples', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 13}, ax=axes[1])
    axes[1].set_title("V10 Soft Voting", fontsize=12, pad=10)
    axes[1].set_xlabel('Predicted Label', fontsize=10)
    axes[1].set_ylabel('True Label', fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix_v10_minimal.png", dpi=150)
