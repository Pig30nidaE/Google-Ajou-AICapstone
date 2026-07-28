# =========================================================
# 1. 라이브러리 임포트
# =========================================================
import os
import sys
import pathlib
import textwrap
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
from lightgbm import LGBMClassifier
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
FORWARD_SELECTION_MAX_FEATURES = 40
FORWARD_SELECTION_ESTIMATORS = 100

def load_data():
    print(f"\n[데이터 로드] 환자 단위(Patient-level) 데이터를 불러옵니다...")
    df = pd.read_csv(PATIENT_PATH)
    print(f"  Total Data: {df.shape} | subjects: {df['EMAIL'].nunique()}")
    
    # original_label: CN=0, MCI=1, Dem=2
    # stage1_label: CN=0, Abnormal(MCI+Dem)=1
    df['stage1_label'] = np.where(df[TARGET_COL] == 0, 0, 1)
    
    # stage2_label: MCI=0, Dem=1 (CN은 결측치 처리, 나중에 필터링)
    df['stage2_label'] = np.where(df[TARGET_COL] == 1, 0, np.where(df[TARGET_COL] == 2, 1, np.nan))
    
    all_feats = [c for c in df.columns if c not in DROP_COLS and c not in ['stage1_label', 'stage2_label']]
    return df, all_feats

def add_unsupervised_features(df, features):
    print("\n[피처 엔지니어링] 비지도 학습 기반 피처(PCA, KMeans, GMM, Agglomerative)를 생성합니다...")
    X = df[features].copy()
    
    # 1. 결측치 처리 및 스케일링
    # 비지도 학습 알고리즘은 결측치를 처리하지 못하므로, 간단히 중앙값으로 대체합니다.
    X.fillna(X.median(), inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    new_features_df = pd.DataFrame(index=df.index)
    
    # 2. PCA (Principal Component Analysis)
    n_components = min(5, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca_feats = pca.fit_transform(X_scaled)
    for i in range(n_components):
        new_features_df[f'pca_{i+1}'] = pca_feats[:, i]
        
    # 3. K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    kmeans_dist = kmeans.fit_transform(X_scaled)
    kmeans_labels = kmeans.labels_
    for i in range(3):
        new_features_df[f'kmeans_dist_{i}'] = kmeans_dist[:, i]
    new_features_df['kmeans_label'] = kmeans_labels
    
    # 4. GMM (Gaussian Mixture Model)
    gmm = GaussianMixture(n_components=3, random_state=RANDOM_STATE)
    gmm.fit(X_scaled)
    gmm_probs = gmm.predict_proba(X_scaled)
    for i in range(3):
        new_features_df[f'gmm_prob_{i}'] = gmm_probs[:, i]
        
    # 5. Agglomerative Clustering (Hierarchical)
    agg = AgglomerativeClustering(n_clusters=3)
    agg_labels = agg.fit_predict(X_scaled)
    new_features_df['hierarchical_cluster_label'] = agg_labels
    
    # 병합
    df_new = pd.concat([df, new_features_df], axis=1)
    
    # 새로 추가된 컬럼 목록
    new_cols = new_features_df.columns.tolist()
    print(f"  -> 새로 생성된 비지도 학습 피처 {len(new_cols)}개: {new_cols}")
    
    return df_new, new_cols

def perform_forward_selection(df, all_feats, stage_target, stage_name):
    print(f"\n[전진 선택법] {stage_name} 최적의 피처 개수 탐색...")
    
    # 1. 데이터 필터링 (Stage 2의 경우 CN 제외)
    df_stage = df.dropna(subset=[stage_target]).copy()
    X = df_stage[all_feats]
    y = df_stage[stage_target].astype(int)
    
    # 2. 전체 데이터 대상 SMOTE 및 Feature Importance 계산
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X, y)
    
    base_model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight='balanced', verbose=-1)
    base_model.fit(X_res, y_res)
    
    importance = base_model.feature_importances_
    ranked_features = [feat for _, feat in sorted(zip(importance, all_feats), reverse=True)]
    top_k_features = ranked_features[:FORWARD_SELECTION_MAX_FEATURES]
    
    # 3. 5-Fold 기반 전진 선택법
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    history_scores = []
    
    for k in range(1, len(top_k_features) + 1):
        current_features = top_k_features[:k]
        fold_scores = []
        
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx][current_features], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx][current_features], y.iloc[va_idx]
            
            X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
            
            eval_model = LGBMClassifier(
                random_state=RANDOM_STATE, n_jobs=1, class_weight='balanced', n_estimators=FORWARD_SELECTION_ESTIMATORS, verbose=-1
            )
            eval_model.fit(X_tr_res, y_tr_res, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(30, verbose=False)])
            
            prob = eval_model.predict_proba(X_va)[:, 1]
            fold_scores.append(roc_auc_score(y_va, prob))
            
        history_scores.append(np.mean(fold_scores))
        if k % 10 == 0:
            print(f"  -> {stage_name} Top {k} 피처 AUC: {history_scores[-1]:.4f}")
            
    optimal_k = np.argmax(history_scores) + 1
    optimal_features = top_k_features[:optimal_k]
    print(f"[{stage_name} 피처 선택 완료] 최적 피처 개수: {optimal_k}개")
    
    # 추가된 비지도 학습 피처가 최적 피처에 포함되었는지 확인
    unsupervised_selected = [f for f in optimal_features if f.startswith(('pca_', 'kmeans_', 'gmm_', 'hierarchical_'))]
    if unsupervised_selected:
        print(f"  -> 포함된 비지도 학습 피처: {unsupervised_selected}")
    else:
        print(f"  -> 포함된 비지도 학습 피처 없음")
    
    return optimal_features

def run_grid_search_for_stage(df, features, stage_target, model_name, stage_name):
    print(f"\n[Grid Search] {stage_name} - {model_name} 파라미터 탐색 중...")
    
    df_stage = df.dropna(subset=[stage_target]).copy()
    X = df_stage[features]
    y = df_stage[stage_target].astype(int)
    
    if model_name == "LightGBM":
        param_grid = {
            'num_leaves': [15, 31],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [200, 500],
            'max_depth': [5, 10]
        }
    elif model_name == "CatBoost":
        param_grid = {
            'iterations': [200, 500],
            'learning_rate': [0.05, 0.1],
            'depth': [4, 6]
        }
        
    best_auc = 0
    best_params = None
    
    for params in ParameterGrid(param_grid):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        fold_scores = []
        
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
            
            smote = SMOTE(random_state=RANDOM_STATE)
            X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
            
            if model_name == "LightGBM":
                model = LGBMClassifier(objective="binary", random_state=RANDOM_STATE, n_jobs=1, verbose=-1, **params)
                model.fit(X_tr_res, y_tr_res, eval_set=[(X_va, y_va)], eval_metric='auc', callbacks=[lgb.early_stopping(30, verbose=False)])
                prob = model.predict_proba(X_va)[:, 1]
            elif model_name == "CatBoost":
                model = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, **params)
                model.fit(X_tr_res, y_tr_res, eval_set=[(X_va, y_va)], early_stopping_rounds=30)
                prob = model.predict_proba(X_va)[:, 1]
                
            fold_scores.append(roc_auc_score(y_va, prob))
            
        mean_auc = np.mean(fold_scores)
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params
            
    print(f"  -> {stage_name} {model_name} 최고 AUC: {best_auc:.4f} | 파라미터: {best_params}")
    return best_params

def hierarchical_evaluation(df, stg1_feats, stg2_feats, best_params_lgb, best_params_cat):
    print("\n[최종 평가] 계층적 모델 5-Fold 결합 평가 시작...")
    
    models = ["LightGBM", "CatBoost"]
    final_metrics = {m: {'acc':[], 'prec':[], 'rec':[], 'f1':[], 'auc':[], 'cm':[]} for m in models}
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    for tr_idx, te_idx in skf.split(df, df[TARGET_COL]):
        df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
        
        # Stage 1 Train Data (All)
        X_tr_stg1 = df_tr[stg1_feats]
        y_tr_stg1 = df_tr['stage1_label'].astype(int)
        
        # Stage 2 Train Data (Only Abnormal)
        df_tr_stg2 = df_tr[df_tr['stage1_label'] == 1]
        X_tr_stg2 = df_tr_stg2[stg2_feats]
        y_tr_stg2 = df_tr_stg2['stage2_label'].astype(int)
        
        # Test Data
        y_te = df_te[TARGET_COL].astype(int)
        
        # SMOTE
        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr_stg1_res, y_tr_stg1_res = smote.fit_resample(X_tr_stg1, y_tr_stg1)
        X_tr_stg2_res, y_tr_stg2_res = smote.fit_resample(X_tr_stg2, y_tr_stg2)
        
        for model_name in models:
            # 1. Train Stage 1
            if model_name == "LightGBM":
                model_s1 = LGBMClassifier(objective="binary", random_state=RANDOM_STATE, n_jobs=1, verbose=-1, **best_params_lgb['Stage1'])
                model_s2 = LGBMClassifier(objective="binary", random_state=RANDOM_STATE, n_jobs=1, verbose=-1, **best_params_lgb['Stage2'])
            else:
                model_s1 = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, **best_params_cat['Stage1'])
                model_s2 = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, **best_params_cat['Stage2'])
                
            model_s1.fit(X_tr_stg1_res, y_tr_stg1_res)
            model_s2.fit(X_tr_stg2_res, y_tr_stg2_res)
            
            # 2. Predict Probabilities
            prob1 = model_s1.predict_proba(df_te[stg1_feats])[:, 1] # Prob(Abnormal)
            prob2 = model_s2.predict_proba(df_te[stg2_feats])[:, 1] # Prob(Dem | Abnormal)
            
            # 3. Combine Probabilities
            p_CN = 1.0 - prob1
            p_MCI = prob1 * (1.0 - prob2)
            p_Dem = prob1 * prob2
            
            prob_matrix = np.vstack([p_CN, p_MCI, p_Dem]).T
            final_preds = np.argmax(prob_matrix, axis=1)
            
            # 4. Metrics
            final_metrics[model_name]['acc'].append(accuracy_score(y_te, final_preds))
            final_metrics[model_name]['prec'].append(precision_score(y_te, final_preds, average='macro', zero_division=0))
            final_metrics[model_name]['rec'].append(recall_score(y_te, final_preds, average='macro', zero_division=0))
            final_metrics[model_name]['f1'].append(f1_score(y_te, final_preds, average='macro', zero_division=0))
            final_metrics[model_name]['auc'].append(roc_auc_score(y_te, prob_matrix, multi_class='ovr'))
            final_metrics[model_name]['cm'].append(confusion_matrix(y_te, final_preds))
            
    # Aggregate results
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
    
    # 0. 비지도 학습 기반 피처 추가
    df, unsupervised_feats = add_unsupervised_features(df, all_feats)
    all_feats.extend(unsupervised_feats) # 전체 피처 목록에 추가
    
    # Feature Selection
    opt_feats_s1 = perform_forward_selection(df, all_feats, "stage1_label", "Stage1")
    opt_feats_s2 = perform_forward_selection(df, all_feats, "stage2_label", "Stage2")
    
    # Grid Search
    best_lgb_s1 = run_grid_search_for_stage(df, opt_feats_s1, "stage1_label", "LightGBM", "Stage1")
    best_lgb_s2 = run_grid_search_for_stage(df, opt_feats_s2, "stage2_label", "LightGBM", "Stage2")
    
    best_cat_s1 = run_grid_search_for_stage(df, opt_feats_s1, "stage1_label", "CatBoost", "Stage1")
    best_cat_s2 = run_grid_search_for_stage(df, opt_feats_s2, "stage2_label", "CatBoost", "Stage2")
    
    best_params_lgb = {'Stage1': best_lgb_s1, 'Stage2': best_lgb_s2}
    best_params_cat = {'Stage1': best_cat_s1, 'Stage2': best_cat_s2}
    
    # Final Evaluation
    results = hierarchical_evaluation(df, opt_feats_s1, opt_feats_s2, best_params_lgb, best_params_cat)
    
    print("\n" + "="*60)
    print("FINAL RESULTS (Hierarchical 3-Class + Unsupervised | LightGBM vs CatBoost)")
    print("="*60)
    for model_name, r in results.items():
        print(f"\n[{model_name}]")
        print(f"Accuracy: {r['accuracy']:.4f} | Precision: {r['precision']:.4f} | Recall: {r['recall']:.4f} | Macro F1: {r['f1']:.4f} | OVR AUC: {r['auc']:.4f}")
        
    end_time = datetime.now()
    print(f"\nElapsed: {end_time - start_time}")
    
    # Save Confusion Matrix
    print("\n[시각화] 혼동 행렬 (Confusion Matrix)")
    class_names = ['CN(0)', 'MCI(1)', 'Dem(2)']
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    
    for ax, (model_name, r) in zip(axes, results.items()):
        sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 15}, ax=ax)
        ax.set_title(f"[{model_name}]\nHierarchical + Unsupervised V5", fontsize=14, pad=15)
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix_v5_unsupervised.png", dpi=150)
    # plt.show() 
