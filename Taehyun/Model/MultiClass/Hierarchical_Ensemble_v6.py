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
    
    X.fillna(X.median(), inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    new_features_df = pd.DataFrame(index=df.index)
    
    # PCA
    n_components = min(5, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca_feats = pca.fit_transform(X_scaled)
    for i in range(n_components):
        new_features_df[f'pca_{i+1}'] = pca_feats[:, i]
        
    # K-Means
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    kmeans_dist = kmeans.fit_transform(X_scaled)
    kmeans_labels = kmeans.labels_
    for i in range(3):
        new_features_df[f'kmeans_dist_{i}'] = kmeans_dist[:, i]
    new_features_df['kmeans_label'] = kmeans_labels
    
    # GMM
    gmm = GaussianMixture(n_components=3, random_state=RANDOM_STATE)
    gmm.fit(X_scaled)
    gmm_probs = gmm.predict_proba(X_scaled)
    for i in range(3):
        new_features_df[f'gmm_prob_{i}'] = gmm_probs[:, i]
        
    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=3)
    agg_labels = agg.fit_predict(X_scaled)
    new_features_df['hierarchical_cluster_label'] = agg_labels
    
    df_new = pd.concat([df, new_features_df], axis=1)
    new_cols = new_features_df.columns.tolist()
    print(f"  -> 새로 생성된 비지도 학습 피처 {len(new_cols)}개: {new_cols}")
    
    return df_new, new_cols

def perform_forward_selection(df, all_feats, stage_target, stage_name):
    print(f"\n[전진 선택법] {stage_name} 최적의 피처 개수 탐색...")
    
    df_stage = df.dropna(subset=[stage_target]).copy()
    X = df_stage[all_feats]
    y = df_stage[stage_target].astype(int)
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X, y)
    
    base_model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight='balanced', verbose=-1)
    base_model.fit(X_res, y_res)
    
    importance = base_model.feature_importances_
    ranked_features = [feat for _, feat in sorted(zip(importance, all_feats), reverse=True)]
    top_k_features = ranked_features[:FORWARD_SELECTION_MAX_FEATURES]
    
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
            
    optimal_k = np.argmax(history_scores) + 1
    optimal_features = top_k_features[:optimal_k]
    print(f"[{stage_name} 피처 선택 완료] 최적 피처 개수: {optimal_k}개 (Best AUC: {max(history_scores):.4f})")
    
    return optimal_features

def run_grid_search_for_stage(df, features, stage_target, model_name, stage_name):
    print(f"\n[Grid Search] {stage_name} - {model_name} 파라미터 탐색 중...")
    
    df_stage = df.dropna(subset=[stage_target]).copy()
    X = df_stage[features]
    y = df_stage[stage_target].astype(int)
    
    if model_name == "LightGBM":
        param_grid = {
            'num_leaves': [10, 15, 31, 50],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 500]
        }
    elif model_name == "CatBoost":
        param_grid = {
            'iterations': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'depth': [4, 6, 8]
        }
    elif model_name == "XGBoost":
        param_grid = {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    elif model_name == "RandomForest":
        param_grid = {
            'n_estimators': [200, 500, 800],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10]
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
            elif model_name == "XGBoost":
                model = XGBClassifier(random_state=RANDOM_STATE, n_jobs=1, eval_metric='auc', early_stopping_rounds=30, **params)
                model.fit(X_tr_res, y_tr_res, eval_set=[(X_va, y_va)], verbose=False)
                prob = model.predict_proba(X_va)[:, 1]
            elif model_name == "RandomForest":
                model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **params)
                model.fit(X_tr_res, y_tr_res)
                prob = model.predict_proba(X_va)[:, 1]
                
            fold_scores.append(roc_auc_score(y_va, prob))
            
        mean_auc = np.mean(fold_scores)
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params
            
    print(f"  -> {stage_name} {model_name} 최고 AUC: {best_auc:.4f} | 파라미터: {best_params}")
    return best_params

def hierarchical_evaluation(df, stg1_feats, stg2_feats, best_params_dict):
    print("\n[최종 평가] 4개 모델 앙상블 5-Fold 결합 평가 시작...")
    
    models = ["LightGBM", "CatBoost", "XGBoost", "RandomForest", "Ensemble"]
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
        
        fold_probs = {}
        
        for model_name in ["LightGBM", "CatBoost", "XGBoost", "RandomForest"]:
            # 1. Train Stage 1 & 2
            if model_name == "LightGBM":
                model_s1 = LGBMClassifier(objective="binary", random_state=RANDOM_STATE, n_jobs=1, verbose=-1, **best_params_dict[model_name]['Stage1'])
                model_s2 = LGBMClassifier(objective="binary", random_state=RANDOM_STATE, n_jobs=1, verbose=-1, **best_params_dict[model_name]['Stage2'])
            elif model_name == "CatBoost":
                model_s1 = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, **best_params_dict[model_name]['Stage1'])
                model_s2 = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, **best_params_dict[model_name]['Stage2'])
            elif model_name == "XGBoost":
                model_s1 = XGBClassifier(random_state=RANDOM_STATE, n_jobs=1, **best_params_dict[model_name]['Stage1'])
                model_s2 = XGBClassifier(random_state=RANDOM_STATE, n_jobs=1, **best_params_dict[model_name]['Stage2'])
            elif model_name == "RandomForest":
                model_s1 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **best_params_dict[model_name]['Stage1'])
                model_s2 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **best_params_dict[model_name]['Stage2'])
                
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
            fold_probs[model_name] = prob_matrix
            
            final_preds = np.argmax(prob_matrix, axis=1)
            
            # 4. Metrics
            final_metrics[model_name]['acc'].append(accuracy_score(y_te, final_preds))
            final_metrics[model_name]['prec'].append(precision_score(y_te, final_preds, average='macro', zero_division=0))
            final_metrics[model_name]['rec'].append(recall_score(y_te, final_preds, average='macro', zero_division=0))
            final_metrics[model_name]['f1'].append(f1_score(y_te, final_preds, average='macro', zero_division=0))
            final_metrics[model_name]['auc'].append(roc_auc_score(y_te, prob_matrix, multi_class='ovr'))
            final_metrics[model_name]['cm'].append(confusion_matrix(y_te, final_preds))
            
        # 5. Soft Voting Ensemble
        ensemble_prob = (fold_probs["LightGBM"] + fold_probs["CatBoost"] + fold_probs["XGBoost"] + fold_probs["RandomForest"]) / 4.0
        ensemble_preds = np.argmax(ensemble_prob, axis=1)
        
        final_metrics["Ensemble"]['acc'].append(accuracy_score(y_te, ensemble_preds))
        final_metrics["Ensemble"]['prec'].append(precision_score(y_te, ensemble_preds, average='macro', zero_division=0))
        final_metrics["Ensemble"]['rec'].append(recall_score(y_te, ensemble_preds, average='macro', zero_division=0))
        final_metrics["Ensemble"]['f1'].append(f1_score(y_te, ensemble_preds, average='macro', zero_division=0))
        final_metrics["Ensemble"]['auc'].append(roc_auc_score(y_te, ensemble_prob, multi_class='ovr'))
        final_metrics["Ensemble"]['cm'].append(confusion_matrix(y_te, ensemble_preds))
            
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
    
    # Feature Engineering
    df, unsupervised_feats = add_unsupervised_features(df, all_feats)
    all_feats.extend(unsupervised_feats)
    
    # Feature Selection
    opt_feats_s1 = perform_forward_selection(df, all_feats, "stage1_label", "Stage1")
    opt_feats_s2 = perform_forward_selection(df, all_feats, "stage2_label", "Stage2")
    
    # Grid Search
    base_models = ["LightGBM", "CatBoost", "XGBoost", "RandomForest"]
    best_params_dict = {m: {} for m in base_models}
    
    for m in base_models:
        best_params_dict[m]['Stage1'] = run_grid_search_for_stage(df, opt_feats_s1, "stage1_label", m, "Stage1")
        best_params_dict[m]['Stage2'] = run_grid_search_for_stage(df, opt_feats_s2, "stage2_label", m, "Stage2")
    
    # Final Evaluation
    results = hierarchical_evaluation(df, opt_feats_s1, opt_feats_s2, best_params_dict)
    
    print("\n" + "="*80)
    print("FINAL RESULTS (Hierarchical V6 | LightGBM, CatBoost, XGBoost, RF & Ensemble)")
    print("="*80)
    for model_name, r in results.items():
        print(f"\n[{model_name}]")
        print(f"Accuracy: {r['accuracy']:.4f} | Precision: {r['precision']:.4f} | Recall: {r['recall']:.4f} | Macro F1: {r['f1']:.4f} | OVR AUC: {r['auc']:.4f}")
        
    end_time = datetime.now()
    print(f"\nElapsed: {end_time - start_time}")
    
    # Save Confusion Matrix
    print("\n[시각화] 앙상블 혼동 행렬 (Confusion Matrix)")
    class_names = ['CN(0)', 'MCI(1)', 'Dem(2)']
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    
    for ax, (model_name, r) in zip(axes, results.items()):
        sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 13}, ax=ax)
        ax.set_title(f"{model_name}", fontsize=12, pad=10)
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix_v6_ensemble.png", dpi=150)
    # plt.show()
