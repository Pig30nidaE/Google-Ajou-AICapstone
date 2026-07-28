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

import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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
    
    df['stage1_label'] = np.where(df[TARGET_COL] == 0, 0, 1)
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
    gmm_probs = gmm.fit_predict(X_scaled)
    gmm_probs = gmm.predict_proba(X_scaled)
    for i in range(3):
        new_features_df[f'gmm_prob_{i}'] = gmm_probs[:, i]
        
    agg = AgglomerativeClustering(n_clusters=3)
    new_features_df['hierarchical_cluster_label'] = agg.fit_predict(X_scaled)
    
    df_new = pd.concat([df, new_features_df], axis=1)
    new_cols = new_features_df.columns.tolist()
    return df_new, new_cols

def perform_forward_selection(df, all_feats, stage_target, stage_name):
    print(f"\n[전진 선택법] {stage_name} 최적의 피처 개수 탐색...")
    
    df_stage = df.dropna(subset=[stage_target]).copy()
    X = df_stage[all_feats]
    y = df_stage[stage_target].astype(int)
    
    smotetomek = SMOTETomek(random_state=RANDOM_STATE)
    X_res, y_res = smotetomek.fit_resample(X, y)
    
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
            
            X_tr_res, y_tr_res = smotetomek.fit_resample(X_tr, y_tr)
            
            eval_model = LGBMClassifier(
                random_state=RANDOM_STATE, n_jobs=1, class_weight='balanced', n_estimators=FORWARD_SELECTION_ESTIMATORS, verbose=-1
            )
            eval_model.fit(X_tr_res, y_tr_res, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(30, verbose=False)])
            fold_scores.append(roc_auc_score(y_va, eval_model.predict_proba(X_va)[:, 1]))
            
        history_scores.append(np.mean(fold_scores))
            
    optimal_k = np.argmax(history_scores) + 1
    optimal_features = top_k_features[:optimal_k]
    print(f"[{stage_name} 피처 선택 완료] 최적 피처 개수: {optimal_k}개 (Best AUC: {max(history_scores):.4f})")
    
    return optimal_features

def hierarchical_evaluation(df, stg1_feats, stg2_feats, best_params_dict):
    print("\n[최종 평가] SMOTETomek + Stacking 앙상블 5-Fold 결합 평가 시작...")
    
    models = ["LightGBM", "CatBoost", "XGBoost", "RandomForest", "SoftVoting", "Stacking"]
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
        
        # Test Data
        y_te = df_te[TARGET_COL].astype(int)
        
        # SMOTETomek
        smotetomek = SMOTETomek(random_state=RANDOM_STATE)
        X_tr_stg1_res, y_tr_stg1_res = smotetomek.fit_resample(X_tr_stg1, y_tr_stg1)
        X_tr_stg2_res, y_tr_stg2_res = smotetomek.fit_resample(X_tr_stg2, y_tr_stg2)
        
        fold_probs = {}
        meta_features_s1_tr = []
        meta_features_s2_tr = []
        meta_features_s1_te = []
        meta_features_s2_te = []
        
        base_models_names = ["LightGBM", "CatBoost", "XGBoost", "RandomForest"]
        
        for model_name in base_models_names:
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
            
            # 메타 피처 수집 (Train Data에 대한 예측 확률)
            prob1_tr = model_s1.predict_proba(X_tr_stg1_res)[:, 1]
            prob2_tr = model_s2.predict_proba(X_tr_stg2_res)[:, 1]
            meta_features_s1_tr.append(prob1_tr)
            meta_features_s2_tr.append(prob2_tr)
            
            # Test Data 예측
            prob1 = model_s1.predict_proba(df_te[stg1_feats])[:, 1]
            prob2 = model_s2.predict_proba(df_te[stg2_feats])[:, 1]
            meta_features_s1_te.append(prob1)
            meta_features_s2_te.append(prob2)
            
            # 3. Combine Probabilities for Base Models
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
            
        # 5. Soft Voting Ensemble
        ensemble_prob = sum(fold_probs[m] for m in base_models_names) / 4.0
        ensemble_preds = np.argmax(ensemble_prob, axis=1)
        
        final_metrics["SoftVoting"]['acc'].append(accuracy_score(y_te, ensemble_preds))
        final_metrics["SoftVoting"]['prec'].append(precision_score(y_te, ensemble_preds, average='macro', zero_division=0))
        final_metrics["SoftVoting"]['rec'].append(recall_score(y_te, ensemble_preds, average='macro', zero_division=0))
        final_metrics["SoftVoting"]['f1'].append(f1_score(y_te, ensemble_preds, average='macro', zero_division=0))
        final_metrics["SoftVoting"]['auc'].append(roc_auc_score(y_te, ensemble_prob, multi_class='ovr'))
        final_metrics["SoftVoting"]['cm'].append(confusion_matrix(y_te, ensemble_preds))
        
        # 6. Stacking Ensemble
        meta_X_s1_tr = np.column_stack(meta_features_s1_tr)
        meta_X_s2_tr = np.column_stack(meta_features_s2_tr)
        meta_X_s1_te = np.column_stack(meta_features_s1_te)
        meta_X_s2_te = np.column_stack(meta_features_s2_te)
        
        meta_model_s1 = LogisticRegression(C=0.1, random_state=RANDOM_STATE)
        meta_model_s2 = LogisticRegression(C=0.1, random_state=RANDOM_STATE)
        
        meta_model_s1.fit(meta_X_s1_tr, y_tr_stg1_res)
        meta_model_s2.fit(meta_X_s2_tr, y_tr_stg2_res)
        
        stack_prob1 = meta_model_s1.predict_proba(meta_X_s1_te)[:, 1]
        stack_prob2 = meta_model_s2.predict_proba(meta_X_s2_te)[:, 1]
        
        p_CN_stack = 1.0 - stack_prob1
        p_MCI_stack = stack_prob1 * (1.0 - stack_prob2)
        p_Dem_stack = stack_prob1 * stack_prob2
        
        stack_prob_matrix = np.vstack([p_CN_stack, p_MCI_stack, p_Dem_stack]).T
        stack_preds = np.argmax(stack_prob_matrix, axis=1)
        
        final_metrics["Stacking"]['acc'].append(accuracy_score(y_te, stack_preds))
        final_metrics["Stacking"]['prec'].append(precision_score(y_te, stack_preds, average='macro', zero_division=0))
        final_metrics["Stacking"]['rec'].append(recall_score(y_te, stack_preds, average='macro', zero_division=0))
        final_metrics["Stacking"]['f1'].append(f1_score(y_te, stack_preds, average='macro', zero_division=0))
        final_metrics["Stacking"]['auc'].append(roc_auc_score(y_te, stack_prob_matrix, multi_class='ovr'))
        final_metrics["Stacking"]['cm'].append(confusion_matrix(y_te, stack_preds))
            
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
    df, unsupervised_feats = add_unsupervised_features(df, all_feats)
    all_feats.extend(unsupervised_feats)
    
    opt_feats_s1 = perform_forward_selection(df, all_feats, "stage1_label", "Stage1")
    opt_feats_s2 = perform_forward_selection(df, all_feats, "stage2_label", "Stage2")
    
    # 이전 단계(V6)에서 확인된 가장 안정적이고 뛰어난 하이퍼파라미터 하드코딩 (시간 절약 및 과적합 방지)
    best_params_dict = {
        "LightGBM": {
            "Stage1": {'learning_rate': 0.05, 'n_estimators': 200, 'num_leaves': 15},
            "Stage2": {'learning_rate': 0.1, 'n_estimators': 200, 'num_leaves': 15}
        },
        "CatBoost": {
            "Stage1": {'depth': 4, 'iterations': 200, 'learning_rate': 0.05},
            "Stage2": {'depth': 4, 'iterations': 200, 'learning_rate': 0.05}
        },
        "XGBoost": {
            "Stage1": {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 200},
            "Stage2": {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 200}
        },
        "RandomForest": {
            "Stage1": {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 500},
            "Stage2": {'max_depth': 5, 'min_samples_split': 5, 'n_estimators': 500}
        }
    }
    
    # Final Evaluation (SMOTETomek + Stacking)
    results = hierarchical_evaluation(df, opt_feats_s1, opt_feats_s2, best_params_dict)
    
    print("\n" + "="*80)
    print("FINAL RESULTS (Hierarchical V7 | SMOTETomek + Stacking & SoftVoting)")
    print("="*80)
    for model_name, r in results.items():
        print(f"\n[{model_name}]")
        print(f"Accuracy: {r['accuracy']:.4f} | Precision: {r['precision']:.4f} | Recall: {r['recall']:.4f} | Macro F1: {r['f1']:.4f} | OVR AUC: {r['auc']:.4f}")
        
    end_time = datetime.now()
    print(f"\nElapsed: {end_time - start_time}")
    
    # Save Confusion Matrix
    print("\n[시각화] V7 앙상블 혼동 행렬 (Confusion Matrix)")
    class_names = ['CN(0)', 'MCI(1)', 'Dem(2)']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    sns.heatmap(results['SoftVoting']['cm'], annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 13}, ax=axes[0])
    axes[0].set_title("V7 Soft Voting", fontsize=12, pad=10)
    axes[0].set_xlabel('Predicted Label', fontsize=10)
    axes[0].set_ylabel('True Label', fontsize=10)
    
    sns.heatmap(results['Stacking']['cm'], annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 13}, ax=axes[1])
    axes[1].set_title("V7 Stacking (Logistic Meta)", fontsize=12, pad=10)
    axes[1].set_xlabel('Predicted Label', fontsize=10)
    axes[1].set_ylabel('True Label', fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix_v7_stacking.png", dpi=150)
