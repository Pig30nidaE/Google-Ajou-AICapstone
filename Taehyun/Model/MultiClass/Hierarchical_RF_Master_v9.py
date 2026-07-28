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

from sklearn.ensemble import RandomForestClassifier
import optuna

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
N_OPTUNA_TRIALS = 30
SEED_ENSEMBLE_COUNT = 10
TOP_K_FEATURES = 40

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
    gmm.fit(X_scaled)
    gmm_probs = gmm.predict_proba(X_scaled)
    for i in range(3):
        new_features_df[f'gmm_prob_{i}'] = gmm_probs[:, i]
        
    agg = AgglomerativeClustering(n_clusters=3)
    new_features_df['hierarchical_cluster_label'] = agg.fit_predict(X_scaled)
    
    df_new = pd.concat([df, new_features_df], axis=1)
    new_cols = new_features_df.columns.tolist()
    return df_new, new_cols

def select_top_features_rf(df, all_feats, stage_target, stage_name):
    print(f"\n[피처 선택] {stage_name} - Random Forest 기반 피처 중요도 산출 및 선택 (Top {TOP_K_FEATURES})...")
    df_stage = df.dropna(subset=[stage_target]).copy()
    X = df_stage[all_feats]
    y = df_stage[stage_target].astype(int)
    
    smotetomek = SMOTETomek(random_state=RANDOM_STATE)
    X_res, y_res = smotetomek.fit_resample(X, y)
    
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=1)
    rf.fit(X_res, y_res)
    
    importance = rf.feature_importances_
    ranked_features = [feat for _, feat in sorted(zip(importance, all_feats), reverse=True)]
    top_k_features = ranked_features[:TOP_K_FEATURES]
    
    print(f"-> {stage_name} 최적 피처 선택 완료")
    return top_k_features

def run_optuna_tuning_rf(df, features, stage_target, stage_name):
    print(f"\n[Optuna 최적화] {stage_name} - Random Forest 베이지안 파라미터 튜닝 ({N_OPTUNA_TRIALS} Trials)...")
    df_stage = df.dropna(subset=[stage_target]).copy()
    X = df_stage[features]
    y = df_stage[stage_target].astype(int)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
            'random_state': RANDOM_STATE,
            'n_jobs': 1
        }
        
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        f1_scores = []
        
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
            
            smotetomek = SMOTETomek(random_state=RANDOM_STATE)
            X_tr_res, y_tr_res = smotetomek.fit_resample(X_tr, y_tr)
            
            model = RandomForestClassifier(**params)
            model.fit(X_tr_res, y_tr_res)
            
            preds = model.predict(X_va)
            f1_scores.append(f1_score(y_va, preds, average='macro'))
            
        return np.mean(f1_scores)

    study = optuna.create_study(direction='maximize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    
    print(f"-> {stage_name} Optuna 최적화 완료! Best Macro F1: {study.best_value:.4f}")
    return study.best_params

def hierarchical_rf_seed_evaluation(df, stg1_feats, stg2_feats, params_s1, params_s2):
    print(f"\n[최종 평가] V9 Hierarchical RF Mastery + Seed Ensemble 5-Fold 평가 시작...")
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    acc_list, prec_list, rec_list, f1_list, auc_list = [], [], [], [], []
    final_cm = np.zeros((3, 3), dtype=int)
    
    seeds = [42, 7, 21, 99, 123, 456, 777, 888, 1024, 2024][:SEED_ENSEMBLE_COUNT]
    
    for fold, (tr_idx, te_idx) in enumerate(skf.split(df, df[TARGET_COL])):
        df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
        y_te = df_te[TARGET_COL].astype(int)
        
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
        
        for s in seeds:
            # Stage 1 Seed Model
            ps1 = params_s1.copy()
            ps1['random_state'] = s
            model_s1 = RandomForestClassifier(**ps1)
            model_s1.fit(X_tr_stg1_res, y_tr_stg1_res)
            prob1_fold += model_s1.predict_proba(df_te[stg1_feats])[:, 1]
            
            # Stage 2 Seed Model
            ps2 = params_s2.copy()
            ps2['random_state'] = s
            model_s2 = RandomForestClassifier(**ps2)
            model_s2.fit(X_tr_stg2_res, y_tr_stg2_res)
            prob2_fold += model_s2.predict_proba(df_te[stg2_feats])[:, 1]
            
        prob1_fold /= len(seeds)
        prob2_fold /= len(seeds)
        
        p_CN = 1.0 - prob1_fold
        p_MCI = prob1_fold * (1.0 - prob2_fold)
        p_Dem = prob1_fold * prob2_fold
        
        prob_matrix = np.vstack([p_CN, p_MCI, p_Dem]).T
        fold_preds = np.argmax(prob_matrix, axis=1)
        
        acc_list.append(accuracy_score(y_te, fold_preds))
        prec_list.append(precision_score(y_te, fold_preds, average='macro', zero_division=0))
        rec_list.append(recall_score(y_te, fold_preds, average='macro', zero_division=0))
        f1_list.append(f1_score(y_te, fold_preds, average='macro', zero_division=0))
        auc_list.append(roc_auc_score(y_te, prob_matrix, multi_class='ovr'))
        final_cm += confusion_matrix(y_te, fold_preds)
        
    print("\n" + "="*80)
    print("FINAL RESULTS (V9 Hierarchical RF Mastery + Seed Ensemble)")
    print("="*80)
    print(f"Accuracy: {np.mean(acc_list):.4f} | Precision: {np.mean(prec_list):.4f} | Recall: {np.mean(rec_list):.4f} | Macro F1: {np.mean(f1_list):.4f} | OVR AUC: {np.mean(auc_list):.4f}")
    
    return final_cm

if __name__ == "__main__":
    start_time = datetime.now()
    
    df, all_feats = load_data()
    df, unsupervised_feats = add_unsupervised_features(df, all_feats)
    all_feats.extend(unsupervised_feats)
    
    opt_feats_s1 = select_top_features_rf(df, all_feats, "stage1_label", "Stage1")
    opt_feats_s2 = select_top_features_rf(df, all_feats, "stage2_label", "Stage2")
    
    params_s1 = run_optuna_tuning_rf(df, opt_feats_s1, "stage1_label", "Stage1")
    params_s2 = run_optuna_tuning_rf(df, opt_feats_s2, "stage2_label", "Stage2")
    
    final_cm = hierarchical_rf_seed_evaluation(df, opt_feats_s1, opt_feats_s2, params_s1, params_s2)
    
    end_time = datetime.now()
    print(f"\nElapsed: {end_time - start_time}")
    
    # Save Confusion Matrix
    print("\n[시각화] V9 혼동 행렬 (Confusion Matrix)")
    class_names = ['CN(0)', 'MCI(1)', 'Dem(2)']
    plt.figure(figsize=(6, 5))
    
    sns.heatmap(final_cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 14})
    plt.title("V9 Hierarchical RF Mastery (Seed Ensemble)", fontsize=14, pad=10)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix_v9_rf_mastery.png", dpi=150)
