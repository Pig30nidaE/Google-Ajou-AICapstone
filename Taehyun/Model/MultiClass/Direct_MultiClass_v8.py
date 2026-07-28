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

from catboost import CatBoostClassifier
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

def load_data():
    print(f"\n[데이터 로드] 환자 단위(Patient-level) 데이터를 불러옵니다...")
    df = pd.read_csv(PATIENT_PATH)
    print(f"  Total Data: {df.shape} | subjects: {df['EMAIL'].nunique()}")
    
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

def select_top_features(df, features):
    print(f"\n[피처 선택] CatBoost 기반 불순도 감소 방식 중요도 산출 및 피처 선택 (Top 40)...")
    X = df[features]
    y = df[TARGET_COL].astype(int)
    
    smotetomek = SMOTETomek(random_state=RANDOM_STATE)
    X_res, y_res = smotetomek.fit_resample(X, y)
    
    base_model = CatBoostClassifier(loss_function='MultiClass', random_state=RANDOM_STATE, verbose=0, iterations=150)
    base_model.fit(X_res, y_res)
    
    importance = base_model.feature_importances_
    ranked_features = [feat for _, feat in sorted(zip(importance, features), reverse=True)]
    top_k_features = ranked_features[:40] # 상위 40개 선택
    
    print(f"-> 최적 피처 선택 완료 (총 {len(top_k_features)}개)")
    return top_k_features

def run_optuna_tuning(df, features):
    print(f"\n[Optuna 최적화] 베이지안 최적화를 통한 하이퍼파라미터 튜닝 시작 ({N_OPTUNA_TRIALS} Trials)...")
    X = df[features]
    y = df[TARGET_COL].astype(int)
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 400),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'depth': trial.suggest_int('depth', 3, 7),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'loss_function': 'MultiClass',
            'random_state': RANDOM_STATE,
            'verbose': 0
        }
        
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        f1_scores = []
        
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
            
            smotetomek = SMOTETomek(random_state=RANDOM_STATE)
            X_tr_res, y_tr_res = smotetomek.fit_resample(X_tr, y_tr)
            
            model = CatBoostClassifier(**params)
            model.fit(X_tr_res, y_tr_res)
            
            preds = model.predict(X_va)
            f1_scores.append(f1_score(y_va, preds, average='macro'))
            
        return np.mean(f1_scores)

    study = optuna.create_study(direction='maximize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    
    print(f"-> Optuna 최적화 완료! Best F1: {study.best_value:.4f}")
    print(f"-> Best Params: {study.best_params}")
    return study.best_params

def seed_ensemble_evaluation(df, features, best_params):
    print(f"\n[최종 평가] 다이렉트 다중 분류 (Direct Multi-Class) 및 Seed Ensemble 5-Fold 결합 평가 시작...")
    
    X = df[features]
    y = df[TARGET_COL].astype(int)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    acc_list, prec_list, rec_list, f1_list, auc_list = [], [], [], [], []
    final_cm = np.zeros((3, 3), dtype=int)
    
    seeds = [42, 7, 21, 99, 123, 456, 777, 888, 1024, 2024][:SEED_ENSEMBLE_COUNT]
    
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
        
        smotetomek = SMOTETomek(random_state=RANDOM_STATE)
        X_tr_res, y_tr_res = smotetomek.fit_resample(X_tr, y_tr)
        
        fold_probs = np.zeros((len(y_te), 3))
        
        for s in seeds:
            params = best_params.copy()
            params['random_state'] = s
            params['loss_function'] = 'MultiClass'
            params['verbose'] = 0
            
            model = CatBoostClassifier(**params)
            model.fit(X_tr_res, y_tr_res)
            fold_probs += model.predict_proba(X_te)
            
        fold_probs /= len(seeds) # 확률 평균 (Soft Voting across seeds)
        
        fold_preds = np.argmax(fold_probs, axis=1)
        
        acc_list.append(accuracy_score(y_te, fold_preds))
        prec_list.append(precision_score(y_te, fold_preds, average='macro', zero_division=0))
        rec_list.append(recall_score(y_te, fold_preds, average='macro', zero_division=0))
        f1_list.append(f1_score(y_te, fold_preds, average='macro', zero_division=0))
        auc_list.append(roc_auc_score(y_te, fold_probs, multi_class='ovr'))
        final_cm += confusion_matrix(y_te, fold_preds)
        
    print("\n" + "="*80)
    print("FINAL RESULTS (Direct Multi-Class V8 | SMOTETomek + Optuna + Seed Ensemble)")
    print("="*80)
    print(f"Accuracy: {np.mean(acc_list):.4f} | Precision: {np.mean(prec_list):.4f} | Recall: {np.mean(rec_list):.4f} | Macro F1: {np.mean(f1_list):.4f} | OVR AUC: {np.mean(auc_list):.4f}")
    
    return final_cm

if __name__ == "__main__":
    start_time = datetime.now()
    
    df, all_feats = load_data()
    df, unsupervised_feats = add_unsupervised_features(df, all_feats)
    all_feats.extend(unsupervised_feats)
    
    opt_feats = select_top_features(df, all_feats)
    
    best_params = run_optuna_tuning(df, opt_feats)
    
    final_cm = seed_ensemble_evaluation(df, opt_feats, best_params)
    
    end_time = datetime.now()
    print(f"\nElapsed: {end_time - start_time}")
    
    # Save Confusion Matrix
    print("\n[시각화] V8 혼동 행렬 (Confusion Matrix)")
    class_names = ['CN(0)', 'MCI(1)', 'Dem(2)']
    plt.figure(figsize=(6, 5))
    
    sns.heatmap(final_cm, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 14})
    plt.title("V8 Direct Multi-Class (Seed Ensemble)", fontsize=14, pad=10)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix_v8_multiclass.png", dpi=150)
