import os
import sys
import pathlib
import json
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from imblearn.over_sampling import SMOTE

import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = pathlib.Path(r"c:\ML4")
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "tabular"
CIRCADIAN_PATH = PROCESSED_DIR / "patient_level_circadian_v3.csv"
PLOT_DIR = BASE_DIR / "report" / "plots"
REPORT_DIR = BASE_DIR / "report" / "multi-class"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

TARGET_COL = "original_label"
RANDOM_STATE = 42
N_TRIALS = 30
CV_SPLITS = 5

V44_14_FEATURES = [
    'sleep_score_alignment', 'sleep_hr_5min_max_std', 'sleep_awake_std',
    'sleep_breath_average', 'activity_score_std', 'activity_class_3_count_std',
    'activity_met_min_low_std', 'sleep_restless_std', 'circadian_IV',
    'circadian_IS', 'circadian_RA', 'sleep_wake_bouts_avg',
    'HR_drop_ratio', 'Circadian_Strain'
]

DROP_METADATA_COLS = [
    "EMAIL", "date", "DIAG_NM", "original_label", "label", "fold",
    "SAMPLE_EMAIL", "DIAG_SEQ", "DOCTOR_NM", "MMSE_NUM", "MMSE_KIND", "TOTAL"
]

def load_and_preprocess_data():
    df = pd.read_csv(CIRCADIAN_PATH)
    mmse_cols = [c for c in df.columns if c.startswith('Q') or 'mmse' in c.lower() or c in ['TOTAL', 'DIAG_SEQ', 'DOCTOR_NM']]
    if mmse_cols:
        df.drop(columns=mmse_cols, inplace=True, errors='ignore')
        
    numeric_cols = [c for c in df.columns if c not in DROP_METADATA_COLS and pd.api.types.is_numeric_dtype(df[c])]
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    if 'sleep_hr_average' in df.columns and 'sleep_hr_lowest' in df.columns:
        df['HR_drop_ratio'] = (df['sleep_hr_average'] - df['sleep_hr_lowest']) / (df['sleep_hr_average'] + 1e-5)
    if 'circadian_IV' in df.columns and 'circadian_IS' in df.columns:
        df['Circadian_Strain'] = df['circadian_IV'] / (df['circadian_IS'] + 1e-5)
        
    df['stage1_label'] = np.where(df[TARGET_COL] == 0, 0, 1)
    df['stage2_label'] = np.where(df[TARGET_COL] == 1, 0, np.where(df[TARGET_COL] == 2, 1, np.nan))
    
    cand_feats = [c for c in df.columns if c not in DROP_METADATA_COLS and c not in ['stage1_label', 'stage2_label'] and pd.api.types.is_numeric_dtype(df[c])]
    
    # 비지도 피처 추출
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_mat = imputer.fit_transform(df[cand_feats])
    X_sc = scaler.fit_transform(X_mat)
    
    pca = PCA(n_components=5, random_state=RANDOM_STATE)
    pca_f = pca.fit_transform(X_sc)
    for i in range(pca_f.shape[1]):
        df[f'pca_{i+1}'] = pca_f[:, i]
        
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    km_d = kmeans.fit_transform(X_sc)
    for i in range(3):
        df[f'kmeans_dist_{i}'] = km_d[:, i]
    df['kmeans_label'] = kmeans.labels_
    
    gmm = GaussianMixture(n_components=3, random_state=RANDOM_STATE)
    gmm.fit(X_sc)
    gmm_p = gmm.predict_proba(X_sc)
    for i in range(3):
        df[f'gmm_prob_{i}'] = gmm_p[:, i]
        
    agg = AgglomerativeClustering(n_clusters=3)
    df['hierarchical_cluster_label'] = agg.fit_predict(X_sc)
    
    all_features = [c for c in df.columns if c not in DROP_METADATA_COLS and c not in ['stage1_label', 'stage2_label'] and pd.api.types.is_numeric_dtype(df[c])]
    return df, all_features

def select_stage_features(df, features, target_col, stage_name):
    df_stage = df.dropna(subset=[target_col]).copy().reset_index(drop=True)
    X = df_stage[features]
    y = df_stage[target_col].astype(int)
    
    min_c = y.value_counts().min()
    smote = SMOTE(k_neighbors=max(1, min(5, min_c - 1)), random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X, y)
    
    base_m = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight='balanced', verbose=-1)
    base_m.fit(X_res, y_res)
    
    imp = base_m.feature_importances_
    ranked = [f for _, f in sorted(zip(imp, features), reverse=True)]
    
    v44_present = [f for f in V44_14_FEATURES if f in features]
    prioritized = list(dict.fromkeys(v44_present + ranked[:30]))
    top_candidates = prioritized[:30]
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for k in range(1, len(top_candidates) + 1):
        feats_k = top_candidates[:k]
        fold_aucs = []
        for tr_i, va_i in skf.split(X[feats_k], y):
            X_tr, y_tr = X[feats_k].iloc[tr_i], y.iloc[tr_i]
            X_va, y_va = X[feats_k].iloc[va_i], y.iloc[va_i]
            sm = SMOTE(k_neighbors=max(1, min(3, y_tr.value_counts().min() - 1)), random_state=RANDOM_STATE)
            X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
            m = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight='balanced', n_estimators=60, verbose=-1)
            m.fit(X_tr_res, y_tr_res)
            fold_aucs.append(roc_auc_score(y_va, m.predict_proba(X_va)[:, 1]))
        scores.append(np.mean(fold_aucs))
        
    opt_k = np.argmax(scores) + 1
    opt_feats = top_candidates[:opt_k]
    print(f"[{stage_name} 피처 선택] 총 {len(opt_feats)}개 피처 선정 (선택 Inner AUC: {max(scores):.4f})")
    return opt_feats

# ---------------------------------------------------------
# Optuna 하이퍼파라미터 탐색 함수들
# ---------------------------------------------------------
def tune_stage_model(df_stage, features, target_col, model_name, stage_name, n_trials=30):
    X = df_stage[features]
    y = df_stage[target_col].astype(int)
    
    min_c = y.value_counts().min()
    k_neighbors = max(1, min(4, min_c - 1))
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    
    def objective(trial):
        if model_name == "LightGBM":
            params = {
                'objective': 'binary',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 7, 63),
                'max_depth': trial.suggest_int('max_depth', 2, 6),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 80, 250),
                'random_state': RANDOM_STATE, 'n_jobs': 1, 'verbose': -1
            }
        elif model_name == "CatBoost":
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'depth': trial.suggest_int('depth', 3, 6),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 25.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'iterations': trial.suggest_int('iterations', 80, 250),
                'random_state': RANDOM_STATE, 'verbose': 0, 'thread_count': 1
            }
        elif model_name == "XGBoost":
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 6),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 80, 250),
                'eval_metric': 'logloss', 'random_state': RANDOM_STATE, 'n_jobs': 1
            }
        elif model_name == "RandomForest":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 8),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 6),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.6, 0.8]),
                'random_state': RANDOM_STATE, 'n_jobs': 1
            }
            
        cv_scores = []
        for tr_i, va_i in skf.split(X, y):
            X_tr, y_tr = X.iloc[tr_i], y.iloc[tr_i]
            X_va, y_va = X.iloc[va_i], y.iloc[va_i]
            
            sm = SMOTE(k_neighbors=max(1, min(k_neighbors, y_tr.value_counts().min() - 1)), random_state=RANDOM_STATE)
            X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
            
            if model_name == "LightGBM":
                m = LGBMClassifier(**params)
            elif model_name == "CatBoost":
                m = CatBoostClassifier(**params)
            elif model_name == "XGBoost":
                m = XGBClassifier(**params)
            elif model_name == "RandomForest":
                m = RandomForestClassifier(**params)
                
            m.fit(X_tr_res, y_tr_res)
            prob = m.predict_proba(X_va)[:, 1]
            try:
                cv_scores.append(roc_auc_score(y_va, prob))
            except ValueError:
                cv_scores.append(0.5)
                
        return np.mean(cv_scores)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials)
    
    print(f"  [{stage_name} - {model_name:<12}] Optuna 완료: Best AUC = {study.best_value:.4f}")
    return study.best_params, study.best_value


def run_optuna_pipeline():
    print("=" * 85)
    print(" [H-CRE Optuna Hyperparameter Optimization] 확장 탐색 시작")
    print(f"  탐색 대상 모델: LightGBM, CatBoost, XGBoost, RandomForest | 시행 횟수: {N_TRIALS}회")
    print("=" * 85)
    
    df, all_features = load_and_preprocess_data()
    print(f"데이터 로드 완료: 총 {len(df)}명 (CN: {sum(df[TARGET_COL]==0)}, MCI: {sum(df[TARGET_COL]==1)}, Dem: {sum(df[TARGET_COL]==2)})")
    
    # 1. 피처 선택 (Stage 1 & Stage 2)
    s1_feats = select_stage_features(df, all_features, "stage1_label", "Stage1")
    df_s2 = df.dropna(subset=['stage2_label']).copy().reset_index(drop=True)
    s2_feats = select_stage_features(df_s2, all_features, "stage2_label", "Stage2")
    
    models = ["LightGBM", "CatBoost", "XGBoost", "RandomForest"]
    opt_params = {'Stage1': {}, 'Stage2': {}}
    opt_scores = {'Stage1': {}, 'Stage2': {}}
    
    print("\n--- [Stage 1: CN vs Abnormal Optuna 탐색] ---")
    for m in models:
        best_p, best_s = tune_stage_model(df, s1_feats, "stage1_label", m, "Stage 1", n_trials=N_TRIALS)
        opt_params['Stage1'][m] = best_p
        opt_scores['Stage1'][m] = best_s
        
    print("\n--- [Stage 2: MCI vs Dem Optuna 탐색] ---")
    for m in models:
        best_p, best_s = tune_stage_model(df_s2, s2_feats, "stage2_label", m, "Stage 2", n_trials=N_TRIALS)
        opt_params['Stage2'][m] = best_p
        opt_scores['Stage2'][m] = best_s
        
    print("\n" + "=" * 85)
    print(" 🎯 [Optuna 최적 하이퍼파라미터 추출 결과 요약]")
    print("=" * 85)
    
    for stage in ['Stage1', 'Stage2']:
        print(f"\n--- {stage} 최적 파라미터 ---")
        for m in models:
            print(f"[{m}] (Best AUC: {opt_scores[stage][m]:.4f})")
            for k, v in opt_params[stage][m].items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.5f}")
                else:
                    print(f"    {k}: {v}")
                    
    # JSON 파일로 최적 파라미터 저장
    params_save_path = pathlib.Path(__file__).parent / "h_cre_optuna_best_params.json"
    with open(params_save_path, "w", encoding="utf-8") as f:
        json.dump({'params': opt_params, 'scores': opt_scores}, f, indent=4, ensure_ascii=False)
    print(f"\n최적 파라미터 저장 완료: {params_save_path}")
    
    # 2. 최적 파라미터를 사용한 5-Fold Nested CV 성능 평가
    print("\n" + "=" * 85)
    print(" 📊 [최적 파라미터 적용 5-Fold Out-of-Fold 3-Class 평가]")
    print("=" * 85)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_true_all = df[TARGET_COL].astype(int).values
    oof_probs = {m: np.zeros((len(df), 3)) for m in models + ["Ensemble"]}
    
    for fold, (tr_i, te_i) in enumerate(skf.split(df, df[TARGET_COL]), 1):
        df_tr, df_te = df.iloc[tr_i].copy(), df.iloc[te_i].copy()
        
        # Stage 1
        X_tr_s1, y_tr_s1 = df_tr[s1_feats], df_tr['stage1_label'].astype(int)
        sm_s1 = SMOTE(k_neighbors=max(1, min(4, y_tr_s1.value_counts().min() - 1)), random_state=RANDOM_STATE)
        X_tr_s1_res, y_tr_s1_res = sm_s1.fit_resample(X_tr_s1, y_tr_s1)
        
        # Stage 2
        df_tr_s2 = df_tr[df_tr['stage1_label'] == 1].copy()
        X_tr_s2, y_tr_s2 = df_tr_s2[s2_feats], df_tr_s2['stage2_label'].astype(int)
        sm_s2 = SMOTE(k_neighbors=max(1, min(2, y_tr_s2.value_counts().min() - 1)), random_state=RANDOM_STATE)
        X_tr_s2_res, y_tr_s2_res = sm_s2.fit_resample(X_tr_s2, y_tr_s2)
        
        fold_probs = {}
        for m in models:
            p1 = opt_params['Stage1'][m]
            p2 = opt_params['Stage2'][m]
            
            if m == "LightGBM":
                m1 = LGBMClassifier(**p1, random_state=RANDOM_STATE, n_jobs=1, verbose=-1)
                m2 = LGBMClassifier(**p2, random_state=RANDOM_STATE, n_jobs=1, verbose=-1)
            elif m == "CatBoost":
                m1 = CatBoostClassifier(**p1, random_state=RANDOM_STATE, verbose=0, thread_count=1)
                m2 = CatBoostClassifier(**p2, random_state=RANDOM_STATE, verbose=0, thread_count=1)
            elif m == "XGBoost":
                m1 = XGBClassifier(**p1, random_state=RANDOM_STATE, n_jobs=1)
                m2 = XGBClassifier(**p2, random_state=RANDOM_STATE, n_jobs=1)
            elif m == "RandomForest":
                m1 = RandomForestClassifier(**p1, random_state=RANDOM_STATE, n_jobs=1)
                m2 = RandomForestClassifier(**p2, random_state=RANDOM_STATE, n_jobs=1)
                
            m1.fit(X_tr_s1_res, y_tr_s1_res)
            m2.fit(X_tr_s2_res, y_tr_s2_res)
            
            prob1 = m1.predict_proba(df_te[s1_feats])[:, 1]
            prob2 = m2.predict_proba(df_te[s2_feats])[:, 1]
            
            # Confidence Penalty
            penalty = np.where(prob1 < 0.65, 0.6 + 0.4 * (prob1 / 0.65), 1.0)
            prob2_reg = prob2 * penalty
            
            p_CN = 1.0 - prob1
            p_MCI = prob1 * (1.0 - prob2_reg)
            p_Dem = prob1 * prob2_reg
            
            prob_mat = np.vstack([p_CN, p_MCI, p_Dem]).T
            oof_probs[m][te_i] = prob_mat
            fold_probs[m] = prob_mat
            
        oof_probs["Ensemble"][te_i] = (fold_probs["LightGBM"] + fold_probs["CatBoost"] + fold_probs["XGBoost"] + fold_probs["RandomForest"]) / 4.0

    eval_results = {}
    for m in models + ["Ensemble"]:
        probs = oof_probs[m]
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(y_true_all, preds)
        prec = precision_score(y_true_all, preds, average='macro', zero_division=0)
        rec = recall_score(y_true_all, preds, average='macro', zero_division=0)
        f1 = f1_score(y_true_all, preds, average='macro', zero_division=0)
        auc = roc_auc_score(y_true_all, probs, multi_class='ovr')
        cm = confusion_matrix(y_true_all, preds)
        
        eval_results[m] = {
            'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc, 'cm': cm
        }
        print(f"[{m:<12}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | Macro F1: {f1:.4f} | OVR AUC: {auc:.4f}")
        print(f"             세부 민감도 -> CN: {cm[0,0]/cm[0].sum():.4f}, MCI: {cm[1,1]/cm[1].sum():.4f}, Dem: {cm[2,2]/cm[2].sum():.4f}")

    # 혼동 행렬 시각화
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    class_names = ['CN(0)', 'MCI(1)', 'Dem(2)']
    for ax, m in zip(axes, models + ["Ensemble"]):
        sns.heatmap(eval_results[m]['cm'], annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f"{m}\nAcc: {eval_results[m]['acc']:.3f} | AUC: {eval_results[m]['auc']:.3f}", fontsize=11)
        ax.set_xlabel('예측')
        ax.set_ylabel('실제')
    plt.suptitle("Optuna 튜닝 후 H-CRE 3-Class 혼동 행렬", fontsize=13)
    plt.tight_layout()
    plot_path = pathlib.Path(__file__).parent / "confusion_matrix_h_cre_optuna_tuned.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"혼동 행렬 저장 완료: {plot_path}")
    
    return opt_params, opt_scores, eval_results

if __name__ == "__main__":
    start_t = datetime.now()
    run_optuna_pipeline()
    end_t = datetime.now()
    print(f"\n전체 Optuna 튜닝 소요 시간: {end_t - start_t}")
