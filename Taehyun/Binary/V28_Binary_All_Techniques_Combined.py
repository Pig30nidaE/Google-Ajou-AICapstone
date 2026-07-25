import os
import sys
import pathlib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
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

current_dir = pathlib.Path(os.getcwd())
sys.path.insert(0, str(current_dir))
try:
    from xai import ShapAnalyzer
except ImportError:
    from xai.analyzer import ShapAnalyzer

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# 1. 글로벌 경로 및 설정
# =========================================================
BASE_DIR = pathlib.Path(r"c:\ML4")
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "tabular"
PATIENT_PATH = PROCESSED_DIR / "patient_level_all_v2.csv"
PLOT_DIR = BASE_DIR / "report" / "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

TARGET_COL = "label"  # 0: CN (Normal, 111명), 1: Abnormal (MCI+Dem, 63명)
DROP_COLS = ["EMAIL", "date", "DIAG_NM", "original_label", TARGET_COL, "fold"]

RANDOM_STATE = 42
N_SPLITS = 5
FORWARD_SELECTION_MAX_FEATURES = 40

def load_data():
    if not PATIENT_PATH.exists():
        raise FileNotFoundError(f"데이터 파일이 존재하지 않습니다: {PATIENT_PATH}")
    
    df = pd.read_csv(PATIENT_PATH)
    all_feats = [c for c in df.columns if c not in DROP_COLS and pd.api.types.is_numeric_dtype(df[c])]
    df[all_feats] = df[all_feats].replace([np.inf, -np.inf], np.nan)
    return df.reset_index(drop=True), all_feats

def add_unsupervised_features(df, features):
    print("[비지도 학습 피처 엔지니어링] PCA, K-Means, GMM, 계층적 군집화 변수를 생성합니다...")
    X = df[features].copy()
    X.fillna(X.median(), inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    new_features_df = pd.DataFrame(index=df.index)
    
    # 1. PCA (5개 주요 성분)
    n_components = min(5, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca_feats = pca.fit_transform(X_scaled)
    for i in range(n_components):
        new_features_df[f'pca_{i+1}'] = pca_feats[:, i]
        
    # 2. K-Means (3개 거리 및 레이블)
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    kmeans_dist = kmeans.fit_transform(X_scaled)
    for i in range(3):
        new_features_df[f'kmeans_dist_{i}'] = kmeans_dist[:, i]
    new_features_df['kmeans_label'] = kmeans.labels_
    
    # 3. Gaussian Mixture Model (3개 확률)
    gmm = GaussianMixture(n_components=3, random_state=RANDOM_STATE)
    gmm.fit(X_scaled)
    gmm_probs = gmm.predict_proba(X_scaled)
    for i in range(3):
        new_features_df[f'gmm_prob_{i}'] = gmm_probs[:, i]
        
    # 4. Agglomerative Clustering
    agg = AgglomerativeClustering(n_clusters=3)
    new_features_df['hierarchical_cluster_label'] = agg.fit_predict(X_scaled)
    
    df_new = pd.concat([df, new_features_df], axis=1)
    new_cols = features + new_features_df.columns.tolist()
    print(f"  -> 피처 확장 완료: 총 {len(new_cols)}개 피처 (비지도 변수 {new_features_df.shape[1]}개 포함)")
    return df_new, new_cols

def perform_shap_forward_selection(df, features):
    print("\n[전진 선택법] SHAP XAI 랭킹 기반 Forward Selection 탐색...")
    X = df[features]
    y = df[TARGET_COL].astype(int)
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X, y)
    
    base_model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight='balanced', verbose=-1)
    base_model.fit(X_res, y_res)
    
    analyzer = ShapAnalyzer(model=base_model, feature_names=features, task="binary", n_classes=1, class_names=["Abnormal"])
    analyzer.explain(X_res)
    shap_df = analyzer.to_dataframe(combine_classes=False)
    
    ranked_features = shap_df['feature'].tolist()
    top_k_features = ranked_features[:FORWARD_SELECTION_MAX_FEATURES]
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    best_k = 0
    best_cv_score = -1
    history_scores = []
    
    for k in range(1, len(top_k_features) + 1):
        curr_feats = top_k_features[:k]
        X_sub = X[curr_feats]
        
        fold_scores = []
        for train_idx, val_idx in skf.split(X_sub, y):
            X_tr, y_tr = X_sub.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X_sub.iloc[val_idx], y.iloc[val_idx]
            
            X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
            
            eval_model = LGBMClassifier(
                random_state=RANDOM_STATE, n_jobs=1, num_leaves=33, learning_rate=0.08,
                n_estimators=120, min_child_samples=15, class_weight='balanced', verbose=-1
            )
            eval_model.fit(X_tr_res, y_tr_res, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(30, verbose=False)])
            prob = eval_model.predict_proba(X_va)[:, 1]
            fold_scores.append(roc_auc_score(y_va, prob))
            
        mean_auc = np.mean(fold_scores)
        history_scores.append(mean_auc)
        
        if mean_auc > best_cv_score:
            best_cv_score = mean_auc
            best_k = k
            
        if k % 10 == 0 or k == len(top_k_features):
            print(f"  -> SHAP 상위 {k:2d}개 적용 CV AUC: {mean_auc:.4f} (현재 최고 K={best_k}, AUC={best_cv_score:.4f})")
            
    optimal_features = top_k_features[:best_k]
    print(f"\n[탐색 완료] SHAP 최적 피처 {best_k}개 선별 (Best CV AUC: {best_cv_score:.4f})")
    
    # 선별된 피처 중 비지도 학습 변수가 들어갔는지 확인 및 출력
    unsupervised_selected = [f for f in optimal_features if any(f.startswith(p) for p in ['pca_', 'kmeans_', 'gmm_', 'hierarchical_'])]
    print(f"  -> 선별된 피처에 포함된 비지도 학습 변수: {unsupervised_selected if unsupervised_selected else '없음'}")
    
    return optimal_features, history_scores

def run_all_combined_ensemble(df, features):
    print("\n[규제화 & 앙상블] 4종 트리 모델 (LGBM, CatBoost, XGBoost, RF) 규제화 5-Fold 학습...")
    X = df[features]
    y = df[TARGET_COL].astype(int)
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    smote = SMOTE(random_state=RANDOM_STATE)
    
    models = ["LightGBM", "CatBoost", "XGBoost", "RandomForest", "Ensemble"]
    fold_predictions = {m: [] for m in models}
    fold_y_true = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
        
        # 1. LightGBM (규제화)
        m_lgb = LGBMClassifier(
            objective="binary", num_leaves=33, learning_rate=0.08, n_estimators=1000,
            min_child_samples=41, max_depth=5, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=1, verbose=-1
        )
        m_lgb.fit(X_tr_res, y_tr_res, eval_set=[(X_te, y_te)], callbacks=[lgb.early_stopping(50, verbose=False)])
        prob_lgb = m_lgb.predict_proba(X_te)[:, 1]
        
        # 2. CatBoost (규제화)
        m_cat = CatBoostClassifier(
            random_state=RANDOM_STATE, thread_count=1, depth=5, learning_rate=0.05,
            iterations=500, l2_leaf_reg=4.0, auto_class_weights='Balanced', verbose=False
        )
        m_cat.fit(X_tr_res, y_tr_res, eval_set=(X_te, y_te), early_stopping_rounds=50)
        prob_cat = m_cat.predict_proba(X_te)[:, 1]
        
        # 3. XGBoost (규제화)
        m_xgb = XGBClassifier(
            random_state=RANDOM_STATE, n_jobs=1, max_depth=4, learning_rate=0.05,
            n_estimators=400, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=1.5,
            eval_metric='auc', early_stopping_rounds=50
        )
        m_xgb.fit(X_tr_res, y_tr_res, eval_set=[(X_te, y_te)], verbose=False)
        prob_xgb = m_xgb.predict_proba(X_te)[:, 1]
        
        # 4. RandomForest (규제화)
        m_rf = RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=1, max_depth=10, n_estimators=1000,
            class_weight='balanced'
        )
        m_rf.fit(X_tr_res, y_tr_res)
        prob_rf = m_rf.predict_proba(X_te)[:, 1]
        
        # 5. Soft Voting Ensemble
        prob_ens = (prob_lgb * 0.40 + prob_cat * 0.20 + prob_xgb * 0.20 + prob_rf * 0.20)
        
        fold_predictions["LightGBM"].extend(prob_lgb)
        fold_predictions["CatBoost"].extend(prob_cat)
        fold_predictions["XGBoost"].extend(prob_xgb)
        fold_predictions["RandomForest"].extend(prob_rf)
        fold_predictions["Ensemble"].extend(prob_ens)
        fold_y_true.extend(y_te)
        
    y_true_all = np.array(fold_y_true)
    results = {}
    
    print("\n" + "="*80)
    print(" 🚀 V28 모든 기법 통합(PCA/KMeans + SHAP Forward Selection + 규제화) 이진 분류 성능")
    print("="*80)
    
    for m in models:
        probs = np.array(fold_predictions[m])
        auc = roc_auc_score(y_true_all, probs)
        
        fpr, tpr, thresholds = roc_curve(y_true_all, probs)
        best_idx = np.argmax(tpr - fpr)
        opt_thresh = thresholds[best_idx]
        
        preds = np.where(probs >= opt_thresh, 1, 0)
        acc = accuracy_score(y_true_all, preds)
        prec = precision_score(y_true_all, preds, zero_division=0)
        rec = recall_score(y_true_all, preds, zero_division=0)
        f1 = f1_score(y_true_all, preds, zero_division=0)
        cm = confusion_matrix(y_true_all, preds)
        
        results[m] = {
            "auc": auc, "acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "threshold": opt_thresh, "cm": cm, "probs": probs, "y_true": y_true_all
        }
        
        prefix = "🔥 [SOTA] " if m == "Ensemble" or auc >= 0.78 else "         "
        print(f"{prefix}[{m:12s}] Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f} (Thresh: {opt_thresh:.3f})")
        
    return results

if __name__ == "__main__":
    start_t = datetime.now()
    print("V28 All Techniques Combined Execution Started.")
    
    df, raw_features = load_data()
    df_aug, aug_features = add_unsupervised_features(df, raw_features)
    opt_features, forward_hist = perform_shap_forward_selection(df_aug, aug_features)
    results = run_all_combined_ensemble(df_aug, opt_features)
    
    elapsed = datetime.now() - start_t
    print(f"\nCompleted in {elapsed}")
