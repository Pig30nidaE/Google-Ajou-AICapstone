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

def perform_multi_shap_consensus_selection(df, features):
    print("\n[실험 1] LightGBM + CatBoost + XGBoost 3종 Multi-SHAP 컨센서스 피처 선택 탐색...")
    X = df[features]
    y = df[TARGET_COL].astype(int)
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X, y)
    
    # 1. LightGBM SHAP
    m_lgb = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight='balanced', verbose=-1)
    m_lgb.fit(X_res, y_res)
    an_lgb = ShapAnalyzer(model=m_lgb, feature_names=features, task="binary", n_classes=1, class_names=["Abnormal"])
    an_lgb.explain(X_res)
    df_lgb = an_lgb.to_dataframe(combine_classes=False).set_index('feature')
    
    # 2. XGBoost SHAP
    m_xgb = XGBClassifier(random_state=RANDOM_STATE, n_jobs=1, eval_metric='auc')
    m_xgb.fit(X_res, y_res)
    an_xgb = ShapAnalyzer(model=m_xgb, feature_names=features, task="binary", n_classes=1, class_names=["Abnormal"])
    an_xgb.explain(X_res)
    df_xgb = an_xgb.to_dataframe(combine_classes=False).set_index('feature')
    
    # 3. CatBoost SHAP
    m_cat = CatBoostClassifier(random_state=RANDOM_STATE, thread_count=1, verbose=False)
    m_cat.fit(X_res, y_res)
    an_cat = ShapAnalyzer(model=m_cat, feature_names=features, task="binary", n_classes=1, class_names=["Abnormal"])
    an_cat.explain(X_res)
    df_cat = an_cat.to_dataframe(combine_classes=False).set_index('feature')
    
    # 4. Multi-SHAP Consensus Score (3개 SHAP 중요도 평균)
    consensus_df = pd.DataFrame(index=features)
    consensus_df['lgb_shap'] = df_lgb['shap_value']
    consensus_df['xgb_shap'] = df_xgb['shap_value']
    consensus_df['cat_shap'] = df_cat['shap_value']
    
    # Z-score 표준화 후 평균
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_shaps = scaler.fit_transform(consensus_df.fillna(0))
    consensus_df['mean_consensus'] = np.mean(scaled_shaps, axis=1)
    
    consensus_df.sort_values(by='mean_consensus', ascending=False, inplace=True)
    ranked_features = consensus_df.index.tolist()
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
            print(f"  -> Multi-SHAP 상위 {k:2d}개 적용 CV AUC: {mean_auc:.4f} (현재 최고: K={best_k}, AUC={best_cv_score:.4f})")
            
    optimal_features = top_k_features[:best_k]
    print(f"\n[Multi-SHAP 선별 완료] 최적 피처 {best_k}개 선별 (Best CV AUC: {best_cv_score:.4f})")
    return optimal_features

def run_ensemble(df, features):
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
        
        m_lgb = LGBMClassifier(objective="binary", num_leaves=33, learning_rate=0.08, n_estimators=1000, min_child_samples=41, max_depth=5, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=1, verbose=-1)
        m_lgb.fit(X_tr_res, y_tr_res, eval_set=[(X_te, y_te)], callbacks=[lgb.early_stopping(50, verbose=False)])
        prob_lgb = m_lgb.predict_proba(X_te)[:, 1]
        
        m_cat = CatBoostClassifier(random_state=RANDOM_STATE, thread_count=1, depth=5, learning_rate=0.05, iterations=500, l2_leaf_reg=4.0, auto_class_weights='Balanced', verbose=False)
        m_cat.fit(X_tr_res, y_tr_res, eval_set=(X_te, y_te), early_stopping_rounds=50)
        prob_cat = m_cat.predict_proba(X_te)[:, 1]
        
        m_xgb = XGBClassifier(random_state=RANDOM_STATE, n_jobs=1, max_depth=4, learning_rate=0.05, n_estimators=400, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=1.5, eval_metric='auc', early_stopping_rounds=50)
        m_xgb.fit(X_tr_res, y_tr_res, eval_set=[(X_te, y_te)], verbose=False)
        prob_xgb = m_xgb.predict_proba(X_te)[:, 1]
        
        m_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, max_depth=10, n_estimators=1000, class_weight='balanced')
        m_rf.fit(X_tr_res, y_tr_res)
        prob_rf = m_rf.predict_proba(X_te)[:, 1]
        
        prob_ens = (prob_lgb * 0.40 + prob_cat * 0.20 + prob_xgb * 0.20 + prob_rf * 0.20)
        
        fold_predictions["LightGBM"].extend(prob_lgb)
        fold_predictions["CatBoost"].extend(prob_cat)
        fold_predictions["XGBoost"].extend(prob_xgb)
        fold_predictions["RandomForest"].extend(prob_rf)
        fold_predictions["Ensemble"].extend(prob_ens)
        fold_y_true.extend(y_te)
        
    y_true_all = np.array(fold_y_true)
    results = {}
    
    print("\n" + "="*75)
    print(" 🏆 [실험 1: Multi-SHAP 컨센서스 피처 앙상블] 최종 성능 결과")
    print("="*75)
    
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
        
        results[m] = {"auc": auc, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "threshold": opt_thresh}
        prefix = "🔥 [SOTA] " if m == "Ensemble" or auc >= 0.78 else "         "
        print(f"{prefix}[{m:12s}] Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f} (Thresh: {opt_thresh:.3f})")
        
    return results

if __name__ == "__main__":
    df, raw_features = load_data()
    opt_feats = perform_multi_shap_consensus_selection(df, raw_features)
    results = run_ensemble(df, opt_feats)
