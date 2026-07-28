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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from imblearn.over_sampling import SMOTE
from sklearn.isotonic import IsotonicRegression

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

BASE_DIR = pathlib.Path(r"c:\ML4")
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "tabular"
PATIENT_PATH = PROCESSED_DIR / "patient_level_all_v2.csv"

TARGET_COL = "label"
DROP_COLS = ["EMAIL", "date", "DIAG_NM", "original_label", TARGET_COL, "fold"]
RANDOM_STATE = 42
N_SPLITS = 5

def load_data():
    df = pd.read_csv(PATIENT_PATH)
    raw_feats = [c for c in df.columns if c not in DROP_COLS and pd.api.types.is_numeric_dtype(df[c])]
    df[raw_feats] = df[raw_feats].replace([np.inf, -np.inf], np.nan)
    return df.reset_index(drop=True), raw_feats

def perform_shap_forward_selection(df, features):
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
    top_k_features = ranked_features[:40]
    
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
            
            eval_model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, num_leaves=33, learning_rate=0.08, n_estimators=120, min_child_samples=15, class_weight='balanced', verbose=-1)
            eval_model.fit(X_tr_res, y_tr_res, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(30, verbose=False)])
            prob = eval_model.predict_proba(X_va)[:, 1]
            fold_scores.append(roc_auc_score(y_va, prob))
            
        mean_auc = np.mean(fold_scores)
        history_scores.append(mean_auc)
        
        if mean_auc > best_cv_score:
            best_cv_score = mean_auc
            best_k = k
            
    optimal_features = top_k_features[:best_k]
    print(f"[SHAP 피처 선별 완료] 최적 피처 {best_k}개 선별 (Best CV AUC: {best_cv_score:.4f})")
    return optimal_features

def run_calibrated_ensemble(df, features):
    print("\n[실험 3] Isotonic Regression 확률 보정(Probability Calibration) 앙상블 학습...")
    X = df[features]
    y = df[TARGET_COL].astype(int)
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    smote = SMOTE(random_state=RANDOM_STATE)
    
    models = ["LightGBM", "CatBoost", "XGBoost", "RandomForest"]
    oof_raw_probs = {m: np.zeros(len(df)) for m in models}
    oof_cal_probs = {m: np.zeros(len(df)) for m in models}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
        
        # Models
        m_lgb = LGBMClassifier(objective="binary", num_leaves=33, learning_rate=0.08, n_estimators=1000, min_child_samples=41, max_depth=5, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=1, verbose=-1)
        m_lgb.fit(X_tr_res, y_tr_res, eval_set=[(X_te, y_te)], callbacks=[lgb.early_stopping(50, verbose=False)])
        p_lgb_raw = m_lgb.predict_proba(X_te)[:, 1]
        
        m_cat = CatBoostClassifier(random_state=RANDOM_STATE, thread_count=1, depth=5, learning_rate=0.05, iterations=500, l2_leaf_reg=4.0, auto_class_weights='Balanced', verbose=False)
        m_cat.fit(X_tr_res, y_tr_res, eval_set=(X_te, y_te), early_stopping_rounds=50)
        p_cat_raw = m_cat.predict_proba(X_te)[:, 1]
        
        m_xgb = XGBClassifier(random_state=RANDOM_STATE, n_jobs=1, max_depth=4, learning_rate=0.05, n_estimators=400, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=1.5, eval_metric='auc', early_stopping_rounds=50)
        m_xgb.fit(X_tr_res, y_tr_res, eval_set=[(X_te, y_te)], verbose=False)
        p_xgb_raw = m_xgb.predict_proba(X_te)[:, 1]
        
        m_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, max_depth=10, n_estimators=1000, class_weight='balanced')
        m_rf.fit(X_tr_res, y_tr_res)
        p_rf_raw = m_rf.predict_proba(X_te)[:, 1]
        
        # Isotonic Calibration on Validation Predictions
        # Fit Isotonic Regression on train predictions
        p_lgb_tr = m_lgb.predict_proba(X_tr)[:, 1]
        p_cat_tr = m_cat.predict_proba(X_tr)[:, 1]
        p_xgb_tr = m_xgb.predict_proba(X_tr)[:, 1]
        p_rf_tr = m_rf.predict_proba(X_tr)[:, 1]
        
        iso_lgb = IsotonicRegression(out_of_bounds='clip').fit(p_lgb_tr, y_tr)
        iso_cat = IsotonicRegression(out_of_bounds='clip').fit(p_cat_tr, y_tr)
        iso_xgb = IsotonicRegression(out_of_bounds='clip').fit(p_xgb_tr, y_tr)
        iso_rf = IsotonicRegression(out_of_bounds='clip').fit(p_rf_tr, y_tr)
        
        oof_raw_probs["LightGBM"][test_idx] = p_lgb_raw
        oof_raw_probs["CatBoost"][test_idx] = p_cat_raw
        oof_raw_probs["XGBoost"][test_idx] = p_xgb_raw
        oof_raw_probs["RandomForest"][test_idx] = p_rf_raw
        
        oof_cal_probs["LightGBM"][test_idx] = iso_lgb.predict(p_lgb_raw)
        oof_cal_probs["CatBoost"][test_idx] = iso_cat.predict(p_cat_raw)
        oof_cal_probs["XGBoost"][test_idx] = iso_xgb.predict(p_xgb_raw)
        oof_cal_probs["RandomForest"][test_idx] = iso_rf.predict(p_rf_raw)
        
    y_true_all = y.values
    
    raw_ens = (oof_raw_probs["LightGBM"] * 0.4 + oof_raw_probs["CatBoost"] * 0.2 + oof_raw_probs["XGBoost"] * 0.2 + oof_raw_probs["RandomForest"] * 0.2)
    cal_ens = (oof_cal_probs["LightGBM"] * 0.4 + oof_cal_probs["CatBoost"] * 0.2 + oof_cal_probs["XGBoost"] * 0.2 + oof_cal_probs["RandomForest"] * 0.2)
    
    print("\n" + "="*75)
    print(" 🏆 [실험 3: Isotonic 확률 보정 앙상블 대조 결과]")
    print("="*75)
    
    for name, probs in [("Raw Soft Voting Ensemble", raw_ens), ("Calibrated Soft Voting Ensemble", cal_ens)]:
        auc = roc_auc_score(y_true_all, probs)
        fpr, tpr, thresholds = roc_curve(y_true_all, probs)
        best_idx = np.argmax(tpr - fpr)
        opt_thresh = thresholds[best_idx]
        
        preds = np.where(probs >= opt_thresh, 1, 0)
        acc = accuracy_score(y_true_all, preds)
        prec = precision_score(y_true_all, preds, zero_division=0)
        rec = recall_score(y_true_all, preds, zero_division=0)
        f1 = f1_score(y_true_all, preds, zero_division=0)
        
        prefix = "🔥 [SOTA] " if auc >= 0.78 else "         "
        print(f"{prefix}[{name:30s}] Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f} (Thresh: {opt_thresh:.3f})")

if __name__ == "__main__":
    df, features = load_data()
    opt_feats = perform_shap_forward_selection(df, features)
    run_calibrated_ensemble(df, opt_feats)
