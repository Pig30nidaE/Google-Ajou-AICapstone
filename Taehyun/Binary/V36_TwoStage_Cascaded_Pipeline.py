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

def get_shap_ranked_features(df, features):
    X = df[features]
    y = df[TARGET_COL].astype(int)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X, y)
    
    m = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight='balanced', verbose=-1)
    m.fit(X_res, y_res)
    an = ShapAnalyzer(model=m, feature_names=features, task="binary", n_classes=1, class_names=["Abnormal"])
    an.explain(X_res)
    shap_df = an.to_dataframe(combine_classes=False)
    return shap_df['feature'].tolist()

def compare_hard_vs_soft_cascade(df, features):
    X = df[features]
    y = df[TARGET_COL].astype(int)
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    smote = SMOTE(random_state=RANDOM_STATE)
    
    ranked_feats = get_shap_ranked_features(df, features)
    s1_feats = ranked_feats[:12]  # V35 Ultra Recall (K=12)
    s2_feats = ranked_feats[:16]  # V29 Optuna LightGBM (K=16)
    
    p1_oof = np.zeros(len(df))
    p2_oof = np.zeros(len(df))
    
    for train_idx, test_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        
        # Model 1: V35 Ultra Recall Ensemble
        X_tr1, y_tr1 = smote.fit_resample(X_tr[s1_feats], y_tr)
        m_lgb1 = LGBMClassifier(objective="binary", num_leaves=33, learning_rate=0.08, n_estimators=1000, min_child_samples=41, max_depth=5, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=1, verbose=-1)
        m_lgb1.fit(X_tr1, y_tr1, eval_set=[(X_te[s1_feats], y_te)], callbacks=[lgb.early_stopping(50, verbose=False)])
        
        m_cat1 = CatBoostClassifier(random_state=RANDOM_STATE, thread_count=1, depth=5, learning_rate=0.05, iterations=500, l2_leaf_reg=4.0, auto_class_weights='Balanced', verbose=False)
        m_cat1.fit(X_tr1, y_tr1, eval_set=(X_te[s1_feats], y_te), early_stopping_rounds=50)
        
        m_xgb1 = XGBClassifier(random_state=RANDOM_STATE, n_jobs=1, max_depth=4, learning_rate=0.05, n_estimators=400, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=1.5, eval_metric='auc', early_stopping_rounds=50)
        m_xgb1.fit(X_tr1, y_tr1, eval_set=[(X_te[s1_feats], y_te)], verbose=False)
        
        m_rf1 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, max_depth=10, n_estimators=1000, class_weight='balanced')
        m_rf1.fit(X_tr1, y_tr1)
        
        p1 = 0.32*m_lgb1.predict_proba(X_te[s1_feats])[:,1] + 0.35*m_cat1.predict_proba(X_te[s1_feats])[:,1] + 0.22*m_xgb1.predict_proba(X_te[s1_feats])[:,1] + 0.11*m_rf1.predict_proba(X_te[s1_feats])[:,1]
        p1_oof[test_idx] = p1
        
        # Model 2: V29 Optuna LGBM
        X_tr2, y_tr2 = smote.fit_resample(X_tr[s2_feats], y_tr)
        m_lgb2 = LGBMClassifier(objective="binary", num_leaves=31, learning_rate=0.05, n_estimators=1000, min_child_samples=20, max_depth=6, reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=1, verbose=-1)
        m_lgb2.fit(X_tr2, y_tr2, eval_set=[(X_te[s2_feats], y_te)], callbacks=[lgb.early_stopping(50, verbose=False)])
        p2_oof[test_idx] = m_lgb2.predict_proba(X_te[s2_feats])[:,1]

    y_all = y.values
    
    # 1. Hard Cascade (Strict 2-Stage Filter)
    hard_preds = np.zeros(len(df))
    stage1_passed = p1_oof >= 0.409
    hard_preds[stage1_passed] = np.where(p2_oof[stage1_passed] >= 0.48, 1, 0)
    
    # 2. Soft Fusion (Weighted Probability Blend: 60% V35 + 40% V29)
    soft_probs = 0.60 * p1_oof + 0.40 * p2_oof
    fpr, tpr, thresholds = roc_curve(y_all, soft_probs)
    best_idx = np.argmax(tpr - fpr)
    opt_thresh = thresholds[best_idx]
    soft_preds = np.where(soft_probs >= opt_thresh, 1, 0)

    print("="*80)
    print(" 🏆 [V35 Ultra Recall + V29 Precision 결합 방식 비교 결과]")
    print("="*80)
    
    for name, preds, probs in [
        ("V35 Ultra Recall 단독", np.where(p1_oof >= 0.409, 1, 0), p1_oof),
        ("V29 Optuna LGBM 단독", np.where(p2_oof >= 0.524, 1, 0), p2_oof),
        ("하드 2단계 필터링 (Hard Cascade)", hard_preds, None),
        ("소프트 확률 융합 (Soft Fusion)", soft_preds, soft_probs)
    ]:
        acc = accuracy_score(y_all, preds)
        prec = precision_score(y_all, preds, zero_division=0)
        rec = recall_score(y_all, preds, zero_division=0)
        f1 = f1_score(y_all, preds, zero_division=0)
        auc = roc_auc_score(y_all, probs) if probs is not None else 0.0
        
        auc_str = f"{auc:.4f}" if probs is not None else "N/A"
        print(f"[{name:28s}] Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc_str}")

if __name__ == "__main__":
    df, features = load_data()
    compare_hard_vs_soft_cascade(df, features)
