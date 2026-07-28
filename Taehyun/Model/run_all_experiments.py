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

def add_domain_biomarkers(df, features):
    bio_df = pd.DataFrame(index=df.index)
    if 'SLEEP_REM_mean' in df.columns and 'SLEEP_TOTAL_mean' in df.columns:
        bio_df['sleep_rem_ratio'] = df['SLEEP_REM_mean'] / (df['SLEEP_TOTAL_mean'] + 1e-5)
    if 'SLEEP_DEEP_mean' in df.columns and 'SLEEP_TOTAL_mean' in df.columns:
        bio_df['sleep_deep_ratio'] = df['SLEEP_DEEP_mean'] / (df['SLEEP_TOTAL_mean'] + 1e-5)
    if 'HRV_SDNN_std' in df.columns and 'HRV_SDNN_mean' in df.columns:
        bio_df['hrv_stability'] = df['HRV_SDNN_std'] / (df['HRV_SDNN_mean'] + 1e-5)
    if 'MET_std' in df.columns and 'MET_mean' in df.columns:
        bio_df['activity_volatility'] = df['MET_std'] / (df['MET_mean'] + 1e-5)
        
    df_aug = pd.concat([df, bio_df], axis=1)
    all_features = [c for c in df_aug.columns if c not in DROP_COLS and pd.api.types.is_numeric_dtype(df_aug[c])]
    return df_aug, all_features

def get_shap_ranked_features(df, features, model_type='lgb'):
    X = df[features]
    y = df[TARGET_COL].astype(int)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X, y)
    
    if model_type == 'lgb':
        m = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight='balanced', verbose=-1)
    elif model_type == 'xgb':
        m = XGBClassifier(random_state=RANDOM_STATE, n_jobs=1, eval_metric='auc')
    elif model_type == 'cat':
        m = CatBoostClassifier(random_state=RANDOM_STATE, thread_count=1, verbose=False)
        
    m.fit(X_res, y_res)
    an = ShapAnalyzer(model=m, feature_names=features, task="binary", n_classes=1, class_names=["Abnormal"])
    an.explain(X_res)
    shap_df = an.to_dataframe(combine_classes=False)
    return shap_df['feature'].tolist()

def eval_pipeline(df, features, calibrate=False, weights=(0.4, 0.2, 0.2, 0.2)):
    X = df[features]
    y = df[TARGET_COL].astype(int)
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    smote = SMOTE(random_state=RANDOM_STATE)
    
    models = ["LightGBM", "CatBoost", "XGBoost", "RandomForest"]
    oof_probs = {m: np.zeros(len(df)) for m in models}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
        
        # 1. LightGBM
        m_lgb = LGBMClassifier(objective="binary", num_leaves=33, learning_rate=0.08, n_estimators=1000, min_child_samples=41, max_depth=5, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=1, verbose=-1)
        m_lgb.fit(X_tr_res, y_tr_res, eval_set=[(X_te, y_te)], callbacks=[lgb.early_stopping(50, verbose=False)])
        p_lgb = m_lgb.predict_proba(X_te)[:, 1]
        
        # 2. CatBoost
        m_cat = CatBoostClassifier(random_state=RANDOM_STATE, thread_count=1, depth=5, learning_rate=0.05, iterations=500, l2_leaf_reg=4.0, auto_class_weights='Balanced', verbose=False)
        m_cat.fit(X_tr_res, y_tr_res, eval_set=(X_te, y_te), early_stopping_rounds=50)
        p_cat = m_cat.predict_proba(X_te)[:, 1]
        
        # 3. XGBoost
        m_xgb = XGBClassifier(random_state=RANDOM_STATE, n_jobs=1, max_depth=4, learning_rate=0.05, n_estimators=400, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=1.5, eval_metric='auc', early_stopping_rounds=50)
        m_xgb.fit(X_tr_res, y_tr_res, eval_set=[(X_te, y_te)], verbose=False)
        p_xgb = m_xgb.predict_proba(X_te)[:, 1]
        
        # 4. RandomForest
        m_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, max_depth=10, n_estimators=1000, class_weight='balanced')
        m_rf.fit(X_tr_res, y_tr_res)
        p_rf = m_rf.predict_proba(X_te)[:, 1]
        
        if calibrate:
            iso_lgb = IsotonicRegression(out_of_bounds='clip').fit(m_lgb.predict_proba(X_tr)[:, 1], y_tr)
            iso_cat = IsotonicRegression(out_of_bounds='clip').fit(m_cat.predict_proba(X_tr)[:, 1], y_tr)
            iso_xgb = IsotonicRegression(out_of_bounds='clip').fit(m_xgb.predict_proba(X_tr)[:, 1], y_tr)
            iso_rf = IsotonicRegression(out_of_bounds='clip').fit(m_rf.predict_proba(X_tr)[:, 1], y_tr)
            
            p_lgb = iso_lgb.predict(p_lgb)
            p_cat = iso_cat.predict(p_cat)
            p_xgb = iso_xgb.predict(p_xgb)
            p_rf = iso_rf.predict(p_rf)
            
        oof_probs["LightGBM"][test_idx] = p_lgb
        oof_probs["CatBoost"][test_idx] = p_cat
        oof_probs["XGBoost"][test_idx] = p_xgb
        oof_probs["RandomForest"][test_idx] = p_rf
        
    ens_prob = (
        oof_probs["LightGBM"] * weights[0] +
        oof_probs["CatBoost"] * weights[1] +
        oof_probs["XGBoost"] * weights[2] +
        oof_probs["RandomForest"] * weights[3]
    )
    
    y_all = y.values
    auc = roc_auc_score(y_all, ens_prob)
    fpr, tpr, thresholds = roc_curve(y_all, ens_prob)
    best_idx = np.argmax(tpr - fpr)
    opt_thresh = thresholds[best_idx]
    
    preds = np.where(ens_prob >= opt_thresh, 1, 0)
    acc = accuracy_score(y_all, preds)
    prec = precision_score(y_all, preds, zero_division=0)
    rec = recall_score(y_all, preds, zero_division=0)
    f1 = f1_score(y_all, preds, zero_division=0)
    
    return {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc, "thresh": opt_thresh
    }

if __name__ == "__main__":
    df, raw_features = load_data()
    
    print("="*80)
    print(" 🚀 4가지 성능 향상 실험 개별 검증 시작 (5-Fold CV Out-of-Fold)")
    print("="*80)
    
    # 0. Baseline V26
    r_lgb = get_shap_ranked_features(df, raw_features, 'lgb')
    res_v26 = eval_pipeline(df, r_lgb[:15])
    print(f"\n[기준 V26 베이스라인     ] Acc: {res_v26['acc']:.4f} | Prec: {res_v26['prec']:.4f} | Recall: {res_v26['rec']:.4f} | F1: {res_v26['f1']:.4f} | AUC: {res_v26['auc']:.4f}")
    
    # 실험 1: Multi-SHAP Consensus Feature Selection
    print("\n[실험 1 진행 중...] Multi-SHAP 3종 컨센서스 랭킹 계산...")
    r_xgb = get_shap_ranked_features(df, raw_features, 'xgb')
    r_cat = get_shap_ranked_features(df, raw_features, 'cat')
    
    # Rank averaging
    rank_dict = {f: 0 for f in raw_features}
    for f in raw_features:
        rank_dict[f] = r_lgb.index(f) + r_xgb.index(f) + r_cat.index(f)
    consensus_ranked = sorted(raw_features, key=lambda x: rank_dict[x])
    res_exp1 = eval_pipeline(df, consensus_ranked[:15])
    print(f"  -> [실험 1: Multi-SHAP 컨센서스] Acc: {res_exp1['acc']:.4f} | Prec: {res_exp1['prec']:.4f} | Recall: {res_exp1['rec']:.4f} | F1: {res_exp1['f1']:.4f} | AUC: {res_exp1['auc']:.4f}")

    # 실험 2: Domain Biomarker Features
    print("\n[실험 2 진행 중...] 수면/HRV 생체 비율 변수 추가...")
    df_bio, bio_features = add_domain_biomarkers(df, raw_features)
    r_bio = get_shap_ranked_features(df_bio, bio_features, 'lgb')
    res_exp2 = eval_pipeline(df_bio, r_bio[:15])
    print(f"  -> [실험 2: 생체 비율 지처 추가 ] Acc: {res_exp2['acc']:.4f} | Prec: {res_exp2['prec']:.4f} | Recall: {res_exp2['rec']:.4f} | F1: {res_exp2['f1']:.4f} | AUC: {res_exp2['auc']:.4f}")

    # 실험 3: Isotonic Probability Calibration
    print("\n[실험 3 진행 중...] Isotonic 확률 보정 앙상블...")
    res_exp3 = eval_pipeline(df, r_lgb[:15], calibrate=True)
    print(f"  -> [실험 3: Isotonic 확률 보정  ] Acc: {res_exp3['acc']:.4f} | Prec: {res_exp3['prec']:.4f} | Recall: {res_exp3['rec']:.4f} | F1: {res_exp3['f1']:.4f} | AUC: {res_exp3['auc']:.4f}")

    # 실험 4: Combined Best Techniques (Multi-SHAP + Biomarkers + Calibrated Weights)
    print("\n[실험 4 진행 중...] 시너지 극대화 통합 앙상블...")
    r_bio_lgb = get_shap_ranked_features(df_bio, bio_features, 'lgb')
    r_bio_xgb = get_shap_ranked_features(df_bio, bio_features, 'xgb')
    r_bio_cat = get_shap_ranked_features(df_bio, bio_features, 'cat')
    
    bio_rank_dict = {f: 0 for f in bio_features}
    for f in bio_features:
        bio_rank_dict[f] = r_bio_lgb.index(f) + r_bio_xgb.index(f) + r_bio_cat.index(f)
    bio_consensus = sorted(bio_features, key=lambda x: bio_rank_dict[x])
    
    res_exp4 = eval_pipeline(df_bio, bio_consensus[:18], calibrate=True, weights=(0.35, 0.25, 0.25, 0.15))
    print(f"  -> [실험 4: 통합 최적화 파이프라인] Acc: {res_exp4['acc']:.4f} | Prec: {res_exp4['prec']:.4f} | Recall: {res_exp4['rec']:.4f} | F1: {res_exp4['f1']:.4f} | AUC: {res_exp4['auc']:.4f}")
    
    print("\n" + "="*80)
    print(" 🏆 4개 실험 검증 완료!")
    print("="*80)
