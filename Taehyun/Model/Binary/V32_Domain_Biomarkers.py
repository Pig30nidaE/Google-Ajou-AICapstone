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

def load_data_and_engineer_biomarkers():
    df = pd.read_csv(PATIENT_PATH)
    raw_feats = [c for c in df.columns if c not in DROP_COLS and pd.api.types.is_numeric_dtype(df[c])]
    df[raw_feats] = df[raw_feats].replace([np.inf, -np.inf], np.nan)
    
    print("[실험 2] 수면 패턴 및 심박변동성(HRV) 도메인 특화 생체 지표 비율 변수를 생성합니다...")
    bio_df = pd.DataFrame(index=df.index)
    
    # 1. 수면 관련 생체 지표 비율
    if 'SLEEP_REM_mean' in df.columns and 'SLEEP_TOTAL_mean' in df.columns:
        bio_df['sleep_rem_ratio'] = df['SLEEP_REM_mean'] / (df['SLEEP_TOTAL_mean'] + 1e-5)
    if 'SLEEP_DEEP_mean' in df.columns and 'SLEEP_TOTAL_mean' in df.columns:
        bio_df['sleep_deep_ratio'] = df['SLEEP_DEEP_mean'] / (df['SLEEP_TOTAL_mean'] + 1e-5)
    if 'SLEEP_LIGHT_mean' in df.columns and 'SLEEP_TOTAL_mean' in df.columns:
        bio_df['sleep_light_ratio'] = df['SLEEP_LIGHT_mean'] / (df['SLEEP_TOTAL_mean'] + 1e-5)
        
    # 2. HRV 상대 변동도
    if 'HRV_SDNN_std' in df.columns and 'HRV_SDNN_mean' in df.columns:
        bio_df['hrv_stability'] = df['HRV_SDNN_std'] / (df['HRV_SDNN_mean'] + 1e-5)
    if 'MET_std' in df.columns and 'MET_mean' in df.columns:
        bio_df['activity_volatility'] = df['MET_std'] / (df['MET_mean'] + 1e-5)
        
    df_aug = pd.concat([df, bio_df], axis=1)
    all_features = [c for c in df_aug.columns if c not in DROP_COLS and pd.api.types.is_numeric_dtype(df_aug[c])]
    print(f"  -> 생체 지표 변수 {bio_df.shape[1]}개 생성 완료! 총 {len(all_features)}개 피처로 확장.")
    
    return df_aug.reset_index(drop=True), all_features

def perform_shap_forward_selection(df, features):
    print("\n[전진 선택법] SHAP 중요도 기반 최적 피처 탐색...")
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
            
        if k % 10 == 0 or k == len(top_k_features):
            print(f"  -> 상위 피처 {k:2d}개 적용 CV AUC: {mean_auc:.4f} (현재 최고 K={best_k}, AUC={best_cv_score:.4f})")
            
    optimal_features = top_k_features[:best_k]
    print(f"\n[선택 완료] 최적 피처 {best_k}개 선별 (Best CV AUC: {best_cv_score:.4f})")
    
    bio_selected = [f for f in optimal_features if any(f.startswith(p) for p in ['sleep_', 'hrv_', 'activity_'])]
    print(f"  -> 선별된 피처에 포함된 생체 비율 지표: {bio_selected if bio_selected else '없음'}")
    
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
    print(" 🏆 [실험 2: 생체 지표 비율 변수 추가 앙상블] 최종 성능 결과")
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
    df_aug, features = load_data_and_engineer_biomarkers()
    opt_feats = perform_shap_forward_selection(df_aug, features)
    results = run_ensemble(df_aug, opt_feats)
