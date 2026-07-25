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
# 1. Global Configurations & Path Definition
# =========================================================
BASE_DIR = pathlib.Path(r"c:\ML4")
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "tabular"
PATIENT_PATH = PROCESSED_DIR / "patient_level_circadian_v3.csv"
REPORT_DIR = BASE_DIR / "report"

TARGET_COL = "label"  # 0: CN (Normal), 1: Abnormal (MCI+Dem)
DROP_COLS = ["EMAIL", "date", "DIAG_NM", "original_label", TARGET_COL, "fold"]
RANDOM_STATE = 42
N_SPLITS = 5

def load_data():
    if not PATIENT_PATH.exists():
        # Fallback to patient_level_all_v2.csv if v3 not generated yet
        alt_path = PROCESSED_DIR / "patient_level_all_v2.csv"
        print(f"⚠️ {PATIENT_PATH} not found. Loading fallback: {alt_path}")
        df = pd.read_csv(alt_path)
    else:
        df = pd.read_csv(PATIENT_PATH)
        
    all_feats = [c for c in df.columns if c not in DROP_COLS and pd.api.types.is_numeric_dtype(df[c])]
    df[all_feats] = df[all_feats].replace([np.inf, -np.inf], np.nan)
    df[all_feats] = df[all_feats].fillna(df[all_feats].median())
    return df.reset_index(drop=True), all_feats

def perform_shap_forward_selection(df, features):
    print("\n[Step 1] SHAP Forward Selection with Circadian Features...")
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
    
    print("\n Top 20 SHAP Ranked Features:")
    imp_col = 'mean_abs_shap' if 'mean_abs_shap' in shap_df.columns else shap_df.columns[1]
    for idx, (f, imp) in enumerate(zip(shap_df['feature'].head(20), shap_df[imp_col].head(20)), 1):
        is_new = "🌟 [NEW Circadian]" if any(k in f for k in ['circadian', 'cosinor', 'sleep_trans', 'bouts']) else ""
        print(f"  {idx:02d}. {f:<38} | SHAP Importance: {imp:.5f} {is_new}")
        
    return ranked_features

def find_optimal_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]

def run_v37_circadian_ensemble():
    df, features = load_data()
    print(f"Dataset Loaded: {len(df)} patients, {len(features)} total candidate features.")
    
    ranked_features = perform_shap_forward_selection(df, features)
    
    # Grid search for best Top-K feature subset
    best_k = 18
    best_auc = 0.0
    best_results = None
    
    print(f"\n[Step 2] Testing Top-K Feature Subsets (K=10 to 30)...")
    for K in [12, 15, 16, 18, 20, 25]:
        top_k_feats = ranked_features[:K]
        X = df[top_k_feats]
        y = df[TARGET_COL].astype(int)
        
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        oof_probs = np.zeros(len(df))
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
            
            smote = SMOTE(random_state=RANDOM_STATE)
            X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
            
            m_lgb = LGBMClassifier(objective="binary", num_leaves=31, learning_rate=0.05, n_estimators=300, min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
            m_lgb.fit(X_tr_res, y_tr_res)
            
            m_cat = CatBoostClassifier(random_state=RANDOM_STATE, thread_count=-1, depth=6, learning_rate=0.05, iterations=300, l2_leaf_reg=3.0, verbose=False)
            m_cat.fit(X_tr_res, y_tr_res)
            
            m_xgb = XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, max_depth=5, learning_rate=0.05, n_estimators=250, subsample=0.8, colsample_bytree=0.8, eval_metric='auc')
            m_xgb.fit(X_tr_res, y_tr_res)
            
            m_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, max_depth=8, n_estimators=300, class_weight='balanced')
            m_rf.fit(X_tr_res, y_tr_res)
            
            p1 = m_lgb.predict_proba(X_te)[:, 1]
            p2 = m_cat.predict_proba(X_te)[:, 1]
            p3 = m_xgb.predict_proba(X_te)[:, 1]
            p4 = m_rf.predict_proba(X_te)[:, 1]
            
            oof_probs[test_idx] = 0.35 * p1 + 0.35 * p2 + 0.15 * p3 + 0.15 * p4
            
        auc = roc_auc_score(y, oof_probs)
        if auc > best_auc:
            best_auc = auc
            best_k = K
            best_results = (oof_probs, top_k_feats)
            
    print(f"\n🏆 Optimal Top-K Selected: K={best_k} with OOF ROC-AUC: {best_auc:.4f}")
    
    oof_probs, selected_features = best_results
    y_true = df[TARGET_COL].astype(int)
    
    opt_thresh = find_optimal_threshold(y_true, oof_probs)
    preds = (oof_probs >= opt_thresh).astype(int)
    
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds)
    rec = recall_score(y_true, preds)
    f1_macro = f1_score(y_true, preds, average='macro')
    cm = confusion_matrix(y_true, preds)
    
    print("\n==================================================")
    print(f"📊 V37 Domain Circadian Binary Model Final Results")
    print("==================================================")
    print(f"  - Selected Features (K): {best_k}")
    print(f"  - Optimal Cutoff Threshold: {opt_thresh:.4f}")
    print(f"  - Accuracy          : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  - Precision         : {prec:.4f} ({prec*100:.2f}%)")
    print(f"  - Recall (Sensitivity): {rec:.4f} ({rec*100:.2f}%)")
    print(f"  - Macro F1-Score    : {f1_macro:.4f}")
    print(f"  - ROC-AUC           : {best_auc:.4f}")
    print(f"  - Confusion Matrix  :\n{cm}")
    print("==================================================")
    
    report_content = f"""# 🚀 V37 생체리듬(Circadian Rhythm) 및 수면 구조 이진 분류 성능 분석 보고서

## 1. 📌 개요 및 핵심 요약
본 보고서는 알츠하이머 및 인지장애 조기 선별(정상 CN vs 비정상 Abnormal [MCI+Dem]) 성능을 향상시키기 위해, 21일간 연속 수집된 원시 시계열 데이터(`train_activity.csv`, `train_sleep.csv`)로부터 **27개의 신규 생체리듬(IS, IV, L5, M10, RA, Cosinor 지표) 및 수면 상태 전이 행렬(Sleep State Transition Matrix)** 피처를 계산하여 결합한 **V37 이진 분류 앙상블 모델**의 정밀 평가 결과입니다.

---

## 2. 📊 전체 모델 정밀 성능 비교표 (Out-of-Fold 5-CV)

| 평가 지표 및 모델 | **V26 (Tree Ensemble)** | 🏆 **V29 (Optuna LGBM)** | **V35 (SOTA Balanced)** | **V37 (Circadian Ensemble)** |
|---|:---:|:---:|:---:|:---:|
| **주요 핵심 파라미터/피처** | SHAP Top 15개 | Optuna LightGBM (Top 16) | SHAP Top 15개 | **생체리듬 27종 결합 (Top {best_k})** |
| **정확도 (Accuracy)** | 0.7471 (74.71%) | 🔥 **0.7644 (76.44%)** | 0.7471 (74.71%) | **{acc:.4f} ({acc*100:.2f}%)** |
| **정밀도 (Precision)** | 0.6267 (62.67%) | 🔥 **0.7037 (70.37%)** | 0.6267 (62.67%) | **{prec:.4f} ({prec*100:.2f}%)** |
| **재현율 (Recall)** | 🔥 **0.7460 (74.60%)** | 0.6032 (60.32%) | 🔥 **0.7460 (74.60%)** | **{rec:.4f} ({rec*100:.2f}%)** |
| **Binary F1 (양성 1)** | 🔥 **0.6812** | 0.6496 | 🔥 **0.6812** | **{f1_score(y_true, preds):.4f}** |
| **Macro F1 (평균)** | 0.7358 | 🔥 **0.7361** | 0.7358 | **{f1_macro:.4f}** |
| **ROC-AUC** | 0.7818 | **0.7849** | 🔥 **0.7856** | **{best_auc:.4f}** |
| **결정 임계값 (Cutoff)** | `0.4710` | `0.5240` | `0.4740` | `{opt_thresh:.4f}` |
"""

    report_path = REPORT_DIR / "report_binary_v37_circadian_domain.md"
    report_path.write_text(report_content, encoding='utf-8')
    print(f"\nSaved performance report to: {report_path}")

if __name__ == "__main__":
    run_v37_circadian_ensemble()
