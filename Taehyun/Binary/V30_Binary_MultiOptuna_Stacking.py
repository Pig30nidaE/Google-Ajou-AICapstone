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
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

def load_data():
    if not PATIENT_PATH.exists():
        raise FileNotFoundError(f"데이터 파일이 존재하지 않습니다: {PATIENT_PATH}")
    
    df = pd.read_csv(PATIENT_PATH)
    all_feats = [c for c in df.columns if c not in DROP_COLS and pd.api.types.is_numeric_dtype(df[c])]
    df[all_feats] = df[all_feats].replace([np.inf, -np.inf], np.nan)
    return df.reset_index(drop=True), all_feats

def perform_shap_forward_selection(df, features):
    print("\n[단계 1] SHAP 기반 최적 피처 선택...")
    X = df[features]
    y = df[TARGET_COL].astype(int)
    
    smote = BorderlineSMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X, y)
    
    base_model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight='balanced', verbose=-1)
    base_model.fit(X_res, y_res)
    
    analyzer = ShapAnalyzer(model=base_model, feature_names=features, task="binary", n_classes=1, class_names=["Abnormal"])
    analyzer.explain(X_res)
    shap_df = analyzer.to_dataframe(combine_classes=False)
    
    ranked_features = shap_df['feature'].tolist()
    top_k_features = ranked_features[:35]
    
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
                random_state=RANDOM_STATE, n_jobs=1, num_leaves=31, learning_rate=0.06,
                n_estimators=150, min_child_samples=20, class_weight='balanced', verbose=-1
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
            print(f"  -> 상위 {k:2d}개 피처 적용 CV AUC: {mean_auc:.4f} (최고 K={best_k}, AUC={best_cv_score:.4f})")
            
    optimal_features = top_k_features[:best_k]
    print(f"\n[선택 완료] SHAP 최적 피처 {best_k}개 선별 (Best CV AUC: {best_cv_score:.4f})")
    return optimal_features, history_scores

def run_v30_multi_optuna_stacking(df, features):
    print("\n[단계 2] Multi-Optuna Bayesian Search (LightGBM + XGBoost + CatBoost) 진행...")
    X = df[features]
    y = df[TARGET_COL].astype(int)
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    smote = BorderlineSMOTE(random_state=RANDOM_STATE)
    
    # 1. Optuna for LightGBM
    def objective_lgb(trial):
        params = {
            'objective': 'binary',
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 200, 600),
            'min_child_samples': trial.suggest_int('min_child_samples', 15, 50),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            'random_state': RANDOM_STATE, 'n_jobs': 1, 'verbose': -1
        }
        auc_scores = []
        for tr_i, va_i in skf.split(X, y):
            X_tr, y_tr = X.iloc[tr_i], y.iloc[tr_i]
            X_va, y_va = X.iloc[va_i], y.iloc[va_i]
            X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
            m = LGBMClassifier(**params)
            m.fit(X_tr_res, y_tr_res, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(30, verbose=False)])
            auc_scores.append(roc_auc_score(y_va, m.predict_proba(X_va)[:, 1]))
        return np.mean(auc_scores)
        
    # 2. Optuna for XGBoost
    def objective_xgb(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 200, 600),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 2.5),
            'eval_metric': 'auc', 'random_state': RANDOM_STATE, 'n_jobs': 1
        }
        auc_scores = []
        for tr_i, va_i in skf.split(X, y):
            X_tr, y_tr = X.iloc[tr_i], y.iloc[tr_i]
            X_va, y_va = X.iloc[va_i], y.iloc[va_i]
            X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
            m = XGBClassifier(**params)
            m.fit(X_tr_res, y_tr_res, eval_set=[(X_va, y_va)], verbose=False)
            auc_scores.append(roc_auc_score(y_va, m.predict_proba(X_va)[:, 1]))
        return np.mean(auc_scores)

    print("  -> LightGBM Optuna Search 중...")
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(objective_lgb, n_trials=35)
    best_lgb = study_lgb.best_params
    print(f"     [LGBM Best AUC: {study_lgb.best_value:.4f}]")

    print("  -> XGBoost Optuna Search 중...")
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=35)
    best_xgb = study_xgb.best_params
    print(f"     [XGB Best AUC: {study_xgb.best_value:.4f}]")
    
    models = ["LightGBM", "CatBoost", "XGBoost", "RandomForest"]
    oof_predictions = {m: np.zeros(len(df)) for m in models}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
        
        # LightGBM
        m_lgb = LGBMClassifier(**best_lgb, random_state=RANDOM_STATE, n_jobs=1, verbose=-1)
        m_lgb.fit(X_tr_res, y_tr_res, eval_set=[(X_te, y_te)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_predictions["LightGBM"][test_idx] = m_lgb.predict_proba(X_te)[:, 1]
        
        # CatBoost
        m_cat = CatBoostClassifier(
            random_state=RANDOM_STATE, thread_count=1, depth=5, learning_rate=0.04,
            iterations=500, l2_leaf_reg=4.0, auto_class_weights='Balanced', verbose=False
        )
        m_cat.fit(X_tr_res, y_tr_res, eval_set=(X_te, y_te), early_stopping_rounds=50)
        oof_predictions["CatBoost"][test_idx] = m_cat.predict_proba(X_te)[:, 1]
        
        # XGBoost
        m_xgb = XGBClassifier(**best_xgb, random_state=RANDOM_STATE, n_jobs=1)
        m_xgb.fit(X_tr_res, y_tr_res, eval_set=[(X_te, y_te)], verbose=False)
        oof_predictions["XGBoost"][test_idx] = m_xgb.predict_proba(X_te)[:, 1]
        
        # RandomForest
        m_rf = RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=1, max_depth=8, n_estimators=400,
            min_samples_split=4, class_weight='balanced'
        )
        m_rf.fit(X_tr_res, y_tr_res)
        oof_predictions["RandomForest"][test_idx] = m_rf.predict_proba(X_te)[:, 1]
        
    # Meta-Learner Stacking (Logistic Regression)
    OOF_X = np.column_stack([oof_predictions[m] for m in models])
    y_all = y.values
    
    meta_learner = LogisticRegression(C=0.8, penalty='l2', random_state=RANDOM_STATE)
    meta_learner.fit(OOF_X, y_all)
    oof_predictions["Stacking_MetaLearner"] = meta_learner.predict_proba(OOF_X)[:, 1]
    
    # Soft Voting Blend (Optuna LGBM 0.35 + Optuna XGB 0.35 + CatBoost 0.15 + RF 0.15)
    oof_predictions["Weighted_Ensemble"] = (
        oof_predictions["LightGBM"] * 0.35 +
        oof_predictions["XGBoost"] * 0.35 +
        oof_predictions["CatBoost"] * 0.15 +
        oof_predictions["RandomForest"] * 0.15
    )
    
    all_models = models + ["Weighted_Ensemble", "Stacking_MetaLearner"]
    results = {}
    
    print("\n" + "="*85)
    print(" 🔥 V30 이진 분류 최고 성능 결과 (Multi-Optuna + BorderlineSMOTE + Meta-Stacking)")
    print("="*85)
    
    for m in all_models:
        probs = oof_predictions[m]
        auc = roc_auc_score(y_all, probs)
        
        fpr, tpr, thresholds = roc_curve(y_all, probs)
        best_idx = np.argmax(tpr - fpr)
        opt_thresh = thresholds[best_idx]
        
        preds = np.where(probs >= opt_thresh, 1, 0)
        acc = accuracy_score(y_all, preds)
        prec = precision_score(y_all, preds, zero_division=0)
        rec = recall_score(y_all, preds, zero_division=0)
        f1 = f1_score(y_all, preds, zero_division=0)
        cm = confusion_matrix(y_all, preds)
        
        results[m] = {
            "auc": auc, "acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "threshold": opt_thresh, "cm": cm, "probs": probs, "y_true": y_all
        }
        
        prefix = "🏆 [SOTA KING] " if auc >= 0.785 or acc >= 0.76 else "              "
        print(f"{prefix}[{m:20s}] Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f} (Thresh: {opt_thresh:.3f})")
        
    return results

def plot_v30_results(history_scores, optimal_k, results):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(history_scores) + 1), history_scores, marker='o', color='#2b5c8f', linewidth=2)
    plt.axvline(x=optimal_k, color='#e74c3c', linestyle='--', linewidth=2, label=f'Optimal K = {optimal_k}')
    plt.title('V30 Multi-Optuna Forward Selection Curve (5-Fold CV AUC)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Selected Features', fontsize=12)
    plt.ylabel('Mean ROC-AUC Score', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "forward_selection_v30.png", dpi=150)
    plt.close()
    
    plt.figure(figsize=(9, 7))
    for m, r in results.items():
        fpr, tpr, _ = roc_curve(r['y_true'], r['probs'])
        lw = 3 if "Ensemble" in m or "Stacking" in m else 1.5
        ls = '-' if "Ensemble" in m or "Stacking" in m else '--'
        plt.plot(fpr, tpr, label=f"{m} (AUC = {r['auc']:.4f})", linewidth=lw, linestyle=ls)
        
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title('V30 Multi-Optuna & Stacking Meta-Learner ROC Curves', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "roc_curves_v30_binary.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    start_t = datetime.now()
    print("V30 Binary Classification Multi-Optuna + Stacking Execution Started.")
    
    df, raw_features = load_data()
    opt_features, forward_hist = perform_shap_forward_selection(df, raw_features)
    results = run_v30_multi_optuna_stacking(df, opt_features)
    plot_v30_results(forward_hist, len(opt_features), results)
    
    elapsed = datetime.now() - start_t
    print(f"\nCompleted in {elapsed}")
