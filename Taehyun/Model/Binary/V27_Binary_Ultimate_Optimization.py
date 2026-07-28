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
from scipy.optimize import minimize

import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

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

def perform_shap_forward_selection(df, features):
    print("\n[단계 1] SHAP 중요도 기준 Forward Selection 탐색...")
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
                random_state=RANDOM_STATE, n_jobs=1, num_leaves=31, learning_rate=0.05,
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
            print(f"  -> 상위 {k:2d}개 피처 적용 CV AUC: {mean_auc:.4f} (현재 최고 K={best_k}, AUC={best_cv_score:.4f})")
            
    optimal_features = top_k_features[:best_k]
    print(f"\n[선택 완료] 최적 피처 {best_k}개 선별 (Best CV AUC: {best_cv_score:.4f})")
    return optimal_features, history_scores

def run_ultimate_binary_ensemble(df, features):
    print("\n[단계 2] 5종 앙상블 (LightGBM, CatBoost, XGBoost, RF, ExtraTrees) 및 가중치 최적화 진행...")
    X = df[features]
    y = df[TARGET_COL].astype(int)
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    smote = SMOTE(random_state=RANDOM_STATE)
    
    models = ["LightGBM", "CatBoost", "XGBoost", "RandomForest", "ExtraTrees"]
    oof_probs = {m: np.zeros(len(df)) for m in models}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
        
        # 1. LightGBM
        m_lgb = LGBMClassifier(
            objective="binary", num_leaves=33, learning_rate=0.06, n_estimators=500,
            min_child_samples=25, max_depth=5, class_weight='balanced',
            reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=1, verbose=-1
        )
        m_lgb.fit(X_tr_res, y_tr_res, eval_set=[(X_te, y_te)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_probs["LightGBM"][test_idx] = m_lgb.predict_proba(X_te)[:, 1]
        
        # 2. CatBoost
        m_cat = CatBoostClassifier(
            random_state=RANDOM_STATE, thread_count=1, depth=5, learning_rate=0.04,
            iterations=500, l2_leaf_reg=4.0, auto_class_weights='Balanced', verbose=False
        )
        m_cat.fit(X_tr_res, y_tr_res, eval_set=(X_te, y_te), early_stopping_rounds=50)
        oof_probs["CatBoost"][test_idx] = m_cat.predict_proba(X_te)[:, 1]
        
        # 3. XGBoost
        m_xgb = XGBClassifier(
            random_state=RANDOM_STATE, n_jobs=1, max_depth=4, learning_rate=0.04,
            n_estimators=400, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=1.5,
            eval_metric='auc', early_stopping_rounds=50
        )
        m_xgb.fit(X_tr_res, y_tr_res, eval_set=[(X_te, y_te)], verbose=False)
        oof_probs["XGBoost"][test_idx] = m_xgb.predict_proba(X_te)[:, 1]
        
        # 4. RandomForest
        m_rf = RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=1, max_depth=8, n_estimators=500,
            min_samples_split=4, class_weight='balanced'
        )
        m_rf.fit(X_tr_res, y_tr_res)
        oof_probs["RandomForest"][test_idx] = m_rf.predict_proba(X_te)[:, 1]
        
        # 5. ExtraTrees
        m_et = ExtraTreesClassifier(
            random_state=RANDOM_STATE, n_jobs=1, max_depth=8, n_estimators=500,
            min_samples_split=4, class_weight='balanced'
        )
        m_et.fit(X_tr_res, y_tr_res)
        oof_probs["ExtraTrees"][test_idx] = m_et.predict_proba(X_te)[:, 1]
        
    y_true_all = y.values
    
    # 앙상블 가중치 최적화 (Scipy minimize)
    def loss_func(weights):
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        ens_prob = sum(w * oof_probs[m] for w, m in zip(weights, models))
        return -roc_auc_score(y_true_all, ens_prob)
        
    init_weights = [0.25, 0.25, 0.25, 0.125, 0.125]
    bounds = [(0, 1)] * len(models)
    opt_res = minimize(loss_func, init_weights, bounds=bounds, method='SLSQP')
    best_weights = opt_res.x / np.sum(opt_res.x)
    
    print("\n[최적 가중치 탐색 완료]")
    for m, w in zip(models, best_weights):
        print(f"  -> {m:12s} Weight: {w:.4f}")
        
    oof_probs["Ensemble"] = sum(w * oof_probs[m] for w, m in zip(best_weights, models))
    
    all_models = models + ["Ensemble"]
    results = {}
    
    print("\n" + "="*78)
    print(" 🏆 V27 궁극의 이진 분류 최적화 최종 성능 (5-Fold Out-of-Fold CV)")
    print("="*78)
    
    for m in all_models:
        probs = oof_probs[m]
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
        
        prefix = "🔥 [ULTIMATE SOTA] " if m == "Ensemble" or auc >= 0.78 else "                 "
        print(f"{prefix}[{m:12s}] Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f} (Thresh: {opt_thresh:.3f})")
        
    return results, best_weights

def plot_results(history_scores, optimal_k, results):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(history_scores) + 1), history_scores, marker='o', color='#2b5c8f', linewidth=2)
    plt.axvline(x=optimal_k, color='#e74c3c', linestyle='--', linewidth=2, label=f'Optimal K = {optimal_k}')
    plt.title('V27 SHAP Forward Feature Selection Curve (5-Fold CV AUC)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Selected Features', fontsize=12)
    plt.ylabel('Mean ROC-AUC Score', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "forward_selection_v27.png", dpi=150)
    plt.show()
    
    fig, axes = plt.subplots(1, 6, figsize=(26, 4.5))
    class_names = ['Normal(CN)', 'Abnormal']
    
    for ax, (m, r) in zip(axes, results.items()):
        sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 14}, ax=ax)
        ax.set_title(f"[{m}]\nAUC: {r['auc']:.4f} | Acc: {r['acc']:.4f}", fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix_v27_binary.png", dpi=150)
    plt.show()
    
    plt.figure(figsize=(9, 7))
    for m, r in results.items():
        fpr, tpr, _ = roc_curve(r['y_true'], r['probs'])
        lw = 3 if m == "Ensemble" else 1.5
        ls = '-' if m == "Ensemble" else '--'
        plt.plot(fpr, tpr, label=f"{m} (AUC = {r['auc']:.4f})", linewidth=lw, linestyle=ls)
        
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title('V27 Ultimate Binary Classification ROC Curves', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "roc_curves_v27_binary.png", dpi=150)
    plt.show()
    
    print(f"\n[시각화 완료] {PLOT_DIR} 에 시각화 파일들이 생성되었습니다.")

if __name__ == "__main__":
    start_t = datetime.now()
    print("V27 Binary Classification Ultimate Optimization Execution Started.")
    
    df, raw_features = load_data()
    opt_features, forward_hist = perform_shap_forward_selection(df, raw_features)
    results, weights = run_ultimate_binary_ensemble(df, opt_features)
    plot_results(forward_hist, len(opt_features), results)
    
    elapsed = datetime.now() - start_t
    print(f"\nCompleted in {elapsed}")
