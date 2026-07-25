import os
import sys
import pathlib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Windows CP949 인코딩 문제 방지
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

# xai 패키지 경로 추가
current_dir = pathlib.Path(os.getcwd())
sys.path.insert(0, str(current_dir))
try:
    from xai import ShapAnalyzer
except ImportError:
    from xai.analyzer import ShapAnalyzer

# 한글 폰트 설정
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

# =========================================================
# 2. 데이터 로드
# =========================================================
def load_data():
    if not PATIENT_PATH.exists():
        raise FileNotFoundError(f"데이터 파일이 존재하지 않습니다: {PATIENT_PATH}")
    
    df = pd.read_csv(PATIENT_PATH)
    all_feats = [c for c in df.columns if c not in DROP_COLS and pd.api.types.is_numeric_dtype(df[c])]
    df[all_feats] = df[all_feats].replace([np.inf, -np.inf], np.nan)
    return df.reset_index(drop=True), all_feats

# =========================================================
# 3. SHAP 기반 Forward Selection
# =========================================================
def perform_shap_forward_selection(df, features):
    print("\n[단계 1] SHAP 패키지를 활용한 랭킹 계산 및 Forward Feature Selection 탐색...")
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
                random_state=RANDOM_STATE,
                n_jobs=1,
                num_leaves=33,
                learning_rate=0.08,
                n_estimators=120,
                min_child_samples=15,
                class_weight='balanced',
                verbose=-1
            )
            eval_model.fit(
                X_tr_res, y_tr_res,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(30, verbose=False)]
            )
            prob = eval_model.predict_proba(X_va)[:, 1]
            fold_scores.append(roc_auc_score(y_va, prob))
            
        mean_auc = np.mean(fold_scores)
        history_scores.append(mean_auc)
        
        if mean_auc > best_cv_score:
            best_cv_score = mean_auc
            best_k = k
            
        if k % 10 == 0 or k == len(top_k_features):
            print(f"  -> SHAP 상위 피처 {k:2d}개 적용 시 CV AUC: {mean_auc:.4f} (현재 최고: K={best_k}, AUC={best_cv_score:.4f})")
            
    optimal_features = top_k_features[:best_k]
    print(f"\n[SHAP 피처 탐색 완료] 최적 피처 개수: {best_k}개 (Best CV AUC: {best_cv_score:.4f})")
    return optimal_features, history_scores

# =========================================================
# 4. SOTA 앙상블 모델 학습 및 Grid Search
# =========================================================
def run_sota_binary_ensemble(df, features):
    print("\n[단계 2] 4개 시그니처 알고리즘 (LightGBM, CatBoost, XGBoost, RF) 최적화 5-Fold 학습...")
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
        
        # 1. LightGBM (V2 SOTA Hyperparameters)
        m_lgb = LGBMClassifier(
            objective="binary",
            num_leaves=33,
            learning_rate=0.08,
            n_estimators=1000,
            min_child_samples=41,
            max_depth=5,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=-1
        )
        m_lgb.fit(X_tr_res, y_tr_res, eval_set=[(X_te, y_te)], callbacks=[lgb.early_stopping(50, verbose=False)])
        prob_lgb = m_lgb.predict_proba(X_te)[:, 1]
        
        # 2. CatBoost (Regulated)
        m_cat = CatBoostClassifier(
            random_state=RANDOM_STATE, thread_count=1,
            depth=5, learning_rate=0.05, iterations=500,
            l2_leaf_reg=4.0, auto_class_weights='Balanced', verbose=False
        )
        m_cat.fit(X_tr_res, y_tr_res, eval_set=(X_te, y_te), early_stopping_rounds=50)
        prob_cat = m_cat.predict_proba(X_te)[:, 1]
        
        # 3. XGBoost
        m_xgb = XGBClassifier(
            random_state=RANDOM_STATE, n_jobs=1,
            max_depth=4, learning_rate=0.05, n_estimators=400,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=1.5,
            eval_metric='auc', early_stopping_rounds=50
        )
        m_xgb.fit(X_tr_res, y_tr_res, eval_set=[(X_te, y_te)], verbose=False)
        prob_xgb = m_xgb.predict_proba(X_te)[:, 1]
        
        # 4. RandomForest (V2 SOTA Params)
        m_rf = RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=1,
            max_depth=10, n_estimators=1000, class_weight='balanced'
        )
        m_rf.fit(X_tr_res, y_tr_res)
        prob_rf = m_rf.predict_proba(X_te)[:, 1]
        
        # 5. Soft Voting Ensemble (LightGBM 40% + CatBoost 20% + XGBoost 20% + RF 20%)
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
    print(" V26 이진 분류 SOTA 성능 평가 결과 (5-Fold Out-of-Fold Cross Validation)")
    print("="*75)
    
    for m in models:
        probs = np.array(fold_predictions[m])
        auc = roc_auc_score(y_true_all, probs)
        
        # Youden's J index로 최적 임계값 탐색
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
        
        prefix = "[SOTA] " if auc >= 0.77 or acc >= 0.77 else "       "
        print(f"{prefix}[{m:12s}] Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f} (Thresh: {opt_thresh:.3f})")
        
    return results

# =========================================================
# 5. 시각화 그래프 생성
# =========================================================
def plot_results(history_scores, optimal_k, results):
    # 1. Forward Selection Curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(history_scores) + 1), history_scores, marker='o', color='#2b5c8f', linewidth=2)
    plt.axvline(x=optimal_k, color='#e74c3c', linestyle='--', linewidth=2, label=f'Optimal K = {optimal_k}')
    plt.title('V26 SHAP Forward Selection Curve (5-Fold CV AUC)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Selected Features', fontsize=12)
    plt.ylabel('Mean ROC-AUC Score', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "forward_selection_v26.png", dpi=150)
    plt.close()
    
    # 2. Confusion Matrix Heatmaps
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    class_names = ['Normal(CN)', 'Abnormal']
    
    for ax, (m, r) in zip(axes, results.items()):
        sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 14}, ax=ax)
        ax.set_title(f"[{m}]\nAUC: {r['auc']:.4f} | Acc: {r['acc']:.4f}", fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix_v26_binary.png", dpi=150)
    plt.close()
    
    # 3. ROC Curves
    plt.figure(figsize=(9, 7))
    for m, r in results.items():
        fpr, tpr, _ = roc_curve(r['y_true'], r['probs'])
        lw = 3 if m == "Ensemble" else 1.5
        ls = '-' if m == "Ensemble" else '--'
        plt.plot(fpr, tpr, label=f"{m} (AUC = {r['auc']:.4f})", linewidth=lw, linestyle=ls)
        
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title('V26 Binary Classification ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "roc_curves_v26_binary.png", dpi=150)
    plt.close()
    
    print(f"\n[시각화 저장 완료] 결과 그래프가 {PLOT_DIR} 에 성공적으로 저장되었습니다.")

# =========================================================
# 6. 메인 실행
# =========================================================
if __name__ == "__main__":
    start_t = datetime.now()
    print("V26 Binary Classification Optimization Execution Started.")
    
    # 1. 로드
    df, raw_features = load_data()
    
    # 2. SHAP 피처 선택
    opt_features, forward_hist = perform_shap_forward_selection(df, raw_features)
    
    # 3. SOTA 앙상블 모델 학습 & 평가
    results = run_sota_binary_ensemble(df, opt_features)
    
    # 4. 결과 시각화
    plot_results(forward_hist, len(opt_features), results)
    
    elapsed = datetime.now() - start_t
    print(f"\nCompleted in {elapsed}")
