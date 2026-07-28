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
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = pathlib.Path(r"c:\ML4")
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "tabular"
PATIENT_PATH = PROCESSED_DIR / "patient_level_all_v2.csv"

TARGET_COL = "label"
DROP_COLS = ["EMAIL", "date", "DIAG_NM", "original_label", TARGET_COL, "fold"]
RANDOM_STATE = 42

current_dir = pathlib.Path(os.getcwd())
sys.path.insert(0, str(current_dir))
try:
    from xai import ShapAnalyzer
except ImportError:
    from xai.analyzer import ShapAnalyzer

def load_preprocessed_data():
    all_df = pd.read_csv(PATIENT_PATH)
    feature_cols = [col for col in all_df.columns if col not in DROP_COLS and pd.api.types.is_numeric_dtype(all_df[col])]
    all_df[feature_cols] = all_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    return all_df.reset_index(drop=True), feature_cols

df, features = load_preprocessed_data()

# 1. SHAP Selection
X = df[features]
y = df[TARGET_COL].astype(int)

smote = SMOTE(random_state=RANDOM_STATE)
X_res, y_res = smote.fit_resample(X, y)

base_model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight='balanced', verbose=-1)
base_model.fit(X_res, y_res)

analyzer = ShapAnalyzer(model=base_model, feature_names=features, task="binary", n_classes=1, class_names=["Dementia"])
analyzer.explain(X_res)
shap_df = analyzer.to_dataframe(combine_classes=False)
ranked_features = shap_df['feature'].tolist()

# V2 Forward Selection Optimal Features = 40
selected_feats = ranked_features[:40]
X_sel = df[selected_feats]

# 2. V2 Best LightGBM Parameters: num_leaves=33, lr=0.08, n_est=1000, min_child_samples=41, max_depth=5, class_weight='balanced'
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

oof_lgb_probs = np.zeros(len(df))
oof_rf_probs = np.zeros(len(df))

for train_idx, test_idx in skf.split(X_sel, y):
    X_tr, y_tr = X_sel.iloc[train_idx], y.iloc[train_idx]
    X_te, y_te = X_sel.iloc[test_idx], y.iloc[test_idx]
    
    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
    
    # LGBM
    model_lgb = LGBMClassifier(
        objective="binary", num_leaves=33, learning_rate=0.08, n_estimators=1000,
        min_child_samples=41, max_depth=5, class_weight='balanced', random_state=42, n_jobs=1, verbose=-1
    )
    model_lgb.fit(X_tr_res, y_tr_res, eval_set=[(X_te, y_te)], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb_probs[test_idx] = model_lgb.predict_proba(X_te)[:, 1]
    
    # RF
    model_rf = RandomForestClassifier(max_depth=10, n_estimators=1000, class_weight='balanced', random_state=42, n_jobs=1)
    model_rf.fit(X_tr_res, y_tr_res)
    oof_rf_probs[test_idx] = model_rf.predict_proba(X_te)[:, 1]

print("==========================================================")
print(" V2 모델 5-Fold 전체 지표 산출 결과 (Out-of-Fold)")
print("==========================================================")

for m_name, probs in [("V2 LightGBM", oof_lgb_probs), ("V2 RandomForest", oof_rf_probs)]:
    auc = roc_auc_score(y, probs)
    
    # Standard threshold 0.5 vs Optimal threshold
    for th in [0.5, 'optimal']:
        if th == 'optimal':
            fpr, tpr, thresholds = roc_curve(y, probs)
            opt_thresh = thresholds[np.argmax(tpr - fpr)]
            preds = np.where(probs >= opt_thresh, 1, 0)
            th_str = f"Opt Thresh ({opt_thresh:.3f})"
        else:
            preds = np.where(probs >= 0.5, 1, 0)
            th_str = "Thresh (0.500)"
            
        acc = accuracy_score(y, preds)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)
        
        print(f"[{m_name:15s} | {th_str:18s}] Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | Macro F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
