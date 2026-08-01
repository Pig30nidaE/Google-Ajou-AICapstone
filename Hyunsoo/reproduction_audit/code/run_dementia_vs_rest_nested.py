"""
타겟 재구성: CN+MCI(162) vs Dementia(12) 이진분류.
후보 피처는 서캐디안 리듬 지표 5개(IS/IV/RA/M10/L5)로 좁게 제한 (다중비교 위험 최소화).
leak-free nested CV(outer 5-fold, inner 5-fold forward selection)를 동일하게 재사용한다.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_OUTER = 5
N_INNER = 5

DATA_PATH = Path(__file__).parent / "circadian_features.csv"
CANDIDATE_FEATURES = ["IS", "IV", "M10", "L5", "RA"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df.reset_index(drop=True)


def shap_rank(X, y):
    model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight="balanced", verbose=-1,
                            min_child_samples=5, num_leaves=7)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs)
    return [X.columns[i] for i in order]


def inner_select_features(X_tr, y_tr):
    ranked = shap_rank(X_tr[CANDIDATE_FEATURES], y_tr)
    inner_skf = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE)
    best_k, best_auc = 0, -1
    curve = []

    for k in range(1, len(ranked) + 1):
        curr_feats = ranked[:k]
        X_sub = X_tr[curr_feats]
        fold_scores = []
        for in_tr_idx, in_va_idx in inner_skf.split(X_sub, y_tr):
            X_itr, y_itr = X_sub.iloc[in_tr_idx], y_tr[in_tr_idx]
            X_iva, y_iva = X_sub.iloc[in_va_idx], y_tr[in_va_idx]
            if y_iva.sum() == 0 or y_iva.sum() == len(y_iva):
                continue  # AUC undefined if val fold has only one class
            w_itr = compute_sample_weight("balanced", y_itr)
            m = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, num_leaves=7,
                                min_child_samples=5, n_estimators=100, learning_rate=0.08, verbose=-1)
            m.fit(X_itr, y_itr, sample_weight=w_itr)
            prob = m.predict_proba(X_iva)[:, 1]
            fold_scores.append(roc_auc_score(y_iva, prob))
        mean_auc = float(np.mean(fold_scores)) if fold_scores else np.nan
        curve.append(mean_auc)
        if not np.isnan(mean_auc) and mean_auc > best_auc:
            best_auc, best_k = mean_auc, k

    if best_k == 0:
        best_k = 1
    return ranked[:best_k], best_auc, curve, ranked[0]


def train_models(X_tr, y_tr, features):
    w_tr = compute_sample_weight("balanced", y_tr)
    X_tr_f = X_tr[features]

    models = {}
    m_lgb = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, num_leaves=7, min_child_samples=5,
                            n_estimators=300, learning_rate=0.05, verbose=-1)
    m_lgb.fit(X_tr_f, y_tr, sample_weight=w_tr)
    models["LightGBM"] = m_lgb

    m_cat = CatBoostClassifier(random_state=RANDOM_STATE, thread_count=1, depth=3, learning_rate=0.05,
                                iterations=300, l2_leaf_reg=6.0, verbose=False)
    m_cat.fit(X_tr_f, y_tr, sample_weight=w_tr)
    models["CatBoost"] = m_cat

    m_xgb = XGBClassifier(random_state=RANDOM_STATE, n_jobs=1, max_depth=3, learning_rate=0.05,
                           n_estimators=300, subsample=0.8, colsample_bytree=0.8, eval_metric="auc")
    m_xgb.fit(X_tr_f, y_tr, sample_weight=w_tr)
    models["XGBoost"] = m_xgb

    m_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, max_depth=5, n_estimators=500)
    m_rf.fit(X_tr_f, y_tr, sample_weight=w_tr)
    models["RandomForest"] = m_rf

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_f)
    m_lr = LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000)
    m_lr.fit(X_tr_scaled, y_tr)
    models["LogisticRegression"] = (m_lr, scaler)

    return models


def predict_all(models, X_te, features):
    X_te_f = X_te[features]
    out = {}
    out["LightGBM"] = models["LightGBM"].predict_proba(X_te_f)[:, 1]
    out["CatBoost"] = models["CatBoost"].predict_proba(X_te_f)[:, 1]
    out["XGBoost"] = models["XGBoost"].predict_proba(X_te_f)[:, 1]
    out["RandomForest"] = models["RandomForest"].predict_proba(X_te_f)[:, 1]
    m_lr, scaler = models["LogisticRegression"]
    out["LogisticRegression"] = m_lr.predict_proba(scaler.transform(X_te_f))[:, 1]
    out["Ensemble"] = (out["LightGBM"] + out["CatBoost"] + out["XGBoost"] + out["RandomForest"]) / 4.0
    return out


def main():
    df = load_data()
    y_all = df["dementia_label"].astype(int).values
    X_all = df[CANDIDATE_FEATURES]
    print(f"Loaded n={len(df)} (Dementia={y_all.sum()} / CN+MCI={(y_all==0).sum()})")
    print(f"후보 피처(5개, 좁게 제한): {CANDIDATE_FEATURES}")
    print(f"Outer {N_OUTER}-fold / Inner {N_INNER}-fold nested CV\n")

    outer_skf = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE)
    model_names = ["LightGBM", "CatBoost", "XGBoost", "RandomForest", "LogisticRegression", "Ensemble"]
    oof_preds = {m: np.zeros(len(df)) for m in model_names}
    fold_diag = []

    for fold_i, (tr_idx, te_idx) in enumerate(outer_skf.split(X_all, y_all), 1):
        X_tr, X_te = X_all.iloc[tr_idx].reset_index(drop=True), X_all.iloc[te_idx].reset_index(drop=True)
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]
        print(f"--- Outer Fold {fold_i}/{N_OUTER} (train={len(tr_idx)}[{y_tr.sum()} Dem], "
              f"test={len(te_idx)}[{y_te.sum()} Dem]) ---")

        sel_feats, inner_auc, curve, top1 = inner_select_features(X_tr, y_tr)
        print(f"  선택된 피처: {sel_feats} (inner AUC={inner_auc:.4f}, top1='{top1}')")

        models = train_models(X_tr, y_tr, sel_feats)
        probs = predict_all(models, X_te, sel_feats)
        for m in model_names:
            oof_preds[m][te_idx] = probs[m]

        fold_diag.append({"fold": fold_i, "features": sel_feats, "inner_auc": inner_auc,
                           "curve": curve, "top1": top1})

    print("\n" + "=" * 80)
    print(" CN+MCI vs Dementia — Leak-free Nested CV 최종 결과")
    print("=" * 80)

    final_results = {}
    for m in model_names:
        probs = oof_preds[m]
        auc = roc_auc_score(y_all, probs)
        fpr, tpr, thresholds = roc_curve(y_all, probs)
        opt_thresh = thresholds[np.argmax(tpr - fpr)]
        preds = np.where(probs >= opt_thresh, 1, 0)

        final_results[m] = {
            "auc": float(auc),
            "acc": float(accuracy_score(y_all, preds)),
            "prec": float(precision_score(y_all, preds, zero_division=0)),
            "rec": float(recall_score(y_all, preds, zero_division=0)),
            "f1": float(f1_score(y_all, preds, zero_division=0)),
            "threshold": float(opt_thresh),
            "cm": confusion_matrix(y_all, preds).tolist(),
        }
        r = final_results[m]
        print(f"[{m:20s}] Acc:{r['acc']:.4f} Prec:{r['prec']:.4f} Rec(Dem-sens):{r['rec']:.4f} "
              f"F1:{r['f1']:.4f} AUC:{r['auc']:.4f} CM:{r['cm']}")

    out = {"final_results": final_results, "fold_diagnostics": fold_diag}
    out_path = Path(__file__).parent / "dementia_vs_rest_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
