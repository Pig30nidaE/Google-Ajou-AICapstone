"""
진짜 Nested CV: outer-test 데이터는 SMOTE/가중치, SHAP 랭킹, forward-selection(k 선택),
모델 학습 그 어느 단계에도 노출되지 않는다. 오직 최종 예측(outer-test 추론)에만 사용된다.

기존 파이프라인(V26/V29 및 이전 재현 스크립트)의 데이터 누수:
  1. SHAP 피처 랭킹을 SMOTE-resample된 "전체" 데이터로 미리 계산
     -> outer CV split이 일어나기 전에 이미 전체 라벨 정보가 랭킹에 반영됨.
  2. forward selection으로 k를 고를 때 쓰는 CV split과 최종 성능 보고에 쓰는 CV split이
     완전히 동일(random_state=42 재사용) -> "검증 fold"가 그대로 "최종 성능"으로 재활용.

수정된 구조:
  Outer 5-Fold (진짜 held-out test)
    -> outer-train 안에서만: SHAP 랭킹 -> inner 5-Fold로 forward selection(k 선택)
    -> outer-train 전체로 최종 4개 모델 학습 (class-balanced sample_weight)
    -> outer-test 예측 (최초로 한 번 노출)
  전체 outer-test 예측을 모아 최종 OOF 성능 산출 + fold별 선택 피처 안정성 기록
"""
from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_OUTER = 5
N_INNER = 5
TARGET_COL = "label"
DROP_COLS = ["EMAIL", "date", "DIAG_NM", "original_label", TARGET_COL, "fold"]
FORWARD_MAX_FEATURES = 40

DATA_PATH = Path(__file__).parent / "patient_level_all_v2.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    all_feats = [c for c in df.columns if c not in DROP_COLS and pd.api.types.is_numeric_dtype(df[c])]
    df[all_feats] = df[all_feats].replace([np.inf, -np.inf], np.nan)
    return df.reset_index(drop=True), all_feats


def shap_rank_features(X, y):
    model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight="balanced", verbose=-1)
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


def inner_select_features(X_tr, y_tr, features):
    """outer-train 내부에서만 수행되는 SHAP 랭킹 + inner CV forward selection."""
    ranked = shap_rank_features(X_tr[features], y_tr)
    top_k = ranked[:FORWARD_MAX_FEATURES]

    inner_skf = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE)
    best_k, best_auc = 0, -1
    curve = []

    for k in range(1, len(top_k) + 1):
        curr_feats = top_k[:k]
        X_sub = X_tr[curr_feats]
        fold_scores = []
        for in_tr_idx, in_va_idx in inner_skf.split(X_sub, y_tr):
            X_itr, y_itr = X_sub.iloc[in_tr_idx], y_tr[in_tr_idx]
            X_iva, y_iva = X_sub.iloc[in_va_idx], y_tr[in_va_idx]
            w_itr = compute_sample_weight("balanced", y_itr)

            m = LGBMClassifier(
                random_state=RANDOM_STATE, n_jobs=1, num_leaves=33, learning_rate=0.08,
                n_estimators=120, min_child_samples=15, verbose=-1,
            )
            m.fit(X_itr, y_itr, sample_weight=w_itr, eval_set=[(X_iva, y_iva)],
                  callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
            prob = m.predict_proba(X_iva)[:, 1]
            fold_scores.append(roc_auc_score(y_iva, prob))

        mean_auc = float(np.mean(fold_scores))
        curve.append(mean_auc)
        if mean_auc > best_auc:
            best_auc, best_k = mean_auc, k

    return top_k[:best_k], best_auc, curve, top_k[0]


def train_final_ensemble(X_tr, y_tr, features):
    w_tr = compute_sample_weight("balanced", y_tr)
    X_tr_f = X_tr[features]

    m_lgb = LGBMClassifier(
        objective="binary", num_leaves=33, learning_rate=0.08, n_estimators=1000,
        min_child_samples=41, max_depth=5, random_state=RANDOM_STATE, n_jobs=1, verbose=-1,
    )
    m_lgb.fit(X_tr_f, y_tr, sample_weight=w_tr)

    m_cat = CatBoostClassifier(
        random_state=RANDOM_STATE, thread_count=1, depth=5, learning_rate=0.05,
        iterations=500, l2_leaf_reg=4.0, verbose=False,
    )
    m_cat.fit(X_tr_f, y_tr, sample_weight=w_tr)

    m_xgb = XGBClassifier(
        random_state=RANDOM_STATE, n_jobs=1, max_depth=4, learning_rate=0.05,
        n_estimators=400, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=1.5,
        eval_metric="auc",
    )
    m_xgb.fit(X_tr_f, y_tr, sample_weight=w_tr)

    m_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, max_depth=10, n_estimators=1000)
    m_rf.fit(X_tr_f, y_tr, sample_weight=w_tr)

    return {"LightGBM": m_lgb, "CatBoost": m_cat, "XGBoost": m_xgb, "RandomForest": m_rf}


def predict_ensemble(models, X_te, features):
    X_te_f = X_te[features]
    prob_lgb = models["LightGBM"].predict_proba(X_te_f)[:, 1]
    prob_cat = models["CatBoost"].predict_proba(X_te_f)[:, 1]
    prob_xgb = models["XGBoost"].predict_proba(X_te_f)[:, 1]
    prob_rf = models["RandomForest"].predict_proba(X_te_f)[:, 1]
    prob_ens = prob_lgb * 0.40 + prob_cat * 0.20 + prob_xgb * 0.20 + prob_rf * 0.20
    return {"LightGBM": prob_lgb, "CatBoost": prob_cat, "XGBoost": prob_xgb,
            "RandomForest": prob_rf, "Ensemble": prob_ens}


def main():
    df, raw_features = load_data()
    y_all = df[TARGET_COL].astype(int).values
    y_orig_all = df["original_label"].astype(int).values
    X_all = df[raw_features]

    print(f"Loaded n={len(df)} patients ({(y_orig_all==0).sum()} CN / {(y_orig_all==1).sum()} MCI / {(y_orig_all==2).sum()} Dem)")
    print(f"Outer {N_OUTER}-fold / Inner {N_INNER}-fold nested CV (완전 분리된 leak-free forward selection)\n")

    outer_skf = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE)

    model_names = ["LightGBM", "CatBoost", "XGBoost", "RandomForest", "Ensemble"]
    oof_preds = {m: np.zeros(len(df)) for m in model_names}
    fold_diagnostics = []
    feature_selection_counter = Counter()
    top1_feature_counter = Counter()

    for fold_i, (tr_idx, te_idx) in enumerate(outer_skf.split(X_all, y_all), 1):
        X_tr, X_te = X_all.iloc[tr_idx].reset_index(drop=True), X_all.iloc[te_idx].reset_index(drop=True)
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

        print(f"--- Outer Fold {fold_i}/{N_OUTER} (train={len(tr_idx)}, test={len(te_idx)}) ---")
        selected_features, inner_auc, curve, top1 = inner_select_features(X_tr, y_tr, raw_features)
        print(f"  inner CV로 선택된 피처 수: {len(selected_features)} (inner best AUC={inner_auc:.4f}, "
              f"k=1일 때 AUC={curve[0]:.4f}, top1 feature='{top1}')")

        feature_selection_counter.update(selected_features)
        top1_feature_counter[top1] += 1

        models = train_final_ensemble(X_tr, y_tr, selected_features)
        probs = predict_ensemble(models, X_te, selected_features)
        for m in model_names:
            oof_preds[m][te_idx] = probs[m]

        fold_diagnostics.append({
            "fold": fold_i, "n_features": len(selected_features), "features": selected_features,
            "inner_best_auc": inner_auc, "k1_auc": curve[0], "top1_feature": top1,
            "auc_curve": curve,
        })

    print("\n" + "=" * 80)
    print(" Leak-free Nested CV 최종 결과 (outer-test는 오직 예측에만 사용됨)")
    print("=" * 80)

    dem_mask = y_orig_all == 2
    mci_mask = y_orig_all == 1
    cn_mask = y_orig_all == 0

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
            "dem_recall": float(preds[dem_mask].mean()),
            "mci_recall": float(preds[mci_mask].mean()),
            "cn_specificity": float((preds[cn_mask] == 0).mean()),
        }
        r = final_results[m]
        print(f"[{m:12s}] Acc:{r['acc']:.4f} Prec:{r['prec']:.4f} Rec:{r['rec']:.4f} F1:{r['f1']:.4f} "
              f"AUC:{r['auc']:.4f} | Dem-Rec:{r['dem_recall']:.4f} MCI-Rec:{r['mci_recall']:.4f} CN-Spec:{r['cn_specificity']:.4f}")

    print("\n[피처 선택 안정성] 5개 outer fold 중 몇 번 선택됐는지 (내림차순 상위 20개):")
    for feat, cnt in feature_selection_counter.most_common(20):
        print(f"  {cnt}/5  {feat}")

    print("\n[fold별 k=1 SHAP 최상위 피처]:")
    for cnt, feat in sorted(((c, f) for f, c in top1_feature_counter.items()), reverse=True):
        print(f"  {cnt}/5  {feat}")

    out = {
        "final_results": final_results,
        "fold_diagnostics": fold_diagnostics,
        "feature_selection_counts": dict(feature_selection_counter.most_common()),
        "top1_feature_counts": dict(top1_feature_counter),
    }
    out_path = Path(__file__).parent / "nested_cv_leakfree_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
