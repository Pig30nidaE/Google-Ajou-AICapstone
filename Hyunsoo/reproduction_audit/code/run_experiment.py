"""
V26 (SHAP Forward Selection + SOTA Soft-Voting Ensemble)와
V29 (SHAP Forward Selection + Optuna + Stacking Meta-Learner)의
로직을 노트북과 동일한 하이퍼파라미터로 재현하여,
nia+219@rowan.kr 포함/제외 시 성능을 비교한다.

xai.ShapAnalyzer 커스텀 패키지는 로컬에 없으므로, 표준 shap 라이브러리의
TreeExplainer로 대체한다 (mean(|SHAP value|) 내림차순 랭킹 - 통상적인 SHAP
importance 랭킹과 동일한 방식).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
import optuna
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
N_SPLITS = 5
TARGET_COL = "label"
DROP_COLS = ["EMAIL", "date", "DIAG_NM", "original_label", TARGET_COL, "fold"]

DATA_PATH = Path(__file__).parent / "patient_level_all_v2.csv"
EXCLUDE_EMAIL = "nia+219@rowan.kr"


def load_data(exclude: bool) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(DATA_PATH)
    if exclude:
        before = len(df)
        df = df.loc[df["EMAIL"] != EXCLUDE_EMAIL].reset_index(drop=True)
        assert len(df) == before - 1, "exclusion did not remove exactly one row"
    all_feats = [c for c in df.columns if c not in DROP_COLS and pd.api.types.is_numeric_dtype(df[c])]
    df[all_feats] = df[all_feats].replace([np.inf, -np.inf], np.nan)
    return df.reset_index(drop=True), all_feats


def shap_rank_features(X_res: pd.DataFrame, y_res: np.ndarray, features: list[str]) -> list[str]:
    model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight="balanced", verbose=-1)
    model.fit(X_res, y_res)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_res)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs)
    return [features[i] for i in order]


def compute_metrics(y_true, probs) -> dict:
    auc = roc_auc_score(y_true, probs)
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    best_idx = np.argmax(tpr - fpr)
    opt_thresh = thresholds[best_idx]
    preds = np.where(probs >= opt_thresh, 1, 0)
    return {
        "auc": float(auc),
        "acc": float(accuracy_score(y_true, preds)),
        "prec": float(precision_score(y_true, preds, zero_division=0)),
        "rec": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "threshold": float(opt_thresh),
        "cm": confusion_matrix(y_true, preds).tolist(),
    }


# ---------------------------------------------------------------------------
# V26 : SHAP Forward Selection (top 40) + SOTA soft-voting ensemble
# ---------------------------------------------------------------------------

def v26_forward_selection(df, features):
    X = df[features]
    y = df[TARGET_COL].astype(int)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X, y)

    ranked_features = shap_rank_features(X_res, y_res, features)
    top_k_features = ranked_features[:40]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    best_k, best_cv_score = 0, -1
    history = []

    for k in range(1, len(top_k_features) + 1):
        curr_feats = top_k_features[:k]
        X_sub = X[curr_feats]
        fold_scores = []
        for train_idx, val_idx in skf.split(X_sub, y):
            X_tr, y_tr = X_sub.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X_sub.iloc[val_idx], y.iloc[val_idx]
            X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

            m = LGBMClassifier(
                random_state=RANDOM_STATE, n_jobs=1, num_leaves=33, learning_rate=0.08,
                n_estimators=120, min_child_samples=15, class_weight="balanced", verbose=-1,
            )
            m.fit(X_tr_res, y_tr_res, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
            prob = m.predict_proba(X_va)[:, 1]
            fold_scores.append(roc_auc_score(y_va, prob))

        mean_auc = float(np.mean(fold_scores))
        history.append(mean_auc)
        if mean_auc > best_cv_score:
            best_cv_score, best_k = mean_auc, k

    optimal_features = top_k_features[:best_k]
    return optimal_features, history, best_cv_score


def v26_run_ensemble(df, features):
    X = df[features]
    y = df[TARGET_COL].astype(int)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    smote = SMOTE(random_state=RANDOM_STATE)

    models = ["LightGBM", "CatBoost", "XGBoost", "RandomForest", "Ensemble"]
    fold_predictions = {m: [] for m in models}
    fold_y_true = []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

        m_lgb = LGBMClassifier(
            objective="binary", num_leaves=33, learning_rate=0.08, n_estimators=1000,
            min_child_samples=41, max_depth=5, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=1, verbose=-1,
        )
        m_lgb.fit(X_tr_res, y_tr_res, eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        prob_lgb = m_lgb.predict_proba(X_te)[:, 1]

        m_cat = CatBoostClassifier(
            random_state=RANDOM_STATE, thread_count=1, depth=5, learning_rate=0.05,
            iterations=500, l2_leaf_reg=4.0, auto_class_weights="Balanced", verbose=False,
        )
        m_cat.fit(X_tr_res, y_tr_res, eval_set=(X_te, y_te), early_stopping_rounds=50, verbose=False)
        prob_cat = m_cat.predict_proba(X_te)[:, 1]

        m_xgb = XGBClassifier(
            random_state=RANDOM_STATE, n_jobs=1, max_depth=4, learning_rate=0.05,
            n_estimators=400, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=1.5,
            eval_metric="auc", early_stopping_rounds=50,
        )
        m_xgb.fit(X_tr_res, y_tr_res, eval_set=[(X_te, y_te)], verbose=False)
        prob_xgb = m_xgb.predict_proba(X_te)[:, 1]

        m_rf = RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=1, max_depth=10, n_estimators=1000, class_weight="balanced",
        )
        m_rf.fit(X_tr_res, y_tr_res)
        prob_rf = m_rf.predict_proba(X_te)[:, 1]

        prob_ens = prob_lgb * 0.40 + prob_cat * 0.20 + prob_xgb * 0.20 + prob_rf * 0.20

        fold_predictions["LightGBM"].extend(prob_lgb)
        fold_predictions["CatBoost"].extend(prob_cat)
        fold_predictions["XGBoost"].extend(prob_xgb)
        fold_predictions["RandomForest"].extend(prob_rf)
        fold_predictions["Ensemble"].extend(prob_ens)
        fold_y_true.extend(y_te)

    y_true_all = np.array(fold_y_true)
    results = {}
    for m in models:
        probs = np.array(fold_predictions[m])
        results[m] = compute_metrics(y_true_all, probs)
    return results


def run_v26(df, features, tag):
    print(f"\n### [V26 | {tag}] SHAP forward selection 시작 (n={len(df)}) ###")
    opt_features, history, best_fs_auc = v26_forward_selection(df, features)
    print(f"  -> 선택된 피처 수: {len(opt_features)} (Forward-selection best CV AUC={best_fs_auc:.4f})")
    results = v26_run_ensemble(df, opt_features)
    for m, r in results.items():
        print(f"  [{m:12s}] Acc:{r['acc']:.4f} Prec:{r['prec']:.4f} Rec:{r['rec']:.4f} F1:{r['f1']:.4f} AUC:{r['auc']:.4f}")
    return {"n_features": len(opt_features), "forward_selection_best_auc": best_fs_auc, "results": results}


# ---------------------------------------------------------------------------
# V29 : SHAP Forward Selection (top 35) + Optuna LGBM + Stacking meta-learner
# ---------------------------------------------------------------------------

def v29_forward_selection(df, features):
    X = df[features]
    y = df[TARGET_COL].astype(int)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X, y)

    ranked_features = shap_rank_features(X_res, y_res, features)
    top_k_features = ranked_features[:35]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    best_k, best_cv_score = 0, -1
    history = []

    for k in range(1, len(top_k_features) + 1):
        curr_feats = top_k_features[:k]
        X_sub = X[curr_feats]
        fold_scores = []
        for train_idx, val_idx in skf.split(X_sub, y):
            X_tr, y_tr = X_sub.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X_sub.iloc[val_idx], y.iloc[val_idx]
            X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

            m = LGBMClassifier(
                random_state=RANDOM_STATE, n_jobs=1, num_leaves=31, learning_rate=0.06,
                n_estimators=150, min_child_samples=20, class_weight="balanced", verbose=-1,
            )
            m.fit(X_tr_res, y_tr_res, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
            prob = m.predict_proba(X_va)[:, 1]
            fold_scores.append(roc_auc_score(y_va, prob))

        mean_auc = float(np.mean(fold_scores))
        history.append(mean_auc)
        if mean_auc > best_cv_score:
            best_cv_score, best_k = mean_auc, k

    optimal_features = top_k_features[:best_k]
    return optimal_features, history, best_cv_score


def v29_optimize_and_run(df, features):
    X = df[features]
    y = df[TARGET_COL].astype(int)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    smote = BorderlineSMOTE(random_state=RANDOM_STATE)

    def objective_lgb(trial):
        params = {
            "objective": "binary",
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "random_state": RANDOM_STATE, "n_jobs": 1, "verbose": -1,
        }
        auc_scores = []
        for tr_i, va_i in skf.split(X, y):
            X_tr, y_tr = X.iloc[tr_i], y.iloc[tr_i]
            X_va, y_va = X.iloc[va_i], y.iloc[va_i]
            X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
            m = LGBMClassifier(**params)
            m.fit(X_tr_res, y_tr_res, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
            prob = m.predict_proba(X_va)[:, 1]
            auc_scores.append(roc_auc_score(y_va, prob))
        return np.mean(auc_scores)

    study_lgb = optuna.create_study(direction="maximize")
    study_lgb.optimize(objective_lgb, n_trials=30)
    best_lgb_params = study_lgb.best_params

    models = ["LightGBM", "CatBoost", "XGBoost", "RandomForest"]
    oof_predictions = {m: np.zeros(len(df)) for m in models}

    for train_idx, test_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

        m_lgb = LGBMClassifier(**best_lgb_params, random_state=RANDOM_STATE, n_jobs=1, verbose=-1)
        m_lgb.fit(X_tr_res, y_tr_res, eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof_predictions["LightGBM"][test_idx] = m_lgb.predict_proba(X_te)[:, 1]

        m_cat = CatBoostClassifier(
            random_state=RANDOM_STATE, thread_count=1, depth=5, learning_rate=0.04,
            iterations=500, l2_leaf_reg=4.0, auto_class_weights="Balanced", verbose=False,
        )
        m_cat.fit(X_tr_res, y_tr_res, eval_set=(X_te, y_te), early_stopping_rounds=50, verbose=False)
        oof_predictions["CatBoost"][test_idx] = m_cat.predict_proba(X_te)[:, 1]

        m_xgb = XGBClassifier(
            random_state=RANDOM_STATE, n_jobs=1, max_depth=4, learning_rate=0.04,
            n_estimators=400, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=1.5,
            eval_metric="auc", early_stopping_rounds=50,
        )
        m_xgb.fit(X_tr_res, y_tr_res, eval_set=[(X_te, y_te)], verbose=False)
        oof_predictions["XGBoost"][test_idx] = m_xgb.predict_proba(X_te)[:, 1]

        m_rf = RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=1, max_depth=8, n_estimators=400,
            min_samples_split=4, class_weight="balanced",
        )
        m_rf.fit(X_tr_res, y_tr_res)
        oof_predictions["RandomForest"][test_idx] = m_rf.predict_proba(X_te)[:, 1]

    OOF_X = np.column_stack([oof_predictions[m] for m in models])
    y_all = y.values

    meta_learner = LogisticRegression(C=0.5, penalty="l2", random_state=RANDOM_STATE)
    meta_learner.fit(OOF_X, y_all)
    stacking_probs = meta_learner.predict_proba(OOF_X)[:, 1]
    oof_predictions["Stacking_MetaLearner"] = stacking_probs

    all_models = models + ["Stacking_MetaLearner"]
    results = {}
    for m in all_models:
        probs = oof_predictions[m]
        results[m] = compute_metrics(y_all, probs)
    return results, study_lgb.best_value


def run_v29(df, features, tag):
    print(f"\n### [V29 | {tag}] SHAP forward selection 시작 (n={len(df)}) ###")
    opt_features, history, best_fs_auc = v29_forward_selection(df, features)
    print(f"  -> 선택된 피처 수: {len(opt_features)} (Forward-selection best CV AUC={best_fs_auc:.4f})")
    results, lgb_best_auc = v29_optimize_and_run(df, opt_features)
    print(f"  -> Optuna LightGBM best AUC: {lgb_best_auc:.4f}")
    for m, r in results.items():
        print(f"  [{m:20s}] Acc:{r['acc']:.4f} Prec:{r['prec']:.4f} Rec:{r['rec']:.4f} F1:{r['f1']:.4f} AUC:{r['auc']:.4f}")
    return {"n_features": len(opt_features), "forward_selection_best_auc": best_fs_auc,
            "optuna_lgb_best_auc": lgb_best_auc, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["v26", "v29", "both"], default="both")
    parser.add_argument("--exclude", action="store_true", help="exclude nia+219@rowan.kr")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    tag = "EXCLUDED nia+219" if args.exclude else "FULL (baseline)"
    df, features = load_data(exclude=args.exclude)
    print(f"Loaded n={len(df)} patients, {len(features)} raw features. Tag={tag}")

    output = {"tag": tag, "n_patients": len(df), "exclude": args.exclude}

    if args.which in ("v26", "both"):
        output["v26"] = run_v26(df, features, tag)
    if args.which in ("v29", "both"):
        output["v29"] = run_v29(df, features, tag)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
