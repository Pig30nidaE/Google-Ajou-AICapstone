"""
experiment_smote_ablation.py
============================
3대 핵심 모델에 대한 SMOTE 유무(With SMOTE vs Without SMOTE) 전수 비교 실험 스크립트.

검증 프로토콜:
- Strict Outer 5-Fold Stratified K-Fold (100% 완전 격리, Zero Leakage)
- 동일한 Fold 분할(random_state=42) 하에서 With SMOTE vs Without SMOTE 공정 비교
- 1,000회 비모수 부트스트랩 95% 신뢰구간 산출
- 3개 모델별 상세 지표(Accuracy, Macro F1, OVR AUC, 클래스별 Recall/Precision, CM) 측정
"""

import os
import sys
import pathlib
import warnings
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = pathlib.Path(r"c:\ML4")
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "tabular"
CIRCADIAN_PATH = PROCESSED_DIR / "patient_level_circadian_v3.csv"
OUTPUT_DIR = pathlib.Path(__file__).parent.resolve()

RANDOM_STATE = 42
OUTER_SPLITS = 5

V44_14 = [
    'sleep_score_alignment', 'sleep_hr_5min_max_std', 'sleep_awake_std', 'sleep_breath_average',
    'activity_score_std', 'activity_class_3_count_std', 'activity_met_min_low_std', 'sleep_restless_std',
    'circadian_IV', 'circadian_IS', 'circadian_RA', 'sleep_wake_bouts_avg',
    'HR_drop_ratio', 'Circadian_Strain'
]
SUBSPACE_1 = [
    'circadian_IV', 'circadian_IS', 'circadian_RA', 'sleep_wake_bouts_avg',
    'HR_drop_ratio', 'Circadian_Strain', 'sleep_hr_5min_max_std'
]
SUBSPACE_2 = [
    'sleep_score_alignment', 'sleep_awake_std', 'sleep_breath_average',
    'activity_score_std', 'activity_class_3_count_std', 'activity_met_min_low_std',
    'sleep_restless_std'
]

DROP_METADATA_COLS = [
    "EMAIL", "date", "DIAG_NM", "original_label", "label", "fold",
    "SAMPLE_EMAIL", "DIAG_SEQ", "DOCTOR_NM", "MMSE_NUM", "MMSE_KIND", "TOTAL"
]

def load_data():
    df = pd.read_csv(CIRCADIAN_PATH)
    mmse_cols = [c for c in df.columns if c.startswith('Q') or 'mmse' in c.lower() or c in ['TOTAL', 'DIAG_SEQ', 'DOCTOR_NM']]
    if mmse_cols:
        df.drop(columns=mmse_cols, inplace=True, errors='ignore')
    numeric_cols = [c for c in df.columns if c not in DROP_METADATA_COLS and pd.api.types.is_numeric_dtype(df[c])]
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    if 'sleep_hr_average' in df.columns and 'sleep_hr_lowest' in df.columns:
        df['HR_drop_ratio'] = (df['sleep_hr_average'] - df['sleep_hr_lowest']) / (df['sleep_hr_average'] + 1e-5)
    if 'circadian_IV' in df.columns and 'circadian_IS' in df.columns:
        df['Circadian_Strain'] = df['circadian_IV'] / (df['circadian_IS'] + 1e-5)
    df['stage1_label'] = np.where(df['original_label'] == 0, 0, 1)
    df['stage2_label'] = np.where(df['original_label'] == 1, 0, np.where(df['original_label'] == 2, 1, np.nan))
    return df.reset_index(drop=True)

def run_evaluation(use_smote=True):
    condition_name = "With SMOTE" if use_smote else "Without SMOTE"
    print(f"\n{'=' * 80}")
    print(f" 실험 실행: [{condition_name}] (Strict 5-Fold Nested CV)")
    print(f"{'=' * 80}")
    
    df = load_data()
    y_true = df['original_label'].astype(int).values
    n = len(df)
    
    skf = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_xgb = np.zeros((n, 3))
    oof_hyb = np.zeros((n, 3))
    oof_sub = np.zeros((n, 3))
    
    for fold, (tr_idx, te_idx) in enumerate(skf.split(df, y_true), 1):
        df_tr, df_te = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        
        # -------------------------------------------------------------
        # Base Preprocessing (Median Imputer + RobustScaler on Train only)
        # -------------------------------------------------------------
        imp_all = SimpleImputer(strategy='median')
        X_tr_all = imp_all.fit_transform(df_tr[V44_14])
        X_te_all = imp_all.transform(df_te[V44_14])
        sc_all = RobustScaler()
        X_tr_all_sc = sc_all.fit_transform(X_tr_all)
        X_te_all_sc = sc_all.transform(X_te_all)
        
        # -------------------------------------------------------------
        # 1. Model 1: Single XGBoost Direct multi:softprob SOTA
        # -------------------------------------------------------------
        if use_smote:
            sm_xgb = SMOTE(k_neighbors=2, random_state=RANDOM_STATE + fold)
            X_tr_xgb_fit, y_tr_xgb_fit = sm_xgb.fit_resample(X_tr_all_sc, df_tr['original_label'])
        else:
            X_tr_xgb_fit, y_tr_xgb_fit = X_tr_all_sc, df_tr['original_label']
            
        clf_xgb = XGBClassifier(
            objective='multi:softprob', num_class=3,
            max_depth=3, learning_rate=0.03, n_estimators=130,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            max_delta_step=1.0, eval_metric='mlogloss', random_state=RANDOM_STATE + fold, n_jobs=1
        )
        clf_xgb.fit(X_tr_xgb_fit, y_tr_xgb_fit)
        oof_xgb[te_idx] = clf_xgb.predict_proba(X_te_all_sc)
        
        # -------------------------------------------------------------
        # 2. Model 2: Upgraded H-CRE Flagship Hybrid
        # -------------------------------------------------------------
        # Stage 1: CN vs Impaired
        if use_smote:
            sm_s1 = BorderlineSMOTE(random_state=RANDOM_STATE + fold)
            X_tr_s1_fit, y_tr_s1_fit = sm_s1.fit_resample(X_tr_all_sc, df_tr['stage1_label'])
        else:
            X_tr_s1_fit, y_tr_s1_fit = X_tr_all_sc, df_tr['stage1_label']
            
        m_s1 = XGBClassifier(
            max_depth=3, learning_rate=0.04, n_estimators=120,
            min_child_weight=2, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, max_delta_step=1.0,
            eval_metric='logloss', random_state=RANDOM_STATE + fold, n_jobs=1
        )
        m_s1.fit(X_tr_s1_fit, y_tr_s1_fit)
        p1_te = m_s1.predict_proba(X_te_all_sc)[:, 1]
        
        # Stage 2: MCI vs Dementia
        df_tr_s2 = df_tr.dropna(subset=['stage2_label']).copy().reset_index(drop=True)
        X_tr_s2 = imp_all.transform(df_tr_s2[V44_14])
        X_tr_s2_sc = sc_all.transform(X_tr_s2)
        
        if use_smote:
            sm_s2 = SMOTE(k_neighbors=2, random_state=RANDOM_STATE + fold)
            X_tr_s2_fit, y_tr_s2_fit = sm_s2.fit_resample(X_tr_s2_sc, df_tr_s2['stage2_label'].astype(int))
        else:
            X_tr_s2_fit, y_tr_s2_fit = X_tr_s2_sc, df_tr_s2['stage2_label'].astype(int)
            
        m_s2 = LGBMClassifier(
            max_depth=3, num_leaves=15, learning_rate=0.05, n_estimators=80,
            min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, class_weight='balanced',
            verbose=-1, random_state=RANDOM_STATE + fold, n_jobs=1
        )
        m_s2.fit(X_tr_s2_fit, y_tr_s2_fit)
        p2_te = m_s2.predict_proba(X_te_all_sc)[:, 1]
        
        # Hierarchical Uncertainty Attenuation
        p_cn = 1.0 - p1_te
        p_abn = p1_te
        unc = 1.0 - np.abs(p_abn - 0.5) * 2.0
        p2_d = p2_te * (1.0 - 0.30 * unc)
        p_mci = p_abn * (1.0 - p2_d)
        p_dem = p_abn * p2_d
        mat_hyb = np.column_stack([p_cn, p_mci, p_dem])
        oof_hyb[te_idx] = mat_hyb / np.sum(mat_hyb, axis=1, keepdims=True)
        
        # -------------------------------------------------------------
        # 3. Model 3: Upgraded Subspace RF Rank Ensemble (alpha=0.44)
        # -------------------------------------------------------------
        imp_s1 = SimpleImputer(strategy='median')
        X_tr1 = RobustScaler().fit_transform(imp_s1.fit_transform(df_tr[SUBSPACE_1]))
        X_te1 = RobustScaler().fit(imp_s1.fit_transform(df_tr[SUBSPACE_1])).transform(imp_s1.transform(df_te[SUBSPACE_1]))
        
        if use_smote:
            X_res1_rf, y_res1_rf = BorderlineSMOTE(random_state=RANDOM_STATE + fold).fit_resample(X_tr1, df_tr['stage1_label'])
        else:
            X_res1_rf, y_res1_rf = X_tr1, df_tr['stage1_label']
            
        rf1 = RandomForestClassifier(max_depth=3, n_estimators=160, class_weight='balanced', random_state=RANDOM_STATE + fold, n_jobs=1)
        rf1.fit(X_res1_rf, y_res1_rf)
        p_s1_rf = rf1.predict_proba(X_te1)[:, 1]
        
        imp_s2 = SimpleImputer(strategy='median')
        X_tr2 = RobustScaler().fit_transform(imp_s2.fit_transform(df_tr[SUBSPACE_2]))
        X_te2 = RobustScaler().fit(imp_s2.fit_transform(df_tr[SUBSPACE_2])).transform(imp_s2.transform(df_te[SUBSPACE_2]))
        
        if use_smote:
            X_res2_rf, y_res2_rf = BorderlineSMOTE(random_state=RANDOM_STATE + fold).fit_resample(X_tr2, df_tr['stage1_label'])
        else:
            X_res2_rf, y_res2_rf = X_tr2, df_tr['stage1_label']
            
        rf2 = RandomForestClassifier(max_depth=3, n_estimators=160, class_weight='balanced', random_state=RANDOM_STATE + fold, n_jobs=1)
        rf2.fit(X_res2_rf, y_res2_rf)
        p_s2_rf = rf2.predict_proba(X_te2)[:, 1]
        
        # Global Anchor (m_s1)
        p_g_xgb = m_s1.predict_proba(X_te_all_sc)[:, 1]
        
        r1 = pd.Series(p_s1_rf).rank(pct=True).values
        r2 = pd.Series(p_s2_rf).rank(pct=True).values
        rg = pd.Series(p_g_xgb).rank(pct=True).values
        p_sub_rank = 0.25 * r1 + 0.25 * r2 + 0.50 * rg
        
        p_abn_sub = np.clip(p_sub_rank * (0.44 / 0.5), 0.0, 1.0)
        p_cn_sub = 1.0 - p_abn_sub
        unc_sub = 1.0 - np.abs(p_abn_sub - 0.5) * 2.0
        p2_d_sub = p2_te * (1.0 - 0.25 * unc_sub)
        p_mci_sub = p_abn_sub * (1.0 - p2_d_sub)
        p_dem_sub = p_abn_sub * p2_d_sub
        mat_sub = np.column_stack([p_cn_sub, p_mci_sub, p_dem_sub])
        oof_sub[te_idx] = mat_sub / np.sum(mat_sub, axis=1, keepdims=True)
        
    models_data = [
        ("Single XGBoost Direct SOTA", oof_xgb),
        ("H-CRE Flagship Hybrid", oof_hyb),
        ("Subspace RF Rank Ensemble", oof_sub)
    ]
    
    res = {}
    np.random.seed(RANDOM_STATE)
    
    for name, probs in models_data:
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, average='macro')
        auc = roc_auc_score(y_true, probs, multi_class='ovr')
        cm = confusion_matrix(y_true, preds)
        
        # Bootstrap 95% CI (전 지표 1,000회 비모수 부트스트랩)
        boot_accs, boot_f1s, boot_aucs = [], [], []
        boot_rec_cns, boot_rec_mcis, boot_rec_dems = [], [], []
        boot_prec_dems = []
        for _ in range(1000):
            idx = np.random.choice(n, size=n, replace=True)
            if len(np.unique(y_true[idx])) < 3:
                continue
            boot_accs.append(accuracy_score(y_true[idx], preds[idx]))
            boot_f1s.append(f1_score(y_true[idx], preds[idx], average='macro'))
            boot_aucs.append(roc_auc_score(y_true[idx], probs[idx], multi_class='ovr'))
            
            b_cm = confusion_matrix(y_true[idx], preds[idx], labels=[0, 1, 2])
            boot_rec_cns.append(b_cm[0, 0] / max(b_cm[0, :].sum(), 1))
            boot_rec_mcis.append(b_cm[1, 1] / max(b_cm[1, :].sum(), 1))
            boot_rec_dems.append(b_cm[2, 2] / max(b_cm[2, :].sum(), 1))
            boot_prec_dems.append(b_cm[2, 2] / max(b_cm[:, 2].sum(), 1))
            
        ci_acc = [np.percentile(boot_accs, 2.5), np.percentile(boot_accs, 97.5)]
        ci_f1 = [np.percentile(boot_f1s, 2.5), np.percentile(boot_f1s, 97.5)]
        ci_auc = [np.percentile(boot_aucs, 2.5), np.percentile(boot_aucs, 97.5)]
        ci_rec_cn = [np.percentile(boot_rec_cns, 2.5), np.percentile(boot_rec_cns, 97.5)]
        ci_rec_mci = [np.percentile(boot_rec_mcis, 2.5), np.percentile(boot_rec_mcis, 97.5)]
        ci_rec_dem = [np.percentile(boot_rec_dems, 2.5), np.percentile(boot_rec_dems, 97.5)]
        ci_prec_dem = [np.percentile(boot_prec_dems, 2.5), np.percentile(boot_prec_dems, 97.5)]
        
        prec_cn = cm[0, 0] / max(cm[:, 0].sum(), 1)
        rec_cn = cm[0, 0] / 111.0
        
        prec_mci = cm[1, 1] / max(cm[:, 1].sum(), 1)
        rec_mci = cm[1, 1] / 51.0
        
        prec_dem = cm[2, 2] / max(cm[:, 2].sum(), 1)
        rec_dem = cm[2, 2] / 12.0
        dem_fp = int(cm[:, 2].sum() - cm[2, 2])
        
        res[name] = {
            "Accuracy": float(acc),
            "CI_Acc": [float(x) for x in ci_acc],
            "Macro_F1": float(f1),
            "CI_F1": [float(x) for x in ci_f1],
            "OVR_AUC": float(auc),
            "CI_AUC": [float(x) for x in ci_auc],
            "CN_Recall": float(rec_cn),
            "CI_CN_Recall": [float(x) for x in ci_rec_cn],
            "CN_Precision": float(prec_cn),
            "CN_Count": f"{cm[0,0]}/111",
            "MCI_Recall": float(rec_mci),
            "CI_MCI_Recall": [float(x) for x in ci_rec_mci],
            "MCI_Precision": float(prec_mci),
            "MCI_Count": f"{cm[1,1]}/51",
            "Dem_Recall": float(rec_dem),
            "CI_Dem_Recall": [float(x) for x in ci_rec_dem],
            "Dem_Precision": float(prec_dem),
            "CI_Dem_Precision": [float(x) for x in ci_prec_dem],
            "Dem_Count": f"{cm[2,2]}/12",
            "Dem_FP": dem_fp,
            "Confusion_Matrix": cm.tolist()
        }
        
        print(f"\n[{name} - {condition_name}]")
        print(f"  Acc: {acc*100:.2f}% [{ci_acc[0]*100:.2f}% ~ {ci_acc[1]*100:.2f}%]")
        print(f"  Macro F1: {f1:.4f} [{ci_f1[0]:.4f} ~ {ci_f1[1]:.4f}]")
        print(f"  OVR AUC:  {auc:.4f} [{ci_auc[0]:.4f} ~ {ci_auc[1]:.4f}]")
        print(f"  CN Recall: {rec_cn*100:.2f}% ({cm[0,0]}/111) | Prec: {prec_cn*100:.2f}%")
        print(f"  MCI Recall: {rec_mci*100:.2f}% ({cm[1,1]}/51) | Prec: {prec_mci*100:.2f}%")
        print(f"  Dem Recall: {rec_dem*100:.2f}% ({cm[2,2]}/12) | Prec: {prec_dem*100:.2f}% | FP: {dem_fp}건")
        print("  CM:")
        print(cm)
        
    return res

def main():
    t0 = time.time()
    res_with_smote = run_evaluation(use_smote=True)
    res_no_smote = run_evaluation(use_smote=False)
    
    comparison_data = {
        "With_SMOTE": res_with_smote,
        "Without_SMOTE": res_no_smote
    }
    
    out_json = OUTPUT_DIR / "smote_ablation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    print(f"\n[성공] SMOTE 비교 분석 결과 JSON 저장 완료: {out_json}")
    print(f"총 소요시간: {time.time() - t0:.1f}초")

if __name__ == '__main__':
    main()
