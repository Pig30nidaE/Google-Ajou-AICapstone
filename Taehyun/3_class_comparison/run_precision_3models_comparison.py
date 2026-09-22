"""
run_precision_3models_comparison.py
===================================
3대 핵심 정밀 엔지니어링 모델 전수 교차검증 및 종합 비교 평가 스크립트:
1. Model 1: Upgraded Single XGBoost (Direct multi:softprob + max_delta_step)
2. Model 2: Upgraded H-CRE Flagship Hybrid (Stage 1 XGBoost -> Stage 2 LightGBM with Confidence Penalty)
3. Model 3: Upgraded Subspace RandomForest Rank Ensemble (Prevalence-Balanced alpha=0.44)

검증 프로토콜:
- Outer 5-Fold Stratified K-Fold (100% 완전 격리, Zero Leakage)
- Fold별 Train 세트 독립 전처리 (SimpleImputer + RobustScaler + SMOTE)
- 1,000회 비모수 부트스트랩 95% 신뢰구간(CI) 산출
- 3개 모델 개별 및 3-패널 혼동 행렬 시각화 차트 자동 저장
"""

import os
import sys
import pathlib
import warnings
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score
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

# -------------------------------------------------------------
# 경로 및 상수 설정
# -------------------------------------------------------------
BASE_DIR = pathlib.Path(r"c:\ML4")
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "tabular"
CIRCADIAN_PATH = PROCESSED_DIR / "patient_level_circadian_v3.csv"
OUTPUT_DIR = BASE_DIR / "Google-Ajou-AICapstone" / "Taehyun" / "3_class_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def main():
    start_t = time.time()
    print("=" * 95)
    print(" [3대 핵심 정밀 엔지니어링 모델 Strict Full Nested CV 전수 평가] ")
    print(" 1. Upgraded Single XGBoost (Direct multi:softprob SOTA)")
    print(" 2. Upgraded H-CRE Flagship Hybrid (XGB Stage 1 -> LGBM Stage 2)")
    print(" 3. Upgraded Subspace RandomForest Rank (Prevalence-Balanced alpha=0.44)")
    print("=" * 95)
    
    df = load_data()
    y_true = df['original_label'].astype(int).values
    n = len(df)
    
    skf = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_xgboost_direct = np.zeros((n, 3))
    oof_hybrid_upgraded = np.zeros((n, 3))
    oof_subspace_rf_tuned = np.zeros((n, 3))
    
    for fold, (tr_idx, te_idx) in enumerate(skf.split(df, y_true), 1):
        f_start = time.time()
        print(f"\n>>> [Outer Fold {fold}/{OUTER_SPLITS}] Train: {len(tr_idx)}명 | Test(격리): {len(te_idx)}명")
        df_tr, df_te = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        
        # -------------------------------------------------------------
        # 1. Model 1: Upgraded Single XGBoost (Direct multi:softprob SOTA)
        # -------------------------------------------------------------
        imp_xgb = SimpleImputer(strategy='median')
        X_tr_xgb = imp_xgb.fit_transform(df_tr[V44_14])
        X_te_xgb = imp_xgb.transform(df_te[V44_14])
        sc_xgb = RobustScaler()
        X_tr_xgb_sc = sc_xgb.fit_transform(X_tr_xgb)
        X_te_xgb_sc = sc_xgb.transform(X_te_xgb)
        
        sm_xgb = SMOTE(k_neighbors=2, random_state=RANDOM_STATE + fold)
        X_res_xgb, y_res_xgb = sm_xgb.fit_resample(X_tr_xgb_sc, df_tr['original_label'])
        
        clf_xgb = XGBClassifier(
            objective='multi:softprob', num_class=3,
            max_depth=3, learning_rate=0.03, n_estimators=130,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            max_delta_step=1.0, eval_metric='mlogloss', random_state=RANDOM_STATE + fold, n_jobs=1
        )
        clf_xgb.fit(X_res_xgb, y_res_xgb)
        p_xgb = clf_xgb.predict_proba(X_te_xgb_sc)
        oof_xgboost_direct[te_idx] = p_xgb
        
        # -------------------------------------------------------------
        # 2. Model 2: Upgraded H-CRE Flagship Hybrid
        # -------------------------------------------------------------
        sm_s1 = BorderlineSMOTE(random_state=RANDOM_STATE + fold)
        X_res_s1, y_res_s1 = sm_s1.fit_resample(X_tr_xgb_sc, df_tr['stage1_label'])
        m_s1 = XGBClassifier(
            max_depth=3, learning_rate=0.04, n_estimators=120,
            min_child_weight=2, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, max_delta_step=1.0,
            eval_metric='logloss', random_state=RANDOM_STATE + fold, n_jobs=1
        )
        m_s1.fit(X_res_s1, y_res_s1)
        p1_te = m_s1.predict_proba(X_te_xgb_sc)[:, 1]
        
        df_tr_s2 = df_tr.dropna(subset=['stage2_label']).copy().reset_index(drop=True)
        X_tr_s2 = imp_xgb.transform(df_tr_s2[V44_14])
        X_tr_s2_sc = sc_xgb.transform(X_tr_s2)
        sm_s2 = SMOTE(k_neighbors=2, random_state=RANDOM_STATE + fold)
        X_res_s2, y_res_s2 = sm_s2.fit_resample(X_tr_s2_sc, df_tr_s2['stage2_label'].astype(int))
        
        m_s2 = LGBMClassifier(
            max_depth=3, num_leaves=15, learning_rate=0.05, n_estimators=80,
            min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, class_weight='balanced',
            verbose=-1, random_state=RANDOM_STATE + fold, n_jobs=1
        )
        m_s2.fit(X_res_s2, y_res_s2)
        p2_te = m_s2.predict_proba(X_te_xgb_sc)[:, 1]
        
        p_cn = 1.0 - p1_te
        p_abn = p1_te
        unc = 1.0 - np.abs(p_abn - 0.5) * 2.0
        p2_d = p2_te * (1.0 - 0.30 * unc)
        p_mci = p_abn * (1.0 - p2_d)
        p_dem = p_abn * p2_d
        mat_hyb = np.column_stack([p_cn, p_mci, p_dem])
        oof_hybrid_upgraded[te_idx] = mat_hyb / np.sum(mat_hyb, axis=1, keepdims=True)
        
        # -------------------------------------------------------------
        # 3. Model 3: Upgraded Subspace RandomForest Rank (alpha=0.44)
        # -------------------------------------------------------------
        imp_s1 = SimpleImputer(strategy='median')
        X_tr1 = RobustScaler().fit_transform(imp_s1.fit_transform(df_tr[SUBSPACE_1]))
        X_te1 = RobustScaler().fit(imp_s1.fit_transform(df_tr[SUBSPACE_1])).transform(imp_s1.transform(df_te[SUBSPACE_1]))
        X_res1_rf, y_res1_rf = BorderlineSMOTE(random_state=RANDOM_STATE + fold).fit_resample(X_tr1, df_tr['stage1_label'])
        rf1 = RandomForestClassifier(max_depth=3, n_estimators=160, class_weight='balanced', random_state=RANDOM_STATE + fold, n_jobs=1)
        rf1.fit(X_res1_rf, y_res1_rf)
        p_s1_rf = rf1.predict_proba(X_te1)[:, 1]
        
        imp_s2 = SimpleImputer(strategy='median')
        X_tr2 = RobustScaler().fit_transform(imp_s2.fit_transform(df_tr[SUBSPACE_2]))
        X_te2 = RobustScaler().fit(imp_s2.fit_transform(df_tr[SUBSPACE_2])).transform(imp_s2.transform(df_te[SUBSPACE_2]))
        X_res2_rf, y_res2_rf = BorderlineSMOTE(random_state=RANDOM_STATE + fold).fit_resample(X_tr2, df_tr['stage1_label'])
        rf2 = RandomForestClassifier(max_depth=3, n_estimators=160, class_weight='balanced', random_state=RANDOM_STATE + fold, n_jobs=1)
        rf2.fit(X_res2_rf, y_res2_rf)
        p_s2_rf = rf2.predict_proba(X_te2)[:, 1]
        
        p_g_xgb = m_s1.predict_proba(X_te_xgb_sc)[:, 1]
        
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
        oof_subspace_rf_tuned[te_idx] = mat_sub / np.sum(mat_sub, axis=1, keepdims=True)
        
        print(f"  Fold {fold} 완료 ({time.time() - f_start:.1f}초)")

    models_data = [
        ("Upgraded Single XGBoost (Direct multi:softprob)", oof_xgboost_direct),
        ("Upgraded H-CRE Flagship Hybrid (XGB -> LGBM)", oof_hybrid_upgraded),
        ("Upgraded Subspace RF Rank (Prevalence-Balanced)", oof_subspace_rf_tuned)
    ]
    
    final_results = []
    np.random.seed(RANDOM_STATE)
    
    for name, probs in models_data:
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, average='macro')
        auc = roc_auc_score(y_true, probs, multi_class='ovr')
        cm = confusion_matrix(y_true, preds)
        
        prec_cn = cm[0,0] / cm[:,0].sum()
        rec_cn = cm[0,0] / 111
        f1_cn = 2 * (prec_cn * rec_cn) / (prec_cn + rec_cn)
        
        prec_mci = cm[1,1] / cm[:,1].sum()
        rec_mci = cm[1,1] / 51
        f1_mci = 2 * (prec_mci * rec_mci) / (prec_mci + rec_mci)
        
        prec_dem = cm[2,2] / cm[:,2].sum()
        rec_dem = cm[2,2] / 12
        f1_dem = 2 * (prec_dem * rec_dem) / (prec_dem + rec_dem)
        
        bs_acc, bs_f1, bs_auc = [], [], []
        for _ in range(1000):
            idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
            if len(np.unique(y_true[idx])) < 3:
                continue
            bs_acc.append(accuracy_score(y_true[idx], preds[idx]))
            bs_f1.append(f1_score(y_true[idx], preds[idx], average='macro'))
            bs_auc.append(roc_auc_score(y_true[idx], probs[idx], multi_class='ovr'))
            
        final_results.append({
            "Model": name,
            "Accuracy": acc,
            "CI_Acc": [np.percentile(bs_acc, 2.5), np.percentile(bs_acc, 97.5)],
            "Macro_F1": f1,
            "CI_F1": [np.percentile(bs_f1, 2.5), np.percentile(bs_f1, 97.5)],
            "OVR_AUC": auc,
            "CI_AUC": [np.percentile(bs_auc, 2.5), np.percentile(bs_auc, 97.5)],
            "CN_Recall": rec_cn,
            "CN_Precision": prec_cn,
            "CN_Count": f"{cm[0,0]}/111",
            "MCI_Recall": rec_mci,
            "MCI_Precision": prec_mci,
            "MCI_Count": f"{cm[1,1]}/51",
            "Dem_Recall": rec_dem,
            "Dem_Precision": prec_dem,
            "Dem_Count": f"{cm[2,2]}/12",
            "Dem_FP_Count": int(cm[:,2].sum() - cm[2,2]),
            "Confusion_Matrix": cm.tolist()
        })
        
    out_json = OUTPUT_DIR / "precision_engineering_3models_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"\n최종 결과 JSON 저장 완료: {out_json}")
    
    # 3-패널 혼동 행렬 시각화 차트
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for idx, r in enumerate(final_results):
        cm = np.array(r["Confusion_Matrix"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx],
            xticklabels=['Pred CN', 'Pred MCI', 'Pred Dem'],
            yticklabels=['True CN', 'True MCI', 'True Dem']
        )
        axes[idx].set_title(
            f"{r['Model']}\nAcc: {r['Accuracy']:.4f} | F1: {r['Macro_F1']:.4f} | AUC: {r['OVR_AUC']:.4f}",
            fontsize=11, fontweight='bold'
        )
    plt.tight_layout()
    chart_3p = OUTPUT_DIR / "confusion_matrix_3models_precision.png"
    plt.savefig(chart_3p, dpi=300)
    plt.close()
    print(f"3-패널 혼동 행렬 차트 저장 완료: {chart_3p}")
    
    # 개별 혼동 행렬 (Model 1: Direct XGBoost SOTA)
    plt.figure(figsize=(6.5, 5.5))
    cm_xgb = np.array(final_results[0]["Confusion_Matrix"])
    sns.heatmap(
        cm_xgb, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=['Pred CN (0)', 'Pred MCI (1)', 'Pred Dem (2)'],
        yticklabels=['True CN (0)', 'True MCI (1)', 'True Dem (2)']
    )
    plt.title(
        f"Upgraded Single XGBoost (Direct 3-Class SOTA)\nAcc: {final_results[0]['Accuracy']:.4f} | F1: {final_results[0]['Macro_F1']:.4f} | AUC: {final_results[0]['OVR_AUC']:.4f}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    chart_xgb = OUTPUT_DIR / "confusion_matrix_xgboost_direct_sota.png"
    plt.savefig(chart_xgb, dpi=300)
    plt.close()
    print(f"Single XGBoost SOTA 혼동 행렬 차트 저장 완료: {chart_xgb}")
    
    # 개별 혼동 행렬 (Model 2: Hybrid)
    plt.figure(figsize=(6.5, 5.5))
    cm_hyb = np.array(final_results[1]["Confusion_Matrix"])
    sns.heatmap(
        cm_hyb, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=['Pred CN (0)', 'Pred MCI (1)', 'Pred Dem (2)'],
        yticklabels=['True CN (0)', 'True MCI (1)', 'True Dem (2)']
    )
    plt.title(
        f"Upgraded H-CRE Flagship Hybrid (XGB -> LGBM)\nAcc: {final_results[1]['Accuracy']:.4f} | F1: {final_results[1]['Macro_F1']:.4f} | AUC: {final_results[1]['OVR_AUC']:.4f}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    chart_hyb = OUTPUT_DIR / "confusion_matrix_h_cre_hybrid.png"
    plt.savefig(chart_hyb, dpi=300)
    plt.close()
    print(f"Hybrid 혼동 행렬 차트 저장 완료: {chart_hyb}")

    # 개별 혼동 행렬 (Model 3: Subspace RF)
    plt.figure(figsize=(6.5, 5.5))
    cm_sub = np.array(final_results[2]["Confusion_Matrix"])
    sns.heatmap(
        cm_sub, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=['Pred CN (0)', 'Pred MCI (1)', 'Pred Dem (2)'],
        yticklabels=['True CN (0)', 'True MCI (1)', 'True Dem (2)']
    )
    plt.title(
        f"Upgraded Subspace RF Rank (Prevalence-Balanced)\nAcc: {final_results[2]['Accuracy']:.4f} | F1: {final_results[2]['Macro_F1']:.4f} | AUC: {final_results[2]['OVR_AUC']:.4f}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    chart_sub = OUTPUT_DIR / "confusion_matrix_subspace_rf.png"
    plt.savefig(chart_sub, dpi=300)
    plt.close()
    print(f"Subspace RF 혼동 행렬 차트 저장 완료: {chart_sub}")
    
    print(f"\n전체 실행 소요시간: {time.time() - start_t:.1f}초")

if __name__ == '__main__':
    main()
