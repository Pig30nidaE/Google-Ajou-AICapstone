import os
import sys
import pathlib
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, brier_score_loss
)
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
CURRENT_FILE_DIR = pathlib.Path(__file__).resolve().parent
TAEHYUN_DIR = CURRENT_FILE_DIR.parent
DATA_DIR = TAEHYUN_DIR / "data"

if not DATA_DIR.exists():
    if (pathlib.Path.cwd().parent / "data").exists():
        DATA_DIR = pathlib.Path.cwd().parent / "data"
    elif pathlib.Path(r"c:\ML4\Google-Ajou-AICapstone\Taehyun\data").exists():
        DATA_DIR = pathlib.Path(r"c:\ML4\Google-Ajou-AICapstone\Taehyun\data")
    else:
        DATA_DIR = pathlib.Path("./data")

CIRCADIAN_PATH = DATA_DIR / "patient_level_circadian_v3.csv"
PLOT_DIR = CURRENT_FILE_DIR / "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

TARGET_COL = "label"

DROP_COLS = [
    "EMAIL", "date", "DIAG_NM", "original_label", TARGET_COL, "fold", 
    "SAMPLE_EMAIL", "DIAG_SEQ", "DOCTOR_NM", "MMSE_NUM", "MMSE_KIND",
    "TOTAL"
]

RANDOM_STATE = 42
OUTER_SPLITS = 5
INNER_SPLITS = 5
BOOTSTRAP_ROUNDS = 1000

# [전체 14개 핵심 도메인 피처]
ALL_14_FEATURES = [
    'sleep_score_alignment',          # 수면 시간대 규칙성
    'sleep_hr_5min_max_std',          # 수면 중 최대 심박수 변동성
    'sleep_awake_std',                # 수면 중 각성 시간 변동성
    'sleep_breath_average',           # 수면 평균 호흡수
    'activity_score_std',             # 활동 점수 변동성
    'activity_class_3_count_std',     # 중강도 활동 빈도 변동성
    'activity_met_min_low_std',       # 저강도 활동 대사량(MET) 변동성
    'sleep_restless_std',             # 수면 중 뒤척임 변동성
    'circadian_IV',                   # 일내 분절화 지수 (Intradaily Variability)
    'circadian_IS',                   # 일간 안정성 지수 (Interdaily Stability)
    'circadian_RA',                   # 상대 진폭 지수 (Relative Amplitude)
    'sleep_wake_bouts_avg',           # 수면 중 미세 각성 빈도
    'HR_drop_ratio',                  # [Kim & Park 2026] 자율신경 회복력 (야간 심박 하강률)
    'Circadian_Strain'                # [2026 SOTA] 일주기 리듬 파괴 스트레스 (IV / IS)
]

# [서브스페이스 1: 일주기 생체 시계 및 자율신경계 전문 모델용 (7종)]
SUBSPACE_CIRCADIAN_AUTONOMIC = [
    'circadian_IV', 'circadian_IS', 'circadian_RA', 'sleep_wake_bouts_avg',
    'HR_drop_ratio', 'Circadian_Strain', 'sleep_hr_5min_max_std'
]

# [서브스페이스 2: 야간 수면 구조 및 주간 활동 라이프로그 전문 모델용 (7종)]
SUBSPACE_SLEEP_ACTIVITY = [
    'sleep_score_alignment', 'sleep_awake_std', 'sleep_breath_average',
    'activity_score_std', 'activity_class_3_count_std', 'activity_met_min_low_std', 'sleep_restless_std'
]


def load_dataset():
    df = pd.read_csv(CIRCADIAN_PATH)
    
    # 1. MMSE 제거
    mmse_cols = [c for c in df.columns if c.startswith('Q') or 'mmse' in c.lower() or c in ['TOTAL', 'DIAG_SEQ', 'DOCTOR_NM']]
    if mmse_cols:
        df.drop(columns=mmse_cols, inplace=True, errors='ignore')
        
    raw_numeric = [c for c in df.columns if c not in DROP_COLS and pd.api.types.is_numeric_dtype(df[c])]
    df[raw_numeric] = df[raw_numeric].replace([np.inf, -np.inf], np.nan)
    
    # 2. 핵심 파생 피처
    if 'sleep_hr_average' in df.columns and 'sleep_hr_lowest' in df.columns:
        df['HR_drop_ratio'] = (df['sleep_hr_average'] - df['sleep_hr_lowest']) / (df['sleep_hr_average'] + 1e-5)
    if 'circadian_IV' in df.columns and 'circadian_IS' in df.columns:
        df['Circadian_Strain'] = df['circadian_IV'] / (df['circadian_IS'] + 1e-5)
        
    return df.reset_index(drop=True)


def get_adaptive_cv_splitter(df, n_splits=5, random_state=42, group_col='EMAIL', split_name="Outer"):
    n_rows = len(df)
    n_unique = df[group_col].nunique() if group_col in df.columns else n_rows
    if n_unique == n_rows:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state), None, "StratifiedKFold"
    else:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state), df[group_col].values, "StratifiedGroupKFold"


def optimize_pareto_thresholds(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
    # Tier 2: Youden's J (Balanced Diagnosis)
    best_idx = np.argmax(tpr - fpr)
    th_diag = thresholds[best_idx]
    
    # Tier 1: Ultra-Early Screening (Recall >= 0.75)
    idx_scr = np.where(tpr >= 0.75)[0]
    th_screen = thresholds[idx_scr[0]] if len(idx_scr) > 0 else thresholds[np.argmax(tpr - fpr)]
    
    # Tier 3: High-Specificity Confirmation (Specificity >= 0.80)
    idx_conf = np.where((1 - fpr) >= 0.80)[0]
    th_conf = thresholds[idx_conf[-1]] if len(idx_conf) > 0 else thresholds[np.argmax(tpr - fpr)]
    
    return {
        "tier1_screen": th_screen,
        "tier2_diag": th_diag,
        "tier3_conf": th_conf
    }


def fit_predict_subspace_pipeline(X_tr_df, y_tr, X_te_df, random_state=42, use_smote=True):
    """
    서브스페이스 특화 모델 분해 학습 및 예측 파이프라인 (SMOTE 적용 여부 제어)
    """
    # [1. Subspace 1: Circadian-Autonomic Specialist CatBoost]
    imp1 = SimpleImputer(strategy='median')
    X_tr1 = imp1.fit_transform(X_tr_df[SUBSPACE_CIRCADIAN_AUTONOMIC])
    X_te1 = imp1.transform(X_te_df[SUBSPACE_CIRCADIAN_AUTONOMIC])
    sc1 = RobustScaler()
    X_tr1_sc = sc1.fit_transform(X_tr1)
    X_te1_sc = sc1.transform(X_te1)
    
    if use_smote:
        smote1 = BorderlineSMOTE(random_state=random_state)
        X_tr1_res, y_tr1_res = smote1.fit_resample(X_tr1_sc, y_tr)
    else:
        X_tr1_res, y_tr1_res = X_tr1_sc, y_tr
        
    cb_circ = CatBoostClassifier(
        depth=3, l2_leaf_reg=6.0, learning_rate=0.04, iterations=130, 
        auto_class_weights='Balanced', verbose=False, random_state=random_state, thread_count=1
    )
    cb_circ.fit(X_tr1_res, y_tr1_res)
    p_cb_circ = cb_circ.predict_proba(X_te1_sc)[:, 1]
    
    # [2. Subspace 2: Sleep-Activity Specialist LightGBM]
    imp2 = SimpleImputer(strategy='median')
    X_tr2 = imp2.fit_transform(X_tr_df[SUBSPACE_SLEEP_ACTIVITY])
    X_te2 = imp2.transform(X_te_df[SUBSPACE_SLEEP_ACTIVITY])
    sc2 = RobustScaler()
    X_tr2_sc = sc2.fit_transform(X_tr2)
    X_te2_sc = sc2.transform(X_te2)
    
    if use_smote:
        smote2 = BorderlineSMOTE(random_state=random_state)
        X_tr2_res, y_tr2_res = smote2.fit_resample(X_tr2_sc, y_tr)
    else:
        X_tr2_res, y_tr2_res = X_tr2_sc, y_tr
        
    lgb_slp = LGBMClassifier(
        max_depth=3, num_leaves=7, learning_rate=0.04, n_estimators=110, 
        reg_alpha=0.2, reg_lambda=0.2, min_child_samples=18, class_weight='balanced', 
        verbose=-1, random_state=random_state, n_jobs=1
    )
    lgb_slp.fit(X_tr2_res, y_tr2_res)
    p_lgb_slp = lgb_slp.predict_proba(X_te2_sc)[:, 1]
    
    # [3. Global Cross-Domain All 14 Features Models]
    imp_all = SimpleImputer(strategy='median')
    X_tr_all = imp_all.fit_transform(X_tr_df[ALL_14_FEATURES])
    X_te_all = imp_all.transform(X_te_df[ALL_14_FEATURES])
    sc_all = RobustScaler()
    X_tr_all_sc = sc_all.fit_transform(X_tr_all)
    X_te_all_sc = sc_all.transform(X_te_all)
    
    if use_smote:
        smote_all = BorderlineSMOTE(random_state=random_state)
        X_tr_all_res, y_tr_all_res = smote_all.fit_resample(X_tr_all_sc, y_tr)
    else:
        X_tr_all_res, y_tr_all_res = X_tr_all_sc, y_tr
    
    # CatBoost All
    cb_all = CatBoostClassifier(
        depth=3, l2_leaf_reg=4.0, learning_rate=0.035, iterations=140, 
        auto_class_weights='Balanced', verbose=False, random_state=random_state, thread_count=1
    )
    cb_all.fit(X_tr_all_res, y_tr_all_res)
    p_cb_all = cb_all.predict_proba(X_te_all_sc)[:, 1]
    
    # XGBoost All
    xgb_all = XGBClassifier(
        max_depth=3, learning_rate=0.04, n_estimators=100, subsample=0.8, 
        colsample_bytree=0.8, eval_metric='auc', random_state=random_state, n_jobs=1
    )
    xgb_all.fit(X_tr_all_res, y_tr_all_res)
    p_xgb_all = xgb_all.predict_proba(X_te_all_sc)[:, 1]
    
    # RBF SVM All
    svm_all = SVC(
        C=1.0, kernel='rbf', probability=True, class_weight='balanced', random_state=random_state
    )
    svm_all.fit(X_tr_all_res, y_tr_all_res)
    p_svm_all = svm_all.predict_proba(X_te_all_sc)[:, 1]
    
    # RandomForest All
    rf_all = RandomForestClassifier(
        max_depth=3, n_estimators=180, max_features='sqrt', min_samples_leaf=2, 
        class_weight='balanced', random_state=random_state, n_jobs=1
    )
    rf_all.fit(X_tr_all_res, y_tr_all_res)
    p_rf_all = rf_all.predict_proba(X_te_all_sc)[:, 1]
    
    probs_dict = {
        "CatBoost_Circadian_Specialist": p_cb_circ,
        "LightGBM_Sleep_Specialist": p_lgb_slp,
        "CatBoost_Global": p_cb_all,
        "XGBoost_Global": p_xgb_all,
        "RBF_SVM_Global": p_svm_all,
        "RandomForest_Global": p_rf_all
    }
    return probs_dict


def compute_bootstrap_ci(y_true, y_probs, y_preds, n_bootstraps=BOOTSTRAP_ROUNDS, random_state=42):
    """
    95% 신뢰구간 (95% Confidence Interval, Percentile Bootstrap) 산출
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    
    boot_metrics = {
        'auc': [], 'acc': [], 'prec': [], 'rec': [], 'spec': [], 'f1': []
    }
    
    for _ in range(n_bootstraps):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
            
        y_t_b = y_true[idx]
        y_prob_b = y_probs[idx]
        y_pred_b = y_preds[idx]
        
        auc = roc_auc_score(y_t_b, y_prob_b)
        acc = accuracy_score(y_t_b, y_pred_b)
        prec = precision_score(y_t_b, y_pred_b, zero_division=0)
        rec = recall_score(y_t_b, y_pred_b, zero_division=0)
        f1 = f1_score(y_t_b, y_pred_b, zero_division=0)
        cm = confusion_matrix(y_t_b, y_pred_b, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        boot_metrics['auc'].append(auc)
        boot_metrics['acc'].append(acc)
        boot_metrics['prec'].append(prec)
        boot_metrics['rec'].append(rec)
        boot_metrics['spec'].append(spec)
        boot_metrics['f1'].append(f1)
        
    ci_dict = {}
    for k in boot_metrics:
        arr = np.array(boot_metrics[k])
        ci_dict[k] = (np.percentile(arr, 2.5), np.percentile(arr, 97.5))
    return ci_dict


def run_nested_cv(use_smote=True):
    smote_label = "With BorderlineSMOTE" if use_smote else "Without SMOTE (Baseline)"
    print(f"\n{'='*80}")
    print(f" >>> Nested CV 평가 시작: [{smote_label}]")
    print(f"{'='*80}")
    
    df = load_dataset()
    X = df[ALL_14_FEATURES].copy()
    y = df[TARGET_COL].astype(int).copy()
    
    outer_splitter, outer_groups, _ = get_adaptive_cv_splitter(
        df, n_splits=OUTER_SPLITS, random_state=RANDOM_STATE, split_name="Outer"
    )
    
    base_names = [
        "CatBoost_Circadian_Specialist", "LightGBM_Sleep_Specialist", 
        "CatBoost_Global", "XGBoost_Global", "RBF_SVM_Global", "RandomForest_Global"
    ]
    ensemble_names = [
        "V44_Subspace_Decomposed_Rank_Ensemble",
        "V44_Subspace_Soft_Ensemble",
        "Stacking_MetaLearner"
    ]
    all_names = base_names + ensemble_names
    
    outer_oof_probs = {m: np.zeros(len(df)) for m in all_names}
    outer_oof_preds_t1 = {m: np.zeros(len(df)) for m in all_names}
    outer_oof_preds_t2 = {m: np.zeros(len(df)) for m in all_names}
    outer_oof_preds_t3 = {m: np.zeros(len(df)) for m in all_names}
    
    splits = outer_splitter.split(X, y, groups=outer_groups) if outer_groups is not None else outer_splitter.split(X, y)
    
    for outer_fold, (train_idx, test_idx) in enumerate(splits, 1):
        df_outer_tr = df.iloc[train_idx].reset_index(drop=True)
        X_tr, y_tr = X.iloc[train_idx].reset_index(drop=True), y.iloc[train_idx].reset_index(drop=True)
        X_te, y_te = X.iloc[test_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)
        
        # [Inner Loop: 5-Fold CV for Subspace Models & Thresholds]
        inner_splitter, inner_groups, _ = get_adaptive_cv_splitter(
            df_outer_tr, n_splits=INNER_SPLITS, random_state=RANDOM_STATE + outer_fold, split_name="Inner"
        )
        inner_oof_probs = {m: np.zeros(len(X_tr)) for m in base_names}
        in_splits = inner_splitter.split(X_tr, y_tr, groups=inner_groups) if inner_groups is not None else inner_splitter.split(X_tr, y_tr)
        
        for in_tr_idx, in_va_idx in in_splits:
            X_in_tr, y_in_tr = X_tr.iloc[in_tr_idx].copy(), y_tr.iloc[in_tr_idx].copy()
            X_in_va = X_tr.iloc[in_va_idx].copy()
            
            in_preds_dict = fit_predict_subspace_pipeline(
                X_in_tr, y_in_tr, X_in_va, random_state=RANDOM_STATE, use_smote=use_smote
            )
            for m in base_names:
                inner_oof_probs[m][in_va_idx] = in_preds_dict[m]
                
        # Inner Ensembles
        ranks_in = np.column_stack([pd.Series(inner_oof_probs[m]).rank(pct=True).values for m in base_names])
        p_sub_rank_in = (
            0.25 * ranks_in[:, 0] +   # CatBoost Circadian Specialist
            0.15 * ranks_in[:, 1] +   # LightGBM Sleep Specialist
            0.25 * ranks_in[:, 2] +   # CatBoost Global
            0.20 * ranks_in[:, 3] +   # XGBoost Global
            0.10 * ranks_in[:, 4] +   # RBF SVM Global
            0.05 * ranks_in[:, 5]     # RandomForest Global
        )
        
        p_sub_soft_in = (
            0.25 * inner_oof_probs["CatBoost_Circadian_Specialist"] + 
            0.15 * inner_oof_probs["LightGBM_Sleep_Specialist"] + 
            0.25 * inner_oof_probs["CatBoost_Global"] + 
            0.20 * inner_oof_probs["XGBoost_Global"] + 
            0.10 * inner_oof_probs["RBF_SVM_Global"] + 
            0.05 * inner_oof_probs["RandomForest_Global"]
        )
        
        inner_mat = np.column_stack([inner_oof_probs[m] for m in base_names])
        meta_lr = LogisticRegression(class_weight='balanced', C=0.5, penalty='l2', random_state=RANDOM_STATE)
        meta_lr.fit(inner_mat, y_tr)
        p_meta_in = meta_lr.predict_proba(inner_mat)[:, 1]
        
        # Inner Thresholds
        inner_th = {}
        for m in base_names:
            inner_th[m] = optimize_pareto_thresholds(y_tr.values, inner_oof_probs[m])
        inner_th["V44_Subspace_Decomposed_Rank_Ensemble"] = optimize_pareto_thresholds(y_tr.values, p_sub_rank_in)
        inner_th["V44_Subspace_Soft_Ensemble"] = optimize_pareto_thresholds(y_tr.values, p_sub_soft_in)
        inner_th["Stacking_MetaLearner"] = optimize_pareto_thresholds(y_tr.values, p_meta_in)
        
        # [Outer Fit & Predict on Outer Test]
        out_preds_dict = fit_predict_subspace_pipeline(
            X_tr, y_tr, X_te, random_state=RANDOM_STATE, use_smote=use_smote
        )
        for m in base_names:
            prob_te = out_preds_dict[m]
            outer_oof_probs[m][test_idx] = prob_te
            outer_oof_preds_t1[m][test_idx] = np.where(prob_te >= inner_th[m]["tier1_screen"], 1, 0)
            outer_oof_preds_t2[m][test_idx] = np.where(prob_te >= inner_th[m]["tier2_diag"], 1, 0)
            outer_oof_preds_t3[m][test_idx] = np.where(prob_te >= inner_th[m]["tier3_conf"], 1, 0)
            
        ranks_te = np.column_stack([pd.Series(out_preds_dict[m]).rank(pct=True).values for m in base_names])
        p_sub_rank_te = (
            0.25 * ranks_te[:, 0] + 
            0.15 * ranks_te[:, 1] + 
            0.25 * ranks_te[:, 2] + 
            0.20 * ranks_te[:, 3] + 
            0.10 * ranks_te[:, 4] + 
            0.05 * ranks_te[:, 5]
        )
        outer_oof_probs["V44_Subspace_Decomposed_Rank_Ensemble"][test_idx] = p_sub_rank_te
        outer_oof_preds_t1["V44_Subspace_Decomposed_Rank_Ensemble"][test_idx] = np.where(p_sub_rank_te >= inner_th["V44_Subspace_Decomposed_Rank_Ensemble"]["tier1_screen"], 1, 0)
        outer_oof_preds_t2["V44_Subspace_Decomposed_Rank_Ensemble"][test_idx] = np.where(p_sub_rank_te >= inner_th["V44_Subspace_Decomposed_Rank_Ensemble"]["tier2_diag"], 1, 0)
        outer_oof_preds_t3["V44_Subspace_Decomposed_Rank_Ensemble"][test_idx] = np.where(p_sub_rank_te >= inner_th["V44_Subspace_Decomposed_Rank_Ensemble"]["tier3_conf"], 1, 0)
        
        p_sub_soft_te = (
            0.25 * out_preds_dict["CatBoost_Circadian_Specialist"] + 
            0.15 * out_preds_dict["LightGBM_Sleep_Specialist"] + 
            0.25 * out_preds_dict["CatBoost_Global"] + 
            0.20 * out_preds_dict["XGBoost_Global"] + 
            0.10 * out_preds_dict["RBF_SVM_Global"] + 
            0.05 * out_preds_dict["RandomForest_Global"]
        )
        outer_oof_probs["V44_Subspace_Soft_Ensemble"][test_idx] = p_sub_soft_te
        outer_oof_preds_t1["V44_Subspace_Soft_Ensemble"][test_idx] = np.where(p_sub_soft_te >= inner_th["V44_Subspace_Soft_Ensemble"]["tier1_screen"], 1, 0)
        outer_oof_preds_t2["V44_Subspace_Soft_Ensemble"][test_idx] = np.where(p_sub_soft_te >= inner_th["V44_Subspace_Soft_Ensemble"]["tier2_diag"], 1, 0)
        outer_oof_preds_t3["V44_Subspace_Soft_Ensemble"][test_idx] = np.where(p_sub_soft_te >= inner_th["V44_Subspace_Soft_Ensemble"]["tier3_conf"], 1, 0)
        
        outer_mat_te = np.column_stack([out_preds_dict[m] for m in base_names])
        p_meta_te = meta_lr.predict_proba(outer_mat_te)[:, 1]
        outer_oof_probs["Stacking_MetaLearner"][test_idx] = p_meta_te
        outer_oof_preds_t1["Stacking_MetaLearner"][test_idx] = np.where(p_meta_te >= inner_th["Stacking_MetaLearner"]["tier1_screen"], 1, 0)
        outer_oof_preds_t2["Stacking_MetaLearner"][test_idx] = np.where(p_meta_te >= inner_th["Stacking_MetaLearner"]["tier2_diag"], 1, 0)
        outer_oof_preds_t3["Stacking_MetaLearner"][test_idx] = np.where(p_meta_te >= inner_th["Stacking_MetaLearner"]["tier3_conf"], 1, 0)
        
        print(f"  [Fold {outer_fold}] CB Global AUC: {roc_auc_score(y_te, out_preds_dict['CatBoost_Global']):.4f} | V44 Rank Ens AUC: {roc_auc_score(y_te, p_sub_rank_te):.4f}")

    y_true_all = y.values
    results = {}
    
    for m in all_names:
        probs = outer_oof_probs[m]
        p_t1 = outer_oof_preds_t1[m]
        p_t2 = outer_oof_preds_t2[m]
        p_t3 = outer_oof_preds_t3[m]
        
        auc_s = roc_auc_score(y_true_all, probs)
        acc_t2 = accuracy_score(y_true_all, p_t2)
        prec_t2 = precision_score(y_true_all, p_t2, zero_division=0)
        rec_t2 = recall_score(y_true_all, p_t2, zero_division=0)
        f1_t2 = f1_score(y_true_all, p_t2, zero_division=0)
        cm_t2 = confusion_matrix(y_true_all, p_t2)
        tn, fp, fn, tp = cm_t2.ravel()
        spec_t2 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        brier = brier_score_loss(y_true_all, probs)
        
        # 95% 신뢰구간 산출 (Bootstrap 1000회)
        ci_dict = compute_bootstrap_ci(y_true_all, probs, p_t2, n_bootstraps=BOOTSTRAP_ROUNDS, random_state=RANDOM_STATE)
        
        # Tier 1 (Screening)
        rec_t1 = recall_score(y_true_all, p_t1, zero_division=0)
        acc_t1 = accuracy_score(y_true_all, p_t1)
        cm_t1 = confusion_matrix(y_true_all, p_t1)
        spec_t1 = cm_t1.ravel()[0] / (cm_t1.ravel()[0] + cm_t1.ravel()[1])
        
        # Tier 3 (Confirmation)
        cm_t3 = confusion_matrix(y_true_all, p_t3)
        spec_t3 = cm_t3.ravel()[0] / (cm_t3.ravel()[0] + cm_t3.ravel()[1])
        rec_t3 = recall_score(y_true_all, p_t3, zero_division=0)
        acc_t3 = accuracy_score(y_true_all, p_t3)
        
        results[m] = {
            "auc": auc_s, "acc_t2": acc_t2, "prec_t2": prec_t2, "rec_t2": rec_t2, "f1_t2": f1_t2, "spec_t2": spec_t2, "cm_t2": cm_t2, "brier": brier,
            "ci": ci_dict,
            "rec_t1": rec_t1, "acc_t1": acc_t1, "spec_t1": spec_t1,
            "spec_t3": spec_t3, "rec_t3": rec_t3, "acc_t3": acc_t3,
            "probs": probs, "y_true": y_true_all, "p_t2": p_t2
        }
    return results


def main():
    print("="*105)
    print(f" [Ablation Study] V44 Subspace SOTA: BorderlineSMOTE 유무에 따른 성능 비교 (95% 신뢰구간 포함, B={BOOTSTRAP_ROUNDS})")
    print("="*105)
    
    # 1. SMOTE 적용 실행
    res_with_smote = run_nested_cv(use_smote=True)
    
    # 2. SMOTE 미적용 실행
    res_no_smote = run_nested_cv(use_smote=False)
    
    # 3. 비교 결과 정리
    models = list(res_with_smote.keys())
    
    print("\n" + "="*145)
    print(f" {'모델명':<38s} | {'With SMOTE AUC [95% CI]':<26s} | {'No SMOTE AUC [95% CI]':<26s} | {'Δ AUC':<9s} | {'With F1 [95% CI]':<22s} | {'No F1 [95% CI]':<22s}")
    print("="*145)
    
    table_rows = []
    for m in models:
        auc_w = res_with_smote[m]['auc']
        auc_w_ci = res_with_smote[m]['ci']['auc']
        auc_wo = res_no_smote[m]['auc']
        auc_wo_ci = res_no_smote[m]['ci']['auc']
        d_auc = auc_w - auc_wo
        
        acc_w = res_with_smote[m]['acc_t2']
        acc_w_ci = res_with_smote[m]['ci']['acc']
        acc_wo = res_no_smote[m]['acc_t2']
        acc_wo_ci = res_no_smote[m]['ci']['acc']
        d_acc = acc_w - acc_wo
        
        prec_w = res_with_smote[m]['prec_t2']
        prec_w_ci = res_with_smote[m]['ci']['prec']
        prec_wo = res_no_smote[m]['prec_t2']
        prec_wo_ci = res_no_smote[m]['ci']['prec']
        d_prec = prec_w - prec_wo
        
        rec_w = res_with_smote[m]['rec_t2']
        rec_w_ci = res_with_smote[m]['ci']['rec']
        rec_wo = res_no_smote[m]['rec_t2']
        rec_wo_ci = res_no_smote[m]['ci']['rec']
        d_rec = rec_w - rec_wo
        
        f1_w = res_with_smote[m]['f1_t2']
        f1_w_ci = res_with_smote[m]['ci']['f1']
        f1_wo = res_no_smote[m]['f1_t2']
        f1_wo_ci = res_no_smote[m]['ci']['f1']
        d_f1 = f1_w - f1_wo
        
        spec_w = res_with_smote[m]['spec_t2']
        spec_w_ci = res_with_smote[m]['ci']['spec']
        spec_wo = res_no_smote[m]['spec_t2']
        spec_wo_ci = res_no_smote[m]['ci']['spec']
        d_spec = spec_w - spec_wo
        
        sign_auc = "+" if d_auc > 0 else ""
        
        str_auc_w = f"{auc_w:.4f} [{auc_w_ci[0]:.4f}-{auc_w_ci[1]:.4f}]"
        str_auc_wo = f"{auc_wo:.4f} [{auc_wo_ci[0]:.4f}-{auc_wo_ci[1]:.4f}]"
        str_f1_w = f"{f1_w:.4f} [{f1_w_ci[0]:.4f}-{f1_w_ci[1]:.4f}]"
        str_f1_wo = f"{f1_wo:.4f} [{f1_wo_ci[0]:.4f}-{f1_wo_ci[1]:.4f}]"
        
        print(f" {m:<38s} | {str_auc_w:<26s} | {str_auc_wo:<26s} | {sign_auc}{d_auc:<8.4f} | {str_f1_w:<22s} | {str_f1_wo:<22s}")
        
        table_rows.append({
            "Model": m,
            "AUC_With": auc_w, "AUC_With_CI": auc_w_ci, "AUC_No": auc_wo, "AUC_No_CI": auc_wo_ci, "Diff_AUC": d_auc,
            "Acc_With": acc_w, "Acc_With_CI": acc_w_ci, "Acc_No": acc_wo, "Acc_No_CI": acc_wo_ci, "Diff_Acc": d_acc,
            "Prec_With": prec_w, "Prec_With_CI": prec_w_ci, "Prec_No": prec_wo, "Prec_No_CI": prec_wo_ci, "Diff_Prec": d_prec,
            "Rec_With": rec_w, "Rec_With_CI": rec_w_ci, "Rec_No": rec_wo, "Rec_No_CI": rec_wo_ci, "Diff_Rec": d_rec,
            "Spec_With": spec_w, "Spec_With_CI": spec_w_ci, "Spec_No": spec_wo, "Spec_No_CI": spec_wo_ci, "Diff_Spec": d_spec,
            "F1_With": f1_w, "F1_With_CI": f1_w_ci, "F1_No": f1_wo, "F1_No_CI": f1_wo_ci, "Diff_F1": d_f1,
        })
    print("="*145)
    
    # 4. 정확도, 정밀도, 재현율 콘솔 테이블 출력
    print("\n" + "="*170)
    print(" [표 2] 정확도(Accuracy), 정밀도(Precision), 재현율(Recall) (95% 신뢰구간)")
    print("="*170)
    print(f" {'모델명':<38s} | {'With Acc [95% CI]':<24s} | {'No Acc [95% CI]':<24s} | {'With Prec [95% CI]':<24s} | {'No Prec [95% CI]':<24s} | {'With Rec [95% CI]':<24s} | {'No Rec [95% CI]':<24s}")
    print("-"*170)
    for r in table_rows:
        str_acc_w = f"{r['Acc_With']:.4f} [{r['Acc_With_CI'][0]:.4f}-{r['Acc_With_CI'][1]:.4f}]"
        str_acc_wo = f"{r['Acc_No']:.4f} [{r['Acc_No_CI'][0]:.4f}-{r['Acc_No_CI'][1]:.4f}]"
        str_prec_w = f"{r['Prec_With']:.4f} [{r['Prec_With_CI'][0]:.4f}-{r['Prec_With_CI'][1]:.4f}]"
        str_prec_wo = f"{r['Prec_No']:.4f} [{r['Prec_No_CI'][0]:.4f}-{r['Prec_No_CI'][1]:.4f}]"
        str_rec_w = f"{r['Rec_With']:.4f} [{r['Rec_With_CI'][0]:.4f}-{r['Rec_With_CI'][1]:.4f}]"
        str_rec_wo = f"{r['Rec_No']:.4f} [{r['Rec_No_CI'][0]:.4f}-{r['Rec_No_CI'][1]:.4f}]"
        print(f" {r['Model']:<38s} | {str_acc_w:<24s} | {str_acc_wo:<24s} | {str_prec_w:<24s} | {str_prec_wo:<24s} | {str_rec_w:<24s} | {str_rec_wo:<24s}")
    print("="*170)

    # 5. 챔피언 모델 핵심 요약 출력
    champ = [r for r in table_rows if r['Model'] == 'V44_Subspace_Decomposed_Rank_Ensemble'][0]
    print("\n" + "="*95)
    print(" [핵심 챔피언 모델 종합 결과] V44_Subspace_Decomposed_Rank_Ensemble")
    print("="*95)
    print(f" - ROC-AUC:   With SMOTE {champ['AUC_With']:.4f} [95% CI: {champ['AUC_With_CI'][0]:.4f} - {champ['AUC_With_CI'][1]:.4f}] vs Without {champ['AUC_No']:.4f} [95% CI: {champ['AUC_No_CI'][0]:.4f} - {champ['AUC_No_CI'][1]:.4f}] (Δ {champ['Diff_AUC']:+.4f})")
    print(f" - Recall:    With SMOTE {champ['Rec_With']:.4f} [95% CI: {champ['Rec_With_CI'][0]:.4f} - {champ['Rec_With_CI'][1]:.4f}] vs Without {champ['Rec_No']:.4f} [95% CI: {champ['Rec_No_CI'][0]:.4f} - {champ['Rec_No_CI'][1]:.4f}] (Δ {champ['Diff_Rec']:+.4f})")
    print(f" - Precision: With SMOTE {champ['Prec_With']:.4f} [95% CI: {champ['Prec_With_CI'][0]:.4f} - {champ['Prec_With_CI'][1]:.4f}] vs Without {champ['Prec_No']:.4f} [95% CI: {champ['Prec_No_CI'][0]:.4f} - {champ['Prec_No_CI'][1]:.4f}] (Δ {champ['Diff_Prec']:+.4f})")
    print(f" - Accuracy:  With SMOTE {champ['Acc_With']:.4f} [95% CI: {champ['Acc_With_CI'][0]:.4f} - {champ['Acc_With_CI'][1]:.4f}] vs Without {champ['Acc_No']:.4f} [95% CI: {champ['Acc_No_CI'][0]:.4f} - {champ['Acc_No_CI'][1]:.4f}] (Δ {champ['Diff_Acc']:+.4f})")
    print(f" - F1-Score:  With SMOTE {champ['F1_With']:.4f} [95% CI: {champ['F1_With_CI'][0]:.4f}-{champ['F1_With_CI'][1]:.4f}] vs Without {champ['F1_No']:.4f} [95% CI: {champ['F1_No_CI'][0]:.4f}-{champ['F1_No_CI'][1]:.4f}] (Δ {champ['Diff_F1']:+.4f})")
    print("="*95 + "\n")


if __name__ == "__main__":
    main()
