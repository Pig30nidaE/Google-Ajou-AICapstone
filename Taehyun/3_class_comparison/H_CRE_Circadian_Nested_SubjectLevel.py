import os
import sys
import pathlib
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from imblearn.over_sampling import SMOTE

import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ---------------------------------------------------------
# 1. 전역 설정 및 경로
# ---------------------------------------------------------
BASE_DIR = pathlib.Path(r"c:\ML4")
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "tabular"
CIRCADIAN_PATH = PROCESSED_DIR / "patient_level_circadian_v3.csv"
PLOT_DIR = BASE_DIR / "report" / "plots"
REPORT_DIR = BASE_DIR / "report" / "multi-class"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

TARGET_COL = "original_label"
RANDOM_STATE = 42
OUTER_SPLITS = 5
INNER_SPLITS = 3
FORWARD_SELECTION_MAX_FEATURES = 30
FORWARD_SELECTION_ESTIMATORS = 80

# V44 핵심 도메인 피처 14종
V44_14_FEATURES = [
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

DROP_METADATA_COLS = [
    "EMAIL", "date", "DIAG_NM", "original_label", "label", "fold",
    "SAMPLE_EMAIL", "DIAG_SEQ", "DOCTOR_NM", "MMSE_NUM", "MMSE_KIND", "TOTAL"
]


# ---------------------------------------------------------
# 2. 데이터셋 로딩 및 V44 피처 엔지니어링 (피험자별 구조)
# ---------------------------------------------------------
def load_subject_level_circadian_data():
    """
    피험자별 데이터셋(patient_level_circadian_v3.csv) 로드 및 
    V44 핵심 바이오마커 계산, MMSE 누수 방지 컬럼 제거
    """
    df = pd.read_csv(CIRCADIAN_PATH)
    
    # MMSE 관련 임상 인지 점수 컬럼 완전 제거 (누수 원천 방지)
    mmse_cols = [c for c in df.columns if c.startswith('Q') or 'mmse' in c.lower() or c in ['TOTAL', 'DIAG_SEQ', 'DOCTOR_NM']]
    if mmse_cols:
        df.drop(columns=mmse_cols, inplace=True, errors='ignore')
        
    # 수치형 컬럼 무한대 처리
    numeric_cols = [c for c in df.columns if c not in DROP_METADATA_COLS and pd.api.types.is_numeric_dtype(df[c])]
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # V44 핵심 복합 지표 산출
    if 'sleep_hr_average' in df.columns and 'sleep_hr_lowest' in df.columns:
        df['HR_drop_ratio'] = (df['sleep_hr_average'] - df['sleep_hr_lowest']) / (df['sleep_hr_average'] + 1e-5)
    if 'circadian_IV' in df.columns and 'circadian_IS' in df.columns:
        df['Circadian_Strain'] = df['circadian_IV'] / (df['circadian_IS'] + 1e-5)
        
    # 다중 분류 라벨 정의 (3-Class: CN=0, MCI=1, Dem=2)
    # Stage 1: CN(0) vs Abnormal(MCI/Dem=1)
    df['stage1_label'] = np.where(df[TARGET_COL] == 0, 0, 1)
    # Stage 2: MCI(0) vs Dem(1) (CN은 NaN)
    df['stage2_label'] = np.where(df[TARGET_COL] == 1, 0, np.where(df[TARGET_COL] == 2, 1, np.nan))
    
    candidate_features = [
        c for c in df.columns 
        if c not in DROP_METADATA_COLS and c not in ['stage1_label', 'stage2_label'] and pd.api.types.is_numeric_dtype(df[c])
    ]
    
    return df.reset_index(drop=True), candidate_features


# ---------------------------------------------------------
# 3. 누수 없는 비지도 학습 피처 생성기 (Leakage-free Transformer)
# ---------------------------------------------------------
class LeakageFreeUnsupervisedTransformer:
    """
    Outer Train에서만 fit하고, Outer Test에는 transform만 수행하여
    정보 누수(Data Leakage)를 방지하는 비지도 특징 추출 클래스
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=5, random_state=self.random_state)
        self.kmeans = KMeans(n_clusters=3, random_state=self.random_state, n_init=10)
        self.gmm = GaussianMixture(n_components=3, random_state=self.random_state)
        self.agg_centroids_ = None
        self.base_features_ = None

    def fit(self, X_df):
        self.base_features_ = list(X_df.columns)
        X_mat = self.imputer.fit_transform(X_df)
        X_sc = self.scaler.fit_transform(X_mat)
        
        self.pca.fit(X_sc)
        self.kmeans.fit(X_sc)
        self.gmm.fit(X_sc)
        
        # AgglomerativeClustering은 predict가 없으므로 학습 군집 중심점(Centroid) 계산
        agg = AgglomerativeClustering(n_clusters=3)
        agg_labels = agg.fit_predict(X_sc)
        self.agg_centroids_ = np.array([X_sc[agg_labels == c].mean(axis=0) for c in range(3)])
        return self

    def transform(self, X_df):
        X_mat = self.imputer.transform(X_df[self.base_features_])
        X_sc = self.scaler.transform(X_mat)
        
        feat_dict = {}
        # 1. PCA
        pca_feats = self.pca.transform(X_sc)
        for i in range(pca_feats.shape[1]):
            feat_dict[f'pca_{i+1}'] = pca_feats[:, i]
            
        # 2. K-Means
        kmeans_dist = self.kmeans.transform(X_sc)
        for i in range(3):
            feat_dict[f'kmeans_dist_{i}'] = kmeans_dist[:, i]
        feat_dict['kmeans_label'] = self.kmeans.predict(X_sc)
        
        # 3. GMM
        gmm_probs = self.gmm.predict_proba(X_sc)
        for i in range(3):
            feat_dict[f'gmm_prob_{i}'] = gmm_probs[:, i]
            
        # 4. Agglomerative (최근접 중심점 할당)
        dists = np.linalg.norm(X_sc[:, None, :] - self.agg_centroids_[None, :, :], axis=2)
        feat_dict['hierarchical_cluster_label'] = np.argmin(dists, axis=1)
        
        new_feats_df = pd.DataFrame(feat_dict, index=X_df.index)
        return pd.concat([X_df, new_feats_df], axis=1)


# ---------------------------------------------------------
# 4. Inner CV 기반 전진 선택법 (Zero-Leakage Forward Selection)
# ---------------------------------------------------------
def perform_inner_forward_selection(X_train, y_train, candidate_feats, stage_name="Stage1"):
    """
    Outer Train 내부에서 Inner CV로만 최적 피처를 탐색하여
    선택 편향(Selection Bias) 및 누수를 차단하는 전진 선택법
    """
    # 안전한 SMOTE (클래스 최소 빈도 고려)
    min_class_count = pd.Series(y_train).value_counts().min()
    k_neighbors = max(1, min(5, min_class_count - 1))
    smote = SMOTE(k_neighbors=k_neighbors, random_state=RANDOM_STATE)
    
    # 1. 기본 LightGBM 피처 중요도 산출
    X_res, y_res = smote.fit_resample(X_train[candidate_feats], y_train)
    base_model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight='balanced', verbose=-1)
    base_model.fit(X_res, y_res)
    
    importance = base_model.feature_importances_
    ranked_features = [feat for _, feat in sorted(zip(importance, candidate_feats), reverse=True)]
    
    # V44 14개 핵심 피처를 우선 평가군에 강제 보장
    v44_present = [f for f in V44_14_FEATURES if f in candidate_feats]
    prioritized_pool = list(dict.fromkeys(v44_present + ranked_features[:FORWARD_SELECTION_MAX_FEATURES]))
    top_candidates = prioritized_pool[:FORWARD_SELECTION_MAX_FEATURES]
    
    # 2. Inner CV 루프를 통한 최적 피처 개수 k 탐색
    inner_skf = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    history_scores = []
    
    for k in range(1, len(top_candidates) + 1):
        current_features = top_candidates[:k]
        fold_scores = []
        
        for in_tr_idx, in_va_idx in inner_skf.split(X_train[current_features], y_train):
            X_in_tr = X_train[current_features].iloc[in_tr_idx]
            y_in_tr = y_train.iloc[in_tr_idx]
            X_in_va = X_train[current_features].iloc[in_va_idx]
            y_in_va = y_train.iloc[in_va_idx]
            
            in_min_c = pd.Series(y_in_tr).value_counts().min()
            in_k_neighbors = max(1, min(3, in_min_c - 1))
            in_smote = SMOTE(k_neighbors=in_k_neighbors, random_state=RANDOM_STATE)
            X_in_tr_res, y_in_tr_res = in_smote.fit_resample(X_in_tr, y_in_tr)
            
            eval_model = LGBMClassifier(
                random_state=RANDOM_STATE, n_jobs=1, class_weight='balanced', 
                n_estimators=FORWARD_SELECTION_ESTIMATORS, verbose=-1
            )
            eval_model.fit(X_in_tr_res, y_in_tr_res)
            prob = eval_model.predict_proba(X_in_va)[:, 1]
            try:
                fold_scores.append(roc_auc_score(y_in_va, prob))
            except ValueError:
                fold_scores.append(0.5)
                
        history_scores.append(np.mean(fold_scores))
        
    optimal_k = np.argmax(history_scores) + 1
    optimal_features = top_candidates[:optimal_k]
    return optimal_features


# ---------------------------------------------------------
# 5. H-CRE 계층적 2단계 모델 및 이중 규제화 훈련 파이프라인
# ---------------------------------------------------------
def train_predict_h_cre_stage_models(df_tr, df_te, s1_feats, s2_feats):
    """
    H-CRE 모델 핵심 로직:
    Stage 1 분류기 + Stage 2 강규제 분류기 + Confidence Penalty 결합
    """
    y_te = df_te[TARGET_COL].astype(int)
    
    # Stage 1 학습 데이터 준비
    X_tr_s1 = df_tr[s1_feats]
    y_tr_s1 = df_tr['stage1_label'].astype(int)
    
    # Stage 2 학습 데이터 준비 (Abnormal 대상: stage1_label == 1)
    df_tr_s2 = df_tr[df_tr['stage1_label'] == 1]
    X_tr_s2 = df_tr_s2[s2_feats]
    y_tr_s2 = df_tr_s2['stage2_label'].astype(int)
    
    # SMOTE 오버샘플링
    smote1 = SMOTE(k_neighbors=max(1, min(5, y_tr_s1.value_counts().min() - 1)), random_state=RANDOM_STATE)
    X_tr_s1_res, y_tr_s1_res = smote1.fit_resample(X_tr_s1, y_tr_s1)
    
    smote2 = SMOTE(k_neighbors=max(1, min(3, y_tr_s2.value_counts().min() - 1)), random_state=RANDOM_STATE)
    X_tr_s2_res, y_tr_s2_res = smote2.fit_resample(X_tr_s2, y_tr_s2)
    
    model_names = ["LightGBM", "CatBoost", "XGBoost", "RandomForest"]
    fold_probs = {}
    stage1_probs = {}
    stage2_probs = {}
    
    for m in model_names:
        if m == "LightGBM":
            # Optuna Stage 1 최적 파라미터 (Best AUC: 0.7289)
            model_s1 = LGBMClassifier(
                objective="binary", random_state=RANDOM_STATE, n_jobs=1, verbose=-1,
                learning_rate=0.06439, num_leaves=47, max_depth=5, min_child_samples=28,
                subsample=0.8816, colsample_bytree=0.8790, reg_alpha=0.0656, reg_lambda=0.0011, n_estimators=150
            )
            # Optuna Stage 2 최적 규제화 파라미터 (Best AUC: 0.9412)
            model_s2 = LGBMClassifier(
                objective="binary", random_state=RANDOM_STATE, n_jobs=1, verbose=-1,
                learning_rate=0.07181, num_leaves=37, max_depth=3, min_child_samples=22,
                subsample=0.6621, colsample_bytree=0.9955, reg_alpha=0.0045, reg_lambda=0.0072, n_estimators=201
            )
        elif m == "CatBoost":
            # Optuna Stage 1 최적 파라미터 (Best AUC: 0.7216)
            model_s1 = CatBoostClassifier(
                random_state=RANDOM_STATE, verbose=0, thread_count=1,
                learning_rate=0.01816, depth=3, l2_leaf_reg=18.557, subsample=0.6853, iterations=228
            )
            # Optuna Stage 2 최적 규제화 파라미터 (Best AUC: 0.8971)
            model_s2 = CatBoostClassifier(
                random_state=RANDOM_STATE, verbose=0, thread_count=1,
                learning_rate=0.02751, depth=6, l2_leaf_reg=23.427, subsample=0.6321, iterations=250
            )
        elif m == "XGBoost":
            # Optuna Stage 1 최적 파라미터 (Best AUC: 0.7585)
            model_s1 = XGBClassifier(
                random_state=RANDOM_STATE, n_jobs=1, eval_metric='logloss',
                learning_rate=0.08473, max_depth=3, min_child_weight=3,
                subsample=0.5798, colsample_bytree=0.5860, reg_alpha=0.1336, reg_lambda=0.0011, n_estimators=250
            )
            # Optuna Stage 2 최적 규제화 파라미터 (Best AUC: 0.8873)
            model_s2 = XGBClassifier(
                random_state=RANDOM_STATE, n_jobs=1, eval_metric='logloss',
                learning_rate=0.01271, max_depth=2, min_child_weight=1,
                subsample=0.6627, colsample_bytree=0.6943, reg_alpha=0.0122, reg_lambda=2.0651, n_estimators=141
            )
        elif m == "RandomForest":
            # Optuna Stage 1 최적 파라미터 (Best AUC: 0.7083)
            model_s1 = RandomForestClassifier(
                random_state=RANDOM_STATE, n_jobs=1,
                n_estimators=264, max_depth=7, min_samples_split=3, min_samples_leaf=6, max_features='sqrt'
            )
            # Optuna Stage 2 최적 규제화 파라미터 (Best AUC: 0.8725)
            model_s2 = RandomForestClassifier(
                random_state=RANDOM_STATE, n_jobs=1,
                n_estimators=117, max_depth=4, min_samples_split=2, min_samples_leaf=2, max_features=0.6
            )
            
        model_s1.fit(X_tr_s1_res, y_tr_s1_res)
        model_s2.fit(X_tr_s2_res, y_tr_s2_res)
        
        prob1 = model_s1.predict_proba(df_te[s1_feats])[:, 1]
        prob2 = model_s2.predict_proba(df_te[s2_feats])[:, 1]
        stage1_probs[m] = prob1
        stage2_probs[m] = prob2
        
        # [H-CRE 핵심] Stage 1 불확실 환자 대상 Confidence Penalty 감쇠
        confidence_penalty = np.where(prob1 < 0.65, 0.6 + 0.4 * (prob1 / 0.65), 1.0)
        prob2_regularized = prob2 * confidence_penalty
        
        # 계층적 3-Class 확률 합성
        p_CN = 1.0 - prob1
        p_MCI = prob1 * (1.0 - prob2_regularized)
        p_Dem = prob1 * prob2_regularized
        
        prob_matrix = np.vstack([p_CN, p_MCI, p_Dem]).T
        fold_probs[m] = prob_matrix
        
    # [H-CRE 플래그십 하이브리드] Stage 1 XGBoost + Stage 2 LightGBM
    prob1_xgb = stage1_probs["XGBoost"]
    prob2_lgbm = stage2_probs["LightGBM"]
    confidence_penalty_hyb = np.where(prob1_xgb < 0.65, 0.6 + 0.4 * (prob1_xgb / 0.65), 1.0)
    prob2_lgbm_reg = prob2_lgbm * confidence_penalty_hyb
    p_CN_hyb = 1.0 - prob1_xgb
    p_MCI_hyb = prob1_xgb * (1.0 - prob2_lgbm_reg)
    p_Dem_hyb = prob1_xgb * prob2_lgbm_reg
    fold_probs["Hybrid (XGB->LGBM)"] = np.vstack([p_CN_hyb, p_MCI_hyb, p_Dem_hyb]).T
        
    # [H-CRE 앙상블] 4개 모델 소프트 확률 평균
    ensemble_prob = (fold_probs["LightGBM"] + fold_probs["CatBoost"] + fold_probs["XGBoost"] + fold_probs["RandomForest"]) / 4.0
    fold_probs["Ensemble"] = ensemble_prob
    
    return fold_probs


# ---------------------------------------------------------
# 6. 전체 Nested CV 실행 및 OOF 평가
# ---------------------------------------------------------
def run_h_cre_circadian_nested_evaluation():
    print("=" * 85)
    print(" [H-CRE Circadian Nested] 3-Class 계층적 다중분류 피험자별 Nested CV 시작")
    print(f"  Outer Folds: {OUTER_SPLITS} | Inner Folds: {INNER_SPLITS} | 생체리듬 피처 결합 완료")
    print("=" * 85)
    
    df, candidate_feats = load_subject_level_circadian_data()
    n_samples = len(df)
    n_subjects = df['EMAIL'].nunique()
    print(f"  총 피험자 수: {n_subjects}명 (1인 1행 구조, 샘플 수 {n_samples})")
    print(f"  초기 후보 피처 수: {len(candidate_feats)}개 (V44 14종 핵심 지표 포함)")
    print(f"  클래스 분포 (3-Class): CN={sum(df[TARGET_COL]==0)}, MCI={sum(df[TARGET_COL]==1)}, Dem={sum(df[TARGET_COL]==2)}")
    
    # 피험자 단위 독립 분할기 설정 (EMAIL 그룹 분할 지원)
    if n_subjects == n_samples:
        outer_cv = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        splits = outer_cv.split(df, df[TARGET_COL])
    else:
        outer_cv = StratifiedGroupKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        splits = outer_cv.split(df, df[TARGET_COL], groups=df['EMAIL'].values)
        
    model_names = ["LightGBM", "CatBoost", "XGBoost", "RandomForest", "Hybrid (XGB->LGBM)", "Ensemble"]
    oof_probs = {m: np.zeros((n_samples, 3)) for m in model_names}
    y_true_all = df[TARGET_COL].astype(int).values
    
    fold_selected_feats_s1 = []
    fold_selected_feats_s2 = []
    
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        print(f"\n--- [Outer Fold {fold}/{OUTER_SPLITS}] ---")
        df_outer_tr = df.iloc[train_idx].copy().reset_index(drop=True)
        df_outer_te = df.iloc[test_idx].copy().reset_index(drop=True)
        
        # 1. Outer Train 기준 비지도 특징 적합 (누수 원천 차단)
        unsup_transformer = LeakageFreeUnsupervisedTransformer(random_state=RANDOM_STATE + fold)
        unsup_transformer.fit(df_outer_tr[candidate_feats])
        
        df_outer_tr_aug = unsup_transformer.transform(df_outer_tr[candidate_feats])
        df_outer_te_aug = unsup_transformer.transform(df_outer_te[candidate_feats])
        
        df_outer_tr_aug[TARGET_COL] = df_outer_tr[TARGET_COL].values
        df_outer_tr_aug['stage1_label'] = df_outer_tr['stage1_label'].values
        df_outer_tr_aug['stage2_label'] = df_outer_tr['stage2_label'].values
        
        df_outer_te_aug[TARGET_COL] = df_outer_te[TARGET_COL].values
        
        all_aug_feats = [c for c in df_outer_tr_aug.columns if c not in [TARGET_COL, 'stage1_label', 'stage2_label']]
        
        # 2. Inner CV 전진 선택법 (Outer Train 내부에서만 수행)
        opt_s1 = perform_inner_forward_selection(
            df_outer_tr_aug, df_outer_tr_aug['stage1_label'], all_aug_feats, stage_name="Stage1"
        )
        
        df_tr_s2 = df_outer_tr_aug.dropna(subset=['stage2_label']).copy().reset_index(drop=True)
        opt_s2 = perform_inner_forward_selection(
            df_tr_s2, df_tr_s2['stage2_label'].astype(int), all_aug_feats, stage_name="Stage2"
        )
        
        fold_selected_feats_s1.append(opt_s1)
        fold_selected_feats_s2.append(opt_s2)
        print(f"  [Fold {fold}] 선택 피처 수 -> Stage 1: {len(opt_s1)}개, Stage 2: {len(opt_s2)}개")
        
        # 3. H-CRE 계층적 모델 학습 및 Outer Test 예측
        fold_preds_dict = train_predict_h_cre_stage_models(df_outer_tr_aug, df_outer_te_aug, opt_s1, opt_s2)
        
        for m in model_names:
            oof_probs[m][test_idx] = fold_preds_dict[m]
            
        fold_hyb_preds = np.argmax(fold_preds_dict["Hybrid (XGB->LGBM)"], axis=1)
        fold_acc = accuracy_score(df_outer_te[TARGET_COL], fold_hyb_preds)
        try:
            fold_auc = roc_auc_score(df_outer_te[TARGET_COL], fold_preds_dict["Hybrid (XGB->LGBM)"], multi_class='ovr')
        except ValueError:
            fold_auc = np.nan
        print(f"  Fold {fold} Hybrid (XGB->LGBM) 성능: Acc={fold_acc:.4f}, OVR AUC={fold_auc:.4f}")
        
    # ---------------------------------------------------------
    # 7. 종합 평가 및 95% 부트스트랩 신뢰구간 산출
    # ---------------------------------------------------------
    print("\n" + "=" * 85)
    print(" [최종 결과] H-CRE Circadian Nested 3-Class OOF 성능 요약 (Zero-Leakage)")
    print("=" * 85)
    
    results = {}
    
    for m in model_names:
        probs = oof_probs[m]
        preds = np.argmax(probs, axis=1)
        
        acc = accuracy_score(y_true_all, preds)
        prec = precision_score(y_true_all, preds, average='macro', zero_division=0)
        rec = recall_score(y_true_all, preds, average='macro', zero_division=0)
        f1 = f1_score(y_true_all, preds, average='macro', zero_division=0)
        auc = roc_auc_score(y_true_all, probs, multi_class='ovr')
        cm = confusion_matrix(y_true_all, preds)
        
        # 클래스별 세부 재현율 (Sensitivity)
        rec_cn = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0.0
        rec_mci = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0.0
        rec_dem = cm[2, 2] / cm[2].sum() if cm[2].sum() > 0 else 0.0
        
        # 1,000회 부트스트랩 95% CI (앙상블 및 하이브리드 모델 대상)
        ci_acc, ci_f1, ci_auc = None, None, None
        if m in ["Ensemble", "Hybrid (XGB->LGBM)"]:
            np.random.seed(RANDOM_STATE)
            bs_accs, bs_f1s, bs_aucs = [], [], []
            for _ in range(1000):
                idx = np.random.choice(len(y_true_all), size=len(y_true_all), replace=True)
                if len(np.unique(y_true_all[idx])) < 3:
                    continue
                bs_accs.append(accuracy_score(y_true_all[idx], preds[idx]))
                bs_f1s.append(f1_score(y_true_all[idx], preds[idx], average='macro', zero_division=0))
                try:
                    bs_aucs.append(roc_auc_score(y_true_all[idx], probs[idx], multi_class='ovr'))
                except ValueError:
                    pass
            ci_acc = (np.percentile(bs_accs, 2.5), np.percentile(bs_accs, 97.5))
            ci_f1 = (np.percentile(bs_f1s, 2.5), np.percentile(bs_f1s, 97.5))
            ci_auc = (np.percentile(bs_aucs, 2.5), np.percentile(bs_aucs, 97.5))
            
        results[m] = {
            'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc, 'cm': cm,
            'rec_cn': rec_cn, 'rec_mci': rec_mci, 'rec_dem': rec_dem,
            'ci_acc': ci_acc, 'ci_f1': ci_f1, 'ci_auc': ci_auc,
            'probs': probs, 'preds': preds
        }
        
        ci_str = f" [95% CI: {ci_auc[0]:.4f} ~ {ci_auc[1]:.4f}]" if ci_auc else ""
        print(f"[{m:<20}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | Macro F1: {f1:.4f} | OVR AUC: {auc:.4f}{ci_str}")
        print(f"                       세부 민감도 -> CN: {rec_cn:.4f}, MCI: {rec_mci:.4f}, Dem: {rec_dem:.4f}")
        
    # ---------------------------------------------------------
    # 8. 혼동 행렬 시각화 저장
    # ---------------------------------------------------------
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    class_names = ['CN(0)', 'MCI(1)', 'Dem(2)']
    fig, axes = plt.subplots(1, len(model_names), figsize=(4.2 * len(model_names), 4.2))
    
    for ax, m in zip(axes, model_names):
        cm = results[m]['cm']
        cmap = 'Greens' if 'Hybrid' in m else ('Blues' if 'Ensemble' not in m else 'Purples')
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=class_names, yticklabels=class_names, 
                    annot_kws={"size": 12, "weight": "bold" if "Hybrid" in m else "normal"}, ax=ax, cbar=False)
        title_suffix = " (Flagship)" if 'Hybrid' in m else ""
        ax.set_title(f"{m}{title_suffix}\nAcc: {results[m]['acc']:.3f} | F1: {results[m]['f1']:.3f} | AUC: {results[m]['auc']:.3f}", 
                     fontsize=10, pad=8, weight='bold' if 'Hybrid' in m else 'normal')
        ax.set_xlabel('예측 클래스', fontsize=9)
        ax.set_ylabel('실제 클래스', fontsize=9)
        
    plt.suptitle("H-CRE (Hierarchical Circadian Regularized Ensemble) 3-Class 혼동 행렬 (N=174 OOF)", fontsize=13, y=1.03)
    plt.tight_layout()
    cm_plot_path = pathlib.Path(__file__).parent / "confusion_matrix_h_cre_baseline.png"
    plt.savefig(cm_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[시각화 완료] 혼동 행렬 이미지 저장: {cm_plot_path}")
    
    # ---------------------------------------------------------
    # 9. 상세 마크다운 결과 보고서 저장
    # ---------------------------------------------------------
    report_path = pathlib.Path(__file__).parent / "report_3class_h_cre_circadian_nested_summary.md"
    ens_ci_acc = results['Ensemble']['ci_acc']
    ens_ci_f1 = results['Ensemble']['ci_f1']
    ens_ci_auc = results['Ensemble']['ci_auc']
    
    report_content = f"""# H-CRE Circadian Nested 3-Class 다중분류 최종 결과 보고서

## 1. 실험 배경 및 개요
- **목적**: 계층적 다중분류 H-CRE 모델의 전처리/피처 선택 누수를 원천 차단하고, **14종 일주기 생체리듬 피처**를 이식하여 **피험자별(Patient-level) Zero-Leakage Nested CV** 체계로 구성.
- **데이터셋**: `patient_level_circadian_v3.csv` (총 피험자 $N=174$명, 1인 1행 구조)
  - CN(정상): 111명
  - MCI(경도인지장애): 51명
  - Dem(치매): 12명
- **검증 프로토콜**: Outer 5-Fold Stratified CV $\\times$ Inner 3-Fold Stratified CV (Zero Data Leakage)
- **적용 기법**:
  1. 14종 핵심 생체리듬 피처 (`HR_drop_ratio`, `Circadian_Strain`, `circadian_IV/IS/RA`, 수면/활동 변동성)
  2. Outer Train 격리 비지도 피처 추출기 (PCA, K-Means, GMM, Agglomerative)
  3. Inner CV 기반 피처 전진 선택법
  4. Stage 2 모델 강력 규제화(L1/L2 패널티, 깊이 제한) 및 Confidence Penalty 감쇠
  5. 4개 모델(LightGBM, CatBoost, XGBoost, RandomForest) 기반 소프트 앙상블

---

## 2. 모델별 Out-of-Fold (N=174) 3-Class 최종 성능

| 모델명 | 정확도(Accuracy) | 정밀도(Precision) | 재현율(Recall) | Macro F1 | **OVR ROC-AUC** | CN 민감도 | MCI 민감도 | Dem 민감도 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LightGBM** | {results['LightGBM']['acc']:.4f} | {results['LightGBM']['prec']:.4f} | {results['LightGBM']['rec']:.4f} | {results['LightGBM']['f1']:.4f} | {results['LightGBM']['auc']:.4f} | {results['LightGBM']['rec_cn']:.4f} | {results['LightGBM']['rec_mci']:.4f} | {results['LightGBM']['rec_dem']:.4f} |
| **CatBoost** | {results['CatBoost']['acc']:.4f} | {results['CatBoost']['prec']:.4f} | {results['CatBoost']['rec']:.4f} | {results['CatBoost']['f1']:.4f} | {results['CatBoost']['auc']:.4f} | {results['CatBoost']['rec_cn']:.4f} | {results['CatBoost']['rec_mci']:.4f} | {results['CatBoost']['rec_dem']:.4f} |
| **XGBoost** | {results['XGBoost']['acc']:.4f} | {results['XGBoost']['prec']:.4f} | {results['XGBoost']['rec']:.4f} | {results['XGBoost']['f1']:.4f} | {results['XGBoost']['auc']:.4f} | {results['XGBoost']['rec_cn']:.4f} | {results['XGBoost']['rec_mci']:.4f} | {results['XGBoost']['rec_dem']:.4f} |
| **RandomForest** | {results['RandomForest']['acc']:.4f} | {results['RandomForest']['prec']:.4f} | {results['RandomForest']['rec']:.4f} | {results['RandomForest']['f1']:.4f} | {results['RandomForest']['auc']:.4f} | {results['RandomForest']['rec_cn']:.4f} | {results['RandomForest']['rec_mci']:.4f} | {results['RandomForest']['rec_dem']:.4f} |
| **Ensemble** | **{results['Ensemble']['acc']:.4f}** | **{results['Ensemble']['prec']:.4f}** | **{results['Ensemble']['rec']:.4f}** | **{results['Ensemble']['f1']:.4f}** | **{results['Ensemble']['auc']:.4f}** | **{results['Ensemble']['rec_cn']:.4f}** | **{results['Ensemble']['rec_mci']:.4f}** | **{results['Ensemble']['rec_dem']:.4f}** |

> **Ensemble 95% 신뢰구간 (Bootstrap B=1,000)**:
> - **Accuracy**: {results['Ensemble']['acc']:.4f} [{ens_ci_acc[0]:.4f} ~ {ens_ci_acc[1]:.4f}]
> - **Macro F1**: {results['Ensemble']['f1']:.4f} [{ens_ci_f1[0]:.4f} ~ {ens_ci_f1[1]:.4f}]
> - **OVR ROC-AUC**: {results['Ensemble']['auc']:.4f} [{ens_ci_auc[0]:.4f} ~ {ens_ci_auc[1]:.4f}]

---

## 3. 혼동 행렬 시각화
![H-CRE 혼동 행렬](file:///{cm_plot_path.as_posix()})

---

## 4. 모델 특징 및 변경 요약
1. **데이터셋 교체**: 생체리듬 특화 `patient_level_circadian_v3.csv`
2. **신규 바이오마커 추가**: `HR_drop_ratio`, `Circadian_Strain`, `circadian_IV`, `circadian_IS` 등 핵심 생체리듬 14종 피처 반영
3. **엄격한 이중 교차 검증 (Nested CV)**: 비지도 학습(PCA/KMeans/GMM) 및 전진 선택법(Forward Selection)이 모두 Outer Train 내부에서만 수행되어 누수를 완벽 차단.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[보고서 작성 완료] {report_path}")
    
    return results


if __name__ == "__main__":
    start_time = datetime.now()
    results = run_h_cre_circadian_nested_evaluation()
    end_time = datetime.now()
    print(f"\n총 실행 소요 시간: {end_time - start_time}")
