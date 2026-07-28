# =========================================================
# 1. 라이브러리 임포트
# =========================================================
import os
import sys
import pathlib
import textwrap
import numpy as np
import pandas as pd
# =========================================================
# 1. 라이브러리 임포트
# =========================================================
import os
import sys
import pathlib
import textwrap
import numpy as np
import pandas as pd
from itertools import product
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold 

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# xai 패키지 import
current_dir = pathlib.Path(os.getcwd())
sys.path.insert(0, str(current_dir))
try:
    from xai import ShapAnalyzer
except ImportError:
    print("Error: 'xai' 패키지를 찾을 수 없습니다. 경로를 확인해주세요.")

start_time = datetime.now()
print("Start:", start_time)

RANDOM_STATE = 42
FORWARD_SELECTION_MAX_FEATURES = 40
FORWARD_SELECTION_ESTIMATORS = 120

# =========================================================
# 2. 전처리 완료 데이터 로드
# =========================================================
BASE_DIR = pathlib.Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "tabular"
PATIENT_PATH = PROCESSED_DIR / "patient_level_all.csv"

TARGET_COL = "label"
DROP_COLS = ["EMAIL", "date", "DIAG_NM", "original_label", TARGET_COL, "fold"]


def load_preprocessed_data():
    if not PATIENT_PATH.exists():
        raise FileNotFoundError(
            "전처리된 CSV를 찾을 수 없습니다. 먼저 `python preprocessing.py`를 실행해주세요.\n"
            f"path: {PATIENT_PATH}"
        )

    print("[데이터 로드] 환자 단위(Patient-level) 데이터를 불러옵니다...")
    all_df = pd.read_csv(PATIENT_PATH)

    if TARGET_COL not in all_df.columns:
        raise ValueError(f"'{TARGET_COL}' 컬럼이 전처리 데이터에 없습니다.")

    feature_cols = [
        col for col in all_df.columns
        if col not in DROP_COLS and pd.api.types.is_numeric_dtype(all_df[col])
    ]

    if not feature_cols:
        raise ValueError("학습에 사용할 숫자형 피처가 없습니다.")

    all_df[feature_cols] = all_df[feature_cols].replace([np.inf, -np.inf], np.nan)

    print(f"  Total Data: {all_df.shape} | subjects: {all_df['EMAIL'].nunique()}")
    print(f"  target: {TARGET_COL}")
    print(f"  features: {len(feature_cols)}")
    print("\n[Label distribution]")
    print(all_df["DIAG_NM"].value_counts())

    return all_df.reset_index(drop=True), feature_cols


all_df, all_features = load_preprocessed_data()

# =========================================================
# 4. 논문 방식의 Forward Feature Selection 함수
# =========================================================
def perform_forward_selection(all_data, all_feats, target_col):
    X_train = all_data[all_feats]
    y_train = all_data[target_col]
    
    print("\n[단계 1] SHAP을 통해 전체 피처의 중요도 순위를 계산합니다...")
    base_model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight='balanced', verbose=-1)
    base_model.fit(X_train, y_train)

    analyzer = ShapAnalyzer(model=base_model, feature_names=all_feats, task="binary", n_classes=1, class_names=["Dementia"])
    analyzer.explain(X_train)
    shap_df = analyzer.to_dataframe(combine_classes=False)
    
    # 중요도 순으로 줄 세우기
    ranked_features = shap_df['feature'].tolist()
    total_feats = min(len(ranked_features), FORWARD_SELECTION_MAX_FEATURES)
    ranked_features = ranked_features[:total_feats]
    
    print(f"\n[단계 2] 총 {total_feats}개의 피처에 대해 전진 선택법(Forward Selection)을 시작합니다.")
    print(f" SHAP 중요도 상위 {FORWARD_SELECTION_MAX_FEATURES}개까지만 탐색합니다.")
    print(" 1개부터 하나씩 추가하며 CV(교차검증) 성능을 측정합니다. (시간이 소요됩니다.)")
    
    # 논문의 방식을 그대로 재현하기 위해 환자 그룹화 없이 단순 5-Fold 적용 (데이터 누수 모사)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    best_k = 0
    best_cv_score = -1
    history_scores = []
    
    for k in range(1, total_feats + 1):
        current_features = ranked_features[:k]
        X_subset = X_train[current_features]
        
        fold_scores = []
        for train_idx, val_idx in cv.split(X_subset, y_train):
            X_tr, y_tr = X_subset.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_subset.iloc[val_idx], y_train.iloc[val_idx]
            
            # 빠른 평가를 위한 기본 파라미터 모델 (과적합 방지를 위해 early_stopping 적용)
            eval_model = LGBMClassifier(
                random_state=RANDOM_STATE,
                n_jobs=1,
                class_weight='balanced',
                n_estimators=FORWARD_SELECTION_ESTIMATORS,
                verbose=-1,
            )
            eval_model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(30, verbose=False)]
            )
            
            # 논문 명시대로 ROC-AUC를 기준으로 Feature 평가
            prob = eval_model.predict_proba(X_va)[:, 1]
            fold_scores.append(roc_auc_score(y_va, prob))
            
        mean_score = np.mean(fold_scores)
        history_scores.append(mean_score)
        
        if mean_score > best_cv_score:
            best_cv_score = mean_score
            best_k = k
            
        if k % 10 == 0 or k == total_feats:
            print(f"  -> 피처 {k:2d}개 누적 시 5-Fold AUC: {mean_score:.4f} (현재 최고기록: {best_k}개 사용 시 {best_cv_score:.4f})")
            
    print(f"\n[최적 피처 탐색 완료] 가장 높은 성능을 낸 상위 {best_k}개의 피처를 최종 선택합니다.")
    return ranked_features[:best_k], history_scores

# =========================================================
# 5. 최종 선택된 피처로 모델 학습 및 Grid Search
# =========================================================
def evaluate_model(model, X, y):
    prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, prob)
    fpr, tpr, thresholds = roc_curve(y, prob)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    
    pred_optimal = np.where(prob >= optimal_threshold, 1, 0)
    acc = accuracy_score(y, pred_optimal)
    prec = precision_score(y, pred_optimal, zero_division=0)
    rec = recall_score(y, pred_optimal, zero_division=0)
    f1 = f1_score(y, pred_optimal, zero_division=0)
    
    return acc, prec, rec, f1, auc, confusion_matrix(y, pred_optimal), optimal_threshold

def run_final_grid_search(all_data, selected_feats, target_col):
    print("\n[단계 3] 선택된 피처들로 전체 데이터 대상 5-Fold Grid Search 및 최종 평가를 진행합니다...")
    X_all = all_data[selected_feats]
    y_all = all_data[target_col]

    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    lgbm_param_list = list(product([33], [0.08], [1000], [41], [5, 10, 15], [None, 'balanced']))
    
    rf_param_list = list(product([5, 10, 20], [1000], [None, 'balanced']))
    models_to_run = [("LightGBM", lgbm_param_list), ("RandomForest", rf_param_list)]
    
    results = {}
    for model_name, param_list in models_to_run:
        best_score, best_params = -1, None
        
        for params in param_list:
            fold_scores = []
            for train_idx, valid_idx in cv_splitter.split(X_all, y_all):
                X_tr, y_tr = X_all.iloc[train_idx], y_all.iloc[train_idx]
                X_va, y_va = X_all.iloc[valid_idx], y_all.iloc[valid_idx]

                if model_name == "LightGBM":
                    model = LGBMClassifier(objective="binary", num_leaves=params[0], learning_rate=params[1], n_estimators=params[2], min_child_samples=params[3], max_depth=params[4], class_weight=params[5], random_state=42, n_jobs=1, verbose=-1)
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc', callbacks=[lgb.early_stopping(50, verbose=False)])
                elif model_name == "RandomForest":
                    model = RandomForestClassifier(max_depth=params[0], n_estimators=params[1], class_weight=params[2], random_state=42, n_jobs=1)
                    model.fit(X_tr, y_tr)

                _, _, _, _, auc, _, _ = evaluate_model(model, X_va, y_va)
                fold_scores.append(auc)

            mean_score = np.mean(fold_scores)
            if mean_score > best_score:
                best_score, best_params = mean_score, params

        print(f"  -> 최고 파라미터 찾기 완료: {best_params}")
        print(f"  -> 찾은 파라미터로 5-Fold 최종 성능 검증 시작...")
        
        final_fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': [], 'cm': []}
        
        for train_idx, test_idx in cv_splitter.split(X_all, y_all):
            X_tr, y_tr = X_all.iloc[train_idx], y_all.iloc[train_idx]
            X_te, y_te = X_all.iloc[test_idx], y_all.iloc[test_idx]
            
            if model_name == "LightGBM":
                final_model = LGBMClassifier(objective="binary", num_leaves=best_params[0], learning_rate=best_params[1], n_estimators=best_params[2], min_child_samples=best_params[3], max_depth=best_params[4], class_weight=best_params[5], random_state=42, n_jobs=1, verbose=-1)
            elif model_name == "RandomForest":
                final_model = RandomForestClassifier(max_depth=best_params[0], n_estimators=best_params[1], class_weight=best_params[2], random_state=42, n_jobs=1)
            
            final_model.fit(X_tr, y_tr)
            acc, prec, rec, f1, auc, cm, th = evaluate_model(final_model, X_te, y_te)
            
            final_fold_metrics['accuracy'].append(acc)
            final_fold_metrics['precision'].append(prec)
            final_fold_metrics['recall'].append(rec)
            final_fold_metrics['f1'].append(f1)
            final_fold_metrics['auc'].append(auc)
            final_fold_metrics['cm'].append(cm)
            
        results[model_name] = {
            "best_params": best_params, 
            "accuracy": np.mean(final_fold_metrics['accuracy']), 
            "precision": np.mean(final_fold_metrics['precision']), 
            "recall": np.mean(final_fold_metrics['recall']), 
            "f1": np.mean(final_fold_metrics['f1']), 
            "auc": np.mean(final_fold_metrics['auc']), 
            "cm": np.sum(final_fold_metrics['cm'], axis=0), 
            "threshold": np.nan
        }
        
    return results

# =========================================================
# 6. 파이프라인 실행 및 시각화
# =========================================================
# 논문 재현 전진 선택법 수행
optimal_features, forward_history = perform_forward_selection(all_df, all_features, TARGET_COL)

# 선택된 피처로 최종 학습
final_results = run_final_grid_search(all_df, optimal_features, TARGET_COL)

print("\n" + "="*60)
print("FINAL RESULTS (LightGBM & RF | 5-Fold CV on Merged Data)")
print("="*60)
for model_name in final_results:
    r = final_results[model_name]
    print(f"\n[{model_name}]")
    print(f"Optimal Threshold: {r['threshold']:.4f}")
    print(f"Accuracy: {r['accuracy']:.4f} | Precision: {r['precision']:.4f} | Recall: {r['recall']:.4f} | F1: {r['f1']:.4f} | AUC: {r['auc']:.4f}")

end_time = datetime.now()
print("\nElapsed:", end_time - start_time)

# --- 시각화 ---
print("\n[시각화 1] 피처 누적에 따른 성능 변화 곡선 (Forward Selection Curve)")
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(forward_history) + 1), forward_history, marker='o', linestyle='-', color='#4c72b0')
plt.axvline(x=len(optimal_features), color='red', linestyle='--', label=f'Optimal K = {len(optimal_features)}')
plt.title('Forward Selection Performance (Cross-Validation AUC)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Top Features Included', fontsize=12)
plt.ylabel('CV Mean ROC-AUC', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n[시각화 2] 혼동 행렬 (Confusion Matrix)")
class_names = ['Normal(0)', 'Dementia(1)']

# 두 모델의 결과를 나란히 그리기 위해 서브플롯 설정
fig, axes = plt.subplots(1, len(final_results), figsize=(6 * len(final_results), 5))
if len(final_results) == 1:
    axes = [axes]

for ax, (model_name, r) in zip(axes, final_results.items()):
    sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 15}, ax=ax)
    param_str = str(r['best_params'])
    wrapped_param = textwrap.fill(param_str, width=40) 
    ax.set_title(f"[{model_name}]\n{wrapped_param}", fontsize=12, pad=15)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)

plt.tight_layout()
plt.show()
