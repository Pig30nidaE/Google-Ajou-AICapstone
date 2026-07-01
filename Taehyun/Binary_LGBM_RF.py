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
# from sklearn.ensemble import RandomForestClassifier # 💡 RF 주석 처리
from sklearn.model_selection import StratifiedKFold 

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
sys.path.insert(0, str(current_dir.parent))
try:
    from xai import ShapAnalyzer
except ImportError:
    print("Error: 'xai' 패키지를 찾을 수 없습니다. 경로를 확인해주세요.")

start_time = datetime.now()
print("Start:", start_time)

# =========================================================
# 2. 데이터 전처리 및 병합 함수 정의 (기존과 동일)
# =========================================================
def preprocess_sleep(df):
    df = df.copy()
    single_val_cols = [col for col in df.columns if df[col].dropna().nunique() <= 1]
    df.drop(columns=single_val_cols, inplace=True, errors='ignore')
    
    cols_to_drop = [c for c in df.columns if '5min' in c or c.startswith('CONVERT(sleep')]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    if 'sleep_bedtime_start' in df.columns and 'sleep_bedtime_end' in df.columns:
        st = pd.to_datetime(df['sleep_bedtime_start'])
        et = pd.to_datetime(df['sleep_bedtime_end'])
        df['sleep_start_float'] = st.dt.hour + st.dt.minute / 60 + st.dt.second / 3600
        df['sleep_end_float'] = et.dt.hour + et.dt.minute / 60 + et.dt.second / 3600
        
        dur = df['sleep_end_float'] - df['sleep_start_float']
        df['sleep_time_calculated'] = np.where(dur < 0, dur + 24, dur)
        df.drop(columns=['sleep_bedtime_start', 'sleep_bedtime_end'], inplace=True)
    return df

def preprocess_activity(df):
    df = df.copy()
    single_val_cols = [col for col in df.columns if df[col].dropna().nunique() <= 1]
    df.drop(columns=single_val_cols, inplace=True, errors='ignore')
    df.drop(columns=['activity_day_start', 'activity_day_end'], inplace=True, errors='ignore')
        
    if 'CONVERT(activity_class_5min USING utf8)' in df.columns:
        def count_classes(val):
            if pd.isna(val): return {}
            return pd.Series([p for p in str(val).split('/') if p]).value_counts().to_dict()
        class_counts = df['CONVERT(activity_class_5min USING utf8)'].apply(count_classes)
        class_df = pd.DataFrame(class_counts.tolist()).fillna(0)
        class_df.columns = [f'activity_class_{c}_count' for c in class_df.columns]
        df = pd.concat([df, class_df], axis=1)
        df.drop(columns=['CONVERT(activity_class_5min USING utf8)'], inplace=True)
        
    if 'CONVERT(activity_met_1min USING utf8)' in df.columns:
        def calc_met_stats(val):
            if pd.isna(val): return [np.nan]*6
            parts = [float(p) for p in str(val).split('/') if p]
            if not parts: return [np.nan]*6
            arr = np.array(parts)
            return [np.mean(arr), np.std(arr), np.var(arr), np.percentile(arr, 25), np.percentile(arr, 50), np.percentile(arr, 75)]
        met_stats = df['CONVERT(activity_met_1min USING utf8)'].apply(calc_met_stats)
        met_cols = ['met_mean', 'met_std', 'met_var', 'met_q1', 'met_q2', 'met_q3']
        met_df = pd.DataFrame(met_stats.tolist(), columns=met_cols)
        df = pd.concat([df, met_df], axis=1)
        df.drop(columns=['CONVERT(activity_met_1min USING utf8)'], inplace=True)
    return df

def merge_and_label(feature_df, label_df, split_name):
    merged_df = pd.merge(feature_df, label_df, left_on='EMAIL', right_on='SAMPLE_EMAIL', how='inner')
    merged_df.drop(columns=['SAMPLE_EMAIL'], inplace=True)
    merged_df['binary_class'] = merged_df['DIAG_NM'].map({'CN': 0, 'MCI': 1, 'Dem': 1})
    merged_df.drop(columns=['DIAG_NM'], inplace=True)
    merged_df.rename(columns={'EMAIL': 'patient_id'}, inplace=True)
    merged_df['split'] = split_name
    return merged_df

# =========================================================
# 3. 데이터 로드 및 전처리 (본인 경로에 맞게 수정 요망)
# =========================================================
print("[데이터 전처리] 원본 데이터 전처리 및 병합을 시작합니다...")
train_sleep = pd.read_csv("C:/ML4/data/train_sleep.csv")
val_sleep = pd.read_csv("C:/ML4/data/val_sleep.csv")
train_act = pd.read_csv("C:/ML4/data/train_activity.csv")
val_act = pd.read_csv("C:/ML4/data/val_activity.csv")

train_label_s = pd.read_csv("C:/ML4/data/training_label_sleep.csv")
val_label_s = pd.read_csv("C:/ML4/data/val_label_sleep.csv")
train_label_a = pd.read_csv("C:/ML4/data/training_label_activity.csv")
val_label_a = pd.read_csv("C:/ML4/data/val_label_activity.csv")

t_sleep_c = preprocess_sleep(train_sleep)
v_sleep_c = preprocess_sleep(val_sleep)
t_act_c = preprocess_activity(train_act)
v_act_c = preprocess_activity(val_act)

t_sleep_f = merge_and_label(t_sleep_c, train_label_s, 'train')
v_sleep_f = merge_and_label(v_sleep_c, val_label_s, 'val')
t_act_f = merge_and_label(t_act_c, train_label_a, 'train')
v_act_f = merge_and_label(v_act_c, val_label_a, 'val')

merge_keys = ['patient_id', 'split', 'binary_class']
train_merged = pd.merge(t_act_f, t_sleep_f, on=merge_keys, how='inner')
val_merged = pd.merge(v_act_f, v_sleep_f, on=merge_keys, how='inner')

train_df = train_merged.reset_index(drop=True)
test_df = val_merged.reset_index(drop=True)

TARGET_COL = 'binary_class' 
DROP_COLS = ['patient_id', 'split', TARGET_COL]
all_features = [col for col in train_df.columns if col not in DROP_COLS]

print(f"데이터 준비 완료: 전체 피처 수 {len(all_features)}개")

# =========================================================
# 4. 논문 방식의 Forward Feature Selection 함수
# =========================================================
def perform_forward_selection(train_data, all_feats, target_col):
    X_train = train_data[all_feats]
    y_train = train_data[target_col]
    
    print("\n[단계 1] SHAP을 통해 전체 피처의 중요도 순위를 계산합니다...")
    base_model = LGBMClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    base_model.fit(X_train, y_train)

    analyzer = ShapAnalyzer(model=base_model, feature_names=all_feats, task="binary", n_classes=1, class_names=["Dementia"])
    analyzer.explain(X_train)
    shap_df = analyzer.to_dataframe(combine_classes=False)
    
    # 중요도 순으로 줄 세우기
    ranked_features = shap_df['feature'].tolist()
    total_feats = len(ranked_features)
    
    print(f"\n[단계 2] 총 {total_feats}개의 피처에 대해 전진 선택법(Forward Selection)을 시작합니다.")
    print(" 💡 1개부터 하나씩 추가하며 CV(교차검증) 성능을 측정합니다. (시간이 소요됩니다.)")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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
            
            # 빠른 평가를 위한 기본 파라미터 모델
            eval_model = LGBMClassifier(random_state=42, n_jobs=-1, class_weight='balanced', verbose=-1)
            eval_model.fit(X_tr, y_tr)
            
            preds = eval_model.predict(X_va)
            fold_scores.append(f1_score(y_va, preds, zero_division=0))
            
        mean_score = np.mean(fold_scores)
        history_scores.append(mean_score)
        
        if mean_score > best_cv_score:
            best_cv_score = mean_score
            best_k = k
            
        if k % 10 == 0 or k == total_feats:
            print(f"  -> 피처 {k:2d}개 누적 시 5-Fold F1: {mean_score:.4f} (현재 최고기록: {best_k}개 사용 시 {best_cv_score:.4f})")
            
    print(f"\n✅ [최적 피처 탐색 완료] 가장 높은 성능을 낸 상위 {best_k}개의 피처를 최종 선택합니다!")
    return ranked_features[:best_k], history_scores

# =========================================================
# 5. 최종 선택된 피처로 모델 학습 및 Grid Search
# =========================================================
def lgbm_f1_eval(y_true, y_pred):
    y_bin = np.where(y_pred > 0.5, 1, 0)
    return 'f1', f1_score(y_true, y_bin, zero_division=0), True

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

def run_final_grid_search(train_data, test_data, selected_feats, target_col):
    print("\n[단계 3] 선택된 피처들로 LightGBM Grid Search 하이퍼파라미터 튜닝을 진행합니다...")
    X_train = train_data[selected_feats]
    y_train = train_data[target_col]
    X_test = test_data[selected_feats]
    y_test = test_data[target_col]

    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lgbm_param_list = list(product([330], [0.08], [1000], [41], [5, 10, 15], [None, 'balanced']))
    
    # 💡 RF는 주석 처리하여 제외
    # rf_param_list = list(product([5, 10, 20], [1000], [None, 'balanced']))
    models_to_run = [("LightGBM", lgbm_param_list)]
    
    results = {}
    for model_name, param_list in models_to_run:
        best_score, best_params = -1, None
        
        for params in param_list:
            fold_scores = []
            for train_idx, valid_idx in cv_splitter.split(X_train, y_train):
                X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
                X_va, y_va = X_train.iloc[valid_idx], y_train.iloc[valid_idx]

                model = LGBMClassifier(objective="binary", num_leaves=params[0], learning_rate=params[1], n_estimators=params[2], min_child_samples=params[3], max_depth=params[4], class_weight=params[5], random_state=42, n_jobs=-1, verbose=-1)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric=lgbm_f1_eval, callbacks=[lgb.early_stopping(50, verbose=False)])

                _, _, _, f1, _, _, _ = evaluate_model(model, X_va, y_va)
                fold_scores.append(f1)

            mean_score = np.mean(fold_scores)
            if mean_score > best_score:
                best_score, best_params = mean_score, params

        print(f"  -> 최고 파라미터 찾기 완료: {best_params}")
        
        # Test 셋을 위한 최종 재학습
        final_model = LGBMClassifier(objective="binary", num_leaves=best_params[0], learning_rate=best_params[1], n_estimators=best_params[2], min_child_samples=best_params[3], max_depth=best_params[4], class_weight=best_params[5], random_state=42, n_jobs=-1, verbose=-1)
        final_model.fit(X_train, y_train)
        
        acc, prec, rec, f1, auc, cm, th = evaluate_model(final_model, X_test, y_test)
        results[model_name] = {"best_params": best_params, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc, "cm": cm, "threshold": th}
        
    return results

# =========================================================
# 6. 파이프라인 실행 및 시각화
# =========================================================
# 논문 재현 전진 선택법 수행
optimal_features, forward_history = perform_forward_selection(train_df, all_features, TARGET_COL)

# 선택된 피처로 최종 학습
final_results = run_final_grid_search(train_df, test_df, optimal_features, TARGET_COL)

print("\n" + "="*60)
print("FINAL RESULTS (LightGBM | Forward Selection | Pre-split Data)")
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
plt.title('Forward Selection Performance (Cross-Validation F1 Score)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Top Features Included', fontsize=12)
plt.ylabel('CV Mean F1 Score', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n[시각화 2] 혼동 행렬 (Confusion Matrix)")
class_names = ['Normal(0)', 'Dementia(1)']
r = final_results["LightGBM"]
plt.figure(figsize=(6, 5))
sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 15})
param_str = str(r['best_params'])
wrapped_param = textwrap.fill(param_str, width=40) 
plt.title(f"[LightGBM]\n{wrapped_param}", fontsize=12, pad=15)
plt.xlabel('Predicted Label', fontsize=11)
plt.ylabel('True Label', fontsize=11)
plt.tight_layout()
plt.show()