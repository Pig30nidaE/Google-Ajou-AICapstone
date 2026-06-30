# Auto-generated Python script converted from a Jupyter notebook.
# Source notebook: Taehyun/previous/Binary_LGBM_RF.ipynb
# Do not edit this generated file if you need exact notebook parity; edit the source notebook or copy this file first.

# Notebook compatibility helpers. Generated to keep notebook shell/magic cells runnable as Python.
import os as _NOTEBOOK_OS
import subprocess as _NOTEBOOK_SUBPROCESS
from pathlib import Path as _NOTEBOOK_PATH


def _NOTEBOOK_RUN_SHELL(command: str) -> None:
    _NOTEBOOK_SUBPROCESS.run(command, shell=True, check=True)


def _NOTEBOOK_RUN_BASH(script: str) -> None:
    _NOTEBOOK_SUBPROCESS.run(script, shell=True, executable="/bin/bash", check=True)


def _NOTEBOOK_CD(path: str) -> None:
    _NOTEBOOK_OS.chdir(_NOTEBOOK_OS.path.expanduser(path))
    print(_NOTEBOOK_PATH.cwd())


# %% cell 1
from google.colab import drive

drive.mount('/content/drive')

# %% cell 2
_NOTEBOOK_RUN_SHELL('pip install xai')

# %% cell 3
# =========================================================
# 1. 라이브러리 임포트 및 데이터 로드
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
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
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

# 한글 폰트 설정 (윈도우: Malgun Gothic, 맥: AppleGothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# xai 패키지 경로 추가 및 import
# 주피터 노트북 환경에 맞춰 __file__ 대신 os.getcwd() 사용
current_dir = pathlib.Path(os.getcwd())
sys.path.insert(0, str(current_dir.parent))
try:
    from xai import ShapAnalyzer
except ImportError:
    print("Error: 'xai' 패키지를 찾을 수 없습니다. 경로를 확인해주세요.")
    # 주피터 환경에서는 커널이 죽지 않도록 sys.exit(1) 대신 경고만 출력합니다.

# ---------------------------------------------------------
# 파일 로드 및 통합
# ---------------------------------------------------------
activity_file = pathlib.Path('/content/drive/MyDrive/GoogleAI_contest/Hyunsoo/previous/privious_TreeModel_preprocessing/rf_lgbm_activity_discrete.csv') # colab 기준 경로 지정
sleep_file = pathlib.Path('/content/drive/MyDrive/GoogleAI_contest/Hyunsoo/previous/privious_TreeModel_preprocessing/rf_lgbm_sleep_discrete.csv')# colab 기준 경로 지정

print("[데이터 로드] 활동 데이터와 수면 데이터를 병합합니다...")
df_activity = pd.read_csv(activity_file)
df_sleep = pd.read_csv(sleep_file)

merge_keys = ['patient_id', 'sample_date', 'split', 'binary_class']
df = pd.merge(df_activity, df_sleep, on=merge_keys, how='inner')

TARGET_COL = 'binary_class'
DROP_COLS = ['patient_id', 'sample_date', 'split', TARGET_COL]
all_features = [col for col in df.columns if col not in DROP_COLS]

print(f"데이터 로드 완료: 총 데이터 수 {len(df)}행, 사용 피처 수 {len(all_features)}개")

# %% [markdown] cell 4
# ### 'xai' 패키지 경로 설정 및 재시도
#
# 위에서 `xai` 패키지를 찾을 수 없다는 오류 메시지가 출력되었습니다. 이는 Python 인터프리터가 `ShapAnalyzer` 클래스를 포함하는 `xai` 모듈의 위치를 찾지 못했기 때문입니다.
#
# 아래 셀에서 `XAI_PACKAGE_DIR` 변수에 `xai` 패키지가 설치된 (또는 위치한) 디렉토리의 절대 경로를 입력하여 `sys.path`에 추가해주세요. 일반적으로 `xai`는 현재 노트북 파일의 상위 디렉토리에 위치할 수 있습니다. 경로를 수정한 후 이 셀을 실행하면 `ShapAnalyzer`를 다시 임포트하여 오류가 해결되었는지 확인할 수 있습니다.

# %% cell 5
# 'xai' 패키지가 있는 디렉토리 경로를 여기에 입력하세요.
# 예: '/content/drive/MyDrive/my_project/xai_package_folder'
XAI_PACKAGE_DIR = '' # 여기에 올바른 경로를 입력하세요.

# 현재 sys.path에 XAI_PACKAGE_DIR이 없으면 추가합니다.
if XAI_PACKAGE_DIR and XAI_PACKAGE_DIR not in sys.path:
    sys.path.insert(0, XAI_PACKAGE_DIR)

try:
    # 기존 import 시도 (cell 51239b5f)에서 'xai' 패키지를 찾지 못했을 경우,
    # 여기에 다시 import 시도합니다.
    from xai import ShapAnalyzer
    print("'xai.ShapAnalyzer' 임포트 성공!")
except ImportError:
    print("경로를 추가한 후에도 'xai' 패키지를 찾을 수 없습니다. XAI_PACKAGE_DIR을 다시 확인해주세요.")
    print(f"현재 sys.path: {sys.path}")

# 이제 run_experiment 함수를 포함하는 셀을 다시 실행할 수 있습니다.

# %% cell 6
# 'xai' 패키지 설치 후 다시 임포트 시도

try:
    from xai import ShapAnalyzer
    print("'xai.ShapAnalyzer' 임포트 성공!")
except ImportError:
    print("여전히 'xai' 패키지를 찾을 수 없습니다. 설치가 제대로 되었는지 확인해주세요.")

# %% cell 7
# =========================================================
# 2. 함수 정의 (평가, 모델링 파이프라인)
# =========================================================

# LightGBM 실시간 F1 평가용 커스텀 함수
def lgbm_f1_eval(y_true, y_pred):
    y_bin = np.where(y_pred > 0.5, 1, 0)
    f1 = f1_score(y_true, y_bin, zero_division=0)
    return 'f1', f1, True

# Evaluation Function (최적 Threshold 탐색 및 Precision/Recall 추가)
def evaluate_model(model, X, y):
    prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, prob)

    fpr, tpr, thresholds = roc_curve(y, prob)
    J = tpr - fpr
    optimal_idx = np.argmax(J)
    optimal_threshold = thresholds[optimal_idx]

    pred_optimal = np.where(prob >= optimal_threshold, 1, 0)

    acc = accuracy_score(y, pred_optimal)
    prec = precision_score(y, pred_optimal, zero_division=0)
    rec = recall_score(y, pred_optimal, zero_division=0)
    f1 = f1_score(y, pred_optimal, zero_division=0)

    report = classification_report(y, pred_optimal, zero_division=0)
    cm = confusion_matrix(y, pred_optimal)

    return acc, prec, rec, f1, auc, report, cm, optimal_threshold

# 파이프라인 함수: SHAP -> Scale -> CV -> Train -> Evaluate
def run_experiment(train_df, test_df, cv_type, all_features, target_col):
    print(f"\n[{cv_type} 환경] 모델 파이프라인 시작...")

    X_train_full = train_df[all_features]
    y_train_full = train_df[target_col]
    X_test_full = test_df[all_features]
    y_test = test_df[target_col]

    groups_train = train_df['patient_id'] if cv_type == "GroupKFold" else None

    # 1. SHAP 피처 셀렉션
    print("  -> Base 모델 학습 및 SHAP Top 20 피처 추출 중...")
    proxy_model = LGBMClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    proxy_model.fit(X_train_full, y_train_full)

    analyzer = ShapAnalyzer(
        model=proxy_model, feature_names=all_features,
        task="binary", n_classes=1, class_names=["Dementia"]
    )
    analyzer.explain(X_train_full)
    shap_df = analyzer.to_dataframe(combine_classes=False)
    best_features = shap_df.head(20)['feature'].tolist()

    X_train_selected = X_train_full[best_features]
    X_test_selected = X_test_full[best_features]

    # 2. 정규화
    print("  -> 선택된 피처 MinMaxScaler 적용 중...")
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_selected), columns=X_train_selected.columns, index=X_train_selected.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_selected), columns=X_test_selected.columns, index=X_test_selected.index)

    # 3. K-Fold 설정 및 모델 파라미터
    smote = SMOTE(random_state=42)
    if cv_type == "GroupKFold":
        cv_splitter = GroupKFold(n_splits=5)
    else:
        cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lgbm_param_list_no_smote = list(product([330], [0.08], [1000], [41], [5, 10, 15], [None, 'balanced']))
    rf_param_list_no_smote = list(product([5, 10, 20], [1000], [None, 'balanced',{0: 1, 1: 5},{0: 1, 1: 10},{0: 1, 1: 20},{0: 1, 1: 50}]))
    lgbm_param_list_smote = list(product([330], [0.08], [1000], [41], [5, 10, 15], [None]))
    rf_param_list_smote = list(product([5, 10, 20], [1000], [None]))

    results = {"NO_SMOTE": {}, "SMOTE": {}}

    # 4. 모델 훈련 및 평가 로직 (반복문)
    models_to_run = [
        ("NO_SMOTE", "LightGBM", lgbm_param_list_no_smote),
        ("NO_SMOTE", "RandomForest", rf_param_list_no_smote),
        ("SMOTE", "LightGBM", lgbm_param_list_smote),
        ("SMOTE", "RandomForest", rf_param_list_smote)
    ]

    for smote_type, model_name, param_list in models_to_run:
        best_score, best_params = -1, None

        for params in param_list:
            fold_scores = []
            split_args = (X_train_scaled, y_train_full)
            kwargs = {'groups': groups_train} if cv_type == "GroupKFold" else {}

            for train_idx, valid_idx in cv_splitter.split(*split_args, **kwargs):
                X_train, y_train = X_train_scaled.iloc[train_idx], y_train_full.iloc[train_idx]
                X_valid, y_valid = X_train_scaled.iloc[valid_idx], y_train_full.iloc[valid_idx]

                if smote_type == "SMOTE":
                    X_train, y_train = smote.fit_resample(X_train, y_train)

                if model_name == "LightGBM":
                    model = LGBMClassifier(
                        objective="binary", num_leaves=params[0], learning_rate=params[1],
                        n_estimators=params[2], min_child_samples=params[3], max_depth=params[4],
                        class_weight=params[5], random_state=42, n_jobs=-1, verbose=-1
                    )
                    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric=lgbm_f1_eval, callbacks=[lgb.early_stopping(50, verbose=False)])
                else:
                    model = RandomForestClassifier(
                        max_depth=params[0], n_estimators=params[1],
                        class_weight=params[2], random_state=42, n_jobs=-1, verbose=0
                    )
                    model.fit(X_train, y_train)

                _, _, _, f1, _, _, _, _ = evaluate_model(model, X_valid, y_valid)
                fold_scores.append(f1)

            mean_score = np.mean(fold_scores)
            if mean_score > best_score:
                best_score, best_params = mean_score, params

        # 최종 모델 재학습 및 Test 평가
        X_train_final, y_train_final = X_train_scaled, y_train_full
        if smote_type == "SMOTE":
            X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train_full)

        if model_name == "LightGBM":
            final_model = LGBMClassifier(objective="binary", num_leaves=best_params[0], learning_rate=best_params[1], n_estimators=best_params[2], min_child_samples=best_params[3], max_depth=best_params[4], class_weight=best_params[5], random_state=42, n_jobs=-1, verbose=-1)
        else:
            final_model = RandomForestClassifier(max_depth=best_params[0], n_estimators=best_params[1], class_weight=best_params[2], random_state=42, n_jobs=-1, verbose=0)

        final_model.fit(X_train_final, y_train_final)
        acc, prec, rec, f1, auc, report, cm, th = evaluate_model(final_model, X_test_scaled, y_test)

        results[smote_type][model_name] = {
            "best_params": best_params, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "auc": auc, "report": report, "cm": cm, "threshold": th
        }
        print(f"  -> [{smote_type}] {model_name} 학습 완료 (AUC: {auc:.4f})")

    return results

# %% cell 8
# =========================================================
# 3. 모델 학습 및 결과 출력
# =========================================================
start_time = datetime.now()
print("Start:", start_time)

# EXPERIMENT 1: Group K-Fold (환자 독립성 보장 - 누수 없음)
train_df_gkf = df[df['split'] == 'train'].reset_index(drop=True)
test_df_gkf = df[df['split'] == 'val'].reset_index(drop=True)
results_gkf = run_experiment(train_df_gkf, test_df_gkf, "GroupKFold", all_features, TARGET_COL)

# EXPERIMENT 2: Stratified K-Fold (랜덤 분할 - 데이터 누수 발생)
train_df_skf, test_df_skf = train_test_split(df, test_size=0.3, stratify=df[TARGET_COL], random_state=42)
train_df_skf = train_df_skf.reset_index(drop=True)
test_df_skf = test_df_skf.reset_index(drop=True)
results_skf = run_experiment(train_df_skf, test_df_skf, "StratifiedKFold", all_features, TARGET_COL)

# 최종 콘솔 출력
print("\n" + "="*60)
print("FINAL RESULTS COMPARISON (Binary Classification)")
print("="*60)

for cv_name, res_dict in [("Group K-Fold", results_gkf), ("Stratified K-Fold", results_skf)]:
    print(f"\n\n▶▶ {cv_name} 결과 ◀◀")
    for smote_type in res_dict:
        for model_name in res_dict[smote_type]:
            r = res_dict[smote_type][model_name]
            print(f"\n[{smote_type} - {model_name}]")
            print(f"Optimal Threshold: {r['threshold']:.4f} | Best Params: {r['best_params']}")
            print(f"Accuracy: {r['accuracy']:.4f} | Precision: {r['precision']:.4f} | Recall: {r['recall']:.4f} | F1: {r['f1']:.4f} | AUC: {r['auc']:.4f}")

end_time = datetime.now()
print("\nEnd:", end_time)
print("Elapsed:", end_time - start_time)

# %% cell 9
# =========================================================
# 4. 시각화 (Jupyter Notebook 인라인 출력)
# =========================================================
print("\n[시각화] 막대 차트(LGBM -> RF) 및 개별 혼동 행렬 시각화 결과입니다.")

# 공통 함수: 특정 모델의 성능을 Bar Chart로 그리는 함수
def plot_bar_chart_for_model(target_model_name):
    model_labels = []
    accuracies, precisions, recalls, f1_scores, auc_scores = [], [], [], [], []

    for cv_name, cv_prefix, res_dict in [("GroupKFold", "GKF", results_gkf), ("StratifiedKFold", "SKF", results_skf)]:
        for dataset_type in ["NO_SMOTE", "SMOTE"]:
            label = f"{cv_prefix}\n{dataset_type}"
            model_labels.append(label)
            r = res_dict[dataset_type][target_model_name]
            accuracies.append(r['accuracy'])
            precisions.append(r['precision'])
            recalls.append(r['recall'])
            f1_scores.append(r['f1'])
            auc_scores.append(r['auc'])

    x = np.arange(len(model_labels))
    width = 0.15

    plt.figure(figsize=(12, 7))
    ax_bar = plt.gca()

    rects1 = ax_bar.bar(x - 2*width, accuracies, width, label='Accuracy', color='#4c72b0')
    rects2 = ax_bar.bar(x - width, precisions, width, label='Precision', color='#dd8452')
    rects3 = ax_bar.bar(x, recalls, width, label='Recall', color='#8172b3')
    rects4 = ax_bar.bar(x + width, f1_scores, width, label='F1', color='#55a868')
    rects5 = ax_bar.bar(x + 2*width, auc_scores, width, label='AUC', color='#c44e52')

    ax_bar.set_ylabel('Scores', fontsize=12)
    ax_bar.set_title(f'Performance Comparison: {target_model_name} (GKF vs SKF)', fontsize=16, fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(model_labels, fontsize=11)
    ax_bar.legend(loc='upper right', bbox_to_anchor=(1, 1.15), ncol=5)
    ax_bar.set_ylim(0, 1.15)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax_bar.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)

    plt.tight_layout()
    plt.show()

# 1 & 2) 막대 그래프 그리기
plot_bar_chart_for_model("LightGBM")
plot_bar_chart_for_model("RandomForest")

# 3) 혼동 행렬 시각화
class_names = ['Normal(0)', 'Dementia(1)']

# Group K-Fold 혼동 행렬 (블루 톤)
for ds in ["NO_SMOTE", "SMOTE"]:
    for model in ["LightGBM", "RandomForest"]:
        r = results_gkf[ds][model]

        plt.figure(figsize=(6, 5))
        sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={"size": 15})

        param_str = str(r['best_params'])
        wrapped_param = textwrap.fill(param_str, width=40)

        plt.title(f"[Group K-Fold]\n{ds} - {model}\n{wrapped_param}", fontsize=12, pad=15)
        plt.xlabel('Predicted Label', fontsize=11)
        plt.ylabel('True Label', fontsize=11)
        plt.tight_layout()
        plt.show()

# Stratified K-Fold 혼동 행렬 (레드 톤)
for ds in ["NO_SMOTE", "SMOTE"]:
    for model in ["LightGBM", "RandomForest"]:
        r = results_skf[ds][model]

        plt.figure(figsize=(6, 5))
        sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Reds',
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={"size": 15})

        param_str = str(r['best_params'])
        wrapped_param = textwrap.fill(param_str, width=40)

        plt.title(f"[Stratified K-Fold]\n{ds} - {model}\n{wrapped_param}", fontsize=12, pad=15)
        plt.xlabel('Predicted Label', fontsize=11)
        plt.ylabel('True Label', fontsize=11)
        plt.tight_layout()
        plt.show()
