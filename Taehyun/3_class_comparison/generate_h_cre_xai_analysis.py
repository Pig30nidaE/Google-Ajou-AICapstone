import os
import sys
import pathlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

COMP_DIR = pathlib.Path(r"c:\ML4\Google-Ajou-AICapstone\Taehyun\3_class_comparison")
sys.path.insert(0, str(COMP_DIR))
from H_CRE_Optuna_Tuning import load_and_preprocess_data, select_stage_features

def main():
    params_file = COMP_DIR / "h_cre_optuna_best_params.json"
    with open(params_file, 'r', encoding='utf-8') as f:
        best_info = json.load(f)
    opt_params = best_info['params']

    df, all_features = load_and_preprocess_data()
    s1_feats = select_stage_features(df, all_features, 'stage1_label', 'Stage1')
    s2_feats = select_stage_features(df, all_features, 'stage2_label', 'Stage2')

    # =========================================================
    # 1. Stage 1 (XGBoost): CN vs Abnormal (10 features)
    # =========================================================
    X_s1 = df[s1_feats].copy()
    y_s1 = df['stage1_label'].astype(int)
    
    sm_s1 = SMOTE(k_neighbors=4, random_state=42)
    X_s1_res, y_s1_res = sm_s1.fit_resample(X_s1, y_s1)
    
    xgb_params = opt_params['Stage1']['XGBoost']
    model_s1 = XGBClassifier(**xgb_params, random_state=42, n_jobs=1)
    model_s1.fit(X_s1_res, y_s1_res)
    
    explainer_s1 = shap.TreeExplainer(model_s1)
    shap_values_s1 = explainer_s1.shap_values(X_s1)
    
    mean_abs_shap_s1 = np.abs(shap_values_s1).mean(axis=0)
    s1_ranking_df = pd.DataFrame({
        'Feature': s1_feats,
        'Mean_Abs_SHAP': mean_abs_shap_s1
    }).sort_values(by='Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
    
    # Calculate feature effect direction (Pearson corr between feature value and SHAP value)
    corr_s1 = [np.corrcoef(X_s1[f], shap_values_s1[:, i])[0, 1] for i, f in enumerate(s1_feats)]
    s1_ranking_df['Direction'] = [
        "위험 증가 (+)" if c > 0.1 else ("보호 요인 (-)" if c < -0.1 else "비선형 복합") 
        for c in [corr_s1[s1_feats.index(f)] for f in s1_ranking_df['Feature']]
    ]
    
    # =========================================================
    # 2. Stage 2 (LightGBM): MCI vs Dementia (22 features)
    # =========================================================
    df_abnormal = df[df['stage1_label'] == 1].copy().reset_index(drop=True)
    X_s2 = df_abnormal[s2_feats].copy()
    y_s2 = df_abnormal['stage2_label'].astype(int)
    
    sm_s2 = SMOTE(k_neighbors=3, random_state=42)
    X_s2_res, y_s2_res = sm_s2.fit_resample(X_s2, y_s2)
    
    lgb_params = opt_params['Stage2']['LightGBM']
    model_s2 = LGBMClassifier(**lgb_params, random_state=42, n_jobs=1, verbose=-1)
    model_s2.fit(X_s2_res, y_s2_res)
    
    explainer_s2 = shap.TreeExplainer(model_s2)
    shap_values_s2 = explainer_s2.shap_values(X_s2)
    if isinstance(shap_values_s2, list):
        shap_vals_s2_dem = shap_values_s2[1]
    else:
        shap_vals_s2_dem = shap_values_s2
        
    mean_abs_shap_s2 = np.abs(shap_vals_s2_dem).mean(axis=0)
    s2_ranking_df = pd.DataFrame({
        'Feature': s2_feats,
        'Mean_Abs_SHAP': mean_abs_shap_s2
    }).sort_values(by='Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
    
    corr_s2 = [np.corrcoef(X_s2[f], shap_vals_s2_dem[:, i])[0, 1] for i, f in enumerate(s2_feats)]
    s2_ranking_df['Direction'] = [
        "치매 위험 증가 (+)" if c > 0.1 else ("치매 위험 감소 (-)" if c < -0.1 else "비선형 복합") 
        for c in [corr_s2[s2_feats.index(f)] for f in s2_ranking_df['Feature']]
    ]

    print("=" * 80)
    print("[XAI 분석 1] Stage 1 (XGBoost: 정상 vs 인지저하) 피처 기여도 Top 10")
    print("=" * 80)
    for idx, row in s1_ranking_df.iterrows():
        print(f"{idx+1:2d}. {row['Feature']:<30} | Mean |SHAP|: {row['Mean_Abs_SHAP']:.4f} | 영향 방향: {row['Direction']}")

    print("\n" + "=" * 80)
    print("[XAI 분석 2] Stage 2 (LightGBM: MCI vs 치매) 피처 기여도 Top 15 (총 22개 중)")
    print("=" * 80)
    for idx, row in s2_ranking_df.head(15).iterrows():
        print(f"{idx+1:2d}. {row['Feature']:<30} | Mean |SHAP|: {row['Mean_Abs_SHAP']:.4f} | 영향 방향: {row['Direction']}")

    # =========================================================
    # 3. 고해상도 시각화 차트 생성 (2-패널 막대 차트)
    # =========================================================
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Subplot 1: Stage 1 XGBoost SHAP Importance
    y_pos1 = np.arange(len(s1_ranking_df))
    colors1 = ['#e74c3c' if '+' in d else ('#2980b9' if '-' in d else '#7f8c8d') for d in s1_ranking_df['Direction']]
    ax1.barh(y_pos1, s1_ranking_df['Mean_Abs_SHAP'], color=colors1, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax1.set_yticks(y_pos1)
    ax1.set_yticklabels(s1_ranking_df['Feature'], fontsize=11)
    ax1.invert_yaxis()
    ax1.set_xlabel('평균 절대 SHAP 값 (Mean |SHAP Value|)', fontsize=12, labelpad=10)
    ax1.set_title('[Stage 1] 조기 선별 (CN vs Abnormal) 피처 기여도\n(XGBoost 분류기, N=174)', fontsize=13, weight='bold', pad=12)
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    
    for i, v in enumerate(s1_ranking_df['Mean_Abs_SHAP']):
        ax1.text(v + 0.005, i, f"{v:.4f}", va='center', fontsize=10, weight='bold')

    # Subplot 2: Stage 2 LightGBM SHAP Importance (Top 12)
    top12_s2 = s2_ranking_df.head(12)
    y_pos2 = np.arange(len(top12_s2))
    colors2 = ['#d35400' if '+' in d else ('#27ae60' if '-' in d else '#8e44ad') for d in top12_s2['Direction']]
    ax2.barh(y_pos2, top12_s2['Mean_Abs_SHAP'], color=colors2, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(top12_s2['Feature'], fontsize=11)
    ax2.invert_yaxis()
    ax2.set_xlabel('평균 절대 SHAP 값 (Mean |SHAP Value|)', fontsize=12, labelpad=10)
    ax2.set_title('[Stage 2] 치매 정밀 감별 (MCI vs Dementia) 피처 기여도 Top 12\n(LightGBM 분류기, 소표본 규제화)', fontsize=13, weight='bold', pad=12)
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    
    for i, v in enumerate(top12_s2['Mean_Abs_SHAP']):
        ax2.text(v + 0.005, i, f"{v:.4f}", va='center', fontsize=10, weight='bold')

    plt.suptitle("H-CRE (Hierarchical Circadian Regularized Ensemble) XAI 피처 기여도 분석 (TreeSHAP)", fontsize=16, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_plot = COMP_DIR / "shap_feature_importance_h_cre.png"
    plt.savefig(out_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[XAI 차트 저장 완료] {out_plot}")
    
    # Save CSV tables
    s1_ranking_df.to_csv(COMP_DIR / "shap_importance_stage1.csv", index=False, encoding='utf-8-sig')
    s2_ranking_df.to_csv(COMP_DIR / "shap_importance_stage2.csv", index=False, encoding='utf-8-sig')
    print("[CSV 파일 저장 완료]")

if __name__ == '__main__':
    main()
