import os
import sys
import pathlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

CURR_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(CURR_DIR))
from H_CRE_Optuna_Tuning import load_and_preprocess_data, select_stage_features
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

def main():
    params_file = CURR_DIR / "h_cre_optuna_best_params.json"
    with open(params_file, 'r', encoding='utf-8') as f:
        best_info = json.load(f)
    opt_params = best_info['params']

    df, all_features = load_and_preprocess_data()
    s1_feats = select_stage_features(df, all_features, 'stage1_label', 'Stage1')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_true_binary = (df['original_label'] > 0).astype(int).values
    oof_prob_abnormal = {'Ensemble': np.zeros(len(df)), 'XGBoost': np.zeros(len(df)), 'LightGBM': np.zeros(len(df))}

    for fold, (tr_i, te_i) in enumerate(skf.split(df, df['original_label']), 1):
        df_tr, df_te = df.iloc[tr_i].copy(), df.iloc[te_i].copy()
        X_tr_s1, y_tr_s1 = df_tr[s1_feats], df_tr['stage1_label'].astype(int)
        sm_s1 = SMOTE(k_neighbors=max(1, min(4, y_tr_s1.value_counts().min() - 1)), random_state=42)
        X_tr_s1_res, y_tr_s1_res = sm_s1.fit_resample(X_tr_s1, y_tr_s1)
        
        probs_stage1 = {}
        for m in ['LightGBM', 'CatBoost', 'XGBoost', 'RandomForest']:
            p1 = opt_params['Stage1'][m]
            if m == 'LightGBM': m1 = LGBMClassifier(**p1, random_state=42, n_jobs=1, verbose=-1)
            elif m == 'CatBoost': m1 = CatBoostClassifier(**p1, random_state=42, verbose=0, thread_count=1)
            elif m == 'XGBoost': m1 = XGBClassifier(**p1, random_state=42, n_jobs=1)
            elif m == 'RandomForest': m1 = RandomForestClassifier(**p1, random_state=42, n_jobs=1)
            m1.fit(X_tr_s1_res, y_tr_s1_res)
            probs_stage1[m] = m1.predict_proba(df_te[s1_feats])[:, 1]
            
        oof_prob_abnormal['Ensemble'][te_i] = (probs_stage1['LightGBM'] + probs_stage1['CatBoost'] + probs_stage1['XGBoost'] + probs_stage1['RandomForest']) / 4.0
        oof_prob_abnormal['XGBoost'][te_i] = probs_stage1['XGBoost']
        oof_prob_abnormal['LightGBM'][te_i] = probs_stage1['LightGBM']

    print('--- H-CRE Stage 1 Binary (CN vs Abnormal) Metrics ---')
    for m in ['Ensemble', 'XGBoost', 'LightGBM']:
        p = oof_prob_abnormal[m]
        pred = (p >= 0.5).astype(int)
        auc = roc_auc_score(y_true_binary, p)
        acc = accuracy_score(y_true_binary, pred)
        rec = recall_score(y_true_binary, pred)
        tn, fp, fn, tp = confusion_matrix(y_true_binary, pred).ravel()
        spec = tn / (tn + fp)
        f1 = f1_score(y_true_binary, pred)
        print(f'{m:<10}: AUC={auc:.4f}, Acc={acc:.4f}, Rec={rec:.4f}, Spec={spec:.4f}, F1={f1:.4f}')

if __name__ == '__main__':
    main()
