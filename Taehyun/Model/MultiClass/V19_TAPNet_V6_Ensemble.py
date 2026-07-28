import os
import sys
import pathlib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# 1. 글로벌 설정 및 변수
# =========================================================
BASE_DIR = pathlib.Path(r"c:\ML4")
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "tabular"
PATIENT_PATH = PROCESSED_DIR / "patient_level_all_v2.csv"
PLOT_DIR = BASE_DIR / "report" / "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

TARGET_COL = "original_label"
DROP_COLS = ["EMAIL", "date", "DIAG_NM", "label", TARGET_COL, "fold"]

RANDOM_STATE = 42
N_SPLITS = 5
FORWARD_SELECTION_MAX_FEATURES = 40
FORWARD_SELECTION_ESTIMATORS = 100

# =========================================================
# 2. 시계열(TAPNet) 모듈
# =========================================================
def parse_5min_string(s):
    if pd.isna(s) or s == "..." or str(s).strip() == "":
        return np.zeros(288)
    if "/" in str(s):
        parts = str(s).split("/")
        arr = [float(p) if p.replace('.','',1).isdigit() else 0.0 for p in parts]
    else:
        arr = [float(ch) for ch in str(s) if ch.isdigit()]
        
    arr = np.array(arr)
    if len(arr) == 0:
        return np.zeros(288)
    elif len(arr) >= 288:
        return arr[:288]
    else:
        padded = np.zeros(288)
        padded[:len(arr)] = arr
        return padded

def load_time_series_data(days_limit=21):
    print("\n[1] 원시 시계열 데이터 파싱 중... (This may take a minute)")
    df_act = pd.read_csv(BASE_DIR / "data" / "train_activity.csv")
    df_slp = pd.read_csv(BASE_DIR / "data" / "train_sleep.csv")
    df_lbl = pd.read_csv(BASE_DIR / "data" / "training_label_activity.csv").rename(columns={'SAMPLE_EMAIL': 'EMAIL'})
    
    df_act['date'] = pd.to_datetime(df_act['activity_day_end']).dt.date
    df_slp['date'] = pd.to_datetime(df_slp['sleep_bedtime_end']).dt.date
    
    df = pd.merge(df_act, df_slp, on=['EMAIL', 'date'], how='inner')
    df = pd.merge(df, df_lbl[['EMAIL', 'DIAG_NM']], on='EMAIL', how='inner').drop_duplicates(subset=['EMAIL', 'date'])
    df['label'] = np.where(df['DIAG_NM'] == 'CN', 0, 1)
    
    seq_cols = ['activity_class_5min', 'sleep_hr_5min', 'sleep_hypnogram_5min', 'sleep_rmssd_5min']
    emails = df['EMAIL'].unique()
    
    X_seq_list, y_list, email_list = [], [], []
    
    for email in emails:
        pdf = df[df['EMAIL'] == email].sort_values('date')
        if len(pdf) == 0: continue
        
        pdf = pdf.iloc[-days_limit:]
        days = len(pdf)
        pad_days = max(0, days_limit - days)
            
        seq_day_list = []
        for i in range(days):
            day_arr = [parse_5min_string(pdf.iloc[i][col]) for col in seq_cols]
            seq_day_list.append(np.stack(day_arr))
            
        for _ in range(pad_days):
            seq_day_list.append(np.zeros((4, 288)))
            
        seq_tensor = np.stack(seq_day_list)
        
        X_seq_list.append(seq_tensor)
        y_list.append(pdf['label'].iloc[0])
        email_list.append(email)
        
    X_seq = torch.tensor(np.stack(X_seq_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)
    return X_seq, y, email_list

class TimeSeriesAttentionExtractor(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, embed_dim=16):
        super().__init__()
        self.day_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.inter_lstm = nn.LSTM(2*hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        self.attention = nn.Linear(2*hidden_dim, 1)
        self.fc_embed = nn.Linear(2*hidden_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, 2)
        
    def forward(self, x, extract=False):
        batch_size, days, feat_dim, time_steps = x.shape
        x_reshaped = x.view(batch_size * days, feat_dim, time_steps).transpose(1, 2)
        
        out1, (h_n, _) = self.day_lstm(x_reshaped)
        h_n_fwd, h_n_bwd = h_n[0, :, :], h_n[1, :, :]
        day_rep = torch.cat([h_n_fwd, h_n_bwd], dim=1).view(batch_size, days, -1)
        
        out2, _ = self.inter_lstm(day_rep)
        
        attn_weights = torch.softmax(self.attention(out2), dim=1)
        context_vector = torch.sum(attn_weights * out2, dim=1)
        
        embedding = torch.relu(self.fc_embed(context_vector))
        
        if extract:
            return embedding
        logits = self.classifier(embedding)
        return logits

class SeqDataset(Dataset):
    def __init__(self, X_seq, y):
        self.X_seq = X_seq
        self.y = y
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X_seq[idx], self.y[idx]

def extract_deep_features(X_seq, y, emails):
    print(f"\n[2] TAPNet-like 모델 사전학습(10 Epoch) 및 딥임베딩 추출 중... (Total subjects: {len(y)})")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ds = SeqDataset(X_seq, y)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    
    model = TimeSeriesAttentionExtractor(hidden_dim=32, embed_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for ep in range(10):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
    model.eval()
    embeds = []
    full_loader = DataLoader(ds, batch_size=16, shuffle=False)
    with torch.no_grad():
        for xb, _ in full_loader:
            xb = xb.to(device)
            emb = model(xb, extract=True)
            embeds.append(emb.cpu().numpy())
            
    embeds_arr = np.concatenate(embeds, axis=0)
    embed_df = pd.DataFrame(embeds_arr, columns=[f"tapnet_emb_{i}" for i in range(embeds_arr.shape[1])])
    embed_df['EMAIL'] = emails
    return embed_df

# =========================================================
# 3. Tabular V6 병합 및 비지도 학습 파트
# =========================================================
def load_and_merge_data(embed_df):
    print(f"\n[3] 정형 데이터(Patient-level) 로드 및 TAPNet 임베딩 병합...")
    df = pd.read_csv(PATIENT_PATH)
    
    # TAPNet 임베딩 병합
    df = pd.merge(df, embed_df, on='EMAIL', how='inner')
    print(f"  Merged Data: {df.shape} | subjects: {df['EMAIL'].nunique()}")
    
    df['stage1_label'] = np.where(df[TARGET_COL] == 0, 0, 1)
    df['stage2_label'] = np.where(df[TARGET_COL] == 1, 0, np.where(df[TARGET_COL] == 2, 1, np.nan))
    
    all_feats = [c for c in df.columns if c not in DROP_COLS and c not in ['stage1_label', 'stage2_label']]
    return df, all_feats

def add_unsupervised_features(df, features):
    print("\n[4] 비지도 학습 기반 피처(PCA, KMeans, GMM, Agglomerative)를 생성합니다...")
    X = df[features].copy()
    
    X.fillna(X.median(), inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    new_features_df = pd.DataFrame(index=df.index)
    
    # PCA
    n_components = min(5, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca_feats = pca.fit_transform(X_scaled)
    for i in range(n_components):
        new_features_df[f'pca_{i+1}'] = pca_feats[:, i]
        
    # K-Means
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    kmeans_dist = kmeans.fit_transform(X_scaled)
    for i in range(3):
        new_features_df[f'kmeans_dist_{i}'] = kmeans_dist[:, i]
    new_features_df['kmeans_label'] = kmeans.labels_
    
    # GMM
    gmm = GaussianMixture(n_components=3, random_state=RANDOM_STATE)
    gmm.fit(X_scaled)
    gmm_probs = gmm.predict_proba(X_scaled)
    for i in range(3):
        new_features_df[f'gmm_prob_{i}'] = gmm_probs[:, i]
        
    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=3)
    new_features_df['hierarchical_cluster_label'] = agg.fit_predict(X_scaled)
    
    df_new = pd.concat([df, new_features_df], axis=1)
    new_cols = new_features_df.columns.tolist()
    print(f"  -> 새로 생성된 비지도 학습 피처 {len(new_cols)}개 추가 완료.")
    
    return df_new, new_cols

# =========================================================
# 4. 전진 선택법 (Forward Selection)
# =========================================================
def perform_forward_selection(df, all_feats, stage_target, stage_name):
    print(f"\n[5] 전진 선택법: {stage_name} 최적의 피처 탐색...")
    
    df_stage = df.dropna(subset=[stage_target]).copy()
    # 결측치 단순 대체 (트리 모델이 아닌 SMOTE를 위해)
    X = df_stage[all_feats].fillna(df_stage[all_feats].median())
    y = df_stage[stage_target].astype(int)
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X, y)
    
    base_model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight='balanced', verbose=-1)
    base_model.fit(X_res, y_res)
    
    importance = base_model.feature_importances_
    ranked_features = [feat for _, feat in sorted(zip(importance, all_feats), reverse=True)]
    top_k_features = ranked_features[:FORWARD_SELECTION_MAX_FEATURES]
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    history_scores = []
    
    for k in range(1, len(top_k_features) + 1):
        current_features = top_k_features[:k]
        fold_scores = []
        
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx][current_features], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx][current_features], y.iloc[va_idx]
            
            X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
            
            eval_model = LGBMClassifier(
                random_state=RANDOM_STATE, n_jobs=1, class_weight='balanced', n_estimators=FORWARD_SELECTION_ESTIMATORS, verbose=-1
            )
            eval_model.fit(X_tr_res, y_tr_res, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(30, verbose=False)])
            
            prob = eval_model.predict_proba(X_va)[:, 1]
            fold_scores.append(roc_auc_score(y_va, prob))
            
        history_scores.append(np.mean(fold_scores))
            
    optimal_k = np.argmax(history_scores) + 1
    optimal_features = top_k_features[:optimal_k]
    print(f"[{stage_name} 피처 선택 완료] 최적 피처 개수: {optimal_k}개 (Best AUC: {max(history_scores):.4f})")
    
    # 선택된 피처에 tapnet 임베딩이 몇 개나 포함되었는지 확인
    tapnet_included = [f for f in optimal_features if 'tapnet_emb' in f]
    print(f"  -> 포함된 TAPNet 임베딩 수: {len(tapnet_included)}개 / 16개")
    
    return optimal_features

# =========================================================
# 5. Grid Search 및 최종 평가
# =========================================================
def run_grid_search_for_stage(df, features, stage_target, model_name, stage_name):
    print(f"\n[Grid Search] {stage_name} - {model_name} 파라미터 탐색 중...")
    df_stage = df.dropna(subset=[stage_target]).copy()
    X = df_stage[features].fillna(df_stage[features].median())
    y = df_stage[stage_target].astype(int)
    
    if model_name == "LightGBM":
        param_grid = {'num_leaves': [15, 31], 'learning_rate': [0.05, 0.1], 'n_estimators': [100, 200]}
    elif model_name == "CatBoost":
        param_grid = {'iterations': [100, 200], 'learning_rate': [0.05, 0.1], 'depth': [4, 6]}
    elif model_name == "XGBoost":
        param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
    elif model_name == "RandomForest":
        param_grid = {'n_estimators': [200, 500], 'max_depth': [5, 10], 'min_samples_split': [2, 5]}
        
    best_auc = 0
    best_params = None
    
    for params in ParameterGrid(param_grid):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        fold_scores = []
        
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
            
            smote = SMOTE(random_state=RANDOM_STATE)
            X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
            
            if model_name == "LightGBM":
                model = LGBMClassifier(objective="binary", random_state=RANDOM_STATE, n_jobs=1, verbose=-1, **params)
                model.fit(X_tr_res, y_tr_res, eval_set=[(X_va, y_va)], eval_metric='auc', callbacks=[lgb.early_stopping(30, verbose=False)])
                prob = model.predict_proba(X_va)[:, 1]
            elif model_name == "CatBoost":
                model = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, **params)
                model.fit(X_tr_res, y_tr_res, eval_set=[(X_va, y_va)], early_stopping_rounds=30)
                prob = model.predict_proba(X_va)[:, 1]
            elif model_name == "XGBoost":
                model = XGBClassifier(random_state=RANDOM_STATE, n_jobs=1, eval_metric='auc', early_stopping_rounds=30, **params)
                model.fit(X_tr_res, y_tr_res, eval_set=[(X_va, y_va)], verbose=False)
                prob = model.predict_proba(X_va)[:, 1]
            elif model_name == "RandomForest":
                model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **params)
                model.fit(X_tr_res, y_tr_res)
                prob = model.predict_proba(X_va)[:, 1]
                
            fold_scores.append(roc_auc_score(y_va, prob))
            
        mean_auc = np.mean(fold_scores)
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params
            
    print(f"  -> 최고 AUC: {best_auc:.4f} | 파라미터: {best_params}")
    return best_params

def hierarchical_evaluation(df, stg1_feats, stg2_feats, best_params_dict):
    print("\n[6] 4개 모델 앙상블 5-Fold 결합 평가 시작...")
    
    models = ["LightGBM", "CatBoost", "XGBoost", "RandomForest", "Ensemble"]
    final_metrics = {m: {'acc':[], 'prec':[], 'rec':[], 'f1':[], 'auc':[], 'cm':[]} for m in models}
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    for tr_idx, te_idx in skf.split(df, df[TARGET_COL]):
        df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
        
        X_tr_stg1 = df_tr[stg1_feats].fillna(df_tr[stg1_feats].median())
        y_tr_stg1 = df_tr['stage1_label'].astype(int)
        
        df_tr_stg2 = df_tr[df_tr['stage1_label'] == 1]
        X_tr_stg2 = df_tr_stg2[stg2_feats].fillna(df_tr_stg2[stg2_feats].median())
        y_tr_stg2 = df_tr_stg2['stage2_label'].astype(int)
        
        y_te = df_te[TARGET_COL].astype(int)
        X_te_stg1 = df_te[stg1_feats].fillna(df_te[stg1_feats].median())
        X_te_stg2 = df_te[stg2_feats].fillna(df_te[stg2_feats].median())
        
        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr_stg1_res, y_tr_stg1_res = smote.fit_resample(X_tr_stg1, y_tr_stg1)
        X_tr_stg2_res, y_tr_stg2_res = smote.fit_resample(X_tr_stg2, y_tr_stg2)
        
        fold_probs = {}
        
        for model_name in ["LightGBM", "CatBoost", "XGBoost", "RandomForest"]:
            if model_name == "LightGBM":
                model_s1 = LGBMClassifier(objective="binary", random_state=RANDOM_STATE, n_jobs=1, verbose=-1, **best_params_dict[model_name]['Stage1'])
                model_s2 = LGBMClassifier(objective="binary", random_state=RANDOM_STATE, n_jobs=1, verbose=-1, **best_params_dict[model_name]['Stage2'])
            elif model_name == "CatBoost":
                model_s1 = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, **best_params_dict[model_name]['Stage1'])
                model_s2 = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, **best_params_dict[model_name]['Stage2'])
            elif model_name == "XGBoost":
                model_s1 = XGBClassifier(random_state=RANDOM_STATE, n_jobs=1, **best_params_dict[model_name]['Stage1'])
                model_s2 = XGBClassifier(random_state=RANDOM_STATE, n_jobs=1, **best_params_dict[model_name]['Stage2'])
            elif model_name == "RandomForest":
                model_s1 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **best_params_dict[model_name]['Stage1'])
                model_s2 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **best_params_dict[model_name]['Stage2'])
                
            model_s1.fit(X_tr_stg1_res, y_tr_stg1_res)
            model_s2.fit(X_tr_stg2_res, y_tr_stg2_res)
            
            prob1 = model_s1.predict_proba(X_te_stg1)[:, 1]
            prob2 = model_s2.predict_proba(X_te_stg2)[:, 1]
            
            p_CN = 1.0 - prob1
            p_MCI = prob1 * (1.0 - prob2)
            p_Dem = prob1 * prob2
            
            prob_matrix = np.vstack([p_CN, p_MCI, p_Dem]).T
            fold_probs[model_name] = prob_matrix
            
            final_preds = np.argmax(prob_matrix, axis=1)
            
            final_metrics[model_name]['acc'].append(accuracy_score(y_te, final_preds))
            final_metrics[model_name]['prec'].append(precision_score(y_te, final_preds, average='macro', zero_division=0))
            final_metrics[model_name]['rec'].append(recall_score(y_te, final_preds, average='macro', zero_division=0))
            final_metrics[model_name]['f1'].append(f1_score(y_te, final_preds, average='macro', zero_division=0))
            final_metrics[model_name]['auc'].append(roc_auc_score(y_te, prob_matrix, multi_class='ovr'))
            final_metrics[model_name]['cm'].append(confusion_matrix(y_te, final_preds))
            
        ensemble_prob = (fold_probs["LightGBM"] + fold_probs["CatBoost"] + fold_probs["XGBoost"] + fold_probs["RandomForest"]) / 4.0
        ensemble_preds = np.argmax(ensemble_prob, axis=1)
        
        final_metrics["Ensemble"]['acc'].append(accuracy_score(y_te, ensemble_preds))
        final_metrics["Ensemble"]['prec'].append(precision_score(y_te, ensemble_preds, average='macro', zero_division=0))
        final_metrics["Ensemble"]['rec'].append(recall_score(y_te, ensemble_preds, average='macro', zero_division=0))
        final_metrics["Ensemble"]['f1'].append(f1_score(y_te, ensemble_preds, average='macro', zero_division=0))
        final_metrics["Ensemble"]['auc'].append(roc_auc_score(y_te, ensemble_prob, multi_class='ovr'))
        final_metrics["Ensemble"]['cm'].append(confusion_matrix(y_te, ensemble_preds))
            
    results = {}
    for m in models:
        results[m] = {
            'accuracy': np.mean(final_metrics[m]['acc']),
            'precision': np.mean(final_metrics[m]['prec']),
            'recall': np.mean(final_metrics[m]['rec']),
            'f1': np.mean(final_metrics[m]['f1']),
            'auc': np.mean(final_metrics[m]['auc']),
            'cm': np.sum(final_metrics[m]['cm'], axis=0)
        }
    return results

if __name__ == "__main__":
    start_time = datetime.now()
    
    # 1. TAPNet 임베딩 추출
    X_seq, y_target, emails = load_time_series_data(days_limit=21)
    embed_df = extract_deep_features(X_seq, y_target, emails)
    
    # 2. 정형 데이터 병합
    df, all_feats = load_and_merge_data(embed_df)
    
    # 3. 비지도 학습 피처 생성
    df, unsupervised_feats = add_unsupervised_features(df, all_feats)
    all_feats.extend(unsupervised_feats)
    
    # 4. Feature Selection
    opt_feats_s1 = perform_forward_selection(df, all_feats, "stage1_label", "Stage1")
    opt_feats_s2 = perform_forward_selection(df, all_feats, "stage2_label", "Stage2")
    
    # 5. Grid Search (파라미터 공간 약간 축소로 속도 최적화)
    base_models = ["LightGBM", "CatBoost", "XGBoost", "RandomForest"]
    best_params_dict = {m: {} for m in base_models}
    
    for m in base_models:
        best_params_dict[m]['Stage1'] = run_grid_search_for_stage(df, opt_feats_s1, "stage1_label", m, "Stage1")
        best_params_dict[m]['Stage2'] = run_grid_search_for_stage(df, opt_feats_s2, "stage2_label", m, "Stage2")
    
    # 6. Final Evaluation
    results = hierarchical_evaluation(df, opt_feats_s1, opt_feats_s2, best_params_dict)
    
    print("\n" + "="*80)
    print("FINAL RESULTS (V19_TAPNet_V6_Ensemble)")
    print("="*80)
    for model_name, r in results.items():
        print(f"\n[{model_name}]")
        print(f"Accuracy: {r['accuracy']:.4f} | Precision: {r['precision']:.4f} | Recall: {r['recall']:.4f} | Macro F1: {r['f1']:.4f} | OVR AUC: {r['auc']:.4f}")
        
    end_time = datetime.now()
    print(f"\nElapsed: {end_time - start_time}")
    
    print("\n[시각화] 앙상블 혼동 행렬 (Confusion Matrix)")
    class_names = ['CN(0)', 'MCI(1)', 'Dem(2)']
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    
    for ax, (model_name, r) in zip(axes, results.items()):
        sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 13}, ax=ax)
        ax.set_title(f"{model_name}", fontsize=12, pad=10)
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix_v19_tapnet_v6.png", dpi=150)
