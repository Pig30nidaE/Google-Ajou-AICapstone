import os
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

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
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
RANDOM_STATE = 42
N_SPLITS = 5

STAGE1_TOP20 = [
    'sleep_light_ratio_5min', 'sleep_breath_average', 'sleep_score_alignment_std', 
    'activity_met_min_low_std', 'sleep_hr_5min_max_std', 'sleep_period_id_std', 
    'activity_high_std', 'sleep_light_std', 'activity_met_min_high_std', 
    'sleep_score', 'sleep_period_id', 'activity_class_3_count_std', 
    'sleep_rmssd_5min_max', 'activity_score_recovery_time_std', 'activity_inactivity_alerts', 
    'activity_class_4_ratio_std', 'sleep_score_deep_std', 'sleep_rmssd_5min_var_std', 
    'sleep_light_count_5min_std', 'activity_met_1min_max_std'
]

STAGE2_TOP20 = [
    'activity_met_min_inactive', 'activity_active_ratio_std', 'sleep_rem_count_5min_std', 
    'activity_low_std', 'sleep_score_latency_std', 'activity_total_std', 
    'sleep_hr_lowest_std', 'sleep_deep_std', 'sleep_awake_ratio_5min_std', 
    'sleep_awake_count_5min_std', 'sleep_temperature_delta', 'activity_met_min_low_std', 
    'activity_met_1min_q25', 'activity_met_1min_max_std', 'sleep_score_disturbances', 
    'activity_class_3_ratio_std', 'sleep_score_alignment', 'sleep_rem', 
    'sleep_onset_latency_std', 'sleep_light_count_5min_std'
]

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
    print("\n[1] 원시 시계열 데이터 파싱 중...")
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
        day_rep = torch.cat([h_n[0, :, :], h_n[1, :, :]], dim=1).view(batch_size, days, -1)
        
        out2, _ = self.inter_lstm(day_rep)
        
        attn_weights = torch.softmax(self.attention(out2), dim=1)
        context_vector = torch.sum(attn_weights * out2, dim=1)
        
        embedding = torch.relu(self.fc_embed(context_vector))
        
        if extract:
            return embedding
        return self.classifier(embedding)

class SeqDataset(Dataset):
    def __init__(self, X_seq, y):
        self.X_seq = X_seq; self.y = y
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X_seq[idx], self.y[idx]

def extract_deep_features(X_seq, y, emails):
    print(f"\n[2] TAPNet 사전학습(10 Epoch) 및 딥임베딩 추출 중...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ds = SeqDataset(X_seq, y)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    
    model = TimeSeriesAttentionExtractor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for _ in range(10):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()
            
    model.eval()
    embeds = []
    with torch.no_grad():
        for xb, _ in DataLoader(ds, batch_size=16, shuffle=False):
            embeds.append(model(xb.to(device), extract=True).cpu().numpy())
            
    embeds_arr = np.concatenate(embeds, axis=0)
    embed_df = pd.DataFrame(embeds_arr, columns=[f"tapnet_emb_{i}" for i in range(embeds_arr.shape[1])])
    embed_df['EMAIL'] = emails
    return embed_df

def load_and_prepare_data(embed_df):
    print("\n[3] 정형 데이터(Patient-level) 로드 및 TAPNet 임베딩 병합...")
    df = pd.read_csv(PATIENT_PATH)
    
    df = pd.merge(df, embed_df, on='EMAIL', how='inner')
    df['stage1_label'] = np.where(df[TARGET_COL] == 0, 0, 1)
    df['stage2_label'] = np.where(df[TARGET_COL] == 1, 0, np.where(df[TARGET_COL] == 2, 1, np.nan))
    
    emb_cols = [c for c in embed_df.columns if 'tapnet_emb' in c]
    STG1_FEATURES = STAGE1_TOP20 + emb_cols
    STG2_FEATURES = STAGE2_TOP20 + emb_cols
    
    for f in set(STG1_FEATURES + STG2_FEATURES):
        if df[f].isnull().sum() > 0:
            df[f].fillna(df[f].median(), inplace=True)
            
    return df, STG1_FEATURES, STG2_FEATURES

def run_v20_evaluation(df, STG1_FEATURES, STG2_FEATURES):
    print(f"\n[4] 4개 모델 앙상블 5-Fold 결합 평가 시작...")
    print(f"  Stage1 피처 수: {len(STG1_FEATURES)} (SHAP Top20 + TAPNet 16)")
    print(f"  Stage2 피처 수: {len(STG2_FEATURES)} (SHAP Top20 + TAPNet 16)")
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models_list = ["LightGBM", "CatBoost", "XGBoost", "RandomForest", "Ensemble"]
    final_metrics = {m: {'acc':[], 'prec':[], 'rec':[], 'f1':[], 'auc':[], 'cm':[]} for m in models_list}
    
    for fold, (tr_idx, te_idx) in enumerate(skf.split(df, df[TARGET_COL])):
        df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
        y_te = df_te[TARGET_COL].astype(int)
        
        X_tr_stg1 = df_tr[STG1_FEATURES]
        y_tr_stg1 = df_tr['stage1_label'].astype(int)
        
        df_tr_stg2 = df_tr[df_tr['stage1_label'] == 1]
        X_tr_stg2 = df_tr_stg2[STG2_FEATURES]
        y_tr_stg2 = df_tr_stg2['stage2_label'].astype(int)
        
        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr_stg1_res, y_tr_stg1_res = smote.fit_resample(X_tr_stg1, y_tr_stg1)
        X_tr_stg2_res, y_tr_stg2_res = smote.fit_resample(X_tr_stg2, y_tr_stg2)
        
        fold_probs = {}
        
        for model_name in ["LightGBM", "CatBoost", "XGBoost", "RandomForest"]:
            if model_name == "LightGBM":
                model_s1 = LGBMClassifier(objective="binary", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, class_weight='balanced')
                model_s2 = LGBMClassifier(objective="binary", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, class_weight='balanced')
            elif model_name == "CatBoost":
                model_s1 = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, auto_class_weights='Balanced')
                model_s2 = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, auto_class_weights='Balanced')
            elif model_name == "XGBoost":
                model_s1 = XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss')
                model_s2 = XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss')
            elif model_name == "RandomForest":
                model_s1 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')
                model_s2 = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')
                
            model_s1.fit(X_tr_stg1_res, y_tr_stg1_res)
            model_s2.fit(X_tr_stg2_res, y_tr_stg2_res)
            
            prob1 = model_s1.predict_proba(df_te[STG1_FEATURES])[:, 1]
            prob2 = model_s2.predict_proba(df_te[STG2_FEATURES])[:, 1]
            
            p_CN = 1.0 - prob1
            p_MCI = prob1 * (1.0 - prob2)
            p_Dem = prob1 * prob2
            
            prob_matrix = np.vstack([p_CN, p_MCI, p_Dem]).T
            fold_probs[model_name] = prob_matrix
            
            fold_preds = np.argmax(prob_matrix, axis=1)
            
            final_metrics[model_name]['acc'].append(accuracy_score(y_te, fold_preds))
            final_metrics[model_name]['prec'].append(precision_score(y_te, fold_preds, average='macro', zero_division=0))
            final_metrics[model_name]['rec'].append(recall_score(y_te, fold_preds, average='macro', zero_division=0))
            final_metrics[model_name]['f1'].append(f1_score(y_te, fold_preds, average='macro', zero_division=0))
            final_metrics[model_name]['auc'].append(roc_auc_score(y_te, prob_matrix, multi_class='ovr'))
            final_metrics[model_name]['cm'].append(confusion_matrix(y_te, fold_preds))
            
        ensemble_prob = (fold_probs["LightGBM"] + fold_probs["CatBoost"] + fold_probs["XGBoost"] + fold_probs["RandomForest"]) / 4.0
        ensemble_preds = np.argmax(ensemble_prob, axis=1)
        
        final_metrics["Ensemble"]['acc'].append(accuracy_score(y_te, ensemble_preds))
        final_metrics["Ensemble"]['prec'].append(precision_score(y_te, ensemble_preds, average='macro', zero_division=0))
        final_metrics["Ensemble"]['rec'].append(recall_score(y_te, ensemble_preds, average='macro', zero_division=0))
        final_metrics["Ensemble"]['f1'].append(f1_score(y_te, ensemble_preds, average='macro', zero_division=0))
        final_metrics["Ensemble"]['auc'].append(roc_auc_score(y_te, ensemble_prob, multi_class='ovr'))
        final_metrics["Ensemble"]['cm'].append(confusion_matrix(y_te, ensemble_preds))
        
    print("\n" + "="*80)
    print("FINAL RESULTS (V20: V16 Ensemble + TAPNet Embeddings)")
    print("="*80)
    for model_name in models_list:
        mean_acc = np.mean(final_metrics[model_name]['acc'])
        mean_prec = np.mean(final_metrics[model_name]['prec'])
        mean_rec = np.mean(final_metrics[model_name]['rec'])
        mean_f1 = np.mean(final_metrics[model_name]['f1'])
        mean_auc = np.mean(final_metrics[model_name]['auc'])
        print(f"[{model_name}] Acc: {mean_acc:.4f} | Prec: {mean_prec:.4f} | Rec: {mean_rec:.4f} | F1: {mean_f1:.4f} | AUC: {mean_auc:.4f}")
        
    class_names = ['CN(0)', 'MCI(1)', 'Dem(2)']
    fig, axes = plt.subplots(1, len(models_list), figsize=(5 * len(models_list), 4))
    
    for ax, model_name in zip(axes, models_list):
        cm = np.sum(final_metrics[model_name]['cm'], axis=0)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 13}, ax=ax)
        ax.set_title(f"{model_name}", fontsize=12, pad=10)
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix_v20_tapnet_v16.png", dpi=150)

if __name__ == "__main__":
    start_time = datetime.now()
    
    X_seq, y_target, emails = load_time_series_data()
    embed_df = extract_deep_features(X_seq, y_target, emails)
    
    df, STG1_FEATURES, STG2_FEATURES = load_and_prepare_data(embed_df)
    
    run_v20_evaluation(df, STG1_FEATURES, STG2_FEATURES)
    
    end_time = datetime.now()
    print(f"\nElapsed: {end_time - start_time}")
