import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

PATIENT_PATH = "c:/ML4/data/processed/tabular/patient_level_all_v2.csv"

# ==========================================
# 1. 기존 V16 최우수 피처 세트 (SHAP 20)
# ==========================================
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

# ==========================================
# 2. 시계열 전처리 및 로드 함수 (V17 베이스 + 경량화)
# ==========================================
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
    print("[1] 원시 시계열 데이터 파싱 중... (This may take a minute)")
    df_act = pd.read_csv("c:/ML4/data/train_activity.csv")
    df_slp = pd.read_csv("c:/ML4/data/train_sleep.csv")
    df_lbl = pd.read_csv("c:/ML4/data/training_label_activity.csv").rename(columns={'SAMPLE_EMAIL': 'EMAIL'})
    
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

# ==========================================
# 3. TAPNet-like Attention Extractor 모델
# ==========================================
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
        
        # Attention
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
    for ep in range(10):  # 간단히 10 Epoch 학습
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
    # 전체 데이터 임베딩 추출
    model.eval()
    embeds = []
    full_loader = DataLoader(ds, batch_size=16, shuffle=False)
    with torch.no_grad():
        for xb, _ in full_loader:
            xb = xb.to(device)
            emb = model(xb, extract=True)
            embeds.append(emb.cpu().numpy())
            
    embeds_arr = np.concatenate(embeds, axis=0)
    
    # 이메일을 인덱스로 하는 DataFrame 생성
    embed_df = pd.DataFrame(embeds_arr, columns=[f"tapnet_emb_{i}" for i in range(embeds_arr.shape[1])])
    embed_df['EMAIL'] = emails
    return embed_df

# ==========================================
# 4. 머신러닝 앙상블 학습 (Option A 적용)
# ==========================================
def train_v18_ensemble(embed_df):
    print("\n[3] 딥러닝 임베딩(TAPNet) + Tabular(SHAP 20) 결합 및 Random Forest 평가...")
    tabular_df = pd.read_csv(PATIENT_PATH)
    
    # 병합
    df = pd.merge(tabular_df, embed_df, on='EMAIL', how='inner')
    
    df['stage1_label'] = np.where(df['original_label'] == 0, 0, 1)
    df['stage2_label'] = np.where(df['original_label'] == 1, 0, np.where(df['original_label'] == 2, 1, np.nan))
    
    # Feature 리스트 병합 (SHAP Top20 + 임베딩 16차원)
    emb_cols = [c for c in embed_df.columns if 'tapnet' in c]
    STG1_FEATURES = STAGE1_TOP20 + emb_cols
    STG2_FEATURES = STAGE2_TOP20 + emb_cols
    
    for f in set(STG1_FEATURES + STG2_FEATURES):
        if df[f].isnull().sum() > 0:
            df[f].fillna(df[f].median(), inplace=True)
            
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {'acc':[], 'prec':[], 'rec':[], 'f1':[], 'auc':[]}
    
    for fold, (tr_idx, te_idx) in enumerate(skf.split(df, df['original_label'])):
        df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
        y_te = df_te['original_label'].astype(int)
        
        # Stage 1 Data
        X_tr_stg1 = df_tr[STG1_FEATURES]
        y_tr_stg1 = df_tr['stage1_label'].astype(int)
        
        # Stage 2 Data
        df_tr_stg2 = df_tr[df_tr['stage1_label'] == 1]
        X_tr_stg2 = df_tr_stg2[STG2_FEATURES]
        y_tr_stg2 = df_tr_stg2['stage2_label'].astype(int)
        
        # SMOTE
        smote = SMOTE(random_state=42)
        X1_res, y1_res = smote.fit_resample(X_tr_stg1, y_tr_stg1)
        X2_res, y2_res = smote.fit_resample(X_tr_stg2, y_tr_stg2)
        
        # Model (RandomForest)
        model_s1 = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        model_s2 = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        
        model_s1.fit(X1_res, y1_res)
        model_s2.fit(X2_res, y2_res)
        
        prob1 = model_s1.predict_proba(df_te[STG1_FEATURES])[:, 1]
        prob2 = model_s2.predict_proba(df_te[STG2_FEATURES])[:, 1]
        
        p_CN = 1.0 - prob1
        p_MCI = prob1 * (1.0 - prob2)
        p_Dem = prob1 * prob2
        
        prob_matrix = np.vstack([p_CN, p_MCI, p_Dem]).T
        fold_preds = np.argmax(prob_matrix, axis=1)
        
        metrics['acc'].append(accuracy_score(y_te, fold_preds))
        metrics['prec'].append(precision_score(y_te, fold_preds, average='macro', zero_division=0))
        metrics['rec'].append(recall_score(y_te, fold_preds, average='macro', zero_division=0))
        metrics['f1'].append(f1_score(y_te, fold_preds, average='macro', zero_division=0))
        metrics['auc'].append(roc_auc_score(y_te, prob_matrix, multi_class='ovr'))
        
    print("\n" + "="*60)
    print("V18 FINAL RESULTS (TAPNet Embeddings + SHAP Top20 + RF)")
    print("="*60)
    print(f"Accuracy:  {np.mean(metrics['acc']):.4f}")
    print(f"Precision: {np.mean(metrics['prec']):.4f}")
    print(f"Recall:    {np.mean(metrics['rec']):.4f}")
    print(f"Macro F1:  {np.mean(metrics['f1']):.4f}")
    print(f"ROC AUC:   {np.mean(metrics['auc']):.4f}")
    print("="*60)

if __name__ == "__main__":
    X_seq, y_target, emails = load_time_series_data(days_limit=21)
    embed_df = extract_deep_features(X_seq, y_target, emails)
    train_v18_ensemble(embed_df)
