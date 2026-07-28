import os
import sys
import pathlib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = pathlib.Path(r"c:\ML4")
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed" / "tabular"
PATIENT_PATH = PROCESSED_DIR / "patient_level_all_v2.csv"
OUTPUT_OOF_PATH = PROCESSED_DIR / "oof_lstm_binary_probs.npy"

RANDOM_STATE = 42
N_SPLITS = 5
DAYS_LIMIT = 21

def parse_slash_array(val, target_len=288):
    if pd.isna(val) or val == "..." or str(val).strip() == "":
        return np.zeros(target_len)
    val_str = str(val)
    if "CONVERT(" in val_str:
        return np.zeros(target_len)
    if "/" in val_str:
        parts = val_str.split("/")
        arr = []
        for p in parts:
            p_str = p.strip()
            try:
                arr.append(float(p_str))
            except ValueError:
                arr.append(0.0)
        arr = np.array(arr)
    else:
        arr = np.array([float(ch) for ch in val_str if ch.isdigit()])
    if len(arr) == 0:
        return np.zeros(target_len)
    if len(arr) >= target_len:
        return arr[:target_len]
    else:
        padded = np.zeros(target_len)
        padded[:len(arr)] = arr
        return padded

def load_patient_sequences():
    print("[1] Loading and parsing 21-day continuous sequences...")
    df_act = pd.read_csv(DATA_DIR / "train_activity.csv")
    df_slp = pd.read_csv(DATA_DIR / "train_sleep.csv")
    df_patient = pd.read_csv(PATIENT_PATH)
    
    df_act['date'] = pd.to_datetime(df_act['activity_day_end']).dt.date
    df_slp['date'] = pd.to_datetime(df_slp['sleep_bedtime_end']).dt.date
    
    df_merged = pd.merge(df_act, df_slp, on=['EMAIL', 'date'], how='inner')
    
    unique_emails = df_patient['EMAIL'].values
    label_dict = dict(zip(df_patient['EMAIL'], df_patient['label']))
    
    met_col = 'CONVERT(activity_met_1min USING utf8)' if 'CONVERT(activity_met_1min USING utf8)' in df_act.columns else 'activity_met_1min'
    hr_col = 'CONVERT(sleep_hr_5min USING utf8)' if 'CONVERT(sleep_hr_5min USING utf8)' in df_slp.columns else 'sleep_hr_5min'
    hypno_col = 'CONVERT(sleep_hypnogram_5min USING utf8)' if 'CONVERT(sleep_hypnogram_5min USING utf8)' in df_slp.columns else 'sleep_hypnogram_5min'
    
    X_seq_list = []
    y_list = []
    
    for email in unique_emails:
        patient_data = df_merged[df_merged['EMAIL'] == email].sort_values('date')
        if len(patient_data) == 0:
            seq_feats = np.zeros((DAYS_LIMIT, 7))
        else:
            daily_rows = []
            for _, row in patient_data.iterrows():
                # 7 Daily channels
                met_arr = parse_slash_array(row[met_col], target_len=1440)
                hr_arr = parse_slash_array(row[hr_col], target_len=288)
                hypno_arr = parse_slash_array(row[hypno_col], target_len=288)
                
                met_mean = float(np.mean(met_arr))
                met_max = float(np.max(met_arr))
                hr_valid = hr_arr[hr_arr > 30]
                hr_mean = float(np.mean(hr_valid)) if len(hr_valid) > 0 else 60.0
                
                slp_efficiency = float(row.get('sleep_efficiency', 80.0))
                slp_deep = float(row.get('sleep_deep', 3600.0)) / 3600.0
                slp_rem = float(row.get('sleep_rem', 3600.0)) / 3600.0
                awake_count = float(np.sum(hypno_arr == 4))
                
                daily_rows.append([met_mean, met_max, hr_mean, slp_efficiency, slp_deep, slp_rem, awake_count])
                
            daily_arr = np.array(daily_rows)
            if len(daily_arr) >= DAYS_LIMIT:
                seq_feats = daily_arr[:DAYS_LIMIT]
            else:
                pad_len = DAYS_LIMIT - len(daily_arr)
                seq_feats = np.pad(daily_arr, ((0, pad_len), (0, 0)), mode='edge')
                
        X_seq_list.append(seq_feats)
        y_list.append(label_dict[email])
        
    X_seq = np.array(X_seq_list)  # Shape: (174, 21, 7)
    y = np.array(y_list)          # Shape: (174,)
    
    # Scale channels across patients
    N, T, C = X_seq.shape
    X_reshaped = X_seq.reshape(-1, C)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped).reshape(N, T, C)
    
    print(f"Parsed Sequences Shape: {X_scaled.shape}, Targets: {y.shape}")
    return X_scaled, y

# PyTorch Dataset
class PatientSeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# PyTorch BiLSTM with Attention Pooling
class BiLSTMAttentionModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn_fc = nn.Linear(hidden_size * 2, 1)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        # x: (B, T, C)
        lstm_out, _ = self.lstm(x)  # (B, T, 2*hidden_size)
        
        # Attention weights over time T
        attn_weights = F.softmax(self.attn_fc(lstm_out), dim=1)  # (B, T, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)        # (B, 2*hidden_size)
        
        out = F.relu(self.fc1(context))
        out = self.dropout(out)
        logits = self.fc2(out).squeeze(-1) # (B,)
        return logits

def train_bilstm():
    X_seq, y = load_patient_sequences()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[2] Training PyTorch BiLSTM on device: {device}")
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros(len(y))
    
    pos_weight = torch.tensor([(len(y) - np.sum(y)) / np.sum(y)], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_seq, y)):
        X_tr, y_tr = X_seq[train_idx], y[train_idx]
        X_val, y_val = X_seq[val_idx], y[val_idx]
        
        train_ds = PatientSeqDataset(X_tr, y_tr)
        val_ds = PatientSeqDataset(X_val, y_val)
        
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
        
        model = BiLSTMAttentionModel(input_size=7, hidden_size=64, num_layers=2, dropout=0.3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        best_val_auc = 0.0
        best_probs = None
        
        for epoch in range(60):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                
            model.eval()
            val_preds_list = []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(device)
                    logits = model(bx)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_preds_list.extend(probs)
                    
            val_preds = np.array(val_preds_list)
            val_auc = roc_auc_score(y_val, val_preds)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_probs = val_preds
                
        oof_probs[val_idx] = best_probs
        print(f"  Fold {fold+1}/{N_SPLITS} Best ROC-AUC: {best_val_auc:.4f}")
        
    total_auc = roc_auc_score(y, oof_probs)
    print("==================================================")
    print(f"🔥 PyTorch BiLSTM 5-Fold OOF ROC-AUC: {total_auc:.4f}")
    print("==================================================")
    
    np.save(OUTPUT_OOF_PATH, oof_probs)
    print(f"Saved BiLSTM OOF probabilities to: {OUTPUT_OOF_PATH}")

if __name__ == "__main__":
    train_bilstm()
