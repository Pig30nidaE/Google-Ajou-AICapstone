import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

class CustomRRNN(nn.Module):
    def __init__(self, hidden_size=64, day_hidden_size=64, num_summary_feats=44):
        super(CustomRRNN, self).__init__()
        # 1st level: intra-day sequence
        self.lstm1 = nn.LSTM(4, day_hidden_size, num_layers=1, bias=True, batch_first=True, bidirectional=True)
        # 2nd level: inter-day sequence
        self.lstm2 = nn.LSTM(num_summary_feats + 2*day_hidden_size, hidden_size, num_layers=1, bias=True, batch_first=True, bidirectional=True)
        # Binary Classification
        self.fc = nn.Linear(2*hidden_size, 2)
        
        self.day_hidden_size = day_hidden_size
        self.hidden_size = hidden_size
        
    def forward(self, x, d_):
        # x shape: (batch, days, 4, time_steps)
        batch_size, days, num_features, time_steps = x.shape
        
        # Reshape for lstm1: (batch * days, time_steps, 4)
        x_reshaped = x.view(batch_size * days, num_features, time_steps).transpose(1, 2)
        
        # Pass through lstm1
        out1, (h_n, c_n) = self.lstm1(x_reshaped)
        # out1 shape: (batch * days, time_steps, 2*day_hidden_size)
        
        # We need the last output. Since it's bidirectional, we concatenate the last hidden states of both directions
        # h_n shape: (2, batch * days, day_hidden_size)
        h_n_fwd = h_n[0, :, :]
        h_n_bwd = h_n[1, :, :]
        day_rep = torch.cat([h_n_fwd, h_n_bwd], dim=1) # (batch * days, 2*day_hidden_size)
        
        # Reshape back to (batch, days, 2*day_hidden_size)
        day_rep = day_rep.view(batch_size, days, 2 * self.day_hidden_size)
        
        # Concatenate with daily summary features
        mix_f = torch.cat([day_rep, d_], dim=2) # (batch, days, num_summary_feats + 2*day_hidden_size)
        
        # Pass through lstm2
        out2, (h_n2, c_n2) = self.lstm2(mix_f)
        
        # Take the last hidden state of the sequence of days
        h_n2_fwd = h_n2[0, :, :]
        h_n2_bwd = h_n2[1, :, :]
        patient_rep = torch.cat([h_n2_fwd, h_n2_bwd], dim=1) # (batch, 2*hidden_size)
        
        # Final classification
        logits = self.fc(patient_rep)
        return logits

class PatientDataset(Dataset):
    def __init__(self, X_seq, X_sum, y):
        self.X_seq = X_seq
        self.X_sum = X_sum
        self.y = y
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_sum[idx], self.y[idx]

def parse_5min_string(s):
    if pd.isna(s) or s == "..." or str(s).strip() == "":
        return np.zeros(288)
    if "/" in str(s):
        parts = str(s).split("/")
        arr = [float(p) if p.replace('.','',1).isdigit() else 0.0 for p in parts]
    else:
        arr = [float(ch) for ch in str(s) if ch.isdigit()]
        
    if len(arr) == 0:
        return np.zeros(288)
        
    arr = np.array(arr)
    if len(arr) >= 288:
        return arr[:288]
    else:
        padded = np.zeros(288)
        padded[:len(arr)] = arr
        return padded

def load_data():
    print("Loading datasets...")
    df_act = pd.read_csv("c:/ML4/data/train_activity.csv")
    df_slp = pd.read_csv("c:/ML4/data/train_sleep.csv")
    df_lbl = pd.read_csv("c:/ML4/data/training_label_activity.csv")
    
    df_lbl = df_lbl.rename(columns={'SAMPLE_EMAIL': 'EMAIL'})
    df_act['date'] = pd.to_datetime(df_act['activity_day_end']).dt.date
    df_slp['date'] = pd.to_datetime(df_slp['sleep_bedtime_end']).dt.date
    
    df = pd.merge(df_act, df_slp, on=['EMAIL', 'date'], how='inner')
    df = pd.merge(df, df_lbl[['EMAIL', 'DIAG_NM']], on='EMAIL', how='inner').drop_duplicates(subset=['EMAIL', 'date'])
    
    # 🌟 Binary Target: CN(0) vs MCI/Dem(1)
    df['label'] = np.where(df['DIAG_NM'] == 'CN', 0, 1)
    
    seq_cols = ['activity_class_5min', 'sleep_hr_5min', 'sleep_hypnogram_5min', 'sleep_rmssd_5min']
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['label'] and 'class_5min' not in c]
    
    # Use SHAP top features + others up to ~44 to keep dimensions reasonable
    num_cols = num_cols[:44]
    
    df[num_cols] = df[num_cols].fillna(0)
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    emails = df['EMAIL'].unique()
    
    X_seq_list, X_sum_list, y_list, group_list = [], [], [], []
    
    print("Parsing sequences (This may take a minute)...")
    for email in emails:
        pdf = df[df['EMAIL'] == email].sort_values('date')
        if len(pdf) == 0: continue
        
        pdf = pdf.iloc[-28:]
        days = len(pdf)
        pad_days = max(0, 28 - days)
            
        seq_day_list = []
        for i in range(days):
            day_arr = [parse_5min_string(pdf.iloc[i][col]) for col in seq_cols]
            seq_day_list.append(np.stack(day_arr))
            
        for _ in range(pad_days):
            seq_day_list.append(np.zeros((4, 288)))
            
        seq_tensor = np.stack(seq_day_list)
        
        sum_arr = pdf[num_cols].values
        if pad_days > 0:
            sum_arr = np.vstack([sum_arr, np.zeros((pad_days, len(num_cols)))])
            
        X_seq_list.append(seq_tensor)
        X_sum_list.append(sum_arr)
        y_list.append(pdf['label'].iloc[0])
        group_list.append(email)
        
    X_seq = torch.tensor(np.stack(X_seq_list), dtype=torch.float32)
    X_sum = torch.tensor(np.stack(X_sum_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)
    
    return X_seq, X_sum, y, np.array(group_list), len(num_cols)

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X_seq, X_sum, y, groups, num_sum_feats = load_data()
    print(f"Total Patients: {len(y)}")
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    metrics = {'acc':[], 'prec':[], 'rec':[], 'f1':[], 'auc':[]}
    epochs = 10
    batch_size = 8
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_seq, y.numpy(), groups=groups)):
        print(f"\n--- Fold {fold+1} ---")
        
        train_ds = PatientDataset(X_seq[train_idx], X_sum[train_idx], y[train_idx])
        val_ds = PatientDataset(X_seq[val_idx], X_sum[val_idx], y[val_idx])
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        model = CustomRRNN(hidden_size=64, day_hidden_size=64, num_summary_feats=num_sum_feats).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        best_f1 = 0
        best_preds, best_y, best_probs = [], [], []
        
        for ep in range(epochs):
            model.train()
            train_loss = 0
            for seq_b, sum_b, y_b in tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs} Train", leave=False):
                seq_b, sum_b, y_b = seq_b.to(device), sum_b.to(device), y_b.to(device)
                
                optimizer.zero_grad()
                logits = model(seq_b, sum_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            model.eval()
            val_preds = []
            val_true = []
            val_probs = []
            with torch.no_grad():
                for seq_b, sum_b, y_b in tqdm(val_loader, desc=f"Epoch {ep+1}/{epochs} Val", leave=False):
                    seq_b, sum_b, y_b = seq_b.to(device), sum_b.to(device), y_b.to(device)
                    logits = model(seq_b, sum_b)
                    preds = torch.argmax(logits, dim=1)
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    val_preds.extend(preds.cpu().numpy())
                    val_probs.extend(probs.cpu().numpy())
                    val_true.extend(y_b.cpu().numpy())
            
            v_acc = accuracy_score(val_true, val_preds)
            v_f1 = f1_score(val_true, val_preds, average='macro')
            
            if v_f1 > best_f1:
                best_f1 = v_f1
                best_preds = val_preds
                best_probs = val_probs
                best_y = val_true
                
            if (ep+1) % 5 == 0:
                print(f"  Epoch {ep+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {v_acc:.4f} | Val F1: {v_f1:.4f}")
                
        # 매 폴드 종료 시 파일에 기록하여 에이전트가 확인할 수 있게 함
        with open("c:/ML4/fold_progress.log", "a") as f:
            f.write(f"Fold {fold+1} completed. Best F1: {best_f1:.4f}\n")
            
        try:
            fold_auc = roc_auc_score(best_y, best_probs)
        except ValueError:
            fold_auc = 0.5
            
        metrics['acc'].append(accuracy_score(best_y, best_preds))
        metrics['prec'].append(precision_score(best_y, best_preds, average='macro', zero_division=0))
        metrics['rec'].append(recall_score(best_y, best_preds, average='macro', zero_division=0))
        metrics['f1'].append(best_f1)
        metrics['auc'].append(fold_auc)
        
    print("\n" + "="*50)
    print("V17 FINAL RESULTS (Custom Hierarchical LSTM - Binary)")
    print("="*50)
    print(f"Accuracy:  {np.mean(metrics['acc']):.4f}")
    print(f"Precision: {np.mean(metrics['prec']):.4f}")
    print(f"Recall:    {np.mean(metrics['rec']):.4f}")
    print(f"Macro F1:  {np.mean(metrics['f1']):.4f}")
    print(f"ROC AUC:   {np.mean(metrics['auc']):.4f}")
    print("="*50)

if __name__ == "__main__":
    train_model()
