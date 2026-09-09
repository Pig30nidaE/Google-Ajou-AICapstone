import os
import sys
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
import optuna

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# 1. Raw Data Preprocessing
# ==========================================
# Taehyun 폴더 바로 아래의 data 폴더 경로 설정 (하위 서브폴더 미사용)
CURRENT_FILE_DIR = Path(__file__).resolve().parent
TAEHYUN_DIR = CURRENT_FILE_DIR.parent
DATA_DIR = TAEHYUN_DIR / "data"

if not DATA_DIR.exists():
    if (Path.cwd().parent / "data").exists():
        DATA_DIR = Path.cwd().parent / "data"
    elif Path(r"c:\ML4\Google-Ajou-AICapstone\Taehyun\data").exists():
        DATA_DIR = Path(r"c:\ML4\Google-Ajou-AICapstone\Taehyun\data")
    else:
        DATA_DIR = Path("./data")

def find_csv(filename: str) -> Path:
    return DATA_DIR / filename

def read_csv_flexible(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str, low_memory=False)
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    return pd.DataFrame()

def coerce_numeric_columns(df: pd.DataFrame, keep: set) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col not in keep:
            converted = pd.to_numeric(df[col], errors="coerce")
            if not converted.isna().all():
                df[col] = converted
    return df

def preprocess_label(label_df: pd.DataFrame) -> pd.DataFrame:
    label_df = label_df.copy()
    if "SAMPLE_EMAIL" in label_df.columns:
        label_df = label_df.rename(columns={"SAMPLE_EMAIL": "EMAIL"})
    if label_df.empty: return label_df
    original_label_map = {"CN": 0, "MCI": 1, "Dem": 2, "Dementia": 2}
    label_df["label"] = label_df["DIAG_NM"].map(original_label_map)
    return label_df[["EMAIL", "DIAG_NM", "label"]].dropna().drop_duplicates("EMAIL")

def make_daily_table(activity: pd.DataFrame, sleep: pd.DataFrame, label: pd.DataFrame) -> pd.DataFrame:
    if activity.empty or sleep.empty: return pd.DataFrame()
    activity = coerce_numeric_columns(activity, keep={"EMAIL", "activity_day_start", "activity_day_end"})
    sleep = coerce_numeric_columns(sleep, keep={"EMAIL", "sleep_bedtime_start", "sleep_bedtime_end"})
    
    activity["activity_day_start_dt"] = pd.to_datetime(activity.get("activity_day_start"), errors="coerce")
    sleep["sleep_bedtime_end_dt"] = pd.to_datetime(sleep.get("sleep_bedtime_end"), errors="coerce")
    
    activity["date"] = activity["activity_day_start_dt"].dt.date
    sleep["date"] = sleep["sleep_bedtime_end_dt"].dt.date
    
    daily = activity.merge(sleep, on=["EMAIL", "date"], how="inner", suffixes=("", "_sleep"))
    if not label.empty:
        daily = daily.merge(label, on="EMAIL", how="inner")
        
    drop_cols = [c for c in daily.columns if "CONVERT" in c or "USING utf8" in c]
    drop_cols += ["activity_day_start", "activity_day_end", "activity_day_start_dt", "sleep_bedtime_start", "sleep_bedtime_end", "sleep_bedtime_end_dt", "DIAG_NM", "date"]
    daily = daily.drop(columns=[c for c in drop_cols if c in daily.columns], errors="ignore")
    daily = daily.replace([np.inf, -np.inf], np.nan)
    return daily

def process_raw_to_daily_level():
    print("Loading raw CSV files from data directory...")
    train_act = read_csv_flexible(find_csv("train_activity.csv"))
    val_act = read_csv_flexible(find_csv("val_activity.csv"))
    train_slp = read_csv_flexible(find_csv("train_sleep.csv"))
    val_slp = read_csv_flexible(find_csv("val_sleep.csv"))
    
    train_lbl_name = "training_label.csv" if (DATA_DIR / "training_label.csv").exists() else "training_label_activity.csv"
    val_lbl_name = "val_label.csv" if (DATA_DIR / "val_label.csv").exists() else "val_label_activity.csv"
    
    train_lbl = preprocess_label(read_csv_flexible(find_csv(train_lbl_name)))
    val_lbl = preprocess_label(read_csv_flexible(find_csv(val_lbl_name)))
    
    print("Processing daily tables...")
    train_daily = make_daily_table(train_act, train_slp, train_lbl)
    val_daily = make_daily_table(val_act, val_slp, val_lbl)
    
    all_daily = pd.concat([train_daily, val_daily], ignore_index=True)
    
    TARGET_COL = "label"
    DROP_COLS = ["EMAIL", TARGET_COL]
    features = [c for c in all_daily.columns if c not in DROP_COLS and pd.api.types.is_numeric_dtype(all_daily[c])]
    
    all_daily[features] = all_daily[features].fillna(all_daily[features].median())
    
    X = all_daily[features].values
    y = all_daily[TARGET_COL].astype(int).values
    groups = all_daily["EMAIL"].values
    
    return X, y, groups, features

# ==========================================
# 2. PyTorch LSTM Model Definition
# ==========================================
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=3):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, (hn, _) = self.lstm(x)
        logits = self.fc(hn[-1])
        return logits

class PyTorchLSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_dim=64, epochs=30, batch_size=32, lr=0.001, random_state=42):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.classes_ = np.array([0, 1, 2])
        self.model = None
        
    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        input_dim = X.shape[1]
        self.model = SimpleLSTM(input_dim, self.hidden_dim, num_classes=3)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = self.model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
        return self
        
    def predict_proba(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# ==========================================
# 3. Statistical Utilities
# ==========================================
def get_stats(arr):
    arr = np.array(arr)
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0 or len(arr) == 1:
        return mean, std, mean, mean
    ci = st.t.interval(0.95, len(arr)-1, loc=mean, scale=st.sem(arr))
    return mean, std, ci[0], ci[1]

def compute_multiclass_auc(y_true, probs, all_classes=[0, 1, 2]):
    try:
        if len(np.unique(y_true)) == len(all_classes):
            return roc_auc_score(y_true, probs, multi_class='ovr', labels=all_classes)
        else:
            aucs = []
            for c in all_classes:
                y_c = (y_true == c).astype(int)
                if len(np.unique(y_c)) > 1:
                    aucs.append(roc_auc_score(y_c, probs[:, c]))
            return np.mean(aucs) if len(aucs) > 0 else 0.5
    except Exception:
        return 0.5

CHOI_PAPER_FEATURES = [
    "activity_met_min_medium", "activity_average_met", "activity_cal_active", 
    "activity_score_meet_daily_targets", "activity_score_stay_active", "activity_score",
    "activity_total", "activity_score_recovery_time", "activity_rest",
    "sleep_light", "sleep_deep", "sleep_score_deep", "sleep_awake", "sleep_restless", 
    "sleep_rem", "sleep_score_rem", "sleep_hr_lowest", "sleep_hr_average", "sleep_breath_average",
    "sleep_score_disturbances", "sleep_total", "sleep_score_alignment", "sleep_period_id", 
    "sleep_onset_latency", "sleep_score_latency", "sleep_score", "sleep_midpoint_time", 
    "sleep_midpoint_at_delta", "sleep_score_efficiency", "sleep_duration", "sleep_temperature_deviation",
    "activity_active_low" 
]

def choi_feature_selection(X_train, y_train, feature_names):
    df = pd.DataFrame(X_train, columns=feature_names)
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    median_imp = np.median(importances)
    corr_matrix = df.corr().abs()
    
    to_drop = []
    for i in range(len(feature_names)):
        has_high_corr = False
        for j in range(len(feature_names)):
            if i != j and corr_matrix.iloc[i, j] >= 0.9:
                has_high_corr = True
                break
        if has_high_corr and importances[i] < median_imp:
            to_drop.append(i)
            
    keep_indices = [i for i in range(len(feature_names)) if i not in to_drop]
    return keep_indices

def run_scenario(X, y, groups, feature_names, scenario_name, leakage_allowed, use_nested_cv, n_repeats=5):
    print(f"\n{'='*60}")
    print(f"Starting Scenario: {scenario_name}")
    print(f"Leakage Allowed: {leakage_allowed}, Nested CV: {use_nested_cv}, Repeats: {n_repeats}")
    print(f"{'='*60}")
    
    n_splits = 5
    metrics = {m: {'acc': [], 'auc': [], 'prec': [], 'rec': [], 'f1': []} for m in ['Ensemble']}
    
    for repeat in range(n_repeats):
        if leakage_allowed:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + repeat)
            cv_split = list(cv.split(X, y))
            X_used = X
        else:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42 + repeat)
            cv_split = list(cv.split(X, y, groups))
            X_used = X

        for fold, (train_idx, test_idx) in enumerate(cv_split, 1):
            if leakage_allowed:
                keep_indices = [i for i, f in enumerate(feature_names) if f in CHOI_PAPER_FEATURES or f.replace("activity_active_low", "active_low") in CHOI_PAPER_FEATURES]
                if len(keep_indices) == 0: keep_indices = list(range(len(feature_names)))
                X_tr = X_used[train_idx][:, keep_indices]
                X_te = X_used[test_idx][:, keep_indices]
            else:
                X_tr_raw = X_used[train_idx]
                X_te_raw = X_used[test_idx]
                keep_indices = choi_feature_selection(X_tr_raw, y[train_idx], feature_names)
                X_tr = X_tr_raw[:, keep_indices]
                X_te = X_te_raw[:, keep_indices]
                
            y_tr, y_te = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_te)
            
            smote = SMOTE(random_state=42)
            X_tr_res, y_tr_res = smote.fit_resample(X_tr_scaled, y_tr)
            
            if use_nested_cv:
                def objective(trial):
                    max_depth = trial.suggest_int('max_depth', 5, 20)
                    lr = trial.suggest_float('lr', 0.05, 0.2)
                    
                    rf = RandomForestClassifier(n_estimators=50, max_depth=max_depth, class_weight='balanced', random_state=42)
                    gbm = GradientBoostingClassifier(n_estimators=50, max_depth=max_depth//2, learning_rate=lr, random_state=42)
                    svm = SVC(kernel='rbf', gamma='auto', class_weight='balanced', C=10, probability=True, random_state=42)
                    xgb = XGBClassifier(n_estimators=50, max_depth=max_depth//2, learning_rate=lr, eval_metric='mlogloss', random_state=42)
                    lstm = PyTorchLSTMClassifier(epochs=10, batch_size=32, lr=0.001, random_state=42)
                    
                    model = VotingClassifier(estimators=[('RF', rf), ('GBM', gbm), ('SVM', svm), ('XGB', xgb), ('LSTM', lstm)], voting='soft')
                    
                    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    inner_aucs = []
                    for inner_tr, inner_val in inner_cv.split(X_tr_res, y_tr_res):
                        model.fit(X_tr_res[inner_tr], y_tr_res[inner_tr])
                        probs = model.predict_proba(X_tr_res[inner_val])
                        inner_aucs.append(compute_multiclass_auc(y_tr_res[inner_val], probs, all_classes=[0, 1, 2]))
                    return np.mean(inner_aucs)
                    
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=3) 
                best_depth = study.best_params['max_depth']
                best_lr = study.best_params['lr']
                
                rf = RandomForestClassifier(n_estimators=200, max_depth=best_depth, class_weight='balanced', random_state=42)
                gbm = GradientBoostingClassifier(n_estimators=100, max_depth=best_depth//2, learning_rate=best_lr, random_state=42)
                svm = SVC(kernel='rbf', gamma='auto', class_weight='balanced', C=50, probability=True, random_state=42)
                xgb = XGBClassifier(n_estimators=100, max_depth=best_depth//2, learning_rate=best_lr, eval_metric='mlogloss', random_state=42)
                lstm = PyTorchLSTMClassifier(epochs=30, batch_size=32, lr=0.001, random_state=42)
            else:
                rf = RandomForestClassifier(n_estimators=500, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=30, class_weight='balanced', random_state=42)
                gbm = GradientBoostingClassifier(subsample=1.0, n_estimators=200, max_depth=7, learning_rate=0.2, random_state=42)
                svm = SVC(kernel='rbf', gamma='auto', class_weight='balanced', C=50, probability=True, random_state=42)
                xgb = XGBClassifier(subsample=0.8, reg_lambda=2, n_estimators=300, max_depth=7, learning_rate=0.2, eval_metric='mlogloss', random_state=42)
                lstm = PyTorchLSTMClassifier(epochs=30, batch_size=32, lr=0.001, random_state=42)
            
            model = VotingClassifier(estimators=[('RF', rf), ('GBM', gbm), ('SVM', svm), ('XGB', xgb), ('LSTM', lstm)], voting='soft')
            model.fit(X_tr_res, y_tr_res)
            
            preds = model.predict(X_te_scaled)
            probs = model.predict_proba(X_te_scaled)
            
            acc = accuracy_score(y_te, preds)
            auc = compute_multiclass_auc(y_te, probs, all_classes=[0, 1, 2])
            prec = precision_score(y_te, preds, average='macro', zero_division=0)
            rec = recall_score(y_te, preds, average='macro', zero_division=0)
            f1 = f1_score(y_te, preds, average='macro', zero_division=0)
            
            metrics['Ensemble']['acc'].append(acc)
            metrics['Ensemble']['auc'].append(auc)
            metrics['Ensemble']['prec'].append(prec)
            metrics['Ensemble']['rec'].append(rec)
            metrics['Ensemble']['f1'].append(f1)
            
            print(f"[{scenario_name}] R{repeat+1}-F{fold} Completed. AUC: {auc:.4f}")
            
    print(f"\nFinal Statistical Results (5 Repeats x 5 Folds):")
    for m_name in ['acc', 'auc', 'prec', 'rec', 'f1']:
        mean, std, lower, upper = get_stats(metrics['Ensemble'][m_name])
        if m_name == 'auc':
            print(f"{m_name.upper():<9} : {mean:.4f} ± {std:.4f} (95% CI: {lower:.4f}-{upper:.4f})")
        else:
            print(f"{m_name.upper():<9} : {mean*100:.2f} ± {std*100:.2f}% (95% CI: {lower*100:.2f}-{upper*100:.2f})")

def main():
    X, y, groups, feature_names = process_raw_to_daily_level()
    print(f"\nData Loaded. Shape: {X.shape}, Classes: {np.unique(y)}")
    
    scenarios = [
        {"name": "Scenario1_Leakage_SingleCV", "leak": True, "nested": False},
        {"name": "Scenario2_Leakage_NestedCV", "leak": True, "nested": True},
        {"name": "Scenario3_NoLeakage_SingleCV", "leak": False, "nested": False},
        {"name": "Scenario4_NoLeakage_NestedCV", "leak": False, "nested": True},
    ]
    
    for sc in scenarios:
        run_scenario(X, y, groups, feature_names, sc["name"], sc["leak"], sc["nested"], n_repeats=5)

if __name__ == '__main__':
    main()
