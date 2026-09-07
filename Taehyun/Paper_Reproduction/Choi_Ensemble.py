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
    label_map = {"CN": 0, "MCI": 1, "Dem": 2, "Dementia": 2}
    label_df["label"] = label_df["DIAG_NM"].map(label_map)
    return label_df[["EMAIL", "DIAG_NM", "label"]].drop_duplicates("EMAIL")

def parse_slash_sequence(value, dtype=float) -> np.ndarray:
    if pd.isna(value): return np.array([], dtype=float)
    text = str(value).strip()
    if not text or text == "...": return np.array([], dtype=float)
    values = []
    for token in text.split("/"):
        token = token.strip()
        if not token or token == "...": continue
        try: values.append(dtype(token))
        except ValueError: continue
    return np.array(values, dtype=float)

def numeric_sequence_stats(seq: np.ndarray, prefix: str, remove_zero: bool = False) -> dict:
    arr = np.asarray(seq, dtype=float)
    arr = arr[arr != -1]
    if remove_zero: arr = arr[arr != 0]
    if len(arr) == 0:
        return {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan, f"{prefix}_var": np.nan, f"{prefix}_min": np.nan, f"{prefix}_max": np.nan, f"{prefix}_median": np.nan, f"{prefix}_q25": np.nan, f"{prefix}_q75": np.nan, f"{prefix}_iqr": np.nan, f"{prefix}_valid_count": 0}
    q25, q75 = np.percentile(arr, 25), np.percentile(arr, 75)
    return {f"{prefix}_mean": float(np.mean(arr)), f"{prefix}_std": float(np.std(arr)), f"{prefix}_var": float(np.var(arr)), f"{prefix}_min": float(np.min(arr)), f"{prefix}_max": float(np.max(arr)), f"{prefix}_median": float(np.median(arr)), f"{prefix}_q25": float(q25), f"{prefix}_q75": float(q75), f"{prefix}_iqr": float(q75 - q25), f"{prefix}_valid_count": int(len(arr))}

def activity_class_features(seq: np.ndarray) -> dict:
    arr = np.asarray(seq, dtype=float)
    arr = arr[arr != -1]
    total = len(arr)
    out = {}
    for level in range(6):
        count = int(np.sum(arr == level))
        out[f"activity_class_{level}_count"] = count
        out[f"activity_class_{level}_ratio"] = count / total if total else np.nan
    out["activity_rest_ratio"] = out.get("activity_class_1_ratio", np.nan)
    out["activity_inactive_ratio"] = out.get("activity_class_2_ratio", np.nan)
    out["activity_active_ratio"] = out.get("activity_class_3_ratio", 0) + out.get("activity_class_4_ratio", 0) + out.get("activity_class_5_ratio", 0) if total else np.nan
    out["activity_not_worn_ratio"] = out.get("activity_class_0_ratio", np.nan)
    out["activity_class_valid_count"] = total
    return out

def sleep_hypnogram_features(seq: np.ndarray) -> dict:
    arr = np.asarray(seq, dtype=float)
    arr = arr[(arr != -1) & (arr != 0)]
    total = len(arr)
    out = {}
    stage_map = {1: "deep", 2: "light", 3: "rem", 4: "awake"}
    for level, name in stage_map.items():
        count = int(np.sum(arr == level))
        out[f"sleep_{name}_count_5min"] = count
        out[f"sleep_{name}_ratio_5min"] = count / total if total else np.nan
    out["sleep_stage_transition_count"] = int(np.sum(arr[1:] != arr[:-1])) if total > 1 else 0
    out["sleep_hypnogram_valid_count"] = total
    return out

def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rows = []
    for _, row in df.iterrows():
        feats = {}
        if "CONVERT(activity_class_5min USING utf8)" in row:
            feats.update(activity_class_features(parse_slash_sequence(row["CONVERT(activity_class_5min USING utf8)"], dtype=float)))
        if "CONVERT(activity_met_1min USING utf8)" in row:
            feats.update(numeric_sequence_stats(parse_slash_sequence(row["CONVERT(activity_met_1min USING utf8)"], dtype=float), "activity_met_1min", remove_zero=False))
        if "CONVERT(sleep_hr_5min USING utf8)" in row:
            feats.update(numeric_sequence_stats(parse_slash_sequence(row["CONVERT(sleep_hr_5min USING utf8)"], dtype=float), "sleep_hr_5min", remove_zero=True))
        if "CONVERT(sleep_rmssd_5min USING utf8)" in row:
            feats.update(numeric_sequence_stats(parse_slash_sequence(row["CONVERT(sleep_rmssd_5min USING utf8)"], dtype=float), "sleep_rmssd_5min", remove_zero=True))
        if "CONVERT(sleep_hypnogram_5min USING utf8)" in row:
            feats.update(sleep_hypnogram_features(parse_slash_sequence(row["CONVERT(sleep_hypnogram_5min USING utf8)"], dtype=float)))
        rows.append(feats)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

def make_daily_table(activity: pd.DataFrame, sleep: pd.DataFrame, label: pd.DataFrame) -> pd.DataFrame:
    if activity.empty or sleep.empty: return pd.DataFrame()
    activity = coerce_numeric_columns(activity, keep={"EMAIL", "activity_day_start", "activity_day_end"})
    sleep = coerce_numeric_columns(sleep, keep={"EMAIL", "sleep_bedtime_start", "sleep_bedtime_end"})
    
    activity["activity_day_start_dt"] = pd.to_datetime(activity.get("activity_day_start"), errors="coerce")
    activity["activity_day_end_dt"] = pd.to_datetime(activity.get("activity_day_end"), errors="coerce")
    sleep["sleep_bedtime_start_dt"] = pd.to_datetime(sleep.get("sleep_bedtime_start"), errors="coerce")
    sleep["sleep_bedtime_end_dt"] = pd.to_datetime(sleep.get("sleep_bedtime_end"), errors="coerce")
    
    activity["date"] = activity["activity_day_start_dt"].dt.date
    sleep["date"] = sleep["sleep_bedtime_end_dt"].dt.date
    
    sleep_start = sleep["sleep_bedtime_start_dt"]
    sleep_end = sleep["sleep_bedtime_end_dt"]
    sleep["_sleep_duration_seconds"] = (sleep_end - sleep_start).dt.total_seconds()
    sleep = sleep.sort_values(["EMAIL", "date", "_sleep_duration_seconds"], ascending=[True, True, False]).drop_duplicates(["EMAIL", "date"], keep="first").reset_index(drop=True)
    
    daily = activity.merge(sleep, on=["EMAIL", "date"], how="inner", suffixes=("", "_sleep"))
    if not label.empty:
        daily = daily.merge(label, on="EMAIL", how="left")
    
    daily = add_sequence_features(daily)
    
    drop_cols = ["activity_class_5min", "activity_met_1min", "sleep_hr_5min", "sleep_hypnogram_5min", "sleep_rmssd_5min", "CONVERT(activity_class_5min USING utf8)", "CONVERT(activity_met_1min USING utf8)", "CONVERT(sleep_hr_5min USING utf8)", "CONVERT(sleep_hypnogram_5min USING utf8)", "CONVERT(sleep_rmssd_5min USING utf8)", "activity_day_start", "activity_day_end", "activity_day_start_dt", "activity_day_end_dt", "sleep_bedtime_start", "sleep_bedtime_end", "sleep_bedtime_start_dt", "sleep_bedtime_end_dt", "_sleep_duration_seconds", "date", "DIAG_NM"]
    daily = daily.drop(columns=[c for c in drop_cols if c in daily.columns], errors="ignore")
    daily = daily.replace([np.inf, -np.inf], np.nan)
    return daily

def process_raw_to_daily_level():
    train_act = read_csv_flexible(find_csv("train_activity.csv"))
    val_act = read_csv_flexible(find_csv("val_activity.csv"))
    train_slp = read_csv_flexible(find_csv("train_sleep.csv"))
    val_slp = read_csv_flexible(find_csv("val_sleep.csv"))
    
    train_lbl_name = "training_label.csv" if (DATA_DIR / "training_label.csv").exists() else "training_label_activity.csv"
    val_lbl_name = "val_label.csv" if (DATA_DIR / "val_label.csv").exists() else "val_label_activity.csv"
    
    train_lbl = preprocess_label(read_csv_flexible(find_csv(train_lbl_name)))
    val_lbl = preprocess_label(read_csv_flexible(find_csv(val_lbl_name)))
    
    train_daily = make_daily_table(train_act, train_slp, train_lbl)
    val_daily = make_daily_table(val_act, val_slp, val_lbl)
    
    all_daily = pd.concat([train_daily, val_daily], ignore_index=True)
    
    TARGET_COL = "label"
    DROP_COLS = ["EMAIL", TARGET_COL, "DIAG_NM", "date"]
    features = [c for c in all_daily.columns if c not in DROP_COLS and pd.api.types.is_numeric_dtype(all_daily[c])]
    
    all_daily[features] = all_daily[features].fillna(all_daily[features].median())
    
    X = all_daily[features].values
    y = all_daily[TARGET_COL].astype(int).values
    groups = all_daily["EMAIL"].values
    
    return X, y, groups, features

# ==========================================
# 2. Main Model Execution
# ==========================================
class PaperLSTM(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=128, batch_first=True)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.3)
        
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.3)
        
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = out.transpose(1, 2)
        out = self.bn1(out)
        out = out.transpose(1, 2)
        out = self.drop1(out)
        
        out, _ = self.lstm2(out)
        out = out.transpose(1, 2)
        out = self.bn2(out)
        out = out.transpose(1, 2)
        out = self.drop2(out)
        
        out = out[:, -1, :] 
        return self.fc(out)

class PyTorchLSTMClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, epochs=20, batch_size=32, lr=0.001, random_state=42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        
    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.classes_ = np.unique(y)
        
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        y_t = torch.tensor(y, dtype=torch.long)
        
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = PaperLSTM(input_dim=X.shape[1], num_classes=len(self.classes_))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(bx), by)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            probs = torch.nn.functional.softmax(self.model(X_t), dim=1).numpy()
        return probs

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def get_stats(arr):
    arr = np.array(arr)
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0 or len(arr) == 1:
        return mean, std, mean, mean
    ci = st.t.interval(0.95, len(arr)-1, loc=mean, scale=st.sem(arr))
    return mean, std, ci[0], ci[1]

def compute_multiclass_auc(y_true, y_score, all_classes=(0, 1, 2)):
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return 0.5
    try:
        return roc_auc_score(y_true, y_score, multi_class="ovo", labels=list(all_classes))
    except Exception:
        try:
            return roc_auc_score(y_true, y_score[:, unique_classes], multi_class="ovo")
        except Exception:
            return 0.5

CHOI_PAPER_FEATURES = [
    "activity_high", "activity_steps", "activity_daily_movement", "activity_non_wear", 
    "activity_inactivity_alerts", "activity_score_training_frequency", "activity_score_training_volume", 
    "active_low", "activity_medium", "activity_met_min_inactive", "activity_met_min_low",
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
        {"name": "Scenario2_Leakage_NestedCV", "leak": True, "nested": True},
        {"name": "Scenario3_NoLeakage_SingleCV", "leak": False, "nested": False},
        {"name": "Scenario4_NoLeakage_NestedCV", "leak": False, "nested": True},
    ]
    
    for sc in scenarios:
        run_scenario(X, y, groups, feature_names, sc["name"], sc["leak"], sc["nested"], n_repeats=5)

if __name__ == '__main__':
    main()
