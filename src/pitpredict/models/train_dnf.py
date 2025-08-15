"""Training Skript: Kalibriertes Pre-Race DNF Modell

Ausführung:
    pip install fastf1 pandas numpy scikit-learn pyyaml joblib pyarrow
    python -m src.pitpredict.models.train_dnf

Voraussetzung: ETL erzeugt data/season=2024/driver_race_table.parquet
"""
from __future__ import annotations

import os
import json
import joblib
import warnings
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer  # hinzugefügt

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------- Config Laden ----------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
CONFIG_PATH = os.path.join(ROOT, 'config.yaml')
with open(CONFIG_PATH, 'r') as f:
    CFG = yaml.safe_load(f)

PROCESSED_TABLE = CFG['processed_table']
METRICS_DIR = CFG['metrics_dir']
MODELS_DIR = CFG['models_dir']
HOLDOUT_ROUNDS = set(CFG.get('holdout_rounds', []))
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ----------------------------- Daten Laden -----------------------------------
if not os.path.exists(PROCESSED_TABLE):
    raise FileNotFoundError(f"Processed table nicht gefunden: {PROCESSED_TABLE}")

print(f"[INFO] Lade Tabelle {PROCESSED_TABLE}")
df = pd.read_parquet(PROCESSED_TABLE)

# ----------------------------- Split -----------------------------------------
train_mask = ~df['round'].isin(HOLDOUT_ROUNDS)
train_df = df[train_mask].copy()
holdout_df = df[~train_mask].copy()
print(f"[INFO] Train Runden: {sorted(train_df['round'].unique())}")
print(f"[INFO] Holdout Runden: {sorted(holdout_df['round'].unique())}")

# ----------------------------- Feature Auswahl -------------------------------
exclude_num = {'season', 'round', 'race_id', 'driver', 'driver_number', 'finish_position', 'laps_completed', 'classification_status', 'is_dnf', '__feature_availability__'}
num_features = [c for c in train_df.select_dtypes(include=['int64', 'float64']).columns if c not in exclude_num]
# grid_position sicherstellen
if 'grid_position' not in num_features and 'grid_position' in train_df.columns:
    num_features.append('grid_position')

cat_features = [c for c in ['circuit', 'team', 'degradation_class'] if c in train_df.columns]
print(f"[INFO] Numerische Features ({len(num_features)}): {num_features}")
print(f"[INFO] Kategorische Features ({len(cat_features)}): {cat_features}")

X = train_df[num_features + cat_features]
y = train_df['is_dnf']
groups = train_df['race_id']

# Kurzer Missing Report
missing_summary = X.isna().sum()
if missing_summary.any():
    print("[INFO] Missing Values vor Imputation:")
    print(missing_summary[missing_summary>0].sort_values(ascending=False))

# ----------------------------- Pipeline Def ----------------------------------
preprocess = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=False))
    ]), num_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_features)
])

base_clf = LogisticRegression(class_weight='balanced', max_iter=1000)
pipe = Pipeline([
    ('preprocess', preprocess),
    ('clf', base_clf)
])

# ----------------------------- Cross Validation ------------------------------
cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))

fold_reports = []
oof_pred = np.zeros(len(train_df))

for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = pipe.fit(X_tr, y_tr)
    # Kalibration (Isotonic) nur auf val split
    calib = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
    calib.fit(X_val, y_val)
    prob_val = calib.predict_proba(X_val)[:, 1]
    oof_pred[val_idx] = prob_val

    pr_auc = average_precision_score(y_val, prob_val)
    brier = brier_score_loss(y_val, prob_val)
    ll = log_loss(y_val, prob_val, labels=[0,1])
    fold_reports.append({
        'fold': fold,
        'val_size': int(len(val_idx)),
        'pr_auc': float(pr_auc),
        'brier': float(brier),
        'log_loss': float(ll)
    })
    print(f"[CV] Fold {fold}: PR-AUC={pr_auc:.3f} Brier={brier:.4f} LogLoss={ll:.4f}")

# Gesamte CV Metriken
pr_auc_all = average_precision_score(y, oof_pred)
brier_all = brier_score_loss(y, oof_pred)
logloss_all = log_loss(y, oof_pred, labels=[0,1])
summary = {
    'folds': fold_reports,
    'oof_pr_auc': float(pr_auc_all),
    'oof_brier': float(brier_all),
    'oof_log_loss': float(logloss_all),
    'n_train_samples': int(len(train_df)),
    'n_events': int(y.sum()),
    'event_rate': float(y.mean())
}

with open(os.path.join(METRICS_DIR, 'cv_report.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f"[INFO] CV Report gespeichert -> {os.path.join(METRICS_DIR, 'cv_report.json')}")
np.save(os.path.join(METRICS_DIR, 'oof_pred.npy'), oof_pred)
print(f"[INFO] OOF Predictions gespeichert")

# ----------------------------- Finale Pipeline ------------------------------
print("[INFO] Fit finales Modell (mit Kalibration auf Training CV=5)")
final_preprocess = preprocess  # reuse inkl. Imputation
final_base = LogisticRegression(class_weight='balanced', max_iter=1000)
final_pipe = Pipeline([
    ('preprocess', final_preprocess),
    ('clf', final_base)
])

final_calib = CalibratedClassifierCV(final_pipe, cv=5, method='isotonic')
final_calib.fit(X, y)

model_path = os.path.join(MODELS_DIR, 'dnf_pipeline_calibrated.pkl')
joblib.dump({
    'model': final_calib,
    'num_features': num_features,
    'cat_features': cat_features,
    'config': CFG
}, model_path)
print(f"[INFO] Modell gespeichert -> {model_path}")

# ----------------------------- Holdout Hinweis ------------------------------
if not holdout_df.empty:
    X_hold = holdout_df[num_features + cat_features]
    y_hold = holdout_df['is_dnf']
    hold_prob = final_calib.predict_proba(X_hold)[:,1]
    hold_metrics = {
        'holdout_rounds': sorted(holdout_df['round'].unique().tolist()),
        'hold_pr_auc': float(average_precision_score(y_hold, hold_prob)),
        'hold_brier': float(brier_score_loss(y_hold, hold_prob)),
        'hold_log_loss': float(log_loss(y_hold, hold_prob, labels=[0,1]))
    }
    with open(os.path.join(METRICS_DIR, 'holdout_report.json'), 'w') as f:
        json.dump(hold_metrics, f, indent=2)
    print(f"[INFO] Holdout Report gespeichert -> {os.path.join(METRICS_DIR, 'holdout_report.json')}")

print("[INFO] Fertig.")
