"""Pit Stop Prediction Model mit DNF-Integration

Vorhersage von Pit Stop Timing basierend auf:
1. Race conditions (Position, Reifen-Zustand, Wetter)
2. Strategic factors (Reifenstrategie, Degradation)  
3. DNF-Risiko (integriert DNF-Wahrscheinlichkeiten)
4. Historical patterns (Fahrer/Team-spezifische Tendenzen)

Ausführung:
    python -m src.pitpredict.models.pit_predict --train
    python -m src.pitpredict.models.pit_predict --predict --race_id 2024_21

Features:
- Lap-by-lap Vorhersage der Pit Stop Wahrscheinlichkeit
- Integration des vorhandenen DNF-Modells
- Berücksichtigung von Reifenstrategie und Track-Bedingungen
- Kalibrierte Wahrscheinlichkeiten für präzise Vorhersagen
"""
from __future__ import annotations

import os
import json
import joblib
import warnings
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------- Config und Konstanten ----------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
CONFIG_PATH = os.path.join(ROOT, 'config.yaml')

with open(CONFIG_PATH, 'r') as f:
    CFG = yaml.safe_load(f)

# Pfade aus Config
PROCESSED_TABLE = CFG['processed_table']
METRICS_DIR = CFG['metrics_dir']
MODELS_DIR = CFG['models_dir']
HOLDOUT_ROUNDS = set(CFG.get('holdout_rounds', []))
LAP_DATA_DIR = os.path.join(ROOT, 'data', f'season={CFG["season"]}')

# Erstelle Verzeichnisse
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

@dataclass
class PitStopPredictionConfig:
    """Konfiguration für Pit Stop Predictions"""
    min_laps_for_pitstop: int = 5  # Minimale Runden bevor Pit Stop möglich
    max_stint_length: int = 45     # Maximale Stint-Länge (Runden)
    tyre_life_threshold: int = 15  # Ab wann Reifenwechsel wahrscheinlicher wird
    position_change_weight: float = 0.3  # Gewichtung für Positionsänderungen
    dnf_risk_threshold: float = 0.2  # Schwelle für hohes DNF-Risiko
    
@dataclass  
class RaceState:
    """Aktueller Rennstatus für einen Fahrer"""
    driver: str
    lap_number: int
    position: int
    tyre_age: int
    compound: str
    last_pit_lap: int
    dnf_probability: float
    gap_to_leader: float
    position_trend: float  # Positive = improving, negative = losing positions

# ----------------------------- Daten laden und vorbereiten --------------------------
def load_lap_data(season: int, rounds: List[int]) -> pd.DataFrame:
    """Lade Lap-by-Lap Daten für mehrere Runden"""
    dfs = []
    
    for round_no in rounds:
        lap_file = os.path.join(LAP_DATA_DIR, f'round={round_no:02d}_laps.parquet')
        if os.path.exists(lap_file):
            print(f"[INFO] Lade {lap_file}")
            df = pd.read_parquet(lap_file)
            dfs.append(df)
        else:
            print(f"[WARNING] Datei nicht gefunden: {lap_file}")
    
    if not dfs:
        raise FileNotFoundError("Keine Lap-Daten gefunden")
        
    return pd.concat(dfs, ignore_index=True)

def create_pit_stop_target(lap_data: pd.DataFrame) -> pd.DataFrame:
    """Erstelle Target Variable: Wird in den nächsten 3 Runden ein Pit Stop gemacht?"""
    df = lap_data.copy().sort_values(['race_id', 'driver', 'lap_number'])
    
    # Identifiziere Pit Stops (Runden wo is_pit_out_lap == 1, also nach dem Pit)
    df['is_pitstop_lap'] = df['is_pit_out_lap']
    
    # Erstelle Target: Pit Stop in den nächsten 1-3 Runden
    df['pit_in_next_3_laps'] = 0
    
    for driver_group in df.groupby(['race_id', 'driver'], observed=True):
        driver_df = driver_group[1].copy()
        pit_laps = driver_df[driver_df['is_pitstop_lap'] == 1]['lap_number'].values
        
        for pit_lap in pit_laps:
            # Markiere 1-3 Runden vor dem Pit Stop als positive Samples
            lookback_mask = ((driver_df['lap_number'] >= pit_lap - 3) & 
                           (driver_df['lap_number'] < pit_lap))
            df.loc[driver_group[1].index[lookback_mask], 'pit_in_next_3_laps'] = 1
    
    return df

def engineer_pit_features(lap_data: pd.DataFrame) -> pd.DataFrame:
    """Erstelle Features für Pit Stop Prediction"""
    df = lap_data.copy().sort_values(['race_id', 'driver', 'lap_number'])
    
    # Stint-bezogene Features
    df['laps_since_pit'] = 0  # Runden seit letztem Pit Stop
    df['stint_number'] = 1    # Welcher Stint (1, 2, 3, ...)
    df['compound_age'] = df['tyre_life']  # Reifenalter
    
    # Fahrer-spezifische Berechnung
    for (race_id, driver), group in df.groupby(['race_id', 'driver'], observed=True):
        group = group.sort_values('lap_number')
        
        # Finde Pit Stops für diesen Fahrer
        pit_out_laps = group[group['is_pit_out_lap'] == 1]['lap_number'].values
        
        stint_num = 1
        last_pit = 0
        
        for idx, row in group.iterrows():
            current_lap = row['lap_number']
            
            # Aktualisiere Stint-Nummer wenn Pit Stop passiert ist
            if current_lap in pit_out_laps:
                stint_num += 1
                last_pit = current_lap
                
            df.loc[idx, 'laps_since_pit'] = current_lap - last_pit
            df.loc[idx, 'stint_number'] = stint_num
    
    # Strategische Features
    df['is_hard_compound'] = (df['compound'] == 'HARD').astype(int)
    df['is_medium_compound'] = (df['compound'] == 'MEDIUM').astype(int)
    df['is_soft_compound'] = (df['compound'] == 'SOFT').astype(int)
    
    # Relative Performance Features
    df['position_delta'] = 0  # Positionsänderung seit Stint-Start
    df['avg_laptime_stint'] = 0  # Durchschnittliche Rundenzeit im aktuellen Stint
    df['laptime_degradation'] = 0  # Rundenzeitverschlechterung
    
    for (race_id, driver), group in df.groupby(['race_id', 'driver'], observed=True):
        group = group.sort_values('lap_number')
        
        # Berechne Features pro Stint
        for stint in group['stint_number'].unique():
            stint_mask = (group['stint_number'] == stint)
            stint_laps = group[stint_mask]
            
            if len(stint_laps) > 1:
                start_pos = stint_laps['position'].iloc[0]
                stint_laptimes = stint_laps['lap_time_s'].dropna()
                
                for idx, row in stint_laps.iterrows():
                    # Positionsänderung seit Stint-Start
                    df.loc[idx, 'position_delta'] = row['position'] - start_pos
                    
                    # Durchschnittliche Laptime im Stint (bis jetzt)
                    current_stint_times = stint_laps[stint_laps['lap_number'] <= row['lap_number']]['lap_time_s'].dropna()
                    if len(current_stint_times) > 0:
                        df.loc[idx, 'avg_laptime_stint'] = float(current_stint_times.mean())
                    
                    # Rundenzeitverschlechterung (Trend)
                    if len(stint_laptimes) >= 3:
                        recent_times = stint_laps[stint_laps['lap_number'] <= row['lap_number']]['lap_time_s'].dropna()
                        if len(recent_times) >= 3:
                            # Lineare Regression für Trend
                            x = np.arange(len(recent_times))
                            if np.std(recent_times) > 0:
                                trend = np.polyfit(x, recent_times, 1)[0]  # Steigung
                                df.loc[idx, 'laptime_degradation'] = float(trend)
    
    # Race-State Features
    df['race_progress'] = df['lap_number'] / df['total_laps']  # Rennfortschritt (0-1)
    df['laps_remaining'] = df['total_laps'] - df['lap_number']
    
    # Strategische Überlegungen
    df['pit_window_optimal'] = ((df['laps_since_pit'] >= 15) & 
                               (df['laps_since_pit'] <= 25)).astype(int)
    df['pit_window_critical'] = (df['laps_since_pit'] >= 25).astype(int)
    df['late_race'] = (df['race_progress'] > 0.7).astype(int)
    
    # Wetter-Features falls verfügbar
    if 'air_temp' in df.columns:
        df['temp_change'] = df.groupby(['race_id'], observed=True)['air_temp'].diff().fillna(0)
        df['high_temp'] = (df['air_temp'] > df['air_temp'].median()).astype(int)
    
    return df

# ----------------------------- Model Definition ------------------------------------
class PitStopPredictor:
    """Haupt-Klasse für Pit Stop Prediction mit DNF-Integration"""
    
    def __init__(self, config: PitStopPredictionConfig = None):
        self.config = config or PitStopPredictionConfig()
        self.model = None
        self.feature_names = None
        self.dnf_transformer = None
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Bereite Features für das Training vor"""
        
        # Schließe irrelevante Spalten aus
        exclude_cols = {
            'season', 'round', 'race_id', 'circuit', 'driver', 'driver_number', 
            'team', 'lap_number', 'lap_time_s', 'sector1_time_s', 'sector2_time_s', 
            'sector3_time_s', 'is_pit_out_lap', 'is_pit_in_lap', 'is_pitstop_lap',
            'pit_in_next_3_laps', 'classification_status', 'dnf_reason', 
            'race_finished', 'total_laps'
        }
        
        # Lade DNF-Modell und berechne DNF-Risiko als zusätzliches Feature
        df_with_dnf = df.copy()
        try:
            dnf_model_path = os.path.join(MODELS_DIR, 'dnf_pipeline_calibrated.pkl')
            if os.path.exists(dnf_model_path):
                dnf_data = joblib.load(dnf_model_path)
                dnf_model = dnf_data['model']
                dnf_features = dnf_data['num_features'] + dnf_data['cat_features']
                
                # Prüfe verfügbare Features für DNF-Prediction
                available_dnf_features = [f for f in dnf_features if f in df.columns]
                if len(available_dnf_features) >= len(dnf_features) * 0.5:  # Mindestens 50%
                    X_dnf = df[available_dnf_features]
                    dnf_probs = dnf_model.predict_proba(X_dnf)[:, 1]
                    df_with_dnf['dnf_risk'] = dnf_probs
                    print(f"[INFO] DNF-Risiko berechnet: {dnf_probs.min():.3f} - {dnf_probs.max():.3f}")
                else:
                    df_with_dnf['dnf_risk'] = 0.1  # Default low risk
                    print("[INFO] DNF-Risiko: Standard-Wert verwendet (nicht genug Features)")
            else:
                df_with_dnf['dnf_risk'] = 0.1
                print("[INFO] DNF-Modell nicht gefunden - Standard DNF-Risiko verwendet")
        except Exception as e:
            print(f"[WARNING] DNF-Risiko konnte nicht berechnet werden: {e}")
            df_with_dnf['dnf_risk'] = 0.1
        
        # Numerische Features
        num_features = []
        for col in df_with_dnf.columns:
            if (col not in exclude_cols and 
                df_with_dnf[col].dtype in ['int64', 'float64', 'int32', 'float32', 'int8', 'float16']):
                num_features.append(col)
        
        # Kategorische Features
        cat_features = []
        for col in ['compound']:  # Entferne track_status, das ist numerisch
            if col in df_with_dnf.columns and col not in exclude_cols:
                cat_features.append(col)
                
        print(f"[INFO] Numerische Features ({len(num_features)}): {num_features[:10]}...")  
        print(f"[INFO] Kategorische Features ({len(cat_features)}): {cat_features}")
        
        feature_df = df_with_dnf[num_features + cat_features].copy()
        
        return feature_df, num_features + cat_features
        
    def build_pipeline(self, num_features: List[str], cat_features: List[str]) -> Pipeline:
        """Baue ML-Pipeline mit DNF-Integration"""
        
        # Preprocessing Pipeline
        preprocessor = ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_features)
        ])
        
        # Hauptmodell - Gradient Boosting für bessere Performance
        base_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        )
        
        # Pipeline zusammenbauen (ohne DNF-Transformer für bessere Kompatibilität)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', base_model)
        ])
        
        return pipeline
        
    def train(self, df: pd.DataFrame, target_col: str = 'pit_in_next_3_laps') -> Dict[str, Any]:
        """Trainiere das Pit Stop Prediction Modell"""
        
        # Bereite Features vor
        X, feature_names = self.prepare_features(df)
        y = df[target_col]
        groups = df['race_id']  # Für Group-basierte CV
        
        self.feature_names = feature_names
        
        # Split Features
        num_features = [f for f in feature_names if f not in ['compound']]
        cat_features = [f for f in feature_names if f in ['compound']]
        
        print(f"[INFO] Training mit {len(X)} Samples, {len(feature_names)} Features")
        print(f"[INFO] Target Distribution: {y.value_counts().to_dict()}")
        print(f"[INFO] Positive Rate: {y.mean():.3f}")
        
        # Pipeline bauen
        pipeline = self.build_pipeline(num_features, cat_features)
        
        # Cross-Validation mit Group-based Splits (um Data Leakage zu vermeiden)
        unique_races = df['race_id'].nunique()
        n_splits = max(2, min(5, unique_races))  # Mindestens 2, maximal 5
        
        if unique_races < 2:
            print("[WARNING] Nur ein Rennen verfügbar - verwende einfache CV statt GroupKFold")
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=3, shuffle=True, random_state=42)
            cv_groups = None
        else:
            cv = GroupKFold(n_splits=n_splits)
            cv_groups = groups
        
        fold_scores = []
        oof_predictions = np.zeros(len(X))
        
        print("[INFO] Cross-Validation läuft...")
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=cv_groups), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Training
            fold_pipeline = pipeline.fit(X_train, y_train)
            
            # Validation Predictions
            val_proba = fold_pipeline.predict_proba(X_val)[:, 1]
            oof_predictions[val_idx] = val_proba
            
            # Metriken
            pr_auc = average_precision_score(y_val, val_proba)
            roc_auc = roc_auc_score(y_val, val_proba) 
            brier = brier_score_loss(y_val, val_proba)
            
            fold_scores.append({
                'fold': fold,
                'pr_auc': float(pr_auc),
                'roc_auc': float(roc_auc), 
                'brier': float(brier),
                'n_val': len(val_idx)
            })
            
            print(f"[CV] Fold {fold}: PR-AUC={pr_auc:.3f}, ROC-AUC={roc_auc:.3f}, Brier={brier:.4f}")
        
        # Finale Metriken
        overall_pr_auc = average_precision_score(y, oof_predictions)
        overall_roc_auc = roc_auc_score(y, oof_predictions)
        overall_brier = brier_score_loss(y, oof_predictions)
        
        print(f"[FINAL] PR-AUC={overall_pr_auc:.3f}, ROC-AUC={overall_roc_auc:.3f}, Brier={overall_brier:.4f}")
        
        # Kalibriertes finales Modell trainieren
        print("[INFO] Trainiere kalibriertes finales Modell...")
        calibrated_model = CalibratedClassifierCV(pipeline, cv=3, method='isotonic')
        self.model = calibrated_model.fit(X, y)
        
        # Ergebnisse sammeln
        results = {
            'fold_scores': fold_scores,
            'overall_pr_auc': float(overall_pr_auc),
            'overall_roc_auc': float(overall_roc_auc),
            'overall_brier': float(overall_brier),
            'n_samples': len(X),
            'n_positive': int(y.sum()),
            'positive_rate': float(y.mean()),
            'feature_names': feature_names
        }
        
        return results
    
    def predict_pit_probability(self, race_data: pd.DataFrame) -> pd.DataFrame:
        """Vorhersage von Pit Stop Wahrscheinlichkeiten für Race Data"""
        
        if self.model is None:
            raise ValueError("Modell muss zuerst trainiert werden")
            
        # Features vorbereiten
        X, _ = self.prepare_features(race_data)
        
        # Prediction
        pit_probabilities = self.model.predict_proba(X)[:, 1]
        
        # Ergebnisse zusammenbauen
        results = race_data[['race_id', 'driver', 'lap_number', 'position', 
                           'laps_since_pit', 'compound']].copy()
        results['pit_probability'] = pit_probabilities
        
        return results
    
    def save_model(self, filepath: str, metadata: Dict[str, Any] = None):
        """Speichere das trainierte Modell"""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'config': self.config,
            'metadata': metadata or {}
        }
        
        joblib.dump(model_data, filepath)
        print(f"[INFO] Modell gespeichert: {filepath}")
    
    def load_model(self, filepath: str):
        """Lade ein gespeichertes Modell"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.config = model_data.get('config', PitStopPredictionConfig())
        
        print(f"[INFO] Modell geladen: {filepath}")
        return model_data.get('metadata', {})

# ----------------------------- Training Workflow ----------------------------------
def train_pit_stop_model():
    """Kompletter Training-Workflow"""
    
    print("[INFO] === Pit Stop Prediction Model Training ===")
    
    # Lade alle verfügbaren Lap-Daten
    all_rounds = [r for r in CFG['rounds'] if r not in HOLDOUT_ROUNDS]
    print(f"[INFO] Training Rounds: {all_rounds}")
    
    # Lade Lap-Daten
    lap_data = load_lap_data(CFG['season'], all_rounds)
    print(f"[INFO] Geladen: {len(lap_data)} Lap-Einträge")
    
    # Erstelle Pit Stop Targets
    lap_data = create_pit_stop_target(lap_data)
    print(f"[INFO] Pit Stop Labels erstellt")
    
    # Feature Engineering
    lap_data = engineer_pit_features(lap_data)
    print(f"[INFO] Features engineered: {lap_data.shape}")
    
    # Filtere ungültige Daten
    # Entferne erste paar Runden (zu früh für Pit Stops)
    valid_data = lap_data[lap_data['lap_number'] >= 5].copy()
    
    # Entferne DNF-Fahrer nach ihrem Ausfall
    valid_data = valid_data[valid_data['is_dnf'] == 0].copy()
    
    print(f"[INFO] Valide Daten nach Filterung: {len(valid_data)}")
    print(f"[INFO] Pit Stop Distribution: {valid_data['pit_in_next_3_laps'].value_counts()}")
    
    # Training
    predictor = PitStopPredictor()
    results = predictor.train(valid_data)
    
    # Speichere Modell
    model_path = os.path.join(MODELS_DIR, 'pitstop_predictor_calibrated.pkl')
    predictor.save_model(model_path, results)
    
    # Speichere Metriken
    metrics_path = os.path.join(METRICS_DIR, 'pitstop_cv_report.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[INFO] Training complete! Metriken: {metrics_path}")
    
    # Teste auf Holdout wenn verfügbar
    if HOLDOUT_ROUNDS:
        test_pit_stop_model(predictor)
    
    return predictor, results

def test_pit_stop_model(predictor: PitStopPredictor = None):
    """Teste das Modell auf Holdout-Daten"""
    
    if predictor is None:
        # Lade gespeichertes Modell
        model_path = os.path.join(MODELS_DIR, 'pitstop_predictor_calibrated.pkl')
        predictor = PitStopPredictor()
        predictor.load_model(model_path)
    
    print("[INFO] === Holdout Test ===")
    
    # Lade Holdout-Daten
    holdout_rounds = list(HOLDOUT_ROUNDS)
    print(f"[INFO] Holdout Rounds: {holdout_rounds}")
    
    holdout_data = load_lap_data(CFG['season'], holdout_rounds)
    holdout_data = create_pit_stop_target(holdout_data)
    holdout_data = engineer_pit_features(holdout_data)
    
    # Filtere wie beim Training
    valid_holdout = holdout_data[holdout_data['lap_number'] >= 5].copy()
    valid_holdout = valid_holdout[valid_holdout['is_dnf'] == 0].copy()
    
    print(f"[INFO] Holdout Samples: {len(valid_holdout)}")
    
    # Features und Target
    X_holdout, _ = predictor.prepare_features(valid_holdout)
    y_holdout = valid_holdout['pit_in_next_3_laps']
    
    # Predictions
    holdout_proba = predictor.model.predict_proba(X_holdout)[:, 1]
    
    # Metriken
    pr_auc = average_precision_score(y_holdout, holdout_proba)
    roc_auc = roc_auc_score(y_holdout, holdout_proba)
    brier = brier_score_loss(y_holdout, holdout_proba)
    
    holdout_results = {
        'holdout_rounds': holdout_rounds,
        'holdout_pr_auc': float(pr_auc),
        'holdout_roc_auc': float(roc_auc),
        'holdout_brier': float(brier),
        'n_holdout_samples': len(valid_holdout),
        'n_holdout_positive': int(y_holdout.sum()),
        'holdout_positive_rate': float(y_holdout.mean())
    }
    
    print(f"[HOLDOUT] PR-AUC={pr_auc:.3f}, ROC-AUC={roc_auc:.3f}, Brier={brier:.4f}")
    
    # Speichere Holdout-Ergebnisse  
    holdout_path = os.path.join(METRICS_DIR, 'pitstop_holdout_report.json')
    with open(holdout_path, 'w') as f:
        json.dump(holdout_results, f, indent=2)
        
    return holdout_results

# ----------------------------- Prediction Workflow --------------------------------
def predict_race_pitstops(race_id: str, up_to_lap: int = None):
    """Vorhersage für ein spezifisches Rennen"""
    
    print(f"[INFO] === Pit Stop Prediction für {race_id} ===")
    
    # Parse race_id
    try:
        season, round_no = race_id.split('_')
        season, round_no = int(season), int(round_no)
    except:
        raise ValueError(f"Ungültige race_id: {race_id}. Format: YYYY_RR")
    
    # Lade Modell
    model_path = os.path.join(MODELS_DIR, 'pitstop_predictor_calibrated.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modell nicht gefunden: {model_path}")
    
    predictor = PitStopPredictor()
    predictor.load_model(model_path)
    
    # Lade Race-Daten
    race_data = load_lap_data(season, [round_no])
    race_data = race_data[race_data['race_id'] == race_id].copy()
    
    if race_data.empty:
        raise ValueError(f"Keine Daten für {race_id}")
    
    # Filtere bis zu spezifischer Runde (für Live-Prediction)
    if up_to_lap:
        race_data = race_data[race_data['lap_number'] <= up_to_lap].copy()
    
    # Feature Engineering  
    race_data = engineer_pit_features(race_data)
    
    # Filtere valide Laps
    prediction_data = race_data[race_data['lap_number'] >= 5].copy()
    prediction_data = prediction_data[prediction_data['is_dnf'] == 0].copy()
    
    if prediction_data.empty:
        print("[WARNING] Keine validen Daten für Prediction")
        return pd.DataFrame()
    
    # Predictions
    results = predictor.predict_pit_probability(prediction_data)
    
    # Sortiere nach Wahrscheinlichkeit
    results = results.sort_values('pit_probability', ascending=False)
    
    print(f"[INFO] Predictions für {len(results)} Fahrer-Lap Kombinationen")
    print("\nTop 10 Pit Stop Kandidaten:")
    top_candidates = results.head(10)
    for _, row in top_candidates.iterrows():
        print(f"  {row['driver']} Lap {row['lap_number']}: {row['pit_probability']:.3f} "
              f"(Pos {row['position']}, {row['laps_since_pit']} laps on {row['compound']})")
    
    return results

# ----------------------------- CLI Interface ------------------------------------
def main():
    """Command Line Interface"""
    parser = argparse.ArgumentParser(description="Pit Stop Prediction Model mit DNF-Integration")
    
    parser.add_argument('--train', action='store_true',
                       help='Trainiere das Pit Stop Modell')
    parser.add_argument('--test', action='store_true', 
                       help='Teste auf Holdout-Daten')
    parser.add_argument('--predict', action='store_true',
                       help='Mache Predictions für ein Rennen')
    parser.add_argument('--race_id', type=str,
                       help='Race ID für Prediction (Format: YYYY_RR)')
    parser.add_argument('--up_to_lap', type=int,
                       help='Prediction nur bis zu dieser Runde (für Live-Prediction)')
    
    args = parser.parse_args()
    
    if args.train:
        train_pit_stop_model()
    
    elif args.test:
        test_pit_stop_model()
    
    elif args.predict:
        if not args.race_id:
            print("--race_id erforderlich für Prediction")
            return
        
        results = predict_race_pitstops(args.race_id, args.up_to_lap)
        
        # Optional: Speichere Ergebnisse
        output_path = os.path.join(METRICS_DIR, f'predictions_{args.race_id}.csv')
        if not results.empty:
            results.to_csv(output_path, index=False)
            print(f"[INFO] Predictions gespeichert: {output_path}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
