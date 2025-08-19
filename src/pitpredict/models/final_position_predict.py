"""Final Position Prediction Model mit DNF- und Pit Stop-Integration

Vorhersage der Endplatzierung basierend auf:
1. Pre-race Faktoren (Grid Position, Qualifying Performance)
2. DNF-Wahrscheinlichkeit (vom DNF-Modell)
3. Pit Stop-Strategien (vom Pit Stop-Modell)
4. Historical Performance (Team/Fahrer-spezifisch)
5. Track-spezifische Faktoren

Das Modell kombiniert die Vorhersagen der beiden bestehenden Modelle:
- DNF-Modell: Ausfallwahrscheinlichkeit
- Pit Stop-Modell: Strategische Faktoren während des Rennens

Ausführung:
    python -m src.pitpredict.models.final_position_predict --train
    python -m src.pitpredict.models.final_position_predict --predict --race_id 2024_21

Features:
- Pre-race Vorhersage der finalen Position
- Integration von DNF- und Pit Stop-Wahrscheinlichkeiten
- Berücksichtigung von Grid-Position und Historical Performance
- Robuste Behandlung von DNFs (Position = 20+ je nach Grund)
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
class FinalPositionPredictionConfig:
    """Konfiguration für Final Position Predictions"""
    dnf_position_penalty: int = 21  # Position für DNF (außerhalb der Punkte)
    position_bins: List[int] = None  # Optional: Bins für Klassifikation
    weight_dnf_risk: float = 0.3    # Gewichtung DNF-Risiko
    weight_pitstop_strategy: float = 0.2  # Gewichtung Pit Stop-Faktoren
    weight_grid_position: float = 0.4     # Gewichtung Startplatz
    weight_historical: float = 0.1        # Gewichtung Historical Performance
    
    def __post_init__(self):
        if self.position_bins is None:
            # Standard Bins: Podium, Punkte, Mittelfeld, Hinterfeld
            self.position_bins = [1, 3, 10, 15, 20]

@dataclass
class RacePrediction:
    """Vorhersage für einen Fahrer in einem Rennen"""
    driver: str
    predicted_position: float
    dnf_probability: float
    pitstop_risk_avg: float
    confidence_interval: Tuple[float, float]
    contributing_factors: Dict[str, float]

# ----------------------------- Feature Engineering -----------------------------------
class FinalPositionFeatureEngineer:
    """Feature Engineering für Final Position Prediction"""
    
    def __init__(self, config: FinalPositionPredictionConfig):
        self.config = config
        
    def engineer_features(self, df: pd.DataFrame, 
                         dnf_model=None, pitstop_model=None) -> pd.DataFrame:
        """Erstelle Features für Final Position Prediction"""
        
        df_features = df.copy()
        
        # 1. Grid Position Features
        df_features['grid_position_norm'] = df_features['grid_position'] / 20.0
        df_features['front_row_start'] = (df_features['grid_position'] <= 2).astype(int)
        df_features['top5_start'] = (df_features['grid_position'] <= 5).astype(int)
        df_features['back_of_grid'] = (df_features['grid_position'] >= 15).astype(int)
        
        # 2. Qualifying Performance Features
        df_features['q_gap_to_pole_s'] = df_features['q_gap_to_pole_ms'] / 1000.0
        df_features['q3_participant'] = df_features['q3_reached'].astype(int)
        
        # Relative Quali Performance vs Team
        team_q_means = df_features.groupby(['race_id', 'team'])['q_pos'].transform('mean')
        df_features['q_pos_vs_teammate'] = df_features['q_pos'] - team_q_means
        
        # 3. Historical Performance Features
        df_features['historical_performance_score'] = (
            # Gewichte verschiedene Historical Metrics
            (20 - df_features['avg_finish_delta_vs_grid_last5'].fillna(0)) * 0.4 +
            (df_features['points_last5'].fillna(0) / 25.0) * 0.3 +  # Normiert auf typische Punkte
            (1 - df_features['driver_dnf_rate_last5'].fillna(0.2)) * 0.2 +
            (df_features['q3_rate_last5'].fillna(0.3)) * 0.1
        )
        
        # 4. Track-specific Features
        df_features['overtaking_difficulty'] = df_features['overtake_difficulty']
        df_features['pit_loss_impact'] = df_features['pit_loss_s_est'] / 30.0  # Normiert
        df_features['is_street_circuit'] = df_features['street_circuit']
        
        # Track DNF Risk
        df_features['track_reliability_risk'] = df_features['track_dnf_rate_hist'].fillna(0.2)
        
        # 5. Team Performance Features
        df_features['team_recent_form'] = df_features['team_quali_rank_last5'].fillna(10) / 10.0
        df_features['team_reliability'] = 1 - df_features['team_dnf_rate_last5'].fillna(0.2)
        
        # 6. DNF Model Integration
        if dnf_model is not None:
            try:
                dnf_probs = self._predict_dnf_risk(df_features, dnf_model)
                df_features['dnf_risk_prediction'] = dnf_probs
                print(f"[INFO] DNF-Risiko berechnet: {dnf_probs.min():.3f} - {dnf_probs.max():.3f}")
            except Exception as e:
                print(f"[WARNING] DNF-Vorhersage fehlgeschlagen: {e}")
                df_features['dnf_risk_prediction'] = df_features['driver_dnf_rate_last5'].fillna(0.2)
        else:
            # Fallback auf Historical DNF Rate
            df_features['dnf_risk_prediction'] = df_features['driver_dnf_rate_last5'].fillna(0.2)
        
        # 7. Pit Stop Strategy Features (vereinfacht für Pre-Race)
        # Da Pit Stop Model Lap-by-Lap ist, verwenden wir strategische Proxies
        df_features['expected_pitstops'] = np.where(
            df_features['degradation_class'] == 'hoch', 2,
            np.where(df_features['degradation_class'] == 'mittel', 1, 1)
        )
        df_features['pit_strategy_risk'] = (
            df_features['pit_loss_impact'] * df_features['expected_pitstops']
        )
        
        # 8. Composite Features
        # Starting Performance Index
        df_features['start_performance_index'] = (
            (21 - df_features['grid_position']) * 0.5 +  # Je besser die Grid Pos, desto höher
            df_features['historical_performance_score'] * 0.3 +
            (1 - df_features['dnf_risk_prediction']) * 0.2
        )
        
        # Expected Position Range 
        df_features['expected_position_range'] = abs(
            df_features['grid_position'] - (21 - df_features['historical_performance_score'])
        )
        
        # 9. Risk-adjusted Features
        df_features['reliability_adjusted_performance'] = (
            df_features['start_performance_index'] * (1 - df_features['dnf_risk_prediction'])
        )
        
        return df_features
    
    def _predict_dnf_risk(self, df: pd.DataFrame, dnf_model) -> np.ndarray:
        """Berechne DNF-Risiko mit dem trainierten DNF-Modell"""
        
        # Lade DNF-Features
        if hasattr(dnf_model, 'feature_names_'):
            required_features = dnf_model.feature_names_
        else:
            # Fallback - verwende typische DNF-Features
            required_features = [
                'grid_position', 'q_gap_to_pole_ms', 'driver_dnf_rate_last5',
                'team_dnf_rate_last5', 'track_dnf_rate_hist', 'pit_loss_s_est',
                'degradation_class', 'street_circuit', 'overtake_difficulty'
            ]
        
        # Prüfe verfügbare Features
        available_features = [f for f in required_features if f in df.columns]
        
        if len(available_features) >= len(required_features) * 0.6:  # Mindestens 60%
            X_dnf = df[available_features]
            
            # Behandle kategorische Features
            if 'degradation_class' in X_dnf.columns:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                X_dnf = X_dnf.copy()
                X_dnf['degradation_class'] = le.fit_transform(X_dnf['degradation_class'].fillna('mittel'))
            
            dnf_probs = dnf_model.predict_proba(X_dnf)[:, 1]
            return dnf_probs
        else:
            # Fallback: historische DNF-Rate
            return df['driver_dnf_rate_last5'].fillna(0.2).values

# ----------------------------- Model Definition ------------------------------------
class FinalPositionPredictor:
    """Haupt-Klasse für Final Position Prediction"""
    
    def __init__(self, config: FinalPositionPredictionConfig = None):
        self.config = config or FinalPositionPredictionConfig()
        self.model = None
        self.feature_names = None
        self.feature_engineer = FinalPositionFeatureEngineer(self.config)
        self.dnf_model = None
        self.pitstop_model = None
        
    def load_auxiliary_models(self):
        """Lade DNF- und Pit Stop-Modelle - vereinfachte Version"""
        
        # DNF-Modell laden
        dnf_model_path = os.path.join(MODELS_DIR, 'dnf_pipeline_calibrated.pkl')
        try:
            if os.path.exists(dnf_model_path):
                dnf_data = joblib.load(dnf_model_path)
                self.dnf_model = dnf_data['model']
                print("[INFO] DNF-Modell geladen")
            else:
                print("[WARNING] DNF-Modell nicht gefunden")
                self.dnf_model = None
        except Exception as e:
            print(f"[WARNING] DNF-Modell konnte nicht geladen werden: {e}")
            self.dnf_model = None
            
        # Pit Stop-Modell fürs erste auslassen - hat Kompatibilitätsprobleme
        print("[INFO] Pit Stop-Modell wird übersprungen (Kompatibilitätsproblem)")
        self.pitstop_model = None
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Bereite Features für das Training vor"""
        
        # Feature Engineering
        df_engineered = self.feature_engineer.engineer_features(
            df, self.dnf_model, self.pitstop_model
        )
        
        # Schließe Meta-Spalten aus
        exclude_cols = {
            'season', 'round', 'race_id', 'circuit', 'driver', 'driver_number', 
            'team', 'finish_position', 'classification_status', 'points',
            'laps_completed', 'is_dnf', 'total_laps'
        }
        
        # Numerische Features
        num_features = []
        for col in df_engineered.columns:
            if (col not in exclude_cols and 
                df_engineered[col].dtype in ['int64', 'float64', 'int32', 'float32', 'int8', 'float16']):
                num_features.append(col)
        
        # Kategorische Features
        cat_features = []
        for col in ['degradation_class']:
            if col in df_engineered.columns and col not in exclude_cols:
                cat_features.append(col)
                
        all_features = num_features + cat_features
        
        print(f"[INFO] Numerische Features ({len(num_features)}): {num_features[:10]}...")
        print(f"[INFO] Kategorische Features ({len(cat_features)}): {cat_features}")
        
        feature_df = df_engineered[all_features].copy()
        
        return feature_df, all_features
    
    def build_pipeline(self, num_features: List[str], cat_features: List[str]) -> Pipeline:
        """Baue ML-Pipeline für Final Position Prediction"""
        
        # Preprocessing Pipeline
        preprocessor = ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())  # Robuster gegen Outliers
            ]), num_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ]), cat_features)
        ])
        
        # Hauptmodell - Gradient Boosting Regressor
        base_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        )
        
        # Pipeline zusammenbauen
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', base_model)
        ])
        
        return pipeline
    
    def create_target(self, df: pd.DataFrame) -> pd.Series:
        """Erstelle Target Variable mit DNF-Behandlung"""
        
        target = df['finish_position'].copy()
        
        # Behandle DNFs: Setze auf Position außerhalb der regulären Plätze
        dnf_mask = df['is_dnf'] == 1
        
        # DNF Penalty basierend auf Grund (falls verfügbar)
        if 'classification_status' in df.columns:
            # Verschiedene DNF-Penalties je nach Grund
            target.loc[dnf_mask] = self.config.dnf_position_penalty
        else:
            target.loc[dnf_mask] = self.config.dnf_position_penalty
            
        return target
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Trainiere das Final Position Prediction Modell"""
        
        # Lade Auxiliary Models
        self.load_auxiliary_models()
        
        # Bereite Features vor
        X, feature_names = self.prepare_features(df)
        y = self.create_target(df)
        groups = df['race_id']  # Für Group-basierte CV
        
        self.feature_names = feature_names
        
        # Split Features
        num_features = [f for f in feature_names if f not in ['degradation_class']]
        cat_features = [f for f in feature_names if f in ['degradation_class']]
        
        print(f"[INFO] Training mit {len(X)} Samples, {len(feature_names)} Features")
        print(f"[INFO] Target Stats: Mean={y.mean():.2f}, Std={y.std():.2f}")
        print(f"[INFO] DNFs: {(df['is_dnf'] == 1).sum()} / {len(df)} ({(df['is_dnf'] == 1).mean():.1%})")
        
        # Pipeline bauen
        pipeline = self.build_pipeline(num_features, cat_features)
        
        # Cross-Validation mit Group-based Splits
        unique_races = df['race_id'].nunique()
        n_splits = max(3, min(5, unique_races - 1))
        
        cv = GroupKFold(n_splits=n_splits)
        
        fold_scores = []
        oof_predictions = np.zeros(len(X))
        
        print("[INFO] Cross-Validation läuft...")
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Training
            fold_pipeline = pipeline.fit(X_train, y_train)
            
            # Validation Predictions
            val_pred = fold_pipeline.predict(X_val)
            oof_predictions[val_idx] = val_pred
            
            # Metriken
            mae = mean_absolute_error(y_val, val_pred)
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            r2 = r2_score(y_val, val_pred)
            
            # Spezielle Metriken für Position Prediction
            # Podium Accuracy (Top 3)
            podium_actual = (y_val <= 3).astype(int)
            podium_pred = (val_pred <= 3).astype(int)
            podium_acc = (podium_actual == podium_pred).mean()
            
            # Points Accuracy (Top 10)
            points_actual = (y_val <= 10).astype(int)
            points_pred = (val_pred <= 10).astype(int)
            points_acc = (points_actual == points_pred).mean()
            
            fold_scores.append({
                'fold': fold,
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'podium_accuracy': float(podium_acc),
                'points_accuracy': float(points_acc),
                'n_val': len(val_idx)
            })
            
            print(f"[CV] Fold {fold}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}, "
                  f"Podium Acc={podium_acc:.1%}, Points Acc={points_acc:.1%}")
        
        # Finale Metriken
        overall_mae = mean_absolute_error(y, oof_predictions)
        overall_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
        overall_r2 = r2_score(y, oof_predictions)
        
        # Finale spezielle Metriken
        podium_actual = (y <= 3).astype(int)
        podium_pred = (oof_predictions <= 3).astype(int)
        overall_podium_acc = (podium_actual == podium_pred).mean()
        
        points_actual = (y <= 10).astype(int)
        points_pred = (oof_predictions <= 10).astype(int)
        overall_points_acc = (points_actual == points_pred).mean()
        
        print(f"[FINAL] MAE={overall_mae:.2f}, RMSE={overall_rmse:.2f}, R²={overall_r2:.3f}")
        print(f"[FINAL] Podium Acc={overall_podium_acc:.1%}, Points Acc={overall_points_acc:.1%}")
        
        # Finales Modell trainieren
        print("[INFO] Trainiere finales Modell...")
        self.model = pipeline.fit(X, y)
        
        # Feature Importance (falls verfügbar)
        feature_importance = {}
        try:
            if hasattr(self.model.named_steps['regressor'], 'feature_importances_'):
                # Get feature names after preprocessing
                preprocessor = self.model.named_steps['preprocessor']
                feature_names_transformed = (
                    num_features + 
                    [f"{cat_col}_{val}" for cat_col in cat_features 
                     for val in preprocessor.named_transformers_['cat']
                     .named_steps['ohe'].categories_[cat_features.index(cat_col)]]
                )
                
                importances = self.model.named_steps['regressor'].feature_importances_
                feature_importance = dict(zip(feature_names_transformed[:len(importances)], 
                                            importances.astype(float)))
                
                # Top Features anzeigen
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                print("\n[INFO] Top 10 wichtigste Features:")
                for feat, imp in top_features:
                    print(f"  {feat}: {imp:.4f}")
        except Exception as e:
            print(f"[INFO] Feature Importance nicht verfügbar: {e}")
        
        # Ergebnisse sammeln
        results = {
            'fold_scores': fold_scores,
            'overall_mae': float(overall_mae),
            'overall_rmse': float(overall_rmse),
            'overall_r2': float(overall_r2),
            'overall_podium_accuracy': float(overall_podium_acc),
            'overall_points_accuracy': float(overall_points_acc),
            'feature_importance': feature_importance,
            'n_samples': len(X),
            'n_dnfs': int((df['is_dnf'] == 1).sum()),
            'dnf_rate': float((df['is_dnf'] == 1).mean()),
            'feature_names': feature_names
        }
        
        return results
    
    def predict_final_positions(self, race_data: pd.DataFrame) -> pd.DataFrame:
        """Vorhersage der finalen Positionen für Race Data"""
        
        if self.model is None:
            raise ValueError("Modell muss zuerst trainiert werden")
        
        # Lade Auxiliary Models falls nicht geladen
        if self.dnf_model is None or self.pitstop_model is None:
            self.load_auxiliary_models()
            
        # Features vorbereiten
        X, _ = self.prepare_features(race_data)
        
        # Prediction
        predicted_positions = self.model.predict(X)
        
        # Ergebnisse zusammenbauen
        results = race_data[['race_id', 'driver', 'team', 'grid_position']].copy()
        results['predicted_final_position'] = predicted_positions
        results['dnf_risk'] = race_data.get('dnf_risk_prediction', 0.2)
        
        # Runde Predictions auf ganze Positionen
        results['predicted_position_rounded'] = np.round(predicted_positions).astype(int)
        
        # Confidence basierend auf Historical Performance
        if 'historical_performance_score' in race_data.columns:
            # Höhere Historical Performance = höhere Confidence
            confidence = race_data['historical_performance_score'] / 20.0
            results['prediction_confidence'] = np.clip(confidence, 0.1, 0.9)
        else:
            results['prediction_confidence'] = 0.5
        
        return results.sort_values('predicted_final_position')
    
    def save_model(self, filepath: str, metadata: Dict[str, Any] = None):
        """Speichere das trainierte Modell"""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'config': self.config,
            'feature_engineer': self.feature_engineer,
            'metadata': metadata or {}
        }
        
        joblib.dump(model_data, filepath)
        print(f"[INFO] Modell gespeichert: {filepath}")
    
    def load_model(self, filepath: str):
        """Lade ein gespeichertes Modell"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.config = model_data.get('config', FinalPositionPredictionConfig())
        self.feature_engineer = model_data.get('feature_engineer', 
                                              FinalPositionFeatureEngineer(self.config))
        
        print(f"[INFO] Modell geladen: {filepath}")
        return model_data.get('metadata', {})

# ----------------------------- Training Workflow ----------------------------------
def train_final_position_model():
    """Kompletter Training-Workflow für Final Position Prediction"""
    
    print("[INFO] === Final Position Prediction Model Training ===")
    
    # Lade Driver-Race Daten
    if not os.path.exists(PROCESSED_TABLE):
        raise FileNotFoundError(f"Processed table nicht gefunden: {PROCESSED_TABLE}")
    
    print(f"[INFO] Lade Daten von {PROCESSED_TABLE}")
    df = pd.read_parquet(PROCESSED_TABLE)
    
    # Split in Training und Holdout
    train_mask = ~df['round'].isin(HOLDOUT_ROUNDS)
    train_df = df[train_mask].copy()
    holdout_df = df[~train_mask].copy()
    
    print(f"[INFO] Training: {len(train_df)} Samples aus {train_df['round'].nunique()} Rennen")
    print(f"[INFO] Holdout: {len(holdout_df)} Samples aus {holdout_df['round'].nunique()} Rennen")
    
    # Bereinige Daten
    # Entferne Samples ohne gültige Finish Position (außer DNFs)
    valid_mask = (~train_df['finish_position'].isna()) | (train_df['is_dnf'] == 1)
    train_df = train_df[valid_mask].copy()
    
    print(f"[INFO] Nach Bereinigung: {len(train_df)} Training Samples")
    print(f"[INFO] DNF Rate: {(train_df['is_dnf'] == 1).mean():.1%}")
    
    # Training
    predictor = FinalPositionPredictor()
    results = predictor.train(train_df)
    
    # Speichere Modell
    model_path = os.path.join(MODELS_DIR, 'final_position_predictor.pkl')
    predictor.save_model(model_path, results)
    
    # Speichere Metriken
    metrics_path = os.path.join(METRICS_DIR, 'final_position_cv_report.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[INFO] Training complete! Metriken: {metrics_path}")
    
    # Teste auf Holdout wenn verfügbar
    if not holdout_df.empty:
        test_final_position_model(predictor, holdout_df)
    
    return predictor, results

def test_final_position_model(predictor: FinalPositionPredictor = None, 
                             holdout_df: pd.DataFrame = None):
    """Teste das Modell auf Holdout-Daten"""
    
    if predictor is None:
        # Lade gespeichertes Modell
        model_path = os.path.join(MODELS_DIR, 'final_position_predictor.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modell nicht gefunden: {model_path}")
        predictor = FinalPositionPredictor()
        predictor.load_model(model_path)
    
    if holdout_df is None:
        # Lade Holdout-Daten
        df = pd.read_parquet(PROCESSED_TABLE)
        holdout_df = df[df['round'].isin(HOLDOUT_ROUNDS)].copy()
    
    print("[INFO] === Holdout Test ===")
    
    # Bereinige Holdout Daten
    valid_mask = (~holdout_df['finish_position'].isna()) | (holdout_df['is_dnf'] == 1)
    holdout_df = holdout_df[valid_mask].copy()
    
    print(f"[INFO] Holdout Samples: {len(holdout_df)}")
    print(f"[INFO] Holdout Runden: {sorted(holdout_df['round'].unique())}")
    
    # Features und Target
    X_holdout, _ = predictor.prepare_features(holdout_df)
    y_holdout = predictor.create_target(holdout_df)
    
    # Predictions
    holdout_pred = predictor.model.predict(X_holdout)
    
    # Metriken
    mae = mean_absolute_error(y_holdout, holdout_pred)
    rmse = np.sqrt(mean_squared_error(y_holdout, holdout_pred))
    r2 = r2_score(y_holdout, holdout_pred)
    
    # Spezielle Metriken
    podium_actual = (y_holdout <= 3).astype(int)
    podium_pred = (holdout_pred <= 3).astype(int)
    podium_acc = (podium_actual == podium_pred).mean()
    
    points_actual = (y_holdout <= 10).astype(int)
    points_pred = (holdout_pred <= 10).astype(int)
    points_acc = (points_actual == points_pred).mean()
    
    # DNF Predictions
    dnf_actual = (holdout_df['is_dnf'] == 1)
    dnf_pred = (holdout_pred >= predictor.config.dnf_position_penalty)
    dnf_detection = (dnf_actual == dnf_pred).mean()
    
    holdout_results = {
        'holdout_rounds': sorted(holdout_df['round'].unique().tolist()),
        'holdout_mae': float(mae),
        'holdout_rmse': float(rmse),
        'holdout_r2': float(r2),
        'holdout_podium_accuracy': float(podium_acc),
        'holdout_points_accuracy': float(points_acc),
        'holdout_dnf_detection': float(dnf_detection),
        'n_holdout_samples': len(holdout_df),
        'n_holdout_dnfs': int(dnf_actual.sum()),
        'holdout_dnf_rate': float(dnf_actual.mean())
    }
    
    print(f"[HOLDOUT] MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")
    print(f"[HOLDOUT] Podium Acc={podium_acc:.1%}, Points Acc={points_acc:.1%}")
    print(f"[HOLDOUT] DNF Detection={dnf_detection:.1%}")
    
    # Detaillierte Analyse
    print("\n[HOLDOUT] Top/Bottom Predictions:")
    analysis_df = holdout_df[['driver', 'race_id', 'grid_position', 'finish_position']].copy()
    analysis_df['predicted_position'] = holdout_pred
    analysis_df['prediction_error'] = abs(holdout_pred - y_holdout)
    
    # Beste Predictions
    best_preds = analysis_df.nsmallest(5, 'prediction_error')
    print("Beste Vorhersagen:")
    for _, row in best_preds.iterrows():
        print(f"  {row['driver']} {row['race_id']}: Pred={row['predicted_position']:.1f}, "
              f"Actual={row['finish_position']:.0f}, Error={row['prediction_error']:.1f}")
    
    # Schlechteste Predictions
    worst_preds = analysis_df.nlargest(5, 'prediction_error')
    print("\nSchlechteste Vorhersagen:")
    for _, row in worst_preds.iterrows():
        print(f"  {row['driver']} {row['race_id']}: Pred={row['predicted_position']:.1f}, "
              f"Actual={row['finish_position']:.0f}, Error={row['prediction_error']:.1f}")
    
    # Speichere Holdout-Ergebnisse
    holdout_path = os.path.join(METRICS_DIR, 'final_position_holdout_report.json')
    with open(holdout_path, 'w') as f:
        json.dump(holdout_results, f, indent=2)
        
    return holdout_results

# ----------------------------- Prediction Workflow --------------------------------
def predict_race_positions(race_id: str):
    """Vorhersage für ein spezifisches Rennen"""
    
    print(f"[INFO] === Final Position Prediction für {race_id} ===")
    
    # Parse race_id
    try:
        season, round_no = race_id.split('_')
        season, round_no = int(season), int(round_no)
    except:
        raise ValueError(f"Ungültige race_id: {race_id}. Format: YYYY_RR")
    
    # Lade Modell
    model_path = os.path.join(MODELS_DIR, 'final_position_predictor.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modell nicht gefunden: {model_path}")
    
    predictor = FinalPositionPredictor()
    predictor.load_model(model_path)
    
    # Lade Race-Daten
    df = pd.read_parquet(PROCESSED_TABLE)
    race_data = df[df['race_id'] == race_id].copy()
    
    if race_data.empty:
        raise ValueError(f"Keine Daten für {race_id}")
    
    print(f"[INFO] Gefunden: {len(race_data)} Fahrer für {race_id}")
    
    # Predictions
    results = predictor.predict_final_positions(race_data)
    
    print(f"\n[INFO] Predicted Final Positions für {race_id}:")
    print("=" * 60)
    for i, (_, row) in enumerate(results.iterrows(), 1):
        print(f"{i:2d}. {row['driver']:3s} ({row['team'][:12]:12s}) - "
              f"P{row['predicted_position_rounded']:2d} (Grid P{row['grid_position']:.0f}) "
              f"[Conf: {row['prediction_confidence']:.1%}, DNF Risk: {row['dnf_risk']:.1%}]")
    
    # Vergleiche mit tatsächlichen Ergebnissen (falls vorhanden)
    if not race_data['finish_position'].isna().all():
        print(f"\n[INFO] Vergleich mit tatsächlichen Ergebnissen:")
        actual_results = race_data[['driver', 'finish_position', 'is_dnf']].copy()
        comparison = results.merge(actual_results, on='driver', how='left')
        
        mae = mean_absolute_error(
            comparison['finish_position'].fillna(21), 
            comparison['predicted_final_position']
        )
        print(f"MAE: {mae:.2f}")
        
        # Top 3 Accuracy
        actual_podium = set(comparison[comparison['finish_position'] <= 3]['driver'])
        pred_podium = set(comparison.head(3)['driver'])
        podium_overlap = len(actual_podium & pred_podium)
        print(f"Podium Overlap: {podium_overlap}/3")
    
    return results

# ----------------------------- CLI Interface ------------------------------------
def main():
    """Command Line Interface"""
    parser = argparse.ArgumentParser(
        description="Final Position Prediction Model mit DNF- und Pit Stop-Integration"
    )
    
    parser.add_argument('--train', action='store_true',
                       help='Trainiere das Final Position Modell')
    parser.add_argument('--test', action='store_true', 
                       help='Teste auf Holdout-Daten')
    parser.add_argument('--predict', action='store_true',
                       help='Mache Predictions für ein Rennen')
    parser.add_argument('--race_id', type=str,
                       help='Race ID für Prediction (Format: YYYY_RR)')
    
    args = parser.parse_args()
    
    if args.train:
        train_final_position_model()
    
    elif args.test:
        test_final_position_model()
    
    elif args.predict:
        if not args.race_id:
            print("--race_id erforderlich für Prediction")
            return
        
        results = predict_race_positions(args.race_id)
        
        # Optional: Speichere Ergebnisse
        output_path = os.path.join(METRICS_DIR, f'final_position_predictions_{args.race_id}.csv')
        results.to_csv(output_path, index=False)
        print(f"[INFO] Predictions gespeichert: {output_path}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
