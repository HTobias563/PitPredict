"""Future Race Prediction für 2025+ Rennen

Erweitert das bestehende Final Position Model um die Fähigkeit,
zukünftige Rennen vorherzusagen basierend auf:
- Aktuellsten verfügbaren Daten (2024)
- Extrapolierten Historical Performance Metriken  
- Track-spezifischen Charakteristika
- Angenommenen Team-Transfers

Integration in bestehende PitPredict-Architektur.
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import yaml

# Import der bestehenden Final Position Komponenten
from .final_position_predict import (
    FinalPositionPredictor, 
    FinalPositionPredictionConfig
)

# Config laden
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
CONFIG_PATH = os.path.join(ROOT, 'config.yaml')

with open(CONFIG_PATH, 'r') as f:
    CFG = yaml.safe_load(f)

class FutureRacePredictionConfig(FinalPositionPredictionConfig):
    """Erweiterte Konfiguration für Future Race Predictions"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Future-spezifische Parameter
        self.future_season: int = 2025
        self.baseline_season: int = 2024
        self.team_transfers: Dict[str, str] = {
            'HAM': 'Ferrari',  # Lewis zu Ferrari 2025
            'SAI': 'Williams'  # Sainz zu Williams (hypothetisch)
        }
        
        # Override für erweiterte Konfiguration
        for key, value in kwargs.items():
            setattr(self, key, value)

class FutureRacePredictor(FinalPositionPredictor):
    """Erweiterte Final Position Prediction für zukünftige Rennen"""
    
    def __init__(self, config: FutureRacePredictionConfig = None):
        # Basis Final Position Predictor initialisieren
        base_config = FinalPositionPredictionConfig() if config is None else config
        super().__init__(base_config)
        
        # Future-spezifische Konfiguration
        self.future_config = config or FutureRacePredictionConfig()
        self.baseline_data = None
        self.track_profiles = self._initialize_track_profiles()
        
    def _initialize_track_profiles(self) -> Dict[str, Dict]:
        """Track-Profile für verschiedene Streckentypen"""
        return {
            'netherlands': {
                'circuit': 'Zandvoort',
                'street_circuit': 0,
                'overtake_difficulty': 0.7,  # Schwer zu überholen
                'pit_loss_s_est': 23.5,
                'degradation_class': 'hoch',
                'track_dnf_rate_hist': 0.15,
                'round_type': 'traditional'
            },
            'monaco': {
                'circuit': 'Monaco',
                'street_circuit': 1,
                'overtake_difficulty': 0.9,  # Sehr schwer
                'pit_loss_s_est': 22.0,
                'degradation_class': 'niedrig',
                'track_dnf_rate_hist': 0.25,
                'round_type': 'street'
            },
            'spa': {
                'circuit': 'Spa-Francorchamps',
                'street_circuit': 0,
                'overtake_difficulty': 0.3,  # Leicht zu überholen
                'pit_loss_s_est': 24.0,
                'degradation_class': 'mittel',
                'track_dnf_rate_hist': 0.20,
                'round_type': 'traditional'
            },
            'silverstone': {
                'circuit': 'Silverstone',
                'street_circuit': 0,
                'overtake_difficulty': 0.4,
                'pit_loss_s_est': 22.5,
                'degradation_class': 'hoch',
                'track_dnf_rate_hist': 0.12,
                'round_type': 'traditional'
            },
            'default': {
                'circuit': 'Generic Circuit',
                'street_circuit': 0,
                'overtake_difficulty': 0.5,
                'pit_loss_s_est': 23.0,
                'degradation_class': 'mittel',
                'track_dnf_rate_hist': 0.18,
                'round_type': 'traditional'
            }
        }
    
    def load_baseline_data(self, season: int = None) -> pd.DataFrame:
        """Lade Baseline-Daten für Future Predictions"""
        
        if season is None:
            season = self.future_config.baseline_season
            
        print(f"📊 Lade Baseline-Daten für Saison {season}...")
        
        # Verwende die bestehende Konfiguration
        data_path = CFG['processed_table'].replace('2024', str(season))
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Baseline-Daten nicht gefunden: {data_path}")
            
        self.baseline_data = pd.read_parquet(data_path)
        
        # Lade das trainierte Modell falls noch nicht geladen
        if self.model is None:
            model_path = os.path.join(CFG['models_dir'], 'final_position_predictor.pkl')
            if os.path.exists(model_path):
                self.load_model(model_path)
                print(f"✅ Modell geladen: {model_path}")
            else:
                raise FileNotFoundError(f"Trainiertes Modell nicht gefunden: {model_path}")
        
        return self.baseline_data
    
    def extract_latest_driver_stats(self) -> pd.DataFrame:
        """Extrahiere aktuelle Fahrer-Statistiken für Future Predictions"""
        
        if self.baseline_data is None:
            self.load_baseline_data()
        
        # Nehme die letzten 5 Rennen als "current form"
        latest_races = self.baseline_data['round'].nlargest(5).unique()
        recent_data = self.baseline_data[self.baseline_data['round'].isin(latest_races)]
        
        # Aggregiere Fahrer-Performance
        driver_stats = recent_data.groupby('driver').agg({
            'team': 'last',  # Aktuelles Team
            'finish_position': 'mean',
            'grid_position': 'mean',
            'points': 'sum',
            'is_dnf': 'mean',
            'q_gap_to_pole_ms': 'mean',
            'q_pos': 'mean',
            'Q1_ms': 'mean',
            'Q2_ms': 'mean',
            'Q3_ms': 'mean',
            'q_best_time_ms': 'mean',
            # Historical metrics
            'avg_finish_delta_vs_grid_last5': 'mean',
            'driver_dnf_rate_last5': 'mean',
            'points_last5': 'mean'
        }).round(2)
        
        # Wende Team-Transfers für Future Season an
        for driver, new_team in self.future_config.team_transfers.items():
            if driver in driver_stats.index:
                old_team = driver_stats.loc[driver, 'team']
                driver_stats.loc[driver, 'team'] = new_team
                print(f"🔄 Transfer Update: {driver} {old_team} → {new_team}")
        
        return driver_stats
    
    def create_synthetic_race_data(self,
                                  race_name: str,
                                  grid_positions: Dict[str, int],
                                  track_type: str = 'default',
                                  season: int = None,
                                  round_num: int = 15) -> pd.DataFrame:
        """Erstelle synthetische Race-Daten für Future Race"""
        
        if season is None:
            season = self.future_config.future_season
            
        print(f"🏗️  Erstelle Race-Daten für {race_name} ({season})")
        
        # Basis-Fahrer-Stats von aktueller Saison
        driver_stats = self.extract_latest_driver_stats()
        
        # Track-Profil
        track_profile = self.track_profiles.get(track_type, self.track_profiles['default'])
        
        # Race-Daten erstellen
        race_entries = []
        
        for driver, grid_pos in grid_positions.items():
            if driver not in driver_stats.index:
                print(f"⚠️  Unbekannter Fahrer {driver} - überspringe")
                continue
            
            base_stats = driver_stats.loc[driver]
            
            # Synthetische Race-Entry
            race_entry = {
                # Race Identifikation
                'season': season,
                'round': round_num,
                'race_id': f'{season}_{round_num:02d}',
                'circuit': track_profile['circuit'],
                'driver': driver,
                'driver_number': self._get_driver_number(driver),
                'team': base_stats['team'],
                'grid_position': grid_pos,
                
                # Qualifying-Daten (extrapoliert von Grid Position)
                'q_pos': grid_pos,
                'Q1_ms': base_stats['Q1_ms'] + (grid_pos - base_stats['q_pos']) * 100,
                'Q2_ms': base_stats['Q2_ms'] + (grid_pos - base_stats['q_pos']) * 100,
                'Q3_ms': base_stats['Q3_ms'] + (grid_pos - base_stats['q_pos']) * 100 if grid_pos <= 10 else np.nan,
                'q_best_time_ms': base_stats['q_best_time_ms'] + (grid_pos - base_stats['q_pos']) * 100,
                'q_gap_to_pole_ms': (grid_pos - 1) * 200,  # ~200ms pro Position
                'q3_reached': 1 if grid_pos <= 10 else 0,
                
                # Historical Performance (aus Baseline extrapoliert)
                'points_last5': base_stats['points_last5'] * 0.9,  # Slight decay für neue Saison
                'driver_dnf_rate_last5': base_stats['driver_dnf_rate_last5'],
                'avg_finish_delta_vs_grid_last5': base_stats['avg_finish_delta_vs_grid_last5'],
                'finish_delta_vs_grid': base_stats['avg_finish_delta_vs_grid_last5'],
                
                # Track-spezifische Features
                **track_profile,
                
                # Team-Performance
                'team_quali_rank_last5': self._estimate_team_quali_rank(base_stats['team']),
                'team_dnf_rate_last5': base_stats['driver_dnf_rate_last5'] * 1.1,
                
                # Zusätzliche Required Features für Model Compatibility
                'q_pos_mean_last5': base_stats['q_pos'],
                'q_stdev_last5': 2.0,
                'q3_rate_last5': 0.6 if base_stats['q_pos'] <= 10 else 0.2,
                'q_best_gap_to_pole_ms_last5': base_stats['q_gap_to_pole_ms'],
                
                # Placeholder für Target-Features (werden nicht verwendet für Prediction)
                'finish_position': np.nan,
                'classification_status': None,
                'points': 0,
                'laps_completed': 0,
                'is_dnf': 0,
                'total_laps': 58
            }
            
            race_entries.append(race_entry)
        
        race_df = pd.DataFrame(race_entries)
        print(f"✅ Race-Daten erstellt: {len(race_df)} Fahrer für {race_name}")
        
        return race_df
    
    def _get_driver_number(self, driver_code: str) -> int:
        """Driver Numbers (2024/2025 Saison)"""
        numbers = {
            'VER': 1, 'SAR': 2, 'RIC': 3, 'NOR': 4, 'VET': 5, 'LAT': 6, 'RAI': 7,
            'PIA': 8, 'STR': 18, 'GAS': 10, 'PER': 11, 'ALO': 14, 'LEC': 16,
            'MAG': 20, 'TSU': 22, 'ALB': 23, 'ZHO': 24, 'HUL': 27, 'OCO': 31,
            'RUS': 63, 'HAM': 44, 'BOT': 77, 'SAI': 55, 'LAW': 30, 'COL': 43,
            'BEA': 50, 'DOO': 99
        }
        return numbers.get(driver_code, 99)
    
    def _estimate_team_quali_rank(self, team: str) -> float:
        """Schätze Team-Qualifying-Rank für Future Season"""
        team_ranks_2025 = {
            'Red Bull Racing': 1.5,
            'McLaren': 2.0,
            'Ferrari': 2.5,  # Mit Hamilton-Transfer stärker
            'Mercedes': 4.0,
            'Aston Martin': 6.0,
            'Alpine': 7.0,
            'Williams': 7.5,  # Mit Sainz-Transfer besser
            'RB': 8.0,
            'Haas F1 Team': 8.5,
            'Kick Sauber': 9.5
        }
        return team_ranks_2025.get(team, 7.0)
    
    def predict_future_race(self,
                           race_name: str,
                           grid_positions: Dict[str, int],
                           track_type: str = 'default',
                           season: int = None) -> pd.DataFrame:
        """Hauptmethode für Future Race Prediction"""
        
        if season is None:
            season = self.future_config.future_season
            
        print(f"🚀 FUTURE RACE PREDICTION: {race_name}")
        print("=" * 60)
        
        # 1. Lade Baseline-Daten und trainiertes Modell
        self.load_baseline_data()
        
        # 2. Erstelle synthetische Race-Daten
        future_race_data = self.create_synthetic_race_data(
            race_name, grid_positions, track_type, season
        )
        
        # 3. Nutze bestehende predict_final_positions Methode
        print(f"🎯 Führe Vorhersage durch...")
        results = self.predict_final_positions(future_race_data)
        
        # 4. Erweiterte Analyse für Future Races
        self._analyze_future_results(results, race_name, track_type)
        
        return results
    
    def _analyze_future_results(self, results: pd.DataFrame, race_name: str, track_type: str):
        """Zusätzliche Analyse für Future Race Results"""
        
        print(f"\n🏆 PODIUM VORHERSAGE für {race_name}:")
        print("=" * 50)
        
        podium = results.head(3)
        for i, (_, row) in enumerate(podium.iterrows(), 1):
            print(f"{i}. {row['driver']} ({row['team'][:15]:<15}) - "
                  f"P{row['predicted_position_rounded']:2d} "
                  f"(Grid P{row['grid_position']:.0f}) "
                  f"[DNF Risk: {row['dnf_risk']:.1%}]")
        
        print(f"\n📊 VOLLSTÄNDIGE VORHERSAGE:")
        print("-" * 70)
        
        for i, (_, row) in enumerate(results.iterrows(), 1):
            status = "🏆" if i <= 3 else "🏁" if i <= 10 else "📍"
            print(f"{status} {i:2d}. {row['driver']:3s} ({row['team'][:12]:12s}) - "
                  f"P{row['predicted_position_rounded']:2d} "
                  f"(Grid P{row['grid_position']:.0f}) "
                  f"[Conf: {row['prediction_confidence']:.1%}]")
        
        # Zusätzliche Future-spezifische Insights
        print(f"\n💡 FUTURE RACE INSIGHTS:")
        print("-" * 30)
        
        # Position Changes Analysis
        results['position_change'] = results['grid_position'] - results['predicted_position_rounded']
        
        if results['position_change'].max() > 0:
            top_gainer = results.loc[results['position_change'].idxmax()]
            print(f"🔺 Größter Gewinner: {top_gainer['driver']} "
                  f"(P{top_gainer['grid_position']:.0f} → P{top_gainer['predicted_position_rounded']}, "
                  f"+{top_gainer['position_change']:.0f})")
        
        if results['position_change'].min() < 0:
            biggest_loser = results.loc[results['position_change'].idxmin()]
            print(f"🔻 Größter Verlierer: {biggest_loser['driver']} "
                  f"(P{biggest_loser['grid_position']:.0f} → P{biggest_loser['predicted_position_rounded']}, "
                  f"{biggest_loser['position_change']:.0f})")
        
        # Track-spezifische Hinweise
        track_insights = {
            'netherlands': "Zandvoort: Schwer zu überholen, Qualifying wichtig",
            'monaco': "Monaco: Grid Position entscheidend, Safety Cars möglich",
            'spa': "Spa: Slipstream-Kämpfe, Wetter-Risiko",
            'silverstone': "Silverstone: Balanciertes Racing, Tire Strategy wichtig"
        }
        
        if track_type in track_insights:
            print(f"🏁 Track-Hinweis: {track_insights[track_type]}")

# Factory Functions für Standard-Grids
def get_standard_grids() -> Dict[str, Dict[str, int]]:
    """Standard Grid-Positionen für verschiedene 2025-Rennen"""
    return {
        'netherlands_2025': {
            'VER': 1, 'NOR': 2, 'LEC': 3, 'PIA': 4, 'RUS': 5, 'HAM': 6,
            'ALO': 7, 'PER': 8, 'GAS': 9, 'OCO': 10, 'STR': 11, 'SAI': 12,
            'ALB': 13, 'TSU': 14, 'LAW': 15, 'HUL': 16, 'MAG': 17,
            'ZHO': 18, 'BOT': 19, 'COL': 20
        },
        'monaco_2025': {
            'LEC': 1, 'VER': 2, 'NOR': 3, 'HAM': 4, 'RUS': 5, 'PIA': 6,
            'ALO': 7, 'SAI': 8, 'GAS': 9, 'PER': 10, 'STR': 11, 'ALB': 12,
            'TSU': 13, 'LAW': 14, 'HUL': 15, 'MAG': 16, 'ZHO': 17,
            'BOT': 18, 'COL': 19, 'DOO': 20
        }
    }
