#!/usr/bin/env python3
"""
Future Race Prediction System für 2025 und darüber hinaus

Dieses Script ermöglicht Vorhersagen für zukünftige Rennen basierend auf:
1. Aktuellsten verfügbaren Daten (2024)
2. Extrapolierten Historical Performance Metriken
3. Benutzer-Input für Grid Positions und aktuelle Form

Verwendung: 
    python predict_future_race.py --race_name "Netherlands GP 2025" --grid_positions "VER:1,NOR:2,..."
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from typing import Dict, List, Optional

# PitPredict Module importieren
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from src.pitpredict.models.final_position_predict import FinalPositionPredictor

class FutureRacePredictor:
    """Vorhersage-System für zukünftige Rennen (2025+)"""
    
    def __init__(self):
        self.predictor = FinalPositionPredictor()
        self.model_path = os.path.join(project_root, 'artifacts/models/final_position_predictor.pkl')
        self.base_data = None
        self.track_profiles = self._get_track_profiles()
        
    def _get_track_profiles(self) -> Dict[str, Dict]:
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
    
    def load_latest_data(self):
        """Lade die neuesten verfügbaren Daten (2024)"""
        print("📊 Lade aktuelle Baseline-Daten (2024)...")
        
        data_path = os.path.join(project_root, 'data/season=2024/driver_race_table.parquet')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Baseline-Daten nicht gefunden: {data_path}")
            
        self.base_data = pd.read_parquet(data_path)
        
        # Lade trainiertes Modell
        if os.path.exists(self.model_path):
            self.predictor.load_model(self.model_path)
            print(f"✅ Modell geladen: {self.model_path}")
        else:
            raise FileNotFoundError(f"Trainiertes Modell nicht gefunden: {self.model_path}")
    
    def get_latest_driver_stats(self) -> pd.DataFrame:
        """Extrahiere die neuesten Fahrer-Statistiken aus 2024"""
        
        # Nehme die letzten 5 Rennen als Basis für "current form"
        latest_races = self.base_data['round'].nlargest(5).unique()
        recent_data = self.base_data[self.base_data['round'].isin(latest_races)]
        
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
            'q_best_time_ms': 'mean'
        }).round(2)
        
        # Berechne Performance-Trends
        driver_stats['recent_avg_finish'] = driver_stats['finish_position']
        driver_stats['recent_dnf_rate'] = driver_stats['is_dnf']
        driver_stats['recent_quali_performance'] = driver_stats['q_pos']
        
        # Team-Zuordnung für 2025 (angenommene Transfers)
        team_updates_2025 = {
            'HAM': 'Ferrari',  # Lewis zu Ferrari
            'SAI': 'Williams',  # Sainz zu Williams (hypothetisch)
            # Weitere Transfers hier hinzufügen
        }
        
        for driver, new_team in team_updates_2025.items():
            if driver in driver_stats.index:
                driver_stats.loc[driver, 'team'] = new_team
                print(f"🔄 Transfer Update: {driver} → {new_team}")
        
        return driver_stats
    
    def create_future_race_data(self, 
                               race_name: str,
                               grid_positions: Dict[str, int],
                               track_type: str = 'default',
                               season: int = 2025,
                               round_num: int = 15) -> pd.DataFrame:
        """Erstelle synthetische Race-Daten für zukünftiges Rennen"""
        
        print(f"🏗️  Erstelle Race-Daten für {race_name}")
        
        # Basis-Fahrer-Stats
        driver_stats = self.get_latest_driver_stats()
        
        # Track-Profile
        track_profile = self.track_profiles.get(track_type, self.track_profiles['default'])
        
        # Erstelle Race-DataFrame
        race_data_list = []
        
        for driver, grid_pos in grid_positions.items():
            if driver not in driver_stats.index:
                print(f"⚠️  Unbekannter Fahrer: {driver} - überspringe")
                continue
                
            # Basis-Daten vom Fahrer
            base_stats = driver_stats.loc[driver]
            
            # Synthetische Race-Daten erstellen
            race_entry = {
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
                
                # Historical Performance (aus 2024 extrapoliert)
                'points_last5': base_stats['points'] * 0.8,  # Leicht reduziert für neue Saison
                'driver_dnf_rate_last5': base_stats['recent_dnf_rate'],
                'avg_finish_delta_vs_grid_last5': base_stats['recent_avg_finish'] - base_stats['grid_position'],
                'finish_delta_vs_grid': base_stats['recent_avg_finish'] - base_stats['grid_position'],
                
                # Track-spezifische Features
                **track_profile,
                
                # Team-Performance (simplified)
                'team_quali_rank_last5': self._estimate_team_quali_rank(base_stats['team']),
                'team_dnf_rate_last5': base_stats['recent_dnf_rate'] * 1.2,  # Team-Level
                
                # Zusätzliche Required Features
                'q_pos_mean_last5': base_stats['recent_quali_performance'],
                'q_stdev_last5': 2.0,  # Default Varianz
                'q3_rate_last5': 0.6 if base_stats['recent_quali_performance'] <= 10 else 0.2,
                'q_best_gap_to_pole_ms_last5': base_stats['q_gap_to_pole_ms'],
                
                # Placeholder für fehlende Features
                'finish_position': np.nan,  # Wird vorhergesagt
                'classification_status': None,
                'points': 0,
                'laps_completed': 0,
                'is_dnf': 0,
                'total_laps': 58  # Typische Rundenzahl
            }
            
            race_data_list.append(race_entry)
        
        race_df = pd.DataFrame(race_data_list)
        
        print(f"✅ Race-Daten erstellt: {len(race_df)} Fahrer für {race_name}")
        return race_df
    
    def _get_driver_number(self, driver_code: str) -> int:
        """Driver Numbers (2024/2025)"""
        numbers = {
            'VER': 1, 'SAR': 2, 'RIC': 3, 'NOR': 4, 'PIA': 8, 'GAS': 10,
            'PER': 11, 'ALO': 14, 'LEC': 16, 'STR': 18, 'MAG': 20,
            'TSU': 22, 'ALB': 23, 'ZHO': 24, 'HUL': 27, 'OCO': 31,
            'RUS': 63, 'HAM': 44, 'BOT': 77, 'SAI': 55, 'LAW': 30,
            'COL': 43, 'BEA': 50, 'DOO': 99
        }
        return numbers.get(driver_code, 99)
    
    def _estimate_team_quali_rank(self, team: str) -> float:
        """Schätze Team-Qualifying-Rank für 2025"""
        team_ranks = {
            'Red Bull Racing': 2.0,
            'McLaren': 2.5, 
            'Ferrari': 3.0,
            'Mercedes': 4.0,
            'Aston Martin': 6.0,
            'Alpine': 7.0,
            'Williams': 8.0,
            'RB': 7.5,
            'Haas F1 Team': 8.5,
            'Kick Sauber': 9.0
        }
        return team_ranks.get(team, 7.0)
    
    def predict_future_race(self, 
                           race_name: str,
                           grid_positions: Dict[str, int],
                           track_type: str = 'default',
                           season: int = 2025) -> pd.DataFrame:
        """Hauptfunktion: Vorhersage für zukünftiges Rennen"""
        
        print(f"🚀 FUTURE RACE PREDICTION: {race_name}")
        print("=" * 60)
        
        # 1. Lade aktuelle Daten und Modell
        self.load_latest_data()
        
        # 2. Erstelle synthetische Race-Daten
        future_race_data = self.create_future_race_data(
            race_name, grid_positions, track_type, season
        )
        
        # 3. Mache Prediction
        print(f"🎯 Führe Vorhersage durch...")
        results = self.predictor.predict_final_positions(future_race_data)
        
        # 4. Zeige Ergebnisse
        self._display_results(results, race_name)
        
        return results
    
    def _display_results(self, results: pd.DataFrame, race_name: str):
        """Zeige formatierte Ergebnisse"""
        
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

def parse_grid_positions(grid_string: str) -> Dict[str, int]:
    """Parse Grid Position String wie 'VER:1,NOR:2,LEC:3'"""
    positions = {}
    
    for entry in grid_string.split(','):
        if ':' in entry:
            driver, pos = entry.strip().split(':')
            positions[driver.upper()] = int(pos)
    
    return positions

def get_default_netherlands_grid() -> Dict[str, int]:
    """Default Grid für Niederlande GP 2025 (hypothetisch)"""
    return {
        'VER': 1,   # Verstappen zu Hause
        'NOR': 2,   # Norris starke Form
        'LEC': 3,   # Leclerc mit HAM bei Ferrari
        'PIA': 4,   # Piastri konstant
        'RUS': 5,   # Russell Mercedes
        'HAM': 6,   # Hamilton bei Ferrari (Transfer!)
        'ALO': 7,   # Alonso Aston Martin
        'PER': 8,   # Perez Red Bull
        'GAS': 9,   # Gasly Alpine
        'OCO': 10,  # Ocon Alpine
        'STR': 11,  # Stroll Aston Martin
        'SAI': 12,  # Sainz Williams (Transfer!)
        'ALB': 13,  # Albon Williams
        'TSU': 14,  # Tsunoda RB
        'LAW': 15,  # Lawson RB
        'HUL': 16,  # Hülkenberg Haas
        'MAG': 17,  # Magnussen Haas
        'ZHO': 18,  # Zhou Sauber
        'BOT': 19,  # Bottas Sauber
        'COL': 20   # Colapinto Williams
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Future Race Prediction System')
    parser.add_argument('--race_name', type=str, default='Netherlands GP 2025',
                       help='Name des Rennens')
    parser.add_argument('--track_type', type=str, default='netherlands',
                       choices=['netherlands', 'monaco', 'spa', 'silverstone', 'default'],
                       help='Streckentyp')
    parser.add_argument('--grid_positions', type=str, default=None,
                       help='Grid Positionen als "VER:1,NOR:2,LEC:3"')
    parser.add_argument('--season', type=int, default=2025,
                       help='Saison Jahr')
    
    args = parser.parse_args()
    
    # Grid Positions parsen oder Default verwenden
    if args.grid_positions:
        grid_pos = parse_grid_positions(args.grid_positions)
    else:
        print("🏁 Verwende hypothetische Grid für Niederlande GP 2025:")
        grid_pos = get_default_netherlands_grid()
        
        print("\nGrid Positionen:")
        for pos in range(1, 21):
            driver = [d for d, p in grid_pos.items() if p == pos][0]
            print(f"P{pos:2d}: {driver}")
    
    # Prediction durchführen  
    predictor = FutureRacePredictor()
    
    try:
        results = predictor.predict_future_race(
            args.race_name, 
            grid_pos, 
            args.track_type, 
            args.season
        )
        
        # Speichere Ergebnisse
        output_file = f"artifacts/metrics/future_prediction_{args.race_name.replace(' ', '_').lower()}.csv"
        results.to_csv(output_file, index=False)
        print(f"\n💾 Ergebnisse gespeichert: {output_file}")
        
    except Exception as e:
        print(f"❌ Fehler bei der Vorhersage: {e}")
        print("Stellen Sie sicher, dass das Modell trainiert ist und die Daten verfügbar sind.")
