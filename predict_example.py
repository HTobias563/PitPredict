#!/usr/bin/env python3
"""
Erweiterte Beispiel-Script für Final Position Predictions
Unterstützt sowohl 2024-Daten als auch 2025-Future-Predictions

Verwendung: 
    python predict_example.py                    # 2024 Rennen
    python predict_example.py --future          # 2025 Future Predictions
    python predict_example.py --race_id 2024_21 # Spezifisches 2024 Rennen
"""

import os
import sys
import pandas as pd
import argparse

# PitPredict Module importieren
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
from src.pitpredict.models.final_position_predict import FinalPositionPredictor
from src.pitpredict.models.future_position_predict import (
    FutureRacePredictor, 
    FutureRacePredictionConfig,
    get_standard_grids
)

def predict_race(race_id):
    """Vorhersage für ein spezifisches 2024 Rennen"""
    
    print(f"🏁 Vorhersage für Rennen {race_id}")
    
    # 1. Modell laden
    predictor = FinalPositionPredictor()
    model_path = 'artifacts/models/final_position_predictor.pkl'
    predictor.load_model(model_path)
    
    # 2. Race-Daten laden
    df = pd.read_parquet('data/season=2024/driver_race_table.parquet')
    race_data = df[df['race_id'] == race_id].copy()
    
    if race_data.empty:
        print(f"❌ Keine Daten für Rennen {race_id}")
        return
    
    # 3. Predictions machen
    results = predictor.predict_final_positions(race_data)
    
    # 4. Ergebnisse anzeigen
    _display_race_results(results, race_id)
    
    return results

def predict_future_race(race_name: str, track_type: str = 'netherlands'):
    """Vorhersage für ein 2025 Future Race"""
    
    print(f"🔮 Future Race Vorhersage: {race_name}")
    
    # 1. Future Predictor initialisieren
    config = FutureRacePredictionConfig()
    predictor = FutureRacePredictor(config)
    
    # 2. Standard-Grid laden
    standard_grids = get_standard_grids()
    grid_key = f"{track_type}_2025"
    
    if grid_key in standard_grids:
        grid_positions = standard_grids[grid_key]
    else:
        # Fallback auf Netherlands Grid
        grid_positions = standard_grids['netherlands_2025']
        print(f"⚠️  Kein spezifisches Grid für {track_type}, verwende Netherlands-Grid")
    
    # 3. Future Prediction
    results = predictor.predict_future_race(race_name, grid_positions, track_type, 2025)
    
    return results

def _display_race_results(results: pd.DataFrame, race_identifier: str):
    """Einheitliche Darstellung der Rennergebnisse"""
    
    print(f"\n🎯 PODIUM VORHERSAGE für {race_identifier}:")
    print("=" * 50)
    
    podium = results.head(3)
    for i, (_, row) in enumerate(podium.iterrows(), 1):
        print(f"{i}. {row['driver']} ({row['team'][:15]:<15}) - "
              f"P{row['predicted_position_rounded']:2d} "
              f"(Grid P{row['grid_position']:.0f}) "
              f"[DNF Risk: {row['dnf_risk']:.1%}]")
    
    print(f"\n📊 ALLE POSITIONEN:")
    print("-" * 70)
    
    for i, (_, row) in enumerate(results.iterrows(), 1):
        status = "🏆" if i <= 3 else "🏁" if i <= 10 else "📍"
        print(f"{status} {i:2d}. {row['driver']:3s} ({row['team'][:12]:12s}) - "
              f"P{row['predicted_position_rounded']:2d} "
              f"(Grid P{row['grid_position']:.0f}) "
              f"[Conf: {row['prediction_confidence']:.1%}, "
              f"DNF Risk: {row['dnf_risk']:.1%}]")
    
    return results

def predict_multiple_races(race_ids):
    """Vorhersagen für mehrere Rennen"""
    
    all_predictions = []
    
    for race_id in race_ids:
        print(f"\n" + "="*60)
        results = predict_race(race_id)
        if results is not None:
            all_predictions.append(results)
    
    if all_predictions:
        # Kombiniere alle Predictions
        combined = pd.concat(all_predictions, ignore_index=True)
        
        # Speichere kombinierte Ergebnisse
        output_file = f"artifacts/metrics/combined_predictions_{'_'.join(race_ids)}.csv"
        combined.to_csv(output_file, index=False)
        print(f"\n💾 Alle Predictions gespeichert: {output_file}")
        
        return combined
    
    return None

if __name__ == "__main__":
    # Beispiel-Verwendung
    
    # 1. Einzelnes Rennen
    print("🚀 EINZELNE RACE PREDICTION")
    predict_race('2024_21')  # Las Vegas
    
    print("\n" + "="*80)
    
    # 2. Mehrere Rennen (z.B. Saisonende)
    print("🚀 MULTIPLE RACE PREDICTIONS")
    season_end_races = ['2024_21', '2024_22', '2024_23', '2024_24']
    predict_multiple_races(season_end_races)
    
    print("\n" + "="*80)
    
    # 3. Verfügbare Rennen anzeigen
    print("🚀 VERFÜGBARE RENNEN")
    df = pd.read_parquet('data/season=2024/driver_race_table.parquet')
    available_races = sorted(df['race_id'].unique())
    
    print("Alle verfügbaren Race IDs:")
    for i, race_id in enumerate(available_races):
        if i % 6 == 0:
            print()
        print(f"{race_id:8s}", end="")
    print("\n")
    
    print("📝 Verwendung:")
    print("  python predict_example.py")
    print("  Oder in eigenen Scripten: predict_race('2024_XX')")
