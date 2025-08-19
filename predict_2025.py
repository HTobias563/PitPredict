#!/usr/bin/env python3
"""
Einfache Future Race Predictions - Quick & Easy Interface

Verwendung: python predict_2025.py
"""

import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.predict_future_race import FutureRacePredictor, get_default_netherlands_grid

def main():
    """Hauptfunktion für einfache 2025-Vorhersagen"""
    
    print("🏁 F1 FUTURE RACE PREDICTOR 2025")
    print("=" * 50)
    
    predictor = FutureRacePredictor()
    
    # Verfügbare Strecken
    tracks = {
        '1': ('netherlands', 'Netherlands GP 2025', get_default_netherlands_grid()),
        '2': ('monaco', 'Monaco GP 2025', get_monaco_grid_2025()),
        '3': ('spa', 'Belgian GP 2025', get_spa_grid_2025()),
        '4': ('silverstone', 'British GP 2025', get_silverstone_grid_2025()),
        '5': ('default', 'Generic Race 2025', get_default_netherlands_grid())
    }
    
    print("\nVerfügbare Rennen für 2025-Vorhersage:")
    for key, (track_type, race_name, _) in tracks.items():
        print(f"{key}. {race_name}")
    
    # User Input
    choice = input("\nWähle ein Rennen (1-5) oder drücke Enter für Niederlande: ").strip()
    if not choice:
        choice = '1'
    
    if choice not in tracks:
        print("❌ Ungültige Auswahl, verwende Niederlande GP")
        choice = '1'
    
    track_type, race_name, grid_positions = tracks[choice]
    
    print(f"\n🚀 Starte Vorhersage für {race_name}...")
    
    try:
        results = predictor.predict_future_race(race_name, grid_positions, track_type, 2025)
        
        # Zusätzliche Analyse
        print(f"\n📈 RACE INSIGHTS:")
        print("-" * 30)
        
        # Top Mover (Grid vs Prediction)
        results['position_change'] = results['grid_position'] - results['predicted_position_rounded']
        top_gainers = results.nlargest(3, 'position_change')
        
        print("🔺 Größte Position-Gewinner:")
        for _, row in top_gainers.iterrows():
            if row['position_change'] > 0:
                print(f"   {row['driver']}: P{row['grid_position']:.0f} → P{row['predicted_position_rounded']} "
                      f"(+{row['position_change']:.0f})")
        
        # Biggest Losers
        top_losers = results.nsmallest(3, 'position_change')
        print("\n🔻 Größte Position-Verlierer:")
        for _, row in top_losers.iterrows():
            if row['position_change'] < 0:
                print(f"   {row['driver']}: P{row['grid_position']:.0f} → P{row['predicted_position_rounded']} "
                      f"({row['position_change']:.0f})")
        
        # Championship Implications (Top 10 Points)
        points_system = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
        results['predicted_points'] = 0
        
        for i, (_, row) in enumerate(results.iterrows()):
            if i < 10:
                results.loc[results['driver'] == row['driver'], 'predicted_points'] = points_system[i]
        
        print(f"\n🏆 CHAMPIONSHIP POINTS VORHERSAGE:")
        points_scorers = results[results['predicted_points'] > 0].sort_values('predicted_points', ascending=False)
        for _, row in points_scorers.iterrows():
            print(f"   {row['driver']:3s}: {row['predicted_points']:2.0f} Punkte "
                  f"(P{row['predicted_position_rounded']})")
        
        print(f"\n💡 RACE STRATEGY TIPS:")
        print(f"   • Grid P1 ist nicht garantiert Sieg (siehe {results.iloc[0]['driver']} P{results.iloc[0]['predicted_position_rounded']})")
        print(f"   • Track Overtaking Difficulty: {track_type.title()}")
        print(f"   • Fokus auf Qualifying Performance und Zuverlässigkeit")
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        print("Stelle sicher, dass das Modell trainiert ist.")

def get_monaco_grid_2025():
    """Monaco GP 2025 - Qualifying sehr wichtig"""
    return {
        'LEC': 1,   # Leclerc zu Hause in Monaco
        'VER': 2,   # Verstappen
        'NOR': 3,   # Norris
        'HAM': 4,   # Hamilton bei Ferrari
        'RUS': 5,   # Russell
        'PIA': 6,   # Piastri
        'ALO': 7,   # Alonso
        'SAI': 8,   # Sainz bei Williams
        'GAS': 9,   # Gasly
        'PER': 10,  # Perez
        'STR': 11, 'ALB': 12, 'TSU': 13, 'LAW': 14, 'HUL': 15,
        'MAG': 16, 'ZHO': 17, 'BOT': 18, 'COL': 19, 'DOO': 20
    }

def get_spa_grid_2025():
    """Spa-Francorchamps 2025 - Power Track"""
    return {
        'VER': 1,   # Red Bull Power
        'PIA': 2,   # McLaren stark
        'NOR': 3,   # McLaren
        'LEC': 4,   # Ferrari
        'HAM': 5,   # Hamilton Ferrari
        'RUS': 6,   # Mercedes
        'ALO': 7,   # Alonso
        'PER': 8,   # Perez
        'GAS': 9,   # Alpine
        'SAI': 10,  # Williams
        'STR': 11, 'ALB': 12, 'TSU': 13, 'LAW': 14, 'HUL': 15,
        'MAG': 16, 'ZHO': 17, 'BOT': 18, 'COL': 19, 'DOO': 20
    }

def get_silverstone_grid_2025():
    """Silverstone 2025 - British GP"""
    return {
        'NOR': 1,   # Norris zu Hause
        'RUS': 2,   # Russell zu Hause
        'HAM': 3,   # Hamilton zu Hause (bei Ferrari)
        'VER': 4,   # Verstappen
        'PIA': 5,   # Piastri
        'LEC': 6,   # Leclerc
        'ALO': 7,   # Alonso
        'PER': 8,   # Perez
        'ALB': 9,   # Albon Williams
        'SAI': 10,  # Sainz Williams
        'GAS': 11, 'STR': 12, 'TSU': 13, 'LAW': 14, 'HUL': 15,
        'MAG': 16, 'ZHO': 17, 'BOT': 18, 'COL': 19, 'DOO': 20
    }

if __name__ == "__main__":
    main()
