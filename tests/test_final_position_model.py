#!/usr/bin/env python3
"""Test Script für das Final Position Prediction Model

Dieses Skript testet alle Funktionalitäten des Final Position Modells:
- Model Loading/Saving
- Feature Engineering 
- Training und Cross-Validation
- Holdout Testing
- Race Predictions
- Performance Analyse
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.pitpredict.models.final_position_predict import (
    FinalPositionPredictor, 
    predict_race_positions,
    train_final_position_model,
    test_final_position_model
)
import pandas as pd
import numpy as np
import json

def test_model_functionality():
    """Teste alle Funktionalitäten des Modells"""
    
    print("=== Final Position Prediction Model Test ===\n")
    
    # 1. Test Model Training
    print("1. Teste Model Training...")
    try:
        predictor, results = train_final_position_model()
        print(f"   ✓ Training erfolgreich")
        print(f"   ✓ MAE: {results['overall_mae']:.2f}")
        print(f"   ✓ RMSE: {results['overall_rmse']:.2f}")
        print(f"   ✓ R²: {results['overall_r2']:.3f}")
        print(f"   ✓ Podium Accuracy: {results['overall_podium_accuracy']:.1%}")
        print(f"   ✓ Points Accuracy: {results['overall_points_accuracy']:.1%}")
    except Exception as e:
        print(f"   ✗ Training Error: {e}")
        return
    
    # 2. Test Model Loading
    print("\n2. Teste Model Loading...")
    try:
        predictor_loaded = FinalPositionPredictor()
        metadata = predictor_loaded.load_model('artifacts/models/final_position_predictor.pkl')
        print(f"   ✓ Modell erfolgreich geladen")
        print(f"   ✓ Features: {len(predictor_loaded.feature_names)}")
        print(f"   ✓ Config: {predictor_loaded.config}")
    except Exception as e:
        print(f"   ✗ Loading Error: {e}")
        return
    
    # 3. Test Holdout Predictions
    print("\n3. Teste Holdout Testing...")
    try:
        holdout_results = test_final_position_model(predictor_loaded)
        print(f"   ✓ Holdout Test erfolgreich")
        print(f"   ✓ Holdout MAE: {holdout_results['holdout_mae']:.2f}")
        print(f"   ✓ Holdout R²: {holdout_results['holdout_r2']:.3f}")
        print(f"   ✓ Podium Acc: {holdout_results['holdout_podium_accuracy']:.1%}")
        print(f"   ✓ Points Acc: {holdout_results['holdout_points_accuracy']:.1%}")
    except Exception as e:
        print(f"   ✗ Holdout Error: {e}")
    
    # 4. Test Race Predictions für verschiedene Runden
    print("\n4. Teste Race Predictions...")
    
    test_races = ['2024_21', '2024_22', '2024_08']  # Las Vegas, Qatar, Ungarn
    
    for race_id in test_races:
        try:
            print(f"\n   Testing {race_id}:")
            results = predict_race_positions(race_id)
            
            print(f"   ✓ {len(results)} Predictions generiert")
            print(f"   ✓ Top 3: {', '.join(results.head(3)['driver'])}")
            print(f"   ✓ Avg DNF Risk: {results['dnf_risk'].mean():.1%}")
            print(f"   ✓ Prediction Range: P{results['predicted_position_rounded'].min()}-P{results['predicted_position_rounded'].max()}")
            
        except Exception as e:
            print(f"   ✗ Race {race_id} Error: {e}")
    
    print("\n=== Test Complete ===")

def analyze_model_performance():
    """Analysiere die Model Performance im Detail"""
    
    print("\n=== Model Performance Analyse ===\n")
    
    # Lade gespeicherte Metriken
    try:
        with open('artifacts/metrics/final_position_cv_report.json', 'r') as f:
            cv_results = json.load(f)
        
        with open('artifacts/metrics/final_position_holdout_report.json', 'r') as f:
            holdout_results = json.load(f)
    except Exception as e:
        print(f"Kann Metriken nicht laden: {e}")
        return
    
    print("1. Cross-Validation Performance:")
    print(f"   Overall MAE: {cv_results['overall_mae']:.2f} positions")
    print(f"   Overall RMSE: {cv_results['overall_rmse']:.2f} positions")
    print(f"   Overall R²: {cv_results['overall_r2']:.3f}")
    print(f"   Podium Accuracy: {cv_results['overall_podium_accuracy']:.1%}")
    print(f"   Points Accuracy: {cv_results['overall_points_accuracy']:.1%}")
    
    print(f"\n   Training Data:")
    print(f"   - {cv_results['n_samples']} samples")
    print(f"   - {cv_results['n_dnfs']} DNFs ({cv_results['dnf_rate']:.1%})")
    
    print(f"\n   Fold Performance:")
    for fold in cv_results['fold_scores']:
        print(f"   Fold {fold['fold']}: MAE={fold['mae']:.2f}, "
              f"RMSE={fold['rmse']:.2f}, R²={fold['r2']:.3f}")
    
    print("\n2. Holdout Performance:")
    print(f"   Holdout MAE: {holdout_results['holdout_mae']:.2f} positions")
    print(f"   Holdout RMSE: {holdout_results['holdout_rmse']:.2f} positions")
    print(f"   Holdout R²: {holdout_results['holdout_r2']:.3f}")
    print(f"   Podium Accuracy: {holdout_results['holdout_podium_accuracy']:.1%}")
    print(f"   Points Accuracy: {holdout_results['holdout_points_accuracy']:.1%}")
    print(f"   DNF Detection: {holdout_results['holdout_dnf_detection']:.1%}")
    
    print(f"\n   Holdout Data:")
    print(f"   - {holdout_results['n_holdout_samples']} samples")
    print(f"   - {holdout_results['n_holdout_dnfs']} DNFs ({holdout_results['holdout_dnf_rate']:.1%})")
    print(f"   - Rounds: {holdout_results['holdout_rounds']}")
    
    print("\n3. Feature Importance:")
    if 'feature_importance' in cv_results:
        sorted_features = sorted(cv_results['feature_importance'].items(), 
                               key=lambda x: x[1], reverse=True)[:15]
        for feat, imp in sorted_features:
            print(f"   {feat}: {imp:.4f}")
    
    print("\n4. Model Interpretation:")
    mae = cv_results['overall_mae']
    r2 = cv_results['overall_r2']
    podium_acc = cv_results['overall_podium_accuracy']
    
    if mae <= 2.5:
        mae_rating = "Ausgezeichnet"
    elif mae <= 3.5:
        mae_rating = "Sehr gut"
    elif mae <= 5.0:
        mae_rating = "Gut"
    else:
        mae_rating = "Verbesserungsbedarf"
    
    print(f"   MAE Bewertung: {mae_rating} ({mae:.2f} Positionen)")
    print(f"   R² Erklärung: Das Modell erklärt {r2:.1%} der Varianz")
    print(f"   Podium Precision: {podium_acc:.1%} der Top 3 Vorhersagen sind korrekt")
    
    if r2 > 0.6:
        print("   ✓ Starke Vorhersagekraft")
    elif r2 > 0.4:
        print("   ○ Moderate Vorhersagekraft")
    else:
        print("   ✗ Schwache Vorhersagekraft - weitere Verbesserungen nötig")

def detailed_race_analysis():
    """Detaillierte Analyse spezifischer Rennen"""
    
    print("\n=== Detaillierte Race Analyse ===\n")
    
    # Analysiere verschiedene Race Types
    race_analyses = {
        '2024_08': 'Ungarn (Traditionell)',
        '2024_15': 'Niederlande (Moderne Strecke)',
        '2024_21': 'Las Vegas (Street Circuit)',
        '2024_22': 'Qatar (Sprint Weekend)'
    }
    
    for race_id, description in race_analyses.items():
        try:
            print(f"\n{description} ({race_id}):")
            results = predict_race_positions(race_id)
            
            # Grid vs Prediction Analysis
            grid_positions = []
            predicted_positions = []
            
            # Lade actual data für Vergleich
            import pandas as pd
            df = pd.read_parquet('data/season=2024/driver_race_table.parquet')
            race_data = df[df['race_id'] == race_id].copy()
            
            if not race_data.empty:
                comparison = results.merge(
                    race_data[['driver', 'finish_position', 'is_dnf']], 
                    on='driver', how='left'
                )
                
                # Analysiere Overperformer/Underperformer
                valid_results = comparison.dropna(subset=['finish_position'])
                if not valid_results.empty:
                    valid_results['position_change'] = (
                        valid_results['grid_position'] - valid_results['finish_position']
                    )
                    valid_results['prediction_error'] = abs(
                        valid_results['predicted_final_position'] - valid_results['finish_position']
                    )
                    
                    # Top Overperformer
                    overperformer = valid_results.loc[valid_results['position_change'].idxmax()]
                    print(f"   Größter Overperformer: {overperformer['driver']} "
                          f"(Grid P{overperformer['grid_position']:.0f} → "
                          f"Finish P{overperformer['finish_position']:.0f})")
                    
                    # Beste Prediction
                    best_pred = valid_results.loc[valid_results['prediction_error'].idxmin()]
                    print(f"   Beste Vorhersage: {best_pred['driver']} "
                          f"(Pred P{best_pred['predicted_final_position']:.1f}, "
                          f"Actual P{best_pred['finish_position']:.0f})")
                    
                    # Race Statistics
                    mean_error = valid_results['prediction_error'].mean()
                    print(f"   Durchschnittlicher Vorhersage-Fehler: {mean_error:.2f} Positionen")
            else:
                print("   Keine Vergleichsdaten verfügbar")
                
        except Exception as e:
            print(f"   Fehler bei {race_id}: {e}")

if __name__ == "__main__":
    # Führe alle Tests aus
    test_model_functionality()
    
    # Detaillierte Performance-Analyse
    analyze_model_performance()
    
    # Spezifische Race-Analysen
    detailed_race_analysis()
    
    print("\n" + "="*50)
    print("FINAL POSITION MODEL TESTING COMPLETE")
    print("="*50)
