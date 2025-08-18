#!/usr/bin/env python3
"""Test Script für das Pit Stop Prediction Model"""

from src.pitpredict.models.pit_predict import PitStopPredictor, predict_race_pitstops
import pandas as pd
import numpy as np

def test_model_functionality():
    """Teste alle Funktionalitäten des Modells"""
    
    print("=== Pit Stop Prediction Model Test ===\n")
    
    # 1. Test Model Loading
    print("1. Teste Model Loading...")
    try:
        predictor = PitStopPredictor()
        metadata = predictor.load_model('artifacts/models/pitstop_predictor_calibrated.pkl')
        print(f"   ✓ Modell erfolgreich geladen")
        print(f"   ✓ Features: {len(predictor.feature_names)}")
        print(f"   ✓ Config: {predictor.config}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    
    # 2. Test Race Predictions für verschiedene Runden
    print("\n2. Teste Race Predictions...")
    
    test_races = ['2024_21', '2024_22', '2024_08']  # Verschiedene Rennen
    
    for race_id in test_races:
        try:
            print(f"\n   Testing {race_id}:")
            results = predict_race_pitstops(race_id, up_to_lap=30)
            
            if not results.empty:
                top_3 = results.head(3)
                print(f"   ✓ {len(results)} Predictions generiert")
                print("   ✓ Top 3 Kandidaten:")
                for _, row in top_3.iterrows():
                    print(f"      - {row['driver']} Lap {row['lap_number']}: "
                          f"{row['pit_probability']:.3f} "
                          f"({row['laps_since_pit']} laps on {row['compound']})")
            else:
                print("   ⚠ Keine Predictions generiert")
                
        except Exception as e:
            print(f"   ✗ Error für {race_id}: {e}")
    
    # 3. Test verschiedene Szenarien
    print("\n3. Teste verschiedene Szenarien...")
    
    scenarios = [
        ('2024_21', 20, "Frühe Runden"),
        ('2024_21', 40, "Mittlere Runden"), 
        ('2024_21', None, "Komplettes Rennen")
    ]
    
    for race_id, up_to_lap, description in scenarios:
        try:
            results = predict_race_pitstops(race_id, up_to_lap)
            if not results.empty:
                max_prob = results['pit_probability'].max()
                avg_prob = results['pit_probability'].mean()
                print(f"   ✓ {description}: Max={max_prob:.3f}, Avg={avg_prob:.3f}")
            else:
                print(f"   ⚠ {description}: Keine Daten")
        except Exception as e:
            print(f"   ✗ {description}: Error {e}")
    
    # 4. Analysiere Feature Importance (falls möglich)
    print("\n4. Analysiere Model Insights...")
    try:
        # Lade ein kleines Test-Dataset
        from src.pitpredict.models.pit_predict import load_lap_data, create_pit_stop_target, engineer_pit_features
        
        test_data = load_lap_data(2024, [21])
        test_data = create_pit_stop_target(test_data)
        test_data = engineer_pit_features(test_data)
        
        valid_data = test_data[test_data['lap_number'] >= 5].copy()
        valid_data = valid_data[valid_data['is_dnf'] == 0].copy()
        
        print(f"   ✓ Test Dataset: {len(valid_data)} samples")
        print(f"   ✓ Positive Rate: {valid_data['pit_in_next_3_laps'].mean():.3f}")
        
        # Predictions auf Test-Daten
        test_results = predictor.predict_pit_probability(valid_data)
        print(f"   ✓ Prediction Range: {test_results['pit_probability'].min():.3f} - "
              f"{test_results['pit_probability'].max():.3f}")
        
    except Exception as e:
        print(f"   ✗ Model insights error: {e}")
    
    print("\n=== Test completed! ===")

def analyze_model_performance():
    """Analysiere die Model Performance im Detail"""
    
    print("\n=== Model Performance Analysis ===")
    
    # Lade Reports
    import json
    
    # CV Report
    try:
        with open('artifacts/metrics/pitstop_cv_report.json', 'r') as f:
            cv_report = json.load(f)
        
        print(f"\nCross-Validation Results:")
        print(f"  Overall PR-AUC: {cv_report['overall_pr_auc']:.3f}")
        print(f"  Overall ROC-AUC: {cv_report['overall_roc_auc']:.3f}")
        print(f"  Overall Brier Score: {cv_report['overall_brier']:.4f}")
        print(f"  Training Samples: {cv_report['n_samples']:,}")
        print(f"  Positive Rate: {cv_report['positive_rate']:.3f}")
        
        print(f"\n  Fold-wise Performance:")
        for fold in cv_report['fold_scores']:
            print(f"    Fold {fold['fold']}: PR-AUC={fold['pr_auc']:.3f}, "
                  f"ROC-AUC={fold['roc_auc']:.3f}, Brier={fold['brier']:.4f}")
    
    except Exception as e:
        print(f"  ✗ CV Report error: {e}")
    
    # Holdout Report
    try:
        with open('artifacts/metrics/pitstop_holdout_report.json', 'r') as f:
            holdout_report = json.load(f)
        
        print(f"\nHoldout Test Results:")
        print(f"  Holdout PR-AUC: {holdout_report['holdout_pr_auc']:.3f}")
        print(f"  Holdout ROC-AUC: {holdout_report['holdout_roc_auc']:.3f}")
        print(f"  Holdout Brier Score: {holdout_report['holdout_brier']:.4f}")
        print(f"  Holdout Samples: {holdout_report['n_holdout_samples']:,}")
        print(f"  Holdout Positive Rate: {holdout_report['holdout_positive_rate']:.3f}")
        print(f"  Test Rounds: {holdout_report['holdout_rounds']}")
    
    except Exception as e:
        print(f"  ✗ Holdout Report error: {e}")

if __name__ == "__main__":
    test_model_functionality()
    analyze_model_performance()
