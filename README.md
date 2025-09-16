# PitPredict - F1 Race Outcome Prediction Suite

Ein umfassendes Machine Learning-System zur Vorhersage von Formel-1-Rennergebnissen, bestehend aus drei integrierten Modellen für DNF-Vorhersage, Pit Stop-Strategien und finale Positionen.

## Modell-Übersicht

### 1. **Final Position Prediction Model** 
**Hauptmodell** für die Vorhersage der Endplatzierung vor dem Rennen
- **Performance**: MAE 2.12 Positionen, 96.7% Podium-Accuracy
- **Features**: 42 engineerte Features aus Grid-Position, Qualifying, historischer Performance
- **Integration**: Nutzt DNF- und Pit Stop-Modelle als Features

### 2. **DNF Prediction Model** 
Vorhersage der Ausfallwahrscheinlichkeit (Did Not Finish)
- **Performance**: Hochwertige DNF-Risiko-Einschätzung
- **Features**: Fahrer/Team-Zuverlässigkeit, Track-spezifische Risiken

### 3. **Pit Stop Strategy Model**  
Vorhersage optimaler Pit Stop-Strategien (derzeit in Reparatur)
- **Features**: Reifenverschleiß, Streckencharakteristika, Strategische Faktoren

##  Quick Start

### Installation
```bash
# Repository klonen
git clone <repository-url>
cd PitPredict

# Conda Environment erstellen
conda create -n pitpredict_env python=3.10
conda activate pitpredict_env

# Abhängigkeiten installieren
pip install numpy==1.21.6 pandas==1.5.3 scikit-learn==1.1.3 joblib pyyaml scipy pyarrow
```

### Training der Modelle
```bash
# Final Position Model (Hauptmodell)
python -m src.pitpredict.models.final_position_predict --train

# DNF Model
python -m src.pitpredict.models.train_dnf

# Pit Stop Model (nach Reparatur)
python -m src.pitpredict.models.pit_predict --train
```

### Race Predictions

####  2024 Rennen (Vergangenheitsdaten)
```bash
# Einzelnes Rennen vorhersagen
python predict_example.py --race_id 2024_21

# Alle Rennen
python predict_example.py
```

####  Future Races (2025+)
```bash
# Einfache Vorhersagen
python predict_2025.py

# Erweiterte CLI mit benutzerdefinierten Grid-Positionen
python predict_future_race.py \
  --race_name "Monaco GP 2025" \
  --track_type monaco \
  --season 2025 \
  --grid_positions "VER:1,NOR:2,LEC:3,RUS:4,HAM:5"
```

####  Python API
```python
from src.pitpredict.models.final_position_predict import FinalPositionPredictor
from src.pitpredict.models.future_position_predict import FutureRacePredictor

# 2024 Rennen
predictor = FinalPositionPredictor()
predictor.load_model('artifacts/models/final_position_predictor.pkl')
results = predictor.predict_race('2024_21')

# Future Races
future_predictor = FutureRacePredictor()
grid_positions = ['VER:1', 'NOR:2', 'LEC:3', 'RUS:4', 'HAM:5']
results = future_predictor.predict_future_race(
    "Netherlands GP 2025", 
    grid_positions, 
    "netherlands", 
    2025
)
```

##  Model Performance

| Modell | MAE | R² | Spezialmetrik |
|--------|-----|----|--------------| 
| Final Position | 2.12 | 0.829 | 96.7% Podium Accuracy |
| DNF Prediction | - | - | 76.2% DNF Detection |
| Pit Stop Strategy | - | - | In Reparatur |

##  Architektur

```
PitPredict/
├── src/pitpredict/models/
│   ├── final_position_predict.py    # Hauptmodell
│   ├── train_dnf.py                 # DNF-Modell  
│   └── pit_predict.py               # Pit Stop-Modell
├── data/season=2024/                # Training-Daten
├── artifacts/
│   ├── models/                      # Trainierte Modelle
│   └── metrics/                     # Performance-Metriken
├── docs/                           # Dokumentation
└── tests/                          # Test-Suite
```

##  Wichtigste Features

### Final Position Model Features (Top 10)
1. **start_performance_index** (24.1%) - Kombinierter Performance-Indikator
2. **finish_delta_vs_grid** (21.5%) - Historische Position-Verbesserung
3. **q_gap_to_pole_s** (17.2%) - Qualifying-Gap zur Pole-Position  
4. **reliability_adjusted_performance** (8.5%) - Zuverlässigkeits-adjustierte Performance
5. **q_gap_to_pole_ms** (6.6%) - Qualifying-Gap (Millisekunden)

### Integration
- **DNF-Risiko** wird als Feature in Final Position-Vorhersage integriert
- **Historical Performance** über 5-Rennen-Fenster
- **Track-spezifische Faktoren** (Overtaking, Pit Loss, Street Circuit)

##  Use Cases

### 1. Pre-Race Analysis
```python
from src.pitpredict.models.final_position_predict import FinalPositionPredictor

predictor = FinalPositionPredictor()
predictor.load_model('artifacts/models/final_position_predictor.pkl')

# Predictions für nächstes Rennen
results = predictor.predict_final_positions(race_data)
print("Podium Prediction:", results.head(3)['driver'].tolist())
```

### 2. Strategy Planning
```python
# DNF-Risiko für Fahrer-Bewertung
high_risk_drivers = results[results['dnf_risk'] > 0.3]
print("High DNF Risk:", high_risk_drivers[['driver', 'dnf_risk']])

# Grid Position vs Prediction Analysis
overtaking_potential = results[results['predicted_final_position'] < results['grid_position']]
```

### 3. Performance Analysis
```python
# Holdout-Test für Model-Validierung
python -m src.pitpredict.models.final_position_predict --test

# Ergebnisse:
# MAE: 2.95, R²: 0.691, Podium Acc: 91.3%
```

##  Datenquellen

- **driver_race_table.parquet**: Haupt-Dataset mit Race-Ergebnissen
- **Lap Data**: Detaillierte Lap-by-Lap-Daten für Pit Stop-Modell  
- **FastF1 Cache**: Cached Race-Daten für schnelleren Zugriff
- **Config.yaml**: Zentrale Konfiguration für Pfade und Parameter

##  Testing

```bash
# Vollständiger Test des Final Position Models
python -m tests.test_final_position_model

# Test-Coverage:
#  Model Training & Cross-Validation
#  Model Loading & Persistence  
#  Holdout Testing
#  Race Predictions
#  Performance Analysis
```

##  Dokumentation

- **[Final Position Model Übersicht](docs/FINAL_POSITION_MODEL.md)** - Detaillierte Modell-Dokumentation
- **[API Documentation](docs/FINAL_POSITION_API.md)** - Technische API-Referenz
- **[DNF Features](docs/DNF_FEATURES.md)** - DNF-Modell Details
- **[Pit Stop Model](docs/PITSTOP_MODEL_SUMMARY.md)** - Pit Stop-Modell (Legacy)

## 🔧 Konfiguration

### config.yaml
```yaml
season: 2024
processed_table: 'data/season=2024/driver_race_table.parquet'
models_dir: 'artifacts/models'
metrics_dir: 'artifacts/metrics'
holdout_rounds: [21, 22, 23, 24]  # Abu Dhabi, Las Vegas, Qatar, Finale
```

### Python Environment
- **Python**: 3.10 (Kompatibilität mit Legacy-Paketen)
- **NumPy**: <1.22 (scikit-learn Kompatibilität)
- **Pandas**: <2.0 (Backward Compatibility)
- **Scikit-learn**: <1.2 (OneHotEncoder sparse_output)

##  Aktuelle Entwicklung

###  Fertiggestellt
- Final Position Prediction Model (Produktionsbereit)
- DNF Model Integration
- Comprehensive Test Suite
- Performance Monitoring
- Dokumentation

###  In Arbeit
- Pit Stop Model Reparatur (OneHotEncoder Kompatibilität)
- Real-time Race Updates
- Wetter-Integration
- Erweiterte Feature Engineering

###  Geplant
- Ensemble Methods (Multiple Algorithm Fusion)
- Safety Car Probability Modeling
- Tire Strategy Optimization
- Live Dashboard für Race Day

##  Model Insights

### Was funktioniert sehr gut:
- **Podium-Vorhersagen**: >90% Accuracy auf allen Streckentypen
- **Qualifying-basierte Predictions**: Grid Position + Gap = starker Prädiktor
- **Historical Performance Integration**: 5-Rennen-Fenster optimal
- **DNF-Risiko-Einschätzung**: 76% Detection Rate

### Herausforderungen:
- **Street Circuits**: Höhere Variabilität (MAE ~4.0 vs ~2.5)
- **Extreme Comebacks**: VER P17→P1 schwer vorhersagbar
- **Strategy Surprises**: Unerwartete Pit Stop-Entscheidungen
- **Weather Impact**: Noch nicht vollständig modelliert

##  Erfolge

- **96.7% Podium Accuracy** - Weltklasse-Performance
- **2.12 MAE** - Durchschnittlich nur 2 Positionen Fehler
- **82.9% Varianz erklärt** - Sehr starke Predictive Power
- **Robuste Cross-Validation** - Konsistent über alle Folds

##  Beitragen

Das Projekt ist modular aufgebaut und erweiterbar. Neue Features können in der Feature-Engineering-Pipeline hinzugefügt werden.

### Development Setup
```bash
# Testing Environment
conda activate pitpredict_env
python -m pytest tests/ -v

# Code Quality
flake8 src/
mypy src/
```



---

**Status**:  Produktionsbereit für Final Position Predictions  
**Letzte Aktualisierung**: August 2025  
**Maintainer**: [Entwickler-Info]


