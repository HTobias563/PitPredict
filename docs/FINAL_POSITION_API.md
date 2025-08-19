# Final Position Model - API Documentation

## Klassen-Übersicht

### `FinalPositionPredictor`

Haupt-Klasse für Final Position Predictions mit DNF- und Pit Stop-Integration.

#### Constructor
```python
FinalPositionPredictor(config: FinalPositionPredictionConfig = None)
```

**Parameter:**
- `config`: Optional. Konfigurationsobjekt mit Modell-Parametern

**Attributes:**
- `model`: Trainiertes scikit-learn Pipeline-Objekt
- `feature_names`: Liste der verwendeten Feature-Namen
- `feature_engineer`: Instance der Feature-Engineering-Klasse
- `dnf_model`: Geladenes DNF-Modell (optional)
- `pitstop_model`: Geladenes Pit Stop-Modell (optional)

#### Methoden

##### `load_auxiliary_models()`
Lädt DNF- und Pit Stop-Modelle für Feature-Integration.

```python
predictor.load_auxiliary_models()
```

**Behavior:**
- Lädt DNF-Modell aus `artifacts/models/dnf_pipeline_calibrated.pkl`
- Pit Stop-Modell derzeit deaktiviert (Kompatibilitätsprobleme)
- Graceful Fallback bei Fehlern

##### `prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]`
Bereitet Features für Training/Prediction vor.

```python
feature_df, feature_names = predictor.prepare_features(race_data)
```

**Parameter:**
- `df`: Input DataFrame mit Race-Daten

**Returns:**
- `feature_df`: DataFrame mit engineerten Features
- `feature_names`: Liste der Feature-Namen

**Features Engineering:**
- 42 engineerte Features aus 9 Kategorien
- Numerische und kategorische Preprocessing
- DNF-Risiko-Integration
- Historical Performance Calculations

##### `train(df: pd.DataFrame) -> Dict[str, Any]`
Trainiert das Final Position Prediction Modell.

```python
results = predictor.train(train_df)
```

**Parameter:**
- `df`: Training DataFrame mit Labels

**Returns:**
Dictionary mit Trainings-Ergebnissen:
```python
{
    'fold_scores': [{'fold': 1, 'mae': 2.37, 'rmse': 3.49, ...}],
    'overall_mae': 2.12,
    'overall_rmse': 3.17,
    'overall_r2': 0.829,
    'overall_podium_accuracy': 0.967,
    'overall_points_accuracy': 0.897,
    'feature_importance': {'start_performance_index': 0.2413, ...},
    'n_samples': 399,
    'n_dnfs': 163,
    'dnf_rate': 0.409,
    'feature_names': ['grid_position', ...]
}
```

**Training Process:**
1. Lädt auxiliary models (DNF/Pit Stop)
2. Feature Engineering Pipeline
3. 5-Fold GroupKFold Cross-Validation
4. Final Model Training auf allen Daten
5. Feature Importance Berechnung

##### `predict_final_positions(race_data: pd.DataFrame) -> pd.DataFrame`
Vorhersage der finalen Positionen für Race Data.

```python
predictions = predictor.predict_final_positions(race_df)
```

**Parameter:**
- `race_data`: DataFrame mit Pre-Race Daten

**Returns:**
DataFrame mit Predictions:
```python
{
    'race_id': '2024_21',
    'driver': 'VER',
    'team': 'Red Bull Racing',
    'grid_position': 17,
    'predicted_final_position': 8.2,
    'predicted_position_rounded': 8,
    'dnf_risk': 0.15,
    'prediction_confidence': 0.73
}
```

**Output sortiert nach:** `predicted_final_position` (aufsteigend)

##### `save_model(filepath: str, metadata: Dict[str, Any] = None)`
Speichert das trainierte Modell persistent.

```python
predictor.save_model('artifacts/models/final_position_predictor.pkl', metadata)
```

**Gespeicherte Objekte:**
- Trainiertes Pipeline-Modell
- Feature Names und Config
- Feature Engineer Instance
- Optional: Zusätzliche Metadaten

##### `load_model(filepath: str) -> Dict[str, Any]`
Lädt ein gespeichertes Modell.

```python
metadata = predictor.load_model('artifacts/models/final_position_predictor.pkl')
```

**Returns:**
- Metadata Dictionary aus dem gespeicherten Modell

---

### `FinalPositionFeatureEngineer`

Feature Engineering Pipeline für Position Prediction.

#### Constructor
```python
FinalPositionFeatureEngineer(config: FinalPositionPredictionConfig)
```

#### Hauptmethode

##### `engineer_features(df: pd.DataFrame, dnf_model=None, pitstop_model=None) -> pd.DataFrame`

Erstellt alle 42 Features für das Final Position Modell.

**Feature Kategorien:**

1. **Grid Position Features** (4)
   - `grid_position_norm`: Normalisiert 0-1
   - `front_row_start`: P1-P2 Flag
   - `top5_start`: P1-P5 Flag  
   - `back_of_grid`: P15+ Flag

2. **Qualifying Performance** (3)
   - `q_gap_to_pole_s`: Gap in Sekunden
   - `q3_participant`: Q3 erreicht
   - `q_pos_vs_teammate`: Relative Position

3. **Historical Performance** (1)
   - `historical_performance_score`: Composite Score

4. **Track-Specific** (4)
   - `overtaking_difficulty`: Strecken-Charakteristik
   - `pit_loss_impact`: Normalisierter Pit Loss
   - `is_street_circuit`: Street Circuit Flag
   - `track_reliability_risk`: DNF-Risiko historisch

5. **Team Performance** (2)
   - `team_recent_form`: Quali-Rank normalisiert
   - `team_reliability`: Zuverlässigkeits-Index

6. **DNF Integration** (1)
   - `dnf_risk_prediction`: DNF-Wahrscheinlichkeit

7. **Pit Stop Strategy** (2)
   - `expected_pitstops`: Erwartete Anzahl
   - `pit_strategy_risk`: Kombiniertes Risiko

8. **Composite Features** (3)
   - `start_performance_index`: Haupt-Indikator
   - `expected_position_range`: Positionsabweichung
   - `reliability_adjusted_performance`: Risk-adjustiert

---

### `FinalPositionPredictionConfig`

Konfigurationsdatenklasse für Modell-Parameter.

```python
@dataclass
class FinalPositionPredictionConfig:
    dnf_position_penalty: int = 21      # DNF Position
    position_bins: List[int] = None     # Optional classification bins
    weight_dnf_risk: float = 0.3        # DNF-Gewichtung
    weight_pitstop_strategy: float = 0.2 # Pit Stop-Gewichtung  
    weight_grid_position: float = 0.4   # Grid-Gewichtung
    weight_historical: float = 0.1      # Historical-Gewichtung
```

**Default position_bins:** `[1, 3, 10, 15, 20]` (Podium, Points, Midfield, Backfield)

---

### `RacePrediction`

Datenklasse für einzelne Fahrer-Predictions.

```python
@dataclass
class RacePrediction:
    driver: str                              # Fahrer-Code (3-Letter)
    predicted_position: float                # Predicted Position (float)
    dnf_probability: float                  # DNF-Wahrscheinlichkeit
    pitstop_risk_avg: float                 # Pit Stop-Risiko
    confidence_interval: Tuple[float, float] # 95% Confidence Interval
    contributing_factors: Dict[str, float]   # Feature Contributions
```

---

## Workflow-Funktionen

### `train_final_position_model() -> Tuple[FinalPositionPredictor, Dict]`

Kompletter Training-Workflow.

**Process:**
1. Lädt Driver-Race Tabelle
2. Split in Training/Holdout
3. Data Cleaning
4. Model Training
5. Model & Metrics Persistence
6. Optional Holdout Testing

**Outputs:**
- Model: `artifacts/models/final_position_predictor.pkl`
- Metriken: `artifacts/metrics/final_position_cv_report.json`

### `test_final_position_model(predictor=None, holdout_df=None) -> Dict`

Holdout-Test Workflow.

**Metriken:**
- MAE, RMSE, R²
- Podium/Points Accuracy
- DNF Detection Rate
- Best/Worst Predictions Analysis

**Output:**
- Holdout Report: `artifacts/metrics/final_position_holdout_report.json`

### `predict_race_positions(race_id: str) -> pd.DataFrame`

Single-Race Prediction Workflow.

**Example:**
```python
results = predict_race_positions('2024_21')
# Returns sorted predictions for Las Vegas GP
```

**Features:**
- Automatic Model Loading
- Race Data Extraction  
- Formatted Output
- Optional Comparison with Actual Results
- CSV Export

---

## CLI Interface

### Training
```bash
python -m src.pitpredict.models.final_position_predict --train
```

### Testing  
```bash
python -m src.pitpredict.models.final_position_predict --test
```

### Prediction
```bash
python -m src.pitpredict.models.final_position_predict --predict --race_id 2024_21
```

---

## Error Handling

### Common Exceptions

#### `FileNotFoundError`
- Model file nicht gefunden
- Race data nicht verfügbar
- Config file fehlt

#### `ValueError`
- Ungültige race_id Format
- Leere Race Data
- Model nicht trainiert

#### Integration Failures
- DNF Model loading: Graceful fallback auf historical rates
- Pit Stop Model: Derzeit deaktiviert, logs warning
- Feature Engineering: Robust gegen missing features

### Best Practices

1. **Model Loading**: Immer prüfen ob Model existiert
2. **Data Validation**: Race data vor prediction validieren
3. **Feature Coverage**: Mindestens 60% der erwarteten Features
4. **Fallback Strategies**: Historical rates bei Model-Fehlern

---

## Performance Optimierung

### Batch Predictions
```python
# Für mehrere Rennen
race_ids = ['2024_21', '2024_22', '2024_23']
all_predictions = []

for race_id in race_ids:
    pred = predict_race_positions(race_id)
    all_predictions.append(pred)

combined = pd.concat(all_predictions)
```

### Memory Management
- Model wird nur einmal geladen
- Feature Engineering arbeitet in-place wo möglich
- Pandas Memory Optimierung mit kategorischen Dtypes

### Caching
- DNF Model wird cached nach erstem Load
- Feature Engineering Results können gecached werden
- Preprocessor kann für Batch-Processing wiederverwendet werden

---

## Integration mit anderen Modellen

### DNF Model Pipeline
```python
# DNF-Risiko wird automatisch integriert
predictor.load_auxiliary_models()
# dnf_risk_prediction Feature wird erstellt
```

### Pit Stop Model Integration (Geplant)
```python
# Nach Reparatur der Kompatibilität
pitstop_features = ['pit_window_optimal', 'tire_strategy_risk']
# Integration in engineer_features()
```

### Ensemble Prediction (Future)
```python
# Kombination mehrerer Modelle
final_pred = (
    0.6 * gradient_boost_pred +
    0.3 * random_forest_pred +
    0.1 * linear_regression_pred
)
```

---

## Monitoring und Debugging

### Feature Importance Monitoring
```python
# Nach jedem Training
importance = results['feature_importance']
top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
print("Top 5 Features:", top_features[:5])
```

### Prediction Quality Analysis
```python
# Prediction Confidence Distribution
confidence_dist = predictions['prediction_confidence'].describe()

# Error Analysis by Driver/Team
error_by_driver = abs(predictions['actual'] - predictions['predicted']).groupby('driver').mean()
```

### Cross-Validation Stability
```python
# Fold-by-Fold MAE Variance
mae_variance = np.var([fold['mae'] for fold in results['fold_scores']])
print(f"MAE Stability: {mae_variance:.4f}")
```
