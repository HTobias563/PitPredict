# PitPredict

PitPredict ist ein modulares Machine-Learning-System zur Vorhersage von Formel‑1‑Rennergebnissen. Der Fokus liegt auf reproduzierbaren Trainingspipelines, klarer Modell-Integration und nachvollziehbaren Metriken.

## Inhalt

- Überblick
- Features
- Projektstruktur
- Installation
- Nutzung
- Konfiguration
- Datenquellen
- Qualität & Tests
- Dokumentation
- Roadmap
- Beitragen
- Lizenz

## Überblick

PitPredict besteht aus drei Modulen:

1) Final Position Model: Vorhersage der Endplatzierung vor dem Rennen
2) DNF Model: Ausfallwahrscheinlichkeit (Did Not Finish)
3) Pit Stop Model: Strategievorhersage (derzeit in Überarbeitung)

## Features

- Integrierte Modellkette (DNF + Pit Stop als Features für Final Position)
- Reproduzierbare Trainingsergebnisse via Konfiguration
- Ergebnisartefakte inklusive Metrik-Reports
- Tests für Training, Persistenz und Vorhersagen

## Projektstruktur

```
PitPredict/
├── src/pitpredict/                # Python-Paket (Modelle, Pipeline)
├── data/                          # Trainings- und Feature-Daten
├── artifacts/                     # Modelle, Metriken, Vorhersagen
├── docs/                          # Technische Dokumentation
├── tests/                         # Test-Suite
└── app/                           # App-Prototypen
```

## Installation

### Voraussetzungen

- Python 3.10
- macOS/Linux/Windows

### Lokales Setup

1. Repository klonen
2. Virtuelles Environment erstellen und aktivieren
3. Abhängigkeiten installieren

Beispiel (venv):

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Nutzung

### Training

Final Position Model (Hauptmodell):

```
python -m src.pitpredict.models.final_position_predict --train
```

DNF Model:

```
python -m src.pitpredict.models.train_dnf
```

Pit Stop Model (nach Reparatur):

```
python -m src.pitpredict.models.pit_predict --train
```

### Vorhersagen

Vergangenheitsdaten (2024):

```
python predict_example.py --race_id 2024_21
```

Future Races (2025+):

```
python predict_future_race.py \
  --race_name "Monaco GP 2025" \
  --track_type monaco \
  --season 2025 \
  --grid_positions "VER:1,NOR:2,LEC:3,RUS:4,HAM:5"
```

### Python API

```
from src.pitpredict.models.final_position_predict import FinalPositionPredictor
from src.pitpredict.models.future_position_predict import FutureRacePredictor

predictor = FinalPositionPredictor()
predictor.load_model('artifacts/models/final_position_predictor.pkl')
results = predictor.predict_race('2024_21')

future_predictor = FutureRacePredictor()
grid_positions = ['VER:1', 'NOR:2', 'LEC:3', 'RUS:4', 'HAM:5']
results = future_predictor.predict_future_race(
  "Netherlands GP 2025",
  grid_positions,
  "netherlands",
  2025
)
```

## Konfiguration

Zentrale Einstellungen liegen in [config.yaml](config.yaml):

```
season: 2024
processed_table: data/season=2024/driver_race_table.parquet
models_dir: artifacts/models
metrics_dir: artifacts/metrics
holdout_rounds: [21, 22, 23, 24]
```

## Datenquellen

- driver_race_table.parquet: Haupt-Dataset mit Race-Ergebnissen
- Lap Data: Lap-by-Lap-Daten für Pit-Stop-Modelle
- FastF1 Cache: lokaler Cache für schnellere Runs

## Qualität & Tests

```
python -m pytest tests/ -v
```

## Dokumentation

- [Final Position Model](docs/FINAL_POSITION_MODEL.md)
- [API Referenz](docs/FINAL_POSITION_API.md)
- [DNF Features](docs/DNF_FEATURES.md)
- [Pit Stop Model](docs/PITSTOP_MODEL_SUMMARY.md)

## Roadmap

- Pit Stop Model Fixes (OneHotEncoder-Kompatibilität)
- Wetter-Features
- Live Race Updates
- Ensemble-Ansätze

## Beitragen

Siehe [CONTRIBUTING.md](CONTRIBUTING.md).

## Lizenz

Siehe [LICENSE](LICENSE).

```
