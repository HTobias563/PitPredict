# Beitragen zu PitPredict

Danke für dein Interesse an PitPredict. Dieses Dokument beschreibt den bevorzugten Prozess für Beiträge.

## Voraussetzungen

- Python 3.10
- Ein virtuelles Environment (venv oder conda)

## Lokales Setup

1. Forke das Repository und klone deinen Fork.
2. Erstelle ein virtuelles Environment und installiere die Abhängigkeiten:

   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

## Entwicklung

- Quellcode liegt unter src/pitpredict
- Tests liegen unter tests
- Konfigurationen liegen in config.yaml

## Code-Qualität

Bitte führe vor dem Pull Request die Tests aus:

python -m pytest tests/ -v

Optional (falls lokal verfügbar):

flake8 src/
mypy src/

## Pull Request Richtlinien

- Beschreibe die Motivation und den Lösungsansatz
- Füge Tests hinzu oder aktualisiere bestehende Tests
- Halte Änderungen klein und fokussiert

## Issues

Für Bugs oder Feature-Wünsche bitte ein Issue erstellen und den gewünschten Nutzen klar beschreiben.
