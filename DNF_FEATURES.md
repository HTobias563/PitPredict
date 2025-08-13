# DNF Features Documentation

Das `build_lap_by_lap_dataset.py` Skript wurde erweitert, um DNF (Did Not Finish) Informationen in die Lap-by-Lap Daten einzufügen.

## Neue Spalten

### `is_dnf` (int8)
- **0**: Fahrer hat das Rennen beendet (Finished oder Lapped)
- **1**: Fahrer ist ausgeschieden (Retired)

### `dnf_reason` (category)
- Der Grund für das Ausscheiden/Ergebnis:
  - `"Finished"`: Rennen erfolgreich beendet
  - `"Lapped"`: Überrundet, aber im Rennen geblieben
  - `"Retired"`: Aus dem Rennen ausgeschieden
  - `"Unknown"`: Status nicht verfügbar

### `classification_status` (category)
- Gleiche Werte wie `dnf_reason`, zusätzliche Spalte für Kompatibilität

### `race_finished` (int8)
- **1**: Fahrer hat das Rennen beendet (Finished oder Lapped)
- **0**: Fahrer ist ausgeschieden (Retired)

## Verwendung

```python
import pandas as pd

# Lade Lap-by-Lap Daten mit DNF Informationen
df = pd.read_parquet("data/laps/season=2024/round=08_laps.parquet")

# Finde alle DNFs
dnfs = df[df['is_dnf'] == 1]['driver'].unique()
print(f"DNFs in Monaco 2024: {dnfs}")

# Analysiere DNF-Gründe
dnf_summary = df[df['is_dnf'] == 1][['driver', 'dnf_reason']].drop_duplicates()
print(dnf_summary)

# Vergleiche Performance von Fahrern vor dem Ausscheiden
dnf_laps = df[df['driver'].isin(dnfs)]
```

## Beispiel Ausgabe (Monaco 2024)

```
DNFs: ['OCO', 'PER', 'HUL', 'MAG']

driver dnf_reason
OCO    Retired
PER    Retired  
HUL    Retired
MAG    Retired
```

## Anwendung für Machine Learning

Diese Features sind besonders nützlich für:

1. **Vorhersage von Ausfällen**: Modelle können Patterns erkennen, die zu DNFs führen
2. **Strategieanalyse**: Verstehen, wie Reifenwechsel und Pace DNF-Risiko beeinflussen
3. **Zuverlässigkeitsanalyse**: Team- und Fahrer-spezifische Ausfallraten
4. **Feature Engineering**: DNF-Risiko als zusätzlicher Faktor für Positionsvorhersagen

## Datenqualität

- DNF-Informationen werden aus den offiziellen F1-Ergebnissen (FastF1) extrahiert
- Jede Runde eines Fahrers bekommt den finalen Status des Fahrers zugewiesen
- Falls keine Ergebnisse verfügbar sind, werden Standardwerte gesetzt
