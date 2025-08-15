Hier ist deine Doku in einem kompakten, professionellen Markdown-Format – bereit als README:

# PitPredict

Vorhersagemodell für Boxenstopp-Strategien in der Formel 1

## Zielsetzung

PitPredict sagt für ein **laufendes oder bevorstehendes Rennen** die Reifenstrategie und deren Auswirkung auf das Ergebnis voraus:

1. **Anzahl der Boxenstopps** pro Fahrer (Klassifikation)
2. **Zeitpunkt des/der Stopps** (Regression / Hazard)
3. **Reifenfolge (Compound-Pattern)** (Klassifikation)
4. **Zielplatzierung / Podium** unter Verwendung der vorhergesagten Strategie (Klassifikation/Regression)

> Best Practice: zweistufiges Setup
> **A)** Pit-Stop-Anzahl → **B)** Zeitpunkt des ersten Stopps → **C)** Strategie-Pattern
> **D)** Endmodell für Finish-Position nutzt **die Vorhersagen** aus A+B+C als Features (keine Ground-Truths, um Leakage zu vermeiden).

---

## Daten

Ein Überblick der verfügbaren Quellen/Features steht in **`data_overview`**.
Für das erste Modell wird ein per-Fahrer-pro-Rennen-Frame mit folgenden Spalten erwartet:

```python
dtypes = {
    "season": "int16",
    "round": "int8",
    "race_id": "category",
    "circuit": "category",
    "total_laps": "int16",
    "driver": "category",
    "team": "category",
    "driver_number": "int16",
    "grid": "int8",
    "q1_s": "float32",
    "q2_s": "float32",
    "q3_s": "float32",
    "delta_to_pole_s": "float32",
    "penalty_back_of_grid": "int8",
    "airtemp_mean": "float32",
    "tracktemp_mean": "float32",
    "humidity_mean": "float32",
    "windspeed_mean": "float32",
    "rainflag": "int8",
    "pit_loss_estimate_s": "float32",
    "scvsc_rate_3y": "float32",
    "overtake_index": "float32",
    "team_pace_index_3r": "float32",
    "driver_points_form_3r": "float32",
    "n_stops": "int8",
    "finish_pos": "int8",
    "podium": "int8"
}
```

**Kernschritte zur Datenerstellung**

* Lappendaten laden & bereinigen
* **Strategietabellen** je Fahrer/Rennen bauen (Stints, Stopps, Compounds)
* Feature-Engineering:

  * **Pace/Performance:** delta\_to\_pole\_s, team\_pace\_index\_3r, driver\_points\_form\_3r
  * **Strecke/Überholen:** circuit, overtake\_index, pit\_loss\_estimate\_s, scvsc\_rate\_3y
  * **Startlage/Strafen:** grid, penalty\_back\_of\_grid
  * **Wetter:** airtemp\_mean, tracktemp\_mean, humidity\_mean, windspeed\_mean, rainflag

---

## Labels & Aufgaben

* **A: `n_stops`** (Klassifikation)
* **B: Erster Stopp (Runde/Sekunde)** (Regression oder Hazard/Fine-Timing)
* **C: Strategie-Pattern** (Sequenzklassifikation, z. B. `SOFT→MEDIUM→HARD`)
* **D: `finish_pos` / `podium`** (Klassifikation/Regression) – nutzt Vorhersagen aus A–C als Eingaben

---

## Modellierungsplan

1. **Problem-Frame & Ziele festlegen**
2. **Features bauen** (s. oben) + **Zielgrößen definieren**
3. **Baselines**

   * Heuristiken (z. B. „1-Stop, wenn pit\_loss hoch + geringe Degradation“)
   * Einfache Bäume/Logit/GBMs als Start
4. **Cross-Validation ohne Leakage**

   * Grouped CV **nach Rennen/Season** (ganze Rennen in Train/Test trennen)
   * Metriken:

     * A/C: Accuracy, F1-Macro
     * B: MAE/RMSE oder **Concordance**/Time-to-Event-Metriken bei Hazard
     * D: MAE (Platz), Accuracy\@Podium, ggf. **Ranking-Metriken** (Spearman)
5. **Hyperparam-Tuning & Feature-Importance**
6. **(Optional, stark)**: **Pit-Stop-Hazardmodell** + **Monte-Carlo-Simulation** für Rennverlauf
7. **Inference-Packaging** für „**Next Race**“:

   * Pre-Race-Features generieren
   * A→B→C vorhersagen
   * D (Finish) mit A–C-Preds ausgeben

---

## DNF-Handling

* **Zieldefinition:** DNF als eigenes Label (binär) oder als **Zensierung** bei Zeit-/Runden-Modellen behandeln.
* **Training:**

  * Variante 1: Separates **DNF-Modul** (binäre Klassifikation); DNF-Wahrscheinlichkeit in D einfließen lassen.
  * Variante 2: **Survival/Hazard**-Ansatz, der DNF-Ereignisse im Zeitverlauf modelliert.
* **Evaluation:** Metriken getrennt für „Finishers“ und „DNFs“ reporten; Robustheit gegen Ausreißer prüfen.

---

## Qualitäts- & Leakage-Kontrollen

* Nur **vorher verfügbare** Informationen für A–C nutzen (keine post-hoc Renninfos).
* Für D ausschließlich **A–C Vorhersagen** (keine Ground-Truth-Strategie).
* Strikte **Race-Level-Splits** in CV; Feature-Drift zwischen Saisons beobachten.

---

## Roadmap (Kurz)

* [ ] Data-Pipelines & Strategy-Tables
* [ ] Baselines A/B/C + CV
* [ ] Endmodell D mit A–C-Preds
* [ ] Hazard + Monte-Carlo (optional)
* [ ] Packaging „Next Race“ (CLI/Service)
* [ ] Reports: Metriken, Feature-Importances, Ablation

---

## Reproduzierbarkeit

* Seeds fixieren, Versionen pinnen, Artefakte loggen (Modelle, Metriken, Konfiguration).
* Einheitliche I/O-Schemata und Validierungschecks (Dtypes, Nullraten, Bereichsprüfungen).




# zum training des models ( predict DNFs)
0. welche daten sind vor dem start bekannt ? ewe

# Data Dictionary – F1 (Pre-race Fokus)

| Spaltenname             | Bedeutung                                                                 | Einheit/Format               | Zeitpunkt der Verfügbarkeit           |
|-------------------------|---------------------------------------------------------------------------|------------------------------|---------------------------------------|
| season                  | Saisonjahr                                                                | Integer                      | pre-race                              |
| round                   | Laufnummer innerhalb der Saison                                           | Integer                      | pre-race                              |
| race_id                 | Eindeutige ID des Rennens                                                 | String/Integer               | pre-race                              |
| circuit                 | Streckenname/Kurzname                                                     | String                       | pre-race                              |
| total_laps              | Geplante Rundenzahl                                                        | Integer                      | pre-race *(kann selten abweichen)*    |
| driver                  | Fahrerkürzel/Name                                                         | String                       | pre-race                              |
| team                    | Teamname                                                                   | String                       | pre-race                              |
| driver_number           | Startnummer                                                                | Integer/String               | pre-race                              |
| lap_number              | Rundenindex (1…N)                                                          | Integer                      | in-race                               |
| lap_time_s              | Rundenzeit                                                                 | Sekunden (float)             | in-race                               |
| sector1_time_s          | Sektor 1 Zeit                                                              | Sekunden (float)             | in-race                               |
| sector2_time_s          | Sektor 2 Zeit                                                              | Sekunden (float)             | in-race                               |
| sector3_time_s          | Sektor 3 Zeit                                                              | Sekunden (float)             | in-race                               |
| position                | Position am Ende der Runde                                                 | Integer                      | in-race                               |
| compound                | Reifenmischung der Runde (z. B. S/M/H/INT/WET)                            | Kategorie/String             | in-race *(Start-Compound teils pre)*  |
| tyre_life               | Reifendistanz seit Montage                                                 | Runden (Integer)             | in-race                               |
| is_pit_out_lap          | Flag: Runde direkt nach Boxenstopp                                         | Bool (0/1)                   | in-race                               |
| is_pit_in_lap           | Flag: Runde mit Boxenstopp (Einfahrt)                                      | Bool (0/1)                   | in-race                               |
| fresh_tyre              | Flag: frischer Reifensatz in dieser Runde                                  | Bool (0/1)                   | in-race                               |
| track_status            | Streckenstatus (grün/SC/VSC/gelb … je Code)                                | Kategorie/String             | in-race                               |
| air_temp                | Lufttemperatur                                                             | °C (float)                   | in-race *(pre-race Prognose möglich)* |
| track_temp              | Streckentemperatur                                                         | °C (float)                   | in-race *(pre-race Prognose möglich)* |
| wind_speed              | Windgeschwindigkeit                                                        | m/s oder km/h (float)        | in-race *(pre-race Prognose möglich)* |
| humidity                | Luftfeuchtigkeit                                                           | Prozent % (float)            | in-race *(pre-race Prognose möglich)* |
| is_dnf                  | Flag: Did Not Finish                                                       | Bool (0/1)                   | post-race                             |
| dnf_reason              | Grund für DNF (Unfall, Technik, …)                                         | Kategorie/String             | post-race                             |
| classification_status   | Offizieller Klassifizierungsstatus (Finished, +1 Lap, DNF, DSQ …)          | Kategorie/String             | post-race                             |
| race_finished           | Flag: Fahrer im Ziel klassifiziert (true/false)                            | Bool (0/1)                   | post-race                             |

erste frage 
wie hoch ist die DNF warscheinlichkeit vor dem rennen ? 

new data weil man nur die vor dem rennen nehmen darf und die bisherigen daten sind runden zeiten 
# Pre-Race DNF – Datenliste (für ein gutes Modell)

> Nur **pre-race** verfügbare Infos verwenden; Rolling-Features immer **bis zum VORHERIGEN Rennen** berechnen (kein Leakage).

## 1) Schlüssel & Meta (Pflicht)
- `season`
- `round`
- `race_id`
- `circuit`
- `total_laps`
- `driver_id` / `driver`
- `driver_number`
- `team`

## 2) Startaufstellung & Streckenkontext (Pflicht)
- `grid_position`  *(offizielle Startposition inkl. Strafen)*
- `pit_loss_s_est`  *(Proxy: Boxenzeitverlust dieser Strecke)*
- `degradation_class` *(niedrig/mittel/hoch; Reifenverschleiß-Kategorie)*
- `street_circuit` *(0/1)*
- `overtake_difficulty` *(ordinaler Proxy oder Kategorie)*

## 3) Rolling-Features aus vergangenen Rennen (stark, pre-race)
**Qualifying-Form**
- `q_pos_mean_last5`  *(Ø Quali-Position)*
- `q3_rate_last5`  *(Anteil Q3-Teilnahmen)*
- `q_best_gap_to_pole_ms_last5`  *(Bestzeit-Gap zur Pole, ms)*
- `q_teammate_delta_ms_last5`  *(Ø Quali-Delta zum Teamkollegen, ms)*
- `q_stdev_last5`  *(Streuungsmaß der Quali-Position)*

**Reliability (DNF-Risiko)**
- `driver_dnf_rate_last5`
- `team_dnf_rate_last5`
- `track_dnf_rate_hist`  *(historische DNF-Quote auf diesem Kurs, mehrere Jahre)*

**Form/Pace-Proxies**
- `points_last5`  *(Summe Punkte)*
- `avg_finish_delta_vs_grid_last5`  *(Ø (Finish − Grid); negativ = Plätze gewonnen)*
- `team_quali_rank_last5`  *(Team-Pace-Proxy aus Quali)*

## 4) Optionale Pre-Race-Signale (nur wenn vorab bekannt)
- `forecast_air_temp`, `forecast_wind_speed`, `forecast_humidity`  *(Vorhersagewerte, nicht In-Race-Messungen)*
- `penalties_on_grid`  *(Anzahl/Schwere, die die Startaufstellung betreffen)*
- `start_tyre_compound`  *(falls zuverlässig vor dem Start bekannt)*

## 5) Target (für Training, nicht als Feature)
- `is_dnf` ∈ {0,1}
- *(optional für Analyse)* `dnf_reason`  *(Mechanik/Unfall etc.)*

---

## Minimal-MVP (wenn du schnell starten willst)
- **Meta/Keys:** `season, round, race_id, circuit, total_laps, driver, driver_number, team`
- **Start:** `grid_position`
- **Rolling Reliability:** `driver_dnf_rate_last5, team_dnf_rate_last5, track_dnf_rate_hist`
- **Rolling Quali:** `q_pos_mean_last5, q3_rate_last5`
- **Target:** `is_dnf`

---

## Ausschluss (nicht verwenden für Pre-Race)
- Runden-/Sektorzeiten, `lap_number`, `position` (in-race)
- `compound` je Runde, `tyre_life`, `is_pit_in_lap`, `is_pit_out_lap`
- `track_status` (SC/VSC/gelb)
- gemessene `air_temp`, `track_temp`, `wind_speed`, `humidity` aus dem Rennen
