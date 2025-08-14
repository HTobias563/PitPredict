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

