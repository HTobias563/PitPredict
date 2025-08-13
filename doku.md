# PitPredict
Modell für Boxenstopp-Strategien in der F1


## Ziel 
das ziel meines projektes ist es die reifenstrategie eines live rennens vorherzusagen.
was genau wird dabei vorhergesagt idealerweise.
1. wie viele Pit stops macht ein fahrer
2. wann werden die pit stops gemacht
3. welche reifen werden dabei verwendet
4. welche Endplaztierung wird dadurch erreicht 

### 1. Modell ready Dataset 

welche daten brauche ich und welche sind verfügbar ? 
ein überblick über die verfügbaren daten findet man in data_overview

der ideale data frame für das erst modell:

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

plan: 
Frame the problem + targets
Load & clean the lap data
Build per-driver-per-race “strategy tables” (stints, stops, compounds)
Create features (pace, degradation, pit loss, weather, circuit, team form, etc.)
Define labels for three tasks: total stops, compounds used, final position
Baselines (simple rules + tree models)
Cross-validation that avoids leakage (by race/season) + metrics
Hyperparam tuning + feature importance
(Optional but powerful) Pit-stop hazard model + Monte Carlo simulation
Package inference for “next race” use


Best Practice: zweistufig vorgehen
Teilmodelle für Strategie:
(A) Pit-Stop-Anzahl (Klassifikation)
(B) Erster Stopp (Regression)
(C) Strategie-Pattern (z. B. SOFT->MEDIUM->HARD) (Klassifikation)
Endmodell für Finish Position, das die vorhergesagten A+B+C als Features nutzt – nicht die echten (sonst Leakage!).