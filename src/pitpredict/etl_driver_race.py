"""ETL Skript: Baut eine driver_race_table mit genau einer Zeile pro (race_id, driver).

Ausführung:
    pip install fastf1 pandas numpy scikit-learn pyyaml joblib pyarrow
    python -m src.pitpredict.etl_driver_race

Hinweis: Nur Pre-Race verfügbare Informationen werden als Features verwendet.
"""
from __future__ import annotations

import os
import json
import warnings
from typing import Dict, List

import fastf1
import numpy as np
import pandas as pd
import yaml

# ----------------------------- Konfiguration laden ---------------------------
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.yaml')
with open(CONFIG_PATH, 'r') as f:
    CFG = yaml.safe_load(f)

SEASON = int(CFG['season'])
ROUNDS: List[int] = CFG['rounds']
CACHE_DIR = CFG['cache_dir']
PROCESSED_TABLE = CFG['processed_table']
ARTIFACTS_DIR = CFG['artifacts_dir']
METRICS_DIR = CFG['metrics_dir']
ROLLING_K = int(CFG.get('rolling_k', 5))

os.makedirs(os.path.dirname(PROCESSED_TABLE), exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

fastf1.Cache.enable_cache(CACHE_DIR)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------- Track Meta Mapping ----------------------------
# Minimal beispielhaftes Mapping (kann später erweitert werden)
TRACK_META = {
    'Bahrain Grand Prix': {"pit_loss_s_est": 22.5, "degradation_class": 'hoch', "street_circuit": 0, "overtake_difficulty": 2},
    'Saudi Arabian Grand Prix': {"pit_loss_s_est": 25.0, "degradation_class": 'niedrig', "street_circuit": 1, "overtake_difficulty": 3},
    'Australian Grand Prix': {"pit_loss_s_est": 23.5, "degradation_class": 'mittel', "street_circuit": 0, "overtake_difficulty": 2},
    'Japanese Grand Prix': {"pit_loss_s_est": 20.5, "degradation_class": 'mittel', "street_circuit": 0, "overtake_difficulty": 2},
    'Chinese Grand Prix': {"pit_loss_s_est": 21.5, "degradation_class": 'mittel', "street_circuit": 0, "overtake_difficulty": 2},
    'Miami Grand Prix': {"pit_loss_s_est": 29.0, "degradation_class": 'hoch', "street_circuit": 1, "overtake_difficulty": 3},
    'Emilia Romagna Grand Prix': {"pit_loss_s_est": 24.0, "degradation_class": 'mittel', "street_circuit": 0, "overtake_difficulty": 4},
    'Monaco Grand Prix': {"pit_loss_s_est": 19.5, "degradation_class": 'niedrig', "street_circuit": 1, "overtake_difficulty": 5},
    'Canadian Grand Prix': {"pit_loss_s_est": 18.5, "degradation_class": 'mittel', "street_circuit": 1, "overtake_difficulty": 3},
    'Spanish Grand Prix': {"pit_loss_s_est": 22.0, "degradation_class": 'hoch', "street_circuit": 0, "overtake_difficulty": 2},
    'Austrian Grand Prix': {"pit_loss_s_est": 20.0, "degradation_class": 'mittel', "street_circuit": 0, "overtake_difficulty": 2},
    'British Grand Prix': {"pit_loss_s_est": 21.0, "degradation_class": 'mittel', "street_circuit": 0, "overtake_difficulty": 2},
    'Hungarian Grand Prix': {"pit_loss_s_est": 19.0, "degradation_class": 'hoch', "street_circuit": 0, "overtake_difficulty": 4},
    'Belgian Grand Prix': {"pit_loss_s_est": 21.5, "degradation_class": 'mittel', "street_circuit": 0, "overtake_difficulty": 2},
    'Dutch Grand Prix': {"pit_loss_s_est": 23.0, "degradation_class": 'hoch', "street_circuit": 0, "overtake_difficulty": 3},
    'Italian Grand Prix': {"pit_loss_s_est": 19.0, "degradation_class": 'niedrig', "street_circuit": 0, "overtake_difficulty": 1},
}
DEFAULT_TRACK_META = {"pit_loss_s_est": np.nan, "degradation_class": 'mittel', "street_circuit": 0, "overtake_difficulty": 2}

# ------------------------------- Helper -------------------------------------

def _timedelta_to_ms(td):
    if pd.isna(td):
        return np.nan
    return td.total_seconds() * 1000.0

# -------------------------- Session Ladefunktionen --------------------------

def load_race_result(season: int, rnd: int) -> pd.DataFrame:
    try:
        session = fastf1.get_session(season, rnd, 'R')
        session.load(telemetry=False, weather=False, laps=False)
    except Exception as e:
        print(f"[WARN] Race Session {season} Round {rnd} konnte nicht geladen werden: {e}")
        return pd.DataFrame()
    res = session.results.copy()
    if res is None or res.empty:
        return pd.DataFrame()

    df = pd.DataFrame({
        'season': season,
        'round': rnd,
        'race_id': f"{season}_{rnd:02d}",
        'circuit': session.event['EventName'],
        'total_laps': res['Laps'].max() if 'Laps' in res.columns else np.nan,
        'driver': res['Abbreviation'],
        'driver_number': res['DriverNumber'].astype(str),
        'team': res['TeamName'],
        'grid_position': res['GridPosition'],
        'finish_position': res['Position'],
        'classification_status': res['Status'],
        'points': res['Points'] if 'Points' in res.columns else np.nan,
        'laps_completed': res['Laps'] if 'Laps' in res.columns else np.nan
    })
    # Target
    df['is_dnf'] = (~df['classification_status'].str.contains('Finished', na=False)).astype(int)
    return df

def load_quali(season: int, rnd: int) -> pd.DataFrame:
    for sess_code in ['Q', 'SQ']:
        try:
            session = fastf1.get_session(season, rnd, sess_code)
            session.load(telemetry=False, weather=False, laps=False)
            qual = session.results.copy()
            if qual is None or qual.empty:
                continue
            pole_best = None
            times_cols = ['Q1', 'Q2', 'Q3']
            df = pd.DataFrame({
                'season': season,
                'round': rnd,
                'race_id': f"{season}_{rnd:02d}",
                'driver': qual['Abbreviation'],
                'driver_number': qual['DriverNumber'].astype(str),
                'team': qual['TeamName'],
                'q_pos': qual['Position']
            })
            for c in times_cols:
                df[c + '_ms'] = [ _timedelta_to_ms(t) for t in qual[c] ]
            df['q_best_time_ms'] = df[[c + '_ms' for c in times_cols]].min(axis=1, skipna=True)
            pole_best = df['q_best_time_ms'].min()
            df['q3_reached'] = ~pd.isna(df['Q3_ms'])
            df['q_gap_to_pole_ms'] = df['q_best_time_ms'] - pole_best
            return df
        except Exception as e:
            print(f"[INFO] Quali Typ {sess_code} für Round {rnd} nicht verfügbar: {e}")
            continue
    return pd.DataFrame()

# -------------------------- Rolling Feature Builder -------------------------

def build_rolling_features(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Erzeuge leakage-sichere Rolling-Features (shift(1) vor Aggregation).

    Verwendet groupby().transform statt apply, um Index-Ausrichtungsprobleme zu vermeiden.
    """
    df = df.sort_values(['driver', 'round']).reset_index(drop=True)

    def _roll_mean(series: pd.Series, minp=1):
        return series.shift(1).rolling(k, min_periods=minp).mean()

    def _roll_sum(series: pd.Series, minp=1):
        return series.shift(1).rolling(k, min_periods=minp).sum()

    def _roll_std(series: pd.Series, minp=2):
        return series.shift(1).rolling(k, min_periods=minp).std()

    # Qualifying Form
    if 'q_pos' in df.columns:
        df['q_pos_mean_lastK'] = df.groupby('driver')['q_pos'].transform(_roll_mean)
        df['q_stdev_lastK'] = df.groupby('driver')['q_pos'].transform(_roll_std)
    if 'q3_reached' in df.columns:
        df['q3_rate_lastK'] = df.groupby('driver')['q3_reached'].transform(_roll_mean)
    if 'q_gap_to_pole_ms' in df.columns:
        df['q_best_gap_to_pole_ms_lastK'] = df.groupby('driver')['q_gap_to_pole_ms'].transform(_roll_mean)

    # Reliability
    df['driver_dnf_rate_lastK'] = df.groupby('driver')['is_dnf'].transform(_roll_mean)
    df['team_dnf_rate_lastK'] = df.groupby('team')['is_dnf'].transform(_roll_mean)

    # Track Historie
    df['track_dnf_rate_hist'] = (df.sort_values(['circuit', 'round'])
                                   .groupby('circuit')['is_dnf']
                                   .transform(lambda s: s.shift(1).expanding(min_periods=1).mean()))

    # Form / Pace
    if 'points' in df.columns:
        df['points_lastK'] = df.groupby('driver')['points'].transform(_roll_sum)
    if {'finish_position','grid_position'} <= set(df.columns):
        df['finish_delta_vs_grid'] = df['finish_position'] - df['grid_position']
        df['avg_finish_delta_vs_grid_lastK'] = df.groupby('driver')['finish_delta_vs_grid'].transform(_roll_mean)

    # Team Quali Rank Proxy
    if {'team','round','q_pos'} <= set(df.columns):
        team_mean = (df.groupby(['team','round'])['q_pos']
                       .mean()
                       .reset_index())
        team_mean = team_mean.sort_values(['team','round'])
        team_mean['team_quali_rank_lastK'] = (team_mean.groupby('team')['q_pos']
                                                     .transform(lambda s: s.shift(1).rolling(k, min_periods=1).mean()))
        df = df.merge(team_mean[['team','round','team_quali_rank_lastK']], on=['team','round'], how='left')
        df.rename(columns={'team_quali_rank_lastK':'team_quali_rank_lastK'}, inplace=True)

    return df

# ------------------------------- Main Flow ----------------------------------

def main():
    rows: List[pd.DataFrame] = []
    for rnd in ROUNDS:
        print(f"[INFO] Verarbeite Season={SEASON} Round={rnd}")
        race_df = load_race_result(SEASON, rnd)
        if race_df.empty:
            print(f"[WARN] Keine Race-Daten für Round {rnd}")
            continue

        quali_df = load_quali(SEASON, rnd)
        # Merge Race + Quali
        merged = race_df.merge(quali_df, on=['season', 'round', 'race_id', 'driver', 'driver_number', 'team'], how='left', suffixes=('', '_q'))

        # Track Meta
        track_name = merged['circuit'].iloc[0]
        meta = TRACK_META.get(track_name, DEFAULT_TRACK_META)
        for k, v in meta.items():
            merged[k] = v

        rows.append(merged)

    if not rows:
        print("[ERROR] Keine Daten gesammelt.")
        return

    df_all = pd.concat(rows, ignore_index=True)

    # Rolling Features (leakage-sicher via shift)
    df_all = build_rolling_features(df_all, ROLLING_K)

    # Einheitliche Benennung der im Auftrag geforderten Spalten (5 statt K)
    rename_map = {
        'q_pos_mean_lastK': 'q_pos_mean_last5',
        'q3_rate_lastK': 'q3_rate_last5',
        'q_best_gap_to_pole_ms_lastK': 'q_best_gap_to_pole_ms_last5',
        'q_stdev_lastK': 'q_stdev_last5',
        'driver_dnf_rate_lastK': 'driver_dnf_rate_last5',
        'team_dnf_rate_lastK': 'team_dnf_rate_last5',
        'points_lastK': 'points_last5',
        'avg_finish_delta_vs_grid_lastK': 'avg_finish_delta_vs_grid_last5',
        'team_quali_rank_lastK': 'team_quali_rank_last5'
    }
    df_all.rename(columns=rename_map, inplace=True)

    # Sort & enforce one row per race_id+driver
    df_all.sort_values(['race_id', 'driver'], inplace=True)
    assert not df_all.duplicated(subset=['race_id', 'driver']).any(), "Duplikate in (race_id, driver)"

    # Feature Liste (ohne Target & Ergebnisfelder)
    exclude_cols = {'is_dnf', 'finish_position', 'classification_status', 'laps_completed'}
    feature_cols = [c for c in df_all.columns if c not in exclude_cols]
    feat_path = os.path.join(METRICS_DIR, 'feature_list.json')
    with open(feat_path, 'w') as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Feature-Liste geschrieben: {feat_path}")

    # Speichern
    try:
        df_all.to_parquet(PROCESSED_TABLE, index=False)
        print(f"[INFO] Gespeichert als Parquet: {PROCESSED_TABLE}")
    except Exception as e:
        alt_path = PROCESSED_TABLE.replace('.parquet', '.csv')
        df_all.to_csv(alt_path, index=False)
        print(f"[WARN] Parquet fehlgeschlagen ({e}), CSV gespeichert: {alt_path}")

if __name__ == '__main__':
    main()
