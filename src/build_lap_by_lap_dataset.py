import argparse
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import fastf1
except Exception as e:
    print("FastF1 ist nicht installiert. Bitte zuerst `pip install fastf1` ausführen.", file=sys.stderr)
    raise

# ------------------------- Helper: Dtypes für LAP-BY-LAP Data -------------------------

DTYPES = {
    "season": "int16",
    "round": "int8", 
    "race_id": "category",
    "circuit": "category",
    "total_laps": "int16",
    "driver": "category",
    "team": "category",
    "driver_number": "int16",
    "lap_number": "int16",
    "lap_time_s": "float32",
    "sector1_time_s": "float32",
    "sector2_time_s": "float32", 
    "sector3_time_s": "float32",
    "position": "int8",
    "compound": "category",
    "tyre_life": "int8",
    "is_pit_out_lap": "int8",
    "is_pit_in_lap": "int8",
    "fresh_tyre": "int8",
    "track_status": "int8",
    "air_temp": "float32",
    "track_temp": "float32",
    "wind_speed": "float32",
    "humidity": "float32",
    "is_dnf": "int8",
    "dnf_reason": "category",
    "classification_status": "category",
    "race_finished": "int8"
}

# ------------------------- Feature Builder -------------------------

def _td_to_sec(series):
    """Timedelta -> Sekunden (float).""" 
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()

def build_lap_by_lap_features(session, laps_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Create lap-by-lap features (one row per lap per driver)."""
    
    # Basic session info
    season = int(getattr(session.event, "year", session.event["EventDate"].year))
    round_no = int(session.event.get("RoundNumber", getattr(session.event, "RoundNumber", 0)))
    circuit = str(session.event.get("EventName", getattr(session.event, "EventName", "Unknown")))
    total_laps = int(getattr(session, "total_laps", 0) or 0)
    
    if laps_df is None or laps_df.empty:
        return pd.DataFrame()
    
    # Start with laps data
    df = laps_df.copy()
    
    # Rename columns to match our schema
    df = df.rename(columns={
        "Driver": "driver",
        "LapNumber": "lap_number", 
        "Position": "position",
        "Compound": "compound",
        "TyreLife": "tyre_life",
        "TrackStatus": "track_status"
    })
    
    # Convert times to seconds
    df["lap_time_s"] = _td_to_sec(df.get("LapTime"))
    df["sector1_time_s"] = _td_to_sec(df.get("Sector1Time"))
    df["sector2_time_s"] = _td_to_sec(df.get("Sector2Time")) 
    df["sector3_time_s"] = _td_to_sec(df.get("Sector3Time"))
    
    # Pit stop indicators
    df["is_pit_out_lap"] = df["PitOutTime"].notna().astype("int8")
    df["is_pit_in_lap"] = df["PitInTime"].notna().astype("int8")
    
    # Fresh tyre indicator (tyre_life == 1)
    df["fresh_tyre"] = (df["tyre_life"] == 1).astype("int8")
    
    # Add team info from results
    if hasattr(session, 'results') and session.results is not None:
        team_mapping = session.results.set_index("Abbreviation")["TeamName"].to_dict()
        df["team"] = df["driver"].map(team_mapping)
        
        driver_num_mapping = session.results.set_index("Abbreviation")["DriverNumber"].to_dict()
        df["driver_number"] = df["driver"].map(driver_num_mapping)
        
        # Add DNF information from results
        status_mapping = session.results.set_index("Abbreviation")["Status"].to_dict()
        
        # Map each driver's final race status to all their laps
        df["classification_status"] = df["driver"].map(status_mapping)
        df["dnf_reason"] = df["driver"].map(status_mapping)
        
        # Create binary DNF flag (1 if Retired, 0 if Finished or Lapped)
        df["is_dnf"] = (df["classification_status"] == "Retired").astype("int8")
        df["race_finished"] = df["classification_status"].isin(["Finished", "Lapped"]).astype("int8")
    else:
        # Initialize DNF columns with default values if no results available
        df["is_dnf"] = 0
        df["dnf_reason"] = "Unknown"
        df["classification_status"] = "Unknown"
        df["race_finished"] = 1  # Assume finished if no status available
    
    # Add weather data (match by time)
    if weather_df is not None and not weather_df.empty:
        # Simple approach: add weather for each lap based on lap start time
        # You might want to improve this logic
        weather_avg = weather_df.groupby(weather_df.index // len(weather_df) * len(df)).agg({
            "AirTemp": "mean",
            "TrackTemp": "mean", 
            "WindSpeed": "mean",
            "Humidity": "mean"
        }).reset_index(drop=True)
        
        if len(weather_avg) > 0:
            # Broadcast weather to all laps (simplified)
            df["air_temp"] = weather_avg["AirTemp"].iloc[0] if len(weather_avg) > 0 else np.nan
            df["track_temp"] = weather_avg["TrackTemp"].iloc[0] if len(weather_avg) > 0 else np.nan
            df["wind_speed"] = weather_avg["WindSpeed"].iloc[0] if len(weather_avg) > 0 else np.nan
            df["humidity"] = weather_avg["Humidity"].iloc[0] if len(weather_avg) > 0 else np.nan
    
    # Add meta info
    df["season"] = season
    df["round"] = round_no
    df["race_id"] = f"{season}_{round_no:02d}"
    df["circuit"] = circuit
    df["total_laps"] = total_laps
    
    # Select and order columns
    cols = [c for c in DTYPES.keys() if c in df.columns]
    df = df[cols]
    
    # Clean data types
    df = df.copy()  # Prevent SettingWithCopyWarning
    for c in ["driver", "team", "circuit", "race_id", "compound", "dnf_reason", "classification_status"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    
    for c, dt in DTYPES.items():
        if c in df.columns:
            try:
                df[c] = df[c].astype(dt)
            except Exception:
                pass
    
    return df

# ------------------------- Download & Build Workflow -------------------------

def enable_cache(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    print(f"[Info] FastF1 Cache aktiviert: {cache_dir}")

def fetch_lap_by_lap_features(season: int, round_no: int) -> pd.DataFrame:
    """Load a single race session and build lap-by-lap features."""
    print(f"[Load] Season {season}, Round {round_no} - LAP BY LAP ...")
    session = fastf1.get_session(season, round_no, 'R')
    # Load laps + weather data
    session.load(laps=True, weather=True, telemetry=False)
    
    laps = session.laps
    weather = session.weather_data
    
    if laps is None or laps.empty:
        raise RuntimeError("Keine Lap-Daten vom Session-Objekt erhalten.")
    
    df = build_lap_by_lap_features(session, laps, weather)
    return df

def save_parquet(df: pd.DataFrame, out_dir: Path, season: int, round_no: int):
    out_dir = Path(out_dir) / f"season={season}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"round={round_no:02d}_laps.parquet"
    
    # Ensure column order
    cols = [c for c in DTYPES.keys() if c in df.columns]
    df = df[cols]
    
    # Cast to dtypes
    for c, dt in DTYPES.items():
        if c in df.columns:
            try:
                df[c] = df[c].astype(dt)
            except Exception:
                pass
    
    df.to_parquet(out_path, index=False)
    print(f"[OK] Gespeichert: {out_path} (rows={len(df)} laps)")

def iter_season_rounds(season: int):
    """Yield available round numbers for a season using FastF1 schedule."""
    sched = fastf1.get_event_schedule(season, include_testing=False)
    rounds = sorted(int(r) for r in pd.unique(sched["RoundNumber"]) if pd.notna(r))
    return rounds

# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(description="Erzeuge Lap-by-Lap Features aus FastF1.")
    ap.add_argument("--cache", default="fastf1_cache", type=str, help="Cache-Verzeichnis für FastF1")
    ap.add_argument("--out", default="data/laps", type=str, help="Output-Verzeichnis (Parquet pro Rennen)")
    
    # Single race
    ap.add_argument("--season", type=int, help="Saison (z. B. 2024)")
    ap.add_argument("--round", type=int, help="Rennnummer (z. B. 5)")
    
    # Multiple seasons  
    ap.add_argument("--seasons", nargs="+", type=int, help="Mehrere Saisons (z. B. --seasons 2020 2021 2022)")
    args = ap.parse_args()

    cache_dir = Path(args.cache)
    out_dir = Path(args.out)
    enable_cache(cache_dir)

    # Case 1: Single race
    if args.season and args.round:
        try:
            df = fetch_lap_by_lap_features(args.season, args.round)
            save_parquet(df, out_dir, args.season, args.round)
        except Exception as e:
            print(f"[Error] {args.season} R{args.round}: {e}", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)

    # Case 2: Multiple seasons
    seasons = args.seasons or ([args.season] if args.season else [])
    if not seasons:
        print("Bitte entweder --season & --round oder --seasons angeben.", file=sys.stderr)
        sys.exit(2)

    for s in seasons:
        rounds = iter_season_rounds(s)
        print(f"[Season {s}] Runden: {rounds}")
        for r in rounds:
            try:
                df = fetch_lap_by_lap_features(s, r)
                save_parquet(df, out_dir, s, r)
                time.sleep(0.5)
            except KeyboardInterrupt:
                print("Abgebrochen per Ctrl+C.")
                sys.exit(130)
            except Exception as e:
                print(f"[Warn] {s} R{r}: {e}", file=sys.stderr)
                continue

    print("[Done] Alle Lap-by-Lap Daten verarbeitet.")

if __name__ == "__main__":
    main()
