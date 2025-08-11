

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

# ------------------------- Helper: Dtypes -------------------------

DTYPES = {
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
    "airtemp_mean": "float32",
    "tracktemp_mean": "float32",
    "humidity_mean": "float32",
    "windspeed_mean": "float32",
    "rainflag": "int8",
    "n_stops": "int8",
    "finish_pos": "int8"
}

# ------------------------- Feature Builder -------------------------

def _td_to_sec(series):
    """Timedelta -> Sekunden (float)."""
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    # Try to coerce strings to Timedelta first
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()

def summarise_weather_start(weather_df: pd.DataFrame, window_min: int = 15) -> pd.DataFrame:
    """Aggregate weather in the first `window_min` minutes from race start."""
    if weather_df is None or weather_df.empty:
        return pd.DataFrame([{
            "airtemp_mean": np.nan,
            "tracktemp_mean": np.nan,
            "humidity_mean": np.nan,
            "windspeed_mean": np.nan,
            "rainflag": 0
        }])
    w = weather_df.copy()
    mask = (w["Time"] >= pd.Timedelta(0)) & (w["Time"] <= pd.Timedelta(minutes=window_min))
    w_win = w[mask] if mask.any() else w
    return pd.DataFrame([{
        "airtemp_mean": float(w_win["AirTemp"].mean()),
        "tracktemp_mean": float(w_win["TrackTemp"].mean()),
        "humidity_mean": float(w_win["Humidity"].mean()),
        "windspeed_mean": float(w_win["WindSpeed"].mean()),
        "rainflag": int((w_win["Rainfall"] > 0).any())
    }])

def count_stops_from_laps(laps_df: pd.DataFrame) -> pd.DataFrame:
    """Count pit stops per driver using PitInTime."""
    if laps_df is None or laps_df.empty:
        return pd.DataFrame(columns=["Driver", "n_stops"])
    laps = laps_df.copy()
    laps["is_pit"] = laps["PitInTime"].notna()
    stops = laps.groupby("Driver", dropna=False)["is_pit"].sum().astype("Int64").rename("n_stops").reset_index()
    return stops

def build_prerace_core(session, results_df: pd.DataFrame, weather_df: pd.DataFrame, laps_df: pd.DataFrame, weather_window_min: int = 15) -> pd.DataFrame:
    """Create compact pre-race features for one race (one row per driver)."""
    season = int(getattr(session.event, "year", session.event["EventDate"].year))
    round_no = int(session.event.get("RoundNumber", getattr(session.event, "RoundNumber", 0)))
    circuit = str(session.event.get("EventName", getattr(session.event, "EventName", "Unknown")))
    total_laps = int(getattr(session, "total_laps", 0) or 0)

    wfeat = summarise_weather_start(weather_df, weather_window_min)
    wfeat["_k"] = 1

    # Base results
    res = results_df.rename(columns={
        "Abbreviation": "driver",
        "TeamName": "team",
        "DriverNumber": "driver_number",
        "GridPosition": "grid",
        "ClassifiedPosition": "finish_pos"
    }).copy()

    # Fallback wenn ClassifiedPosition fehlt/NaN
    if "finish_pos" not in res or res["finish_pos"].isna().all():
        res["finish_pos"] = pd.to_numeric(results_df.get("Position"), errors="coerce")

    # Q-Zeiten in Sekunden
    res["q1_s"] = _td_to_sec(res.get("Q1"))
    res["q2_s"] = _td_to_sec(res.get("Q2"))
    res["q3_s"] = _td_to_sec(res.get("Q3"))
    res["best_q_s"] = res[["q1_s", "q2_s", "q3_s"]].min(axis=1, skipna=True)
    pole = res["best_q_s"].min(skipna=True)
    res["delta_to_pole_s"] = res["best_q_s"] - pole

    base = res[["driver", "team", "driver_number", "grid", "finish_pos", "q1_s", "q2_s", "q3_s", "delta_to_pole_s"]].copy()
    base["_k"] = 1
    df = base.merge(wfeat, on="_k").drop(columns="_k")

    # Pit stops
    stops = count_stops_from_laps(laps_df)
    df = df.merge(stops, left_on="driver", right_on="Driver", how="left").drop(columns=["Driver"])
    df["n_stops"] = df["n_stops"].fillna(0).astype("int8")

    # Meta
    df["season"] = season
    df["round"] = round_no
    df["race_id"] = f"{season}_{round_no:02d}"
    df["circuit"] = circuit
    df["total_laps"] = total_laps

    # Dtypes/Downcast
    for c in ["driver", "team", "circuit", "race_id"]:
        df[c] = df[c].astype("category")
    df["driver_number"] = pd.to_numeric(df["driver_number"], errors="coerce").fillna(0).astype("int16")
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce").fillna(99).astype("int8")
    df["finish_pos"] = pd.to_numeric(df["finish_pos"], errors="coerce").fillna(99).astype("int8")
    for c in ["q1_s", "q2_s", "q3_s", "delta_to_pole_s", "airtemp_mean", "tracktemp_mean", "humidity_mean", "windspeed_mean"]:
        if c in df.columns:
            df[c] = df[c].astype("float32")

    return df.drop(columns=["best_q_s"], errors="ignore")

# ------------------------- Download & Build Workflow -------------------------

def enable_cache(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    print(f"[Info] FastF1 Cache aktiviert: {cache_dir}")

def fetch_race_features(season: int, round_no: int, weather_window_min: int = 15) -> pd.DataFrame:
    """Load a single race session and build features."""
    print(f"[Load] Season {season}, Round {round_no} ...")
    session = fastf1.get_session(season, round_no, 'R')
    # Only what we need: laps + weather (no telemetry -> saves time/memory)
    session.load(laps=True, weather=True, telemetry=False)
    res = session.results
    w = session.weather_data
    laps = session.laps
    if res is None or res.empty:
        raise RuntimeError("Leere RESULTS vom Session-Objekt erhalten.")
    df = build_prerace_core(session, res, w, laps, weather_window_min)
    return df

def save_parquet(df: pd.DataFrame, out_dir: Path, season: int, round_no: int):
    out_dir = Path(out_dir) / f"season={season}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"round={round_no:02d}.parquet"
    # Ensure column order roughly consistent
    cols = list(DTYPES.keys())
    cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in DTYPES]
    df = df[cols]
    # Cast to dtypes map where possible
    for c, dt in DTYPES.items():
        if c in df.columns:
            try:
                df[c] = df[c].astype(dt)
            except Exception:
                pass
    df.to_parquet(out_path, index=False)
    print(f"[OK] Gespeichert: {out_path} (rows={len(df)})")

def iter_season_rounds(season: int):
    """Yield available round numbers for a season using FastF1 schedule."""
    sched = fastf1.get_event_schedule(season, include_testing=False)
    # Some schedules use 'RoundNumber', ensure ints and sorted
    rounds = sorted(int(r) for r in pd.unique(sched["RoundNumber"]) if pd.notna(r))
    return rounds

# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(description="Erzeuge kompakte Pre-Race-Features aus FastF1.")
    ap.add_argument("--cache", default="fastf1_cache", type=str, help="Cache-Verzeichnis für FastF1")
    ap.add_argument("--out", default="data/features", type=str, help="Output-Verzeichnis (Parquet pro Rennen)")
    ap.add_argument("--weather-window", default=15, type=int, help="Wetterfenster in Minuten ab Rennstart")
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
            df = fetch_race_features(args.season, args.round, args.weather_window)
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
                df = fetch_race_features(s, r, args.weather_window)
                save_parquet(df, out_dir, s, r)
                time.sleep(0.5)  # kleine Pause, nett zur API
            except KeyboardInterrupt:
                print("Abgebrochen per Ctrl+C.")
                sys.exit(130)
            except Exception as e:
                print(f"[Warn] {s} R{r}: {e}", file=sys.stderr)
                continue

    print("[Done] Alle angefragten Saisons verarbeitet.")

if __name__ == "__main__":
    main()
