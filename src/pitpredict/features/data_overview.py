import fastf1

fastf1.Cache.enable_cache('cache')

session = fastf1.get_session(2023, 'Monaco', 'R')
session.load()

#daten zsmenfassen

# 1. Laps (Rundendaten)
laps = session.laps
print("1. LAPS - Rundendaten")
print(f"   Anzahl Runden: {len(laps)}")
print(f"   Spalten: {len(laps.columns)}")
print(f"   Alle Spalten: {list(laps.columns)}")
print()

# 2. Weather Data
weather = session.weather_data
print("2. WEATHER_DATA - Wetterdaten")
print(f"   Anzahl Messungen: {len(weather)}")
print(f"   Spalten: {len(weather.columns)}")
print(f"   Alle Spalten: {list(weather.columns)}")
print()

# 3. Results
results = session.results
print("3. RESULTS - Endergebnisse")
print(f"   Anzahl Fahrer: {len(results)}")
print(f"   Spalten: {len(results.columns)}")
print(f"   Alle Spalten: {list(results.columns)}")
print()

# 4. Car Data (Telemetrie)
try:
    car_data = session.car_data
    print("4. CAR_DATA - Telemetriedaten")
    print(f"   Anzahl Datenpunkte pro Fahrer: Variiert")
    print(f"   Verfügbare Fahrer: {list(car_data.keys())}")
    # Beispiel für einen Fahrer
    if car_data:
        first_driver = list(car_data.keys())[0]
        driver_data = car_data[first_driver]
        print(f"   Spalten (am Beispiel {first_driver}): {list(driver_data.columns)}")
    print()
except Exception as e:
    print("4. CAR_DATA - Nicht verfügbar")
    print(f"   Fehler: {e}")
    print()

# 5. Position Data (GPS)
try:
    pos_data = session.pos_data
    print("5. POS_DATA - GPS-Positionsdaten")
    print(f"   Anzahl Datenpunkte pro Fahrer: Variiert")
    print(f"   Verfügbare Fahrer: {list(pos_data.keys())}")
    # Beispiel für einen Fahrer
    if pos_data:
        first_driver = list(pos_data.keys())[0]
        driver_data = pos_data[first_driver]
        print(f"   Spalten (am Beispiel {first_driver}): {list(driver_data.columns)}")
    print()
except Exception as e:
    print("5. POS_DATA - Nicht verfügbar")
    print(f"   Fehler: {e}")
    print()

# 6. Race Control Messages
race_messages = session.race_control_messages
print("6. RACE_CONTROL_MESSAGES - Rennleitung")
print(f"   Anzahl Nachrichten: {len(race_messages)}")
print(f"   Spalten: {list(race_messages.columns)}")
print()

# 7. Session Info
print("7. SESSION_INFO - Grundinfos")
print(f"   Event: {session.event}")
print(f"   Datum: {session.date}")
print(f"   Track: {session.event['Location']}")
print()

# Zusätzliche Datenquellen mit allen Spalten
print("=== ZUSÄTZLICHE DATENQUELLEN ===")

# Session Status
try:
    session_status = session.session_status
    print("SESSION_STATUS:")
    print(f"   Spalten: {list(session_status.columns)}")
    print()
except Exception as e:
    print(f"SESSION_STATUS - Fehler: {e}")

# Track Status
try:
    track_status = session.track_status
    print("TRACK_STATUS:")
    print(f"   Spalten: {list(track_status.columns)}")
    print()
except Exception as e:
    print(f"TRACK_STATUS - Fehler: {e}")

# 8. Weitere verfügbare Attribute
print("8. WEITERE VERFÜGBARE DATEN:")
available_attrs = [attr for attr in dir(session) if not attr.startswith('_') and not callable(getattr(session, attr))]
for attr in sorted(available_attrs):
    try:
        data = getattr(session, attr)
        if hasattr(data, '__len__') and hasattr(data, 'columns'):
            print(f"   session.{attr} - {len(data)} Zeilen, {len(data.columns)} Spalten")
        else:
            print(f"   session.{attr} - {type(data).__name__}")
    except:
        print(f"   session.{attr} - Verfügbar")
print()
