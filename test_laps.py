import pandas as pd

# Lade die Lap-by-Lap Daten
df = pd.read_parquet("data/laps/season=2024/round=05_laps.parquet")

print("=== LAP-BY-LAP DATENSATZ ===")
print(f"Anzahl Runden total: {len(df)}")
print(f"Anzahl Spalten: {len(df.columns)}")
print(f"Shape: {df.shape}")

print("\n=== ERSTE 5 RUNDEN ===")
print(df.head())

print("\n=== SPALTEN ===")
print(df.columns.tolist())

print("\n=== FAHRER UND IHRE RUNDEN ===")
print(df.groupby('driver')['lap_number'].count().sort_values(ascending=False))

print("\n=== FEHLENDE WERTE ===")
print(df.isnull().sum())

print("\n=== DATENSATZ-INFO ===")
print(df.info())
