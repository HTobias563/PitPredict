import pandas as pd
df = pd.read_parquet("data/features/season=2024/round=05.parquet")
print(df.head())

print("\n=== DATENSATZ-GRÖßE ===")
print(f"Anzahl Zeilen (Einträge): {len(df)}")
print(f"Anzahl Spalten: {len(df.columns)}")
print(f"Shape (Zeilen, Spalten): {df.shape}")

print("\n=== SPALTEN ===")
print(df.columns.tolist())

print("\n=== FEHLENDE WERTE ===")
print(df.isnull().sum())

print("\n=== DATENSATZ-INFO ===")
print(df.info())