import pandas as pd
df = pd.read_parquet("data/features/season=2024/round=05.parquet")
print(df.head())
