import pandas as pd

df = pd.read_parquet("train_m1_m4.parquet")

print(df.dtypes.value_counts())
print("\nColumns:")
for c in df.columns:
    print(c, "->", df[c].dtype)