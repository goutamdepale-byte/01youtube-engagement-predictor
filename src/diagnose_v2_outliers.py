import os
import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CSV_PATH = os.path.join(_THIS_DIR, "..", "data", "processed", "01training_table_v2_upgrade.csv")

df = pd.read_csv(_CSV_PATH)

print("Shape:", df.shape)
print()

print("target_engagement_rate full distribution:")
print(df["target_engagement_rate"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))
print()

print("Top 10 highest target_engagement_rate values:")
print(df[["video_id", "category_id", "target_engagement_rate"]]
      .sort_values("target_engagement_rate", ascending=False).head(10))
print()

print("Any remaining infinite values in numeric columns?")
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(np.isinf(df[numeric_cols]).sum().sum(), "total infinite values")
print()

print("Any remaining nulls?")
print(df.isnull().sum()[df.isnull().sum() > 0])
print()

print("early_engagement_rate vs target_engagement_rate correlation:")
print(df[["early_engagement_rate", "target_engagement_rate"]].corr())
