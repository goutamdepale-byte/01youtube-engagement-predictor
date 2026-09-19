import os
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CSV_PATH = os.path.join(_THIS_DIR, "..", "data", "processed", "cleaned_data_v2.csv")

df = pd.read_csv(_CSV_PATH)
print("Loaded cleaned_data_v2.csv:", df.shape)
print("dtypes:")
print(df.dtypes)
print()

EARLY_CUTOFF = 6
TARGET_LOW, TARGET_HIGH = 18, 30

early_df = df[df["video_age_hours"] <= EARLY_CUTOFF]
print("Rows with video_age_hours <= 6:", early_df.shape[0])
print("Unique videos in early_df:", early_df["video_id"].nunique())
print()

target_df = df[(df["video_age_hours"] >= TARGET_LOW) & (df["video_age_hours"] <= TARGET_HIGH)]
print("Rows with video_age_hours in [18, 30]:", target_df.shape[0])
print("Unique videos in target_df:", target_df["video_id"].nunique())
print()

print("Sample video_age_hours values (first 10):")
print(df["video_age_hours"].head(10))
print()
print("video_age_hours min/max:", df["video_age_hours"].min(), df["video_age_hours"].max())
