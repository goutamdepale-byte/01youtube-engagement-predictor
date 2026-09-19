import os
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CSV_PATH = os.path.join(_THIS_DIR, "..", "data", "processed", "cleaned_data_v2.csv")

df = pd.read_csv(_CSV_PATH)

print("Reading from:", os.path.abspath(_CSV_PATH))
print("Shape:", df.shape)
print("Unique videos:", df["video_id"].nunique())
print("Unique categories:", df["category_id"].nunique())
print("Category list:", sorted(df["category_id"].unique()))
