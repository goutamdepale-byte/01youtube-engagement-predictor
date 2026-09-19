"""
src/predict_live.py

Day 4 - Step 3 & 4: Convert live YouTube data into the model's feature
format, run the saved model, and return a prediction.

This is the core function Day 5's Streamlit dashboard will call directly.
"""

import re
import joblib
import pandas as pd
import numpy as np

from youtube_api import fetch_live_video_data

MODEL_PATH = "C:/Users/Lenovo/youtube_enagagement_project/models/engagement_model.pkl"
FEATURE_LIST_PATH ="C:/Users/Lenovo/youtube_enagagement_project/models/feature_columns.pkl"

EARLY_CUTOFF_HOURS = 6   # the window the model was actually trained on

_model = joblib.load(MODEL_PATH)
_feature_columns = joblib.load(FEATURE_LIST_PATH)


def parse_duration_to_seconds(duration_str: str) -> int:
    """Same logic as Day 1 — converts ISO 8601 (e.g. PT17M45S) to seconds."""
    if not duration_str:
        return 0
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", str(duration_str))
    if not match:
        return 0
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = int(match.group(3)) if match.group(3) else 0
    return hours * 3600 + minutes * 60 + seconds


def build_feature_vector(raw_data: dict) -> pd.DataFrame:
    """
    Converts raw live video data into a single-row DataFrame matching the
    EXACT columns/order the model was trained on (feature_columns.pkl).

    Any category not seen during training is handled gracefully — all
    one-hot category columns are simply left as 0 (the model treats it as
    "unknown category", which is honest rather than guessing wrong).
    """
    views = max(raw_data["views"], 1)  # avoid divide-by-zero on brand-new videos
    likes = raw_data["likes"]
    comments = raw_data["comments"]
    age_hours = max(raw_data["video_age_hours"], 0.01)

    published_at = raw_data["published_at"]

    features = {
        "duration_seconds": parse_duration_to_seconds(raw_data["duration"]),
        "upload_hour": published_at.hour,
        "upload_dayofweek": published_at.weekday(),
        "is_weekend": int(published_at.weekday() in [5, 6]),
        "early_snapshot_age_hours": age_hours,
        "early_snapshot_count": 1,   # a single live pull, not multiple snapshots
        "early_views": views,
        "early_likes": likes,
        "early_comments": comments,
        "early_views_per_hour": views / age_hours,
        "early_like_rate": (likes / views) * 100,
        "early_comment_rate": (comments / views) * 100,
        "early_engagement_rate": ((likes + comments) / views) * 100,
    }

    # Start every one-hot category column at 0, then set the matching one
    # (if the category was seen during training) to 1
    for col in _feature_columns:
        if col.startswith("cat_"):
            features[col] = 0

    cat_col_name = f"cat_{float(raw_data['category_id'])}"
    category_seen_in_training = cat_col_name in _feature_columns
    if category_seen_in_training:
        features[cat_col_name] = 1

    # Build the DataFrame in the EXACT column order the model expects
    row = pd.DataFrame([features])[_feature_columns]

    return row, category_seen_in_training


def get_live_prediction(video_url: str) -> dict:
    """
    Full pipeline: fetch live data -> build features -> predict.
    Returns a dict with the prediction plus useful context/warnings.
    """
    raw_data = fetch_live_video_data(video_url)
    feature_row, category_seen = build_feature_vector(raw_data)

    predicted_engagement = _model.predict(feature_row)[0]

    warnings = []
    if raw_data["video_age_hours"] > EARLY_CUTOFF_HOURS:
        warnings.append(
            f"This video is {raw_data['video_age_hours']:.1f} hours old, "
            f"outside the model's intended early-prediction window "
            f"(<= {EARLY_CUTOFF_HOURS}h). The prediction may be less reliable."
        )
    if not category_seen:
        warnings.append(
            f"Category ID {raw_data['category_id']} was not present in the "
            f"training data. The model will treat this as an unknown category, "
            f"which may reduce accuracy."
        )

    current_engagement = ((raw_data["likes"] + raw_data["comments"]) / max(raw_data["views"], 1)) * 100

    return {
        "video_id": raw_data["video_id"],
        "title": raw_data["title"],
        "channel": raw_data["channel"],
        "category_id": raw_data["category_id"],
        "video_age_hours": raw_data["video_age_hours"],
        "current_views": raw_data["views"],
        "current_likes": raw_data["likes"],
        "current_comments": raw_data["comments"],
        "current_engagement_rate": round(current_engagement, 3),
        "predicted_engagement_rate_24h": round(float(predicted_engagement), 3),
        "warnings": warnings,
    }


if __name__ == "__main__":
    url = input("Paste a real YouTube video URL to test prediction: ").strip()
    result = get_live_prediction(url)

    print("\n--- Prediction Result ---")
    for k, v in result.items():
        if k == "warnings":
            continue
        print(f"{k}: {v}")

    if result["warnings"]:
        print("\n--- Warnings ---")
        for w in result["warnings"]:
            print(f"⚠ {w}")
