"""
src/predict_live_v3.py

Updated prediction pipeline that loads its model DIRECTLY FROM MONGODB
(the model_store collection, written by lambda_retrain.py) instead of a
local .pkl file. This makes the dashboard automatically pick up whatever
model the weekly Lambda retrain has most recently deployed — no manual
file copying, no local model files at all.

The model is fetched once when this module is first imported (e.g. when
the Streamlit dashboard starts) — not on every single prediction, to avoid
a MongoDB round-trip for every user click. Restarting the dashboard picks
up any newer model deployed since it last started.
"""

import os
import re
import io

import joblib
import pandas as pd

from youtube_api import fetch_live_video_data
from mongo_setup import get_database

EARLY_CUTOFF_HOURS = 6


def load_active_model_from_mongodb():
    """
    Fetches the currently active model + its feature columns and rare
    category grouping from MongoDB's model_store collection.
    """
    db = get_database()
    doc = db["model_store"].find_one({"is_active": True})

    if doc is None:
        raise ValueError(
            "No active model found in MongoDB's model_store collection. "
            "Run lambda_retrain.py (or trigger the retrain Lambda) at "
            "least once before using live predictions."
        )

    model = joblib.load(io.BytesIO(doc["model_bytes"]))
    feature_columns = joblib.load(io.BytesIO(doc["feature_columns_bytes"]))
    rare_categories = joblib.load(io.BytesIO(doc["rare_categories_bytes"]))

    return model, feature_columns, rare_categories, doc["metrics"], doc["trained_at"]


# Loaded once at import time
_model, _feature_columns, _rare_categories, _model_metrics, _model_trained_at = load_active_model_from_mongodb()


def parse_duration_to_seconds(duration_str: str) -> int:
    if not duration_str:
        return 0
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", str(duration_str))
    if not match:
        return 0
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = int(match.group(3)) if match.group(3) else 0
    return hours * 3600 + minutes * 60 + seconds


def get_grouped_category(category_id: float) -> str:
    if category_id in _rare_categories:
        return "Other"
    return str(category_id)


def build_feature_vector(raw_data: dict):
    views = max(raw_data["views"], 1)
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
        "early_snapshot_count": 1,
        "early_views": views,
        "early_likes": likes,
        "early_comments": comments,
        "early_views_per_hour": views / age_hours,
        "early_like_rate": (likes / views) * 100,
        "early_comment_rate": (comments / views) * 100,
        "early_engagement_rate": ((likes + comments) / views) * 100,
    }

    for col in _feature_columns:
        if col.startswith("cat_"):
            features[col] = 0

    grouped_category = get_grouped_category(raw_data["category_id"])
    cat_col_name = f"cat_{grouped_category}"
    category_seen_in_training = cat_col_name in _feature_columns
    if category_seen_in_training:
        features[cat_col_name] = 1

    row = pd.DataFrame([features])[_feature_columns]
    return row, category_seen_in_training, grouped_category


def get_live_prediction(video_url: str) -> dict:
    raw_data = fetch_live_video_data(video_url)
    feature_row, category_seen, grouped_category = build_feature_vector(raw_data)

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
    elif grouped_category == "Other":
        warnings.append(
            f"Category ID {raw_data['category_id']} had too few training "
            f"examples to model individually, so it was grouped with other "
            f"rare categories ('Other'). Prediction may be less precise for "
            f"this specific category."
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
        "model_trained_at": _model_trained_at.isoformat(),
        "model_r2": round(_model_metrics.get("r2_mean", 0), 3),
    }


if __name__ == "__main__":
    print(f"Loaded active model from MongoDB (trained {_model_trained_at}, R2={_model_metrics.get('r2_mean')})")
    url = input("Paste a real YouTube video URL to test prediction: ").strip()
    result = get_live_prediction(url)

    print("\n--- Prediction Result (MongoDB-backed model) ---")
    for k, v in result.items():
        if k == "warnings":
            continue
        print(f"{k}: {v}")

    if result["warnings"]:
        print("\n--- Warnings ---")
        for w in result["warnings"]:
            print(f"⚠ {w}")
