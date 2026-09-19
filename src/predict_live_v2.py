"""
src/predict_live_v2.py

Updated version of predict_live.py using the EXPANDED model (engagement_model_v2.pkl),
trained on 524 videos across 14 categories, with rare categories grouped
into "Other" and outlier-exclusion applied during training.

This is the version Day 5's dashboard should call going forward.
"""

import os
import re
import joblib
import pandas as pd
import numpy as np

from youtube_api import fetch_live_video_data

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_THIS_DIR, "..", "models", "04engagement_model.pkl")
FEATURE_LIST_PATH = os.path.join(_THIS_DIR, "..", "models", "04feature_columns_v2.pkl")
CATEGORY_GROUPING_PATH = os.path.join(_THIS_DIR, "..", "models", "01rare_categories_v2.pkl")

EARLY_CUTOFF_HOURS = 6

_model = joblib.load(MODEL_PATH)
_feature_columns = joblib.load(FEATURE_LIST_PATH)
_rare_categories = joblib.load(CATEGORY_GROUPING_PATH)  # categories grouped into "Other" during training


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


def get_grouped_category(category_id: float) -> str:
    """
    Applies the EXACT SAME rare-category grouping used during training
    (Day 3): categories with too few training samples were grouped into
    "Other" rather than getting their own one-hot column. A live video's
    category must go through this same mapping, or its one-hot encoding
    won't match what the model expects.
    """
    if category_id in _rare_categories:
        return "Other"
    return str(category_id)


def build_feature_vector(raw_data: dict) -> pd.DataFrame:
    """
    Converts raw live video data into a single-row DataFrame matching the
    EXACT columns/order the v2 model was trained on.

    Category handling (v2): the category is first mapped through the same
    "Other" grouping used in training. If the resulting group's one-hot
    column doesn't exist in the model's known columns (e.g. a category ID
    never seen at all during training), it's treated as unknown — same
    honest fallback as v1.
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
    """
    Full pipeline: fetch live data -> build features -> predict.
    Returns a dict with the prediction plus useful context/warnings.
    """
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
    }


if __name__ == "__main__":
    url = input("Paste a real YouTube video URL to test prediction: ").strip()
    result = get_live_prediction(url)

    print("\n--- Prediction Result (v2 model) ---")
    for k, v in result.items():
        if k == "warnings":
            continue
        print(f"{k}: {v}")

    if result["warnings"]:
        print("\n--- Warnings ---")
        for w in result["warnings"]:
            print(f"⚠ {w}")
