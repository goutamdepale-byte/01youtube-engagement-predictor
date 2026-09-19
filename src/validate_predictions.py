"""
src/validate_predictions.py

Cross-check the deployed model: picks real videos from MongoDB that
already have a KNOWN outcome (a snapshot near 24h), reconstructs their
early-window features (as if we were predicting fresh), runs them through
the CURRENTLY ACTIVE model (loaded from MongoDB, same as the dashboard
uses), and shows predicted vs. actual side-by-side.

This is a concrete sanity check beyond the abstract R²/RMSE numbers —
you can see exactly how close (or far) individual predictions are on
real videos.

Run with:
    python src/validate_predictions.py
"""

import os
import sys
import re
import random

import numpy as np
import pandas as pd
import joblib
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mongo_setup import get_database

EARLY_CUTOFF = 6
TARGET_LOW, TARGET_HIGH = 18, 30
SAMPLE_SIZE = 15  # how many videos to spot-check


def load_active_model():
    db = get_database()
    doc = db["model_store"].find_one({"is_active": True})
    if doc is None:
        raise ValueError("No active model found in MongoDB.")
    model = joblib.load(io.BytesIO(doc["model_bytes"]))
    feature_columns = joblib.load(io.BytesIO(doc["feature_columns_bytes"]))
    rare_categories = joblib.load(io.BytesIO(doc["rare_categories_bytes"]))
    return model, feature_columns, rare_categories, doc["metrics"], doc["trained_at"]


def pull_and_reconstruct(db) -> pd.DataFrame:
    videos = pd.DataFrame(list(db["videos_master"].find({}, {"_id": 0})))
    snapshots = pd.DataFrame(list(db["snapshots"].find({}, {"_id": 0})))
    merged = snapshots.merge(videos, on="video_id", how="inner")
    merged["published_at"] = pd.to_datetime(merged["published_at"], utc=True)
    merged["snapshot_time"] = pd.to_datetime(merged["snapshot_time"], utc=True)
    merged["video_age_hours"] = (
        (merged["snapshot_time"] - merged["published_at"]).dt.total_seconds() / 3600
    )
    return merged


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["video_id", "video_age_hours"]).copy()
    df = df.sort_values(["video_id", "snapshot_time"]).reset_index(drop=True)

    for col in ["title", "channel", "category_id", "duration", "published_at"]:
        df[col] = df.groupby("video_id")[col].transform(lambda x: x.ffill().bfill())
    for col in ["views", "likes", "comments"]:
        df[col] = df.groupby("video_id")[col].transform(lambda x: x.ffill().bfill())
    for col in ["views", "likes", "comments"]:
        df[col] = df.groupby("video_id")[col].cummax()

    safe_views = df["views"].clip(lower=1)
    df["like_rate"] = (df["likes"] / safe_views) * 100
    df["comment_rate"] = (df["comments"] / safe_views) * 100
    df["engagement_rate"] = ((df["likes"] + df["comments"]) / safe_views) * 100
    df = df.dropna(subset=["views", "likes", "comments", "like_rate", "comment_rate", "engagement_rate"])

    def parse_duration(s):
        if pd.isna(s):
            return np.nan
        m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", str(s))
        if not m:
            return np.nan
        h = int(m.group(1)) if m.group(1) else 0
        mi = int(m.group(2)) if m.group(2) else 0
        se = int(m.group(3)) if m.group(3) else 0
        return h * 3600 + mi * 60 + se

    df["duration_seconds"] = df["duration"].apply(parse_duration)
    return df


def get_grouped_category(category_id, rare_categories):
    if category_id in rare_categories:
        return "Other"
    return str(category_id)


def build_feature_row(video_df: pd.DataFrame, feature_columns, rare_categories):
    """Builds the early-window feature row for ONE video, matching training format exactly."""
    early = video_df[video_df["video_age_hours"] <= EARLY_CUTOFF]
    if early.empty:
        return None
    early_row = early.loc[early["video_age_hours"].idxmax()]

    features = {
        "duration_seconds": early_row["duration_seconds"],
        "upload_hour": early_row["published_at"].hour,
        "upload_dayofweek": early_row["published_at"].dayofweek,
        "is_weekend": int(early_row["published_at"].dayofweek in [5, 6]),
        "early_snapshot_age_hours": early_row["video_age_hours"],
        "early_snapshot_count": len(early),
        "early_views": early_row["views"],
        "early_likes": early_row["likes"],
        "early_comments": early_row["comments"],
        "early_views_per_hour": early_row["views"] / max(early_row["video_age_hours"], 0.01),
        "early_like_rate": early_row["like_rate"],
        "early_comment_rate": early_row["comment_rate"],
        "early_engagement_rate": early_row["engagement_rate"],
    }

    for col in feature_columns:
        if col.startswith("cat_"):
            features[col] = 0
    grouped = get_grouped_category(early_row["category_id"], rare_categories)
    cat_col = f"cat_{grouped}"
    if cat_col in feature_columns:
        features[cat_col] = 1

    return pd.DataFrame([features])[feature_columns]


def get_actual_target(video_df: pd.DataFrame):
    """Finds the real, known engagement_rate closest to 24h for this video."""
    window = video_df[(video_df["video_age_hours"] >= TARGET_LOW) & (video_df["video_age_hours"] <= TARGET_HIGH)]
    if window.empty:
        return None
    closest = window.iloc[(window["video_age_hours"] - 24).abs().argsort().iloc[0]]
    return closest["engagement_rate"]


def run_validation():
    db = get_database()
    model, feature_columns, rare_categories, metrics, trained_at = load_active_model()
    print(f"Active model: trained {trained_at}, CV R2 = {metrics.get('r2_mean'):.3f}\n")

    raw = pull_and_reconstruct(db)
    cleaned = clean_data(raw)

    all_video_ids = cleaned["video_id"].unique().tolist()
    random.seed(42)
    sample_ids = random.sample(all_video_ids, min(SAMPLE_SIZE, len(all_video_ids)))

    results = []
    for vid in sample_ids:
        video_df = cleaned[cleaned["video_id"] == vid]
        feature_row = build_feature_row(video_df, feature_columns, rare_categories)
        actual = get_actual_target(video_df)

        if feature_row is None or actual is None:
            continue

        predicted = model.predict(feature_row)[0]
        results.append({
            "video_id": vid,
            "actual_engagement_rate": round(actual, 2),
            "predicted_engagement_rate": round(predicted, 2),
            "abs_error": round(abs(actual - predicted), 2),
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print(f"\nSpot-check on {len(results_df)} real videos:")
    print(f"  Mean Absolute Error: {results_df['abs_error'].mean():.2f}")
    print(f"  Median Absolute Error: {results_df['abs_error'].median():.2f}")
    print(f"  Max Absolute Error: {results_df['abs_error'].max():.2f}")
    print(f"\n(Compare this Mean Absolute Error to the model's reported cross-validated "
          f"MAE of {metrics.get('mae_mean'):.2f} — they should be in a similar ballpark, "
          f"since this is just a different random sample of the same kind of data.)")


if __name__ == "__main__":
    run_validation()
