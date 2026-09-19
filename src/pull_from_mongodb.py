"""
src/pull_from_mongodb.py

Phase 5 - Step 5.1: Pull collected data from MongoDB and reconstruct it
into the SAME format as the original raw CSV (youtube_realtime_history_aws_clean.csv)
so it can be fed directly into the existing Day 1-3 pipeline
(01_eda_cleaning.ipynb, 02_feature_engineering.ipynb, 03_modeling.ipynb)
with zero changes to that code.

Output columns match the original raw dataset exactly:
video_id, title, channel, published_at, category_id, duration, views,
likes, comments, snapshot_time, video_age_hours, views_per_hour,
like_rate, comment_rate, engagement_rate

Run with:
    python src/pull_from_mongodb.py
"""

import os
import sys
from datetime import timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mongo_setup import get_database

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "youtube_mongodb_export.csv"
)


def pull_videos_master() -> pd.DataFrame:
    db = get_database()
    docs = list(db["videos_master"].find({}, {"_id": 0}))
    df = pd.DataFrame(docs)
    print(f"Pulled {len(df)} videos from videos_master")
    return df


def pull_snapshots() -> pd.DataFrame:
    db = get_database()
    docs = list(db["snapshots"].find({}, {"_id": 0}))
    df = pd.DataFrame(docs)
    print(f"Pulled {len(df)} snapshots from snapshots")
    return df


def merge_and_compute_fields(videos: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    """
    Joins static video info with time-series snapshots, then computes the
    same derived fields (video_age_hours, views_per_hour, like_rate,
    comment_rate, engagement_rate) that the original raw CSV already had —
    so the output is a drop-in replacement for Day 1's input file.
    """
    merged = snapshots.merge(videos, on="video_id", how="inner")

    # Ensure proper datetime types (Mongo may return these as pandas Timestamps
    # already, but this makes the script robust either way)
    merged["published_at"] = pd.to_datetime(merged["published_at"], utc=True)
    merged["snapshot_time"] = pd.to_datetime(merged["snapshot_time"], utc=True)

    merged["video_age_hours"] = (
        (merged["snapshot_time"] - merged["published_at"]).dt.total_seconds() / 3600
    )

    # Guard against division by zero for the very first snapshot of a video
    safe_age = merged["video_age_hours"].clip(lower=0.01)
    safe_views = merged["views"].clip(lower=1)

    merged["views_per_hour"] = merged["views"] / safe_age
    merged["like_rate"] = (merged["likes"] / safe_views) * 100
    merged["comment_rate"] = (merged["comments"] / safe_views) * 100
    merged["engagement_rate"] = ((merged["likes"] + merged["comments"]) / safe_views) * 100

    # Match original column order exactly
    final_cols = [
        "video_id", "title", "channel", "published_at", "category_id", "duration",
        "views", "likes", "comments", "snapshot_time", "video_age_hours",
        "views_per_hour", "like_rate", "comment_rate", "engagement_rate",
    ]
    return merged[final_cols]


def run_export():
    videos = pull_videos_master()
    snapshots = pull_snapshots()

    if videos.empty or snapshots.empty:
        print("No data found in one or both collections — nothing to export.")
        return None

    combined = merge_and_compute_fields(videos, snapshots)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"\nExported {combined.shape[0]} rows, {combined['video_id'].nunique()} unique videos")
    print(f"Category breakdown:\n{combined.groupby('category_id')['video_id'].nunique()}")
    print(f"\nSaved to: {OUTPUT_PATH}")
    return combined


if __name__ == "__main__":
    run_export()
