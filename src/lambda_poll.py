"""
src/lambda_poll.py

Phase 4 - Lambda-ready version of poll_snapshots.py.

Rewritten WITHOUT pandas — plain dicts/lists only, to keep the Lambda
deployment package small enough for direct upload (no S3 needed).
"""

import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from googleapiclient.discovery import build
from pymongo import MongoClient

STOP_POLLING_AFTER_HOURS = 32


def get_youtube_client():
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY environment variable not set.")
    return build("youtube", "v3", developerKey=api_key)


def get_mongo_db():
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI environment variable not set.")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    return client["youtube_engagement"]


def load_active_video_ids(db) -> list:
    videos_master = db["videos_master"]
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=STOP_POLLING_AFTER_HOURS)

    active_docs = videos_master.find(
        {"published_at": {"$gte": cutoff_time}},
        {"video_id": 1, "_id": 0},
    )
    return [doc["video_id"] for doc in active_docs]


def fetch_current_stats(youtube, video_ids: list) -> list:
    rows = []
    now = datetime.now(timezone.utc)

    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        request = youtube.videos().list(part="statistics", id=",".join(batch))
        response = request.execute()

        for item in response.get("items", []):
            stats = item["statistics"]
            rows.append({
                "video_id": item["id"],
                "snapshot_time": now,
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "comments": int(stats.get("commentCount", 0)),
            })
    return rows


def append_snapshots(db, snapshots: list):
    if not snapshots:
        return
    db["snapshots"].insert_many(snapshots)


def run_poll_cycle() -> dict:
    """Main polling routine — the core logic, callable from Lambda or locally."""
    load_dotenv()

    youtube = get_youtube_client()
    db = get_mongo_db()

    active_ids = load_active_video_ids(db)
    if not active_ids:
        return {"status": "no_active_videos"}

    snapshots = fetch_current_stats(youtube, active_ids)
    append_snapshots(db, snapshots)

    return {"status": "success", "active_video_count": len(active_ids), "snapshots_saved": len(snapshots)}


def lambda_handler(event, context):
    """
    AWS Lambda entry point. EventBridge triggers this every 10 minutes.
    """
    result = run_poll_cycle()
    print(result)  # shows up in CloudWatch Logs
    return result


if __name__ == "__main__":
    # Allows testing this locally before deploying to Lambda
    print(run_poll_cycle())
