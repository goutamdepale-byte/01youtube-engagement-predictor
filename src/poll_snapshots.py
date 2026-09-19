"""
src/poll_snapshots.py

Phase 1 - Step 2: Continuously poll tracked videos every 10 minutes and
record their current stats as a new snapshot row.

Storage-efficient design:
- Static fields (title, channel, category, duration) live ONLY in
  videos_master.parquet (written once by discover_videos.py) — never
  repeated here.
- This file only ever appends small numeric rows: video_id, snapshot_time,
  views, likes, comments — kept as compact int32 dtypes.
- Videos older than STOP_POLLING_AFTER_HOURS are automatically dropped
  from active polling (they've already passed the ~24h target window, so
  further snapshots add no value).

Run with:
    python src/poll_snapshots.py
(runs forever until you stop it with Ctrl+C — leave it running in a
terminal/Anaconda Prompt window during your collection period)
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from youtube_api import youtube
from mongo_setup import get_database
import collection_config as cfg


def load_active_video_ids() -> list:
    """
    Queries MongoDB for videos still within the useful polling window
    (younger than STOP_POLLING_AFTER_HOURS). This query happens directly
    in MongoDB rather than loading everything into pandas first, so it
    stays fast even as videos_master grows.
    """
    db = get_database()
    videos_master = db["videos_master"]

    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=cfg.STOP_POLLING_AFTER_HOURS)

    active_docs = videos_master.find(
        {"published_at": {"$gte": cutoff_time}},
        {"video_id": 1, "_id": 0},
    )
    return [doc["video_id"] for doc in active_docs]


def fetch_current_stats(video_ids: list) -> pd.DataFrame:
    """Batch-fetches current view/like/comment counts (1 unit per 50 IDs)."""
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
    return pd.DataFrame(rows)


def append_snapshots(new_snapshots: pd.DataFrame):
    """
    Inserts new snapshot rows into MongoDB's snapshots collection.
    The TTL index (set up in mongo_setup.py) automatically deletes these
    once they're older than SNAPSHOT_RETENTION_DAYS — no manual cleanup
    needed here.
    """
    if new_snapshots.empty:
        return

    for col, dtype in cfg.SNAPSHOT_DTYPES.items():
        new_snapshots[col] = new_snapshots[col].astype(dtype)

    db = get_database()
    snapshots = db["snapshots"]

    records = new_snapshots.to_dict("records")
    # Ensure snapshot_time is a real datetime (needed for the TTL index to work)
    for record in records:
        if not isinstance(record["snapshot_time"], datetime):
            record["snapshot_time"] = pd.to_datetime(record["snapshot_time"], utc=True).to_pydatetime()

    snapshots.insert_many(records)


def run_one_poll_cycle():
    active_ids = load_active_video_ids()
    if not active_ids:
        print(f"[{datetime.now()}] No active videos to poll "
              f"(run discover_videos.py first, or all tracked videos have aged out).")
        return

    print(f"[{datetime.now()}] Polling {len(active_ids)} active videos...")
    snapshots = fetch_current_stats(active_ids)
    append_snapshots(snapshots)
    print(f"[{datetime.now()}] Saved {len(snapshots)} new snapshot rows.")


if __name__ == "__main__":
    print("Starting continuous polling. Press Ctrl+C to stop.")
    print(f"Polling every {cfg.POLL_INTERVAL_SECONDS} seconds "
          f"({cfg.POLL_INTERVAL_SECONDS / 60:.0f} minutes).")

    try:
        while True:
            run_one_poll_cycle()
            time.sleep(cfg.POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nPolling stopped by user.")
