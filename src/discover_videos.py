"""
src/discover_videos.py

Phase 1 - Step 1: Discover fresh videos across ALL YouTube categories.

Run this ONCE at the start of a collection cycle (e.g., once a day, or
whenever you want to start tracking a new batch of videos).

Quota-conscious design:
- search.list (100 units/call) is used ONCE per category to find fresh
  video IDs published recently (search.list is the only endpoint that can
  filter by exact publish time + category).
- videos.list (1 unit/call, up to 50 IDs per call) is then used to fetch
  full stats for all discovered videos cheaply in batches.

Rough quota cost: ~18 categories x 100 units (search) + a handful of
1-unit batch calls = ~1,900 units out of your 10,000/day free quota,
leaving plenty of room for the polling script to run the same day.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from youtube_api import youtube  # reuse the already-authenticated client
from mongo_setup import get_database
import collection_config as cfg


def get_assignable_categories(region_code: str) -> pd.DataFrame:
    """Fetches the list of YouTube video categories usable in this region."""
    request = youtube.videoCategories().list(part="snippet", regionCode=region_code)
    response = request.execute()

    rows = []
    for item in response.get("items", []):
        if item["snippet"].get("assignable", False):
            rows.append({
                "category_id": item["id"],
                "category_name": item["snippet"]["title"],
            })
    return pd.DataFrame(rows)


def search_fresh_videos_for_category(category_id: str, published_after: str) -> list:
    """Uses search.list to find recently published videos in a category."""
    request = youtube.search().list(
        part="id",
        type="video",
        videoCategoryId=category_id,
        regionCode=cfg.REGION_CODE,
        order="date",
        publishedAfter=published_after,
        maxResults=cfg.MAX_VIDEOS_PER_CATEGORY,
    )
    response = request.execute()
    return [item["id"]["videoId"] for item in response.get("items", [])]


def fetch_video_details_batch(video_ids: list) -> pd.DataFrame:
    """
    Fetches full details for up to 50 video IDs per call (quota-efficient:
    costs 1 unit regardless of how many IDs, up to 50, are requested).
    """
    rows = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(batch),
        )
        response = request.execute()

        for item in response.get("items", []):
            snippet = item["snippet"]
            rows.append({
                "video_id": item["id"],
                "title": snippet.get("title", ""),
                "channel": snippet.get("channelTitle", ""),
                "category_id": float(snippet.get("categoryId", -1)),
                "published_at": snippet.get("publishedAt"),
                "duration": item["contentDetails"].get("duration", "PT0S"),
            })
    return pd.DataFrame(rows)


def discover_all_categories() -> pd.DataFrame:
    """Main discovery routine — loops through every assignable category."""
    categories_df = get_assignable_categories(cfg.REGION_CODE)
    print(f"Found {len(categories_df)} assignable categories for region {cfg.REGION_CODE}")

    published_after = (
        datetime.now(timezone.utc) - timedelta(hours=cfg.FRESHNESS_HOURS)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    all_video_ids = []
    for _, row in categories_df.iterrows():
        cat_id, cat_name = row["category_id"], row["category_name"]
        try:
            ids = search_fresh_videos_for_category(cat_id, published_after)
            print(f"  Category {cat_id} ({cat_name}): found {len(ids)} fresh videos")
            all_video_ids.extend(ids)
        except Exception as e:
            print(f"  Category {cat_id} ({cat_name}): FAILED - {e}")

    print(f"\nTotal videos discovered: {len(all_video_ids)}")

    if not all_video_ids:
        return pd.DataFrame()

    details_df = fetch_video_details_batch(all_video_ids)
    return details_df


def save_to_master(new_videos_df: pd.DataFrame):
    """
    Upserts newly discovered videos into MongoDB's videos_master collection.
    'Upsert' = insert if new, update if the video_id already exists —
    this naturally avoids duplicates without needing a separate dedup step.
    """
    if new_videos_df.empty:
        print("No videos to save.")
        return

    db = get_database()
    videos_master = db["videos_master"]

    records = new_videos_df.to_dict("records")
    upserted_count = 0
    matched_count = 0

    for record in records:
        # published_at comes in as an ISO string from the API — store it as
        # a real datetime so age calculations work correctly later
        record["published_at"] = pd.to_datetime(record["published_at"], utc=True).to_pydatetime()
        record["discovered_at"] = datetime.now(timezone.utc)

        result = videos_master.update_one(
            {"video_id": record["video_id"]},
            {"$setOnInsert": record},
            upsert=True,
        )
        if result.upserted_id is not None:
            upserted_count += 1
        else:
            matched_count += 1

    total_in_db = videos_master.count_documents({})
    print(f"Upserted {upserted_count} new videos, {matched_count} already existed.")
    print(f"Total videos in videos_master: {total_in_db}")


if __name__ == "__main__":
    discovered = discover_all_categories()
    if len(discovered) > 0:
        print("\nCategory breakdown of discovered videos:")
        print(discovered["category_id"].value_counts())
        save_to_master(discovered)
    else:
        print("No videos discovered — check API key/quota and try again.")
