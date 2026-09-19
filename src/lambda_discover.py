"""
src/lambda_discover.py

Phase 4 - Lambda-ready version of discover_videos.py.

Rewritten WITHOUT pandas to keep the deployment package small enough for
direct Lambda upload (avoids needing S3, which would add a small
permanent cost and break our zero-cost design).

Same logic as discover_videos.py, using plain dicts/lists instead of
DataFrames.
"""

import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from googleapiclient.discovery import build
from pymongo import MongoClient

# --- Config (inlined here so this file has zero dependency on collection_config.py
#     or pandas-based modules, keeping the Lambda package minimal) ---
REGION_CODE = "IN"
MAX_VIDEOS_PER_CATEGORY = 40
FRESHNESS_HOURS = 3


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


def get_assignable_categories(youtube, region_code: str) -> list:
    request = youtube.videoCategories().list(part="snippet", regionCode=region_code)
    response = request.execute()

    categories = []
    for item in response.get("items", []):
        if item["snippet"].get("assignable", False):
            categories.append({"id": item["id"], "name": item["snippet"]["title"]})
    return categories


def search_fresh_videos_for_category(youtube, category_id: str, published_after: str) -> list:
    request = youtube.search().list(
        part="id",
        type="video",
        videoCategoryId=category_id,
        regionCode=REGION_CODE,
        order="date",
        publishedAfter=published_after,
        maxResults=MAX_VIDEOS_PER_CATEGORY,
    )
    response = request.execute()
    return [item["id"]["videoId"] for item in response.get("items", [])]


def fetch_video_details_batch(youtube, video_ids: list) -> list:
    """
    Returns a list of plain dicts (no DataFrame) with video details.

    Excludes live streams and upcoming premieres (liveBroadcastContent
    != "none") — these behave very differently from normal uploads
    (chat-driven engagement, unstable duration/view counts while live),
    and would add noise rather than useful signal to the training data.
    """
    results = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        request = youtube.videos().list(part="snippet,contentDetails", id=",".join(batch))
        response = request.execute()

        for item in response.get("items", []):
            snippet = item["snippet"]

            if snippet.get("liveBroadcastContent", "none") != "none":
                continue  # skip live/upcoming content

            results.append({
                "video_id": item["id"],
                "title": snippet.get("title", ""),
                "channel": snippet.get("channelTitle", ""),
                "category_id": float(snippet.get("categoryId", -1)),
                "published_at": datetime.strptime(
                    snippet["publishedAt"], "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=timezone.utc),
                "duration": item["contentDetails"].get("duration", "PT0S"),
            })
    return results


def save_to_master(db, new_videos: list) -> dict:
    """Upserts videos into MongoDB — same dedup logic as the local version."""
    if not new_videos:
        return {"upserted": 0, "matched": 0}

    videos_master = db["videos_master"]
    upserted, matched = 0, 0

    for record in new_videos:
        record["discovered_at"] = datetime.now(timezone.utc)
        result = videos_master.update_one(
            {"video_id": record["video_id"]},
            {"$setOnInsert": record},
            upsert=True,
        )
        if result.upserted_id is not None:
            upserted += 1
        else:
            matched += 1

    return {"upserted": upserted, "matched": matched, "total": videos_master.count_documents({})}


def run_discovery() -> dict:
    """Main discovery routine — the core logic, callable from Lambda or locally."""
    load_dotenv()  # picks up local .env if present; no-op on Lambda (uses env vars directly)

    youtube = get_youtube_client()
    db = get_mongo_db()

    categories = get_assignable_categories(youtube, REGION_CODE)

    published_after = (
        datetime.now(timezone.utc) - timedelta(hours=FRESHNESS_HOURS)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    all_video_ids = []
    category_log = {}
    for cat in categories:
        try:
            ids = search_fresh_videos_for_category(youtube, cat["id"], published_after)
            category_log[cat["name"]] = len(ids)
            all_video_ids.extend(ids)
        except Exception as e:
            category_log[cat["name"]] = f"FAILED: {e}"

    if not all_video_ids:
        return {"status": "no_videos_found", "category_log": category_log}

    details = fetch_video_details_batch(youtube, all_video_ids)
    save_result = save_to_master(db, details)

    return {
        "status": "success",
        "category_log": category_log,
        "discovered_count": len(details),
        "save_result": save_result,
    }


def lambda_handler(event, context):
    """
    AWS Lambda entry point. EventBridge triggers this on a schedule
    (every 3 hours) — 'event' and 'context' are provided by AWS but
    unused here since this runs the same routine every time.
    """
    result = run_discovery()
    print(result)  # shows up in CloudWatch Logs
    return result


if __name__ == "__main__":
    # Allows testing this locally before deploying to Lambda
    print(run_discovery())
