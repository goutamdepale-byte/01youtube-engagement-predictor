"""
src/youtube_api.py

Day 4 - Step 1 & 2: YouTube API connection + live data fetching.

This module:
1. Loads the API key securely from a .env file
2. Extracts a video_id from any YouTube URL format
3. Calls the YouTube Data API v3 to fetch live stats for a video
4. Returns the raw data needed for feature engineering (Step 3, separate file)
"""

import os
import re
from datetime import datetime, timezone

from dotenv import load_dotenv
from googleapiclient.discovery import build

# ---------------------------------------------------------------------------
# Step 1: Load API key securely
# ---------------------------------------------------------------------------
load_dotenv()  # reads the .env file sitting in the project root
API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY or API_KEY == "paste_your_actual_api_key_here":
    raise ValueError(
        "YOUTUBE_API_KEY not found. Open the .env file in the project root "
        "and paste your real API key in place of the placeholder text."
    )

youtube = build("youtube", "v3", developerKey=API_KEY)


# ---------------------------------------------------------------------------
# Step 1b: Extract video_id from any common YouTube URL format
# ---------------------------------------------------------------------------
def extract_video_id(url_or_id: str) -> str:
    """
    Accepts a full YouTube URL (several formats) OR a raw video ID,
    and returns just the 11-character video ID.
    """
    url_or_id = url_or_id.strip()

    # Already looks like a raw video ID (11 chars, no slashes/dots)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id

    patterns = [
        r"(?:v=|\/videos\/|embed\/|youtu\.be\/|\/v\/|watch\?v=)([A-Za-z0-9_-]{11})",
        r"youtube\.com\/shorts\/([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    raise ValueError(f"Could not extract a video ID from: {url_or_id}")


# ---------------------------------------------------------------------------
# Step 2: Fetch live stats for a video
# ---------------------------------------------------------------------------
def fetch_live_video_data(url_or_id: str) -> dict:
    """
    Fetches current live stats for a YouTube video.
    Returns a dict with the raw fields needed for feature engineering.
    Raises ValueError if the video doesn't exist / is private / deleted.
    """
    video_id = extract_video_id(url_or_id)

    request = youtube.videos().list(
        part="snippet,contentDetails,statistics",
        id=video_id
    )
    response = request.execute()

    items = response.get("items", [])
    if not items:
        raise ValueError(
            f"No video found for ID '{video_id}'. It may be private, "
            f"deleted, or the URL/ID is incorrect."
        )

    item = items[0]
    snippet = item["snippet"]
    content_details = item["contentDetails"]
    statistics = item["statistics"]

    published_at = datetime.strptime(
        snippet["publishedAt"], "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    video_age_hours = (now - published_at).total_seconds() / 3600

    raw_data = {
        "video_id": video_id,
        "title": snippet.get("title", ""),
        "channel": snippet.get("channelTitle", ""),
        "published_at": published_at,
        "category_id": float(snippet.get("categoryId", -1)),
        "duration": content_details.get("duration", "PT0S"),
        # statistics fields are sometimes missing (e.g. comments disabled)
        "views": float(statistics.get("viewCount", 0)),
        "likes": float(statistics.get("likeCount", 0)),
        "comments": float(statistics.get("commentCount", 0)),
        "snapshot_time": now,
        "video_age_hours": video_age_hours,
    }
    return raw_data


# ---------------------------------------------------------------------------
# Quick manual test (only runs the URL-parsing part offline;
# the actual API call needs internet + a valid key, run this file directly
# in VS Code to test the live call for real)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "dQw4w9WgXcQ",
    ]
    print("Testing URL parsing (offline, no API call):")
    for url in test_urls:
        print(f"  {url}  ->  {extract_video_id(url)}")

    print("\nNow testing a real live API call...")
    sample_url = input("Paste a real YouTube video URL to test: ").strip()
    data = fetch_live_video_data(sample_url)
    for k, v in data.items():
        print(f"  {k}: {v}")
