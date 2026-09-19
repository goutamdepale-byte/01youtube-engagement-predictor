"""
src/mongo_setup.py

Phase 1 - Step 1.4 & 1.5: Test MongoDB Atlas connection and set up the
TTL (time-to-live) index that auto-expires old snapshot data.

Run this ONCE to verify your connection works and to create the TTL index.
Safe to re-run — creating an index that already exists is a no-op in MongoDB.
"""

import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH = os.path.join(_THIS_DIR, "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH)

MONGODB_URI = os.getenv("MONGODB_URI")

# How long to keep snapshot data before auto-deleting it (sliding window).
# 60 days is a reasonable starting point — adjust later based on how much
# history you actually want to retrain on.
SNAPSHOT_RETENTION_DAYS = 60


def get_database():
    if not MONGODB_URI or MONGODB_URI == "paste_your_mongodb_connection_string_here":
        raise ValueError(
            "MONGODB_URI not found. Open the .env file in the project root "
            "and paste your real MongoDB Atlas connection string in place "
            "of the placeholder text."
        )
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    return client["youtube_engagement"]


def test_connection():
    """Confirms we can actually reach MongoDB Atlas and write/read data."""
    db = get_database()

    # Trigger a real network call to confirm the connection works
    db.command("ping")
    print("Connection successful — MongoDB Atlas is reachable.")

    # Write + read back a test document
    test_collection = db["connection_test"]
    test_doc = {"message": "hello from predict_live project", "tested_at": datetime.now(timezone.utc)}
    result = test_collection.insert_one(test_doc)
    print(f"Test document inserted with id: {result.inserted_id}")

    fetched = test_collection.find_one({"_id": result.inserted_id})
    print(f"Test document read back: {fetched}")

    test_collection.delete_one({"_id": result.inserted_id})
    print("Test document cleaned up.")


def setup_ttl_index():
    """
    Creates a TTL index on the `snapshots` collection so documents older
    than SNAPSHOT_RETENTION_DAYS are AUTOMATICALLY deleted by MongoDB —
    no manual cleanup script needed. This is what makes the "sliding
    window" retraining strategy work without any extra code.

    Requires each snapshot document to have a `snapshot_time` field
    stored as a proper datetime (not a string) — poll_snapshots.py
    already does this correctly.
    """
    db = get_database()
    snapshots = db["snapshots"]

    snapshots.create_index(
        "snapshot_time",
        expireAfterSeconds=SNAPSHOT_RETENTION_DAYS * 24 * 60 * 60,
        name="snapshot_ttl_index",
    )
    print(f"TTL index created on 'snapshots.snapshot_time' — documents older "
          f"than {SNAPSHOT_RETENTION_DAYS} days will be auto-deleted by MongoDB.")


def setup_indexes_for_performance():
    """
    A few extra indexes to keep queries fast as data grows — not strictly
    required at small scale, but good practice and effectively free.
    """
    db = get_database()

    db["videos_master"].create_index("video_id", unique=True)
    db["videos_master"].create_index("published_at")
    db["snapshots"].create_index("video_id")

    print("Performance indexes created on videos_master and snapshots.")


if __name__ == "__main__":
    print("Testing MongoDB Atlas connection...\n")
    test_connection()

    print("\nSetting up TTL index (auto-expiry for old snapshots)...")
    setup_ttl_index()

    print("\nSetting up performance indexes...")
    setup_indexes_for_performance()

    print("\nPhase 1 setup complete. MongoDB Atlas is ready for Phase 2.")
