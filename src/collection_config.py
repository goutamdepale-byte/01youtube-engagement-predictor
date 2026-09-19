"""
src/collection_config.py

Shared settings for Phase 1 data collection (discovery + polling).
Keeping this separate so both scripts stay in sync.
"""

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_THIS_DIR, "..", "data", "realtime")

VIDEOS_MASTER_PATH = os.path.join(DATA_DIR, "videos_master.parquet")
SNAPSHOTS_PATH = os.path.join(DATA_DIR, "snapshots.parquet")

REGION_CODE = "IN"          # India — matches your location/audience
MAX_VIDEOS_PER_CATEGORY = 40   # cap keeps total tracked videos + storage manageable
FRESHNESS_HOURS = 3          # IMPORTANT: keep this well under the model's early-window
                              # cutoff (6h). A video discovered AFTER it's already past
                              # 6h old can NEVER get a valid early-window snapshot (you
                              # can't retroactively poll its past) — it would just waste
                              # API quota. 3h leaves a safe buffer to actually poll it
                              # before it crosses the 6h cutoff.
                              #
                              # To catch MORE videos per category, don't widen this —
                              # instead run discover_videos.py MORE OFTEN (e.g. every
                              # 3 hours via EventBridge/cron). This accumulates a
                              # continuous stream of fresh videos across the day,
                              # instead of one wide-but-wasteful net.
DISCOVERY_INTERVAL_HOURS = 3  # how often to RE-RUN discover_videos.py (not a code loop —
                               # this is the recommended EventBridge/cron schedule)

# Videos older than this are no longer polled — they've already passed the
# ~24h target window, so further snapshots add no value and just waste
# storage + API quota
STOP_POLLING_AFTER_HOURS = 32

POLL_INTERVAL_SECONDS = 600  # 10 minutes, matches your original snapshot design

# Storage-efficient dtypes for the frequently-repeated snapshot table
SNAPSHOT_DTYPES = {
    "views": "int32",
    "likes": "int32",
    "comments": "int32",
}
