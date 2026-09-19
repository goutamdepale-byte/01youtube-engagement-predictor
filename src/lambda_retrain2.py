"""
src/lambda_retrain.py

Phase 7 - Lambda-ready Automated Retraining.

Runs entirely on AWS Lambda (triggered weekly by EventBridge). Unlike
lambda_discover.py / lambda_poll.py, this DOES use pandas + scikit-learn
(unavoidable for modeling) — the deployment package will be larger and
needs to be uploaded via S3 as a one-time deploy step (S3 is NOT used for
ongoing data storage, only as a temporary pass-through to get the larger
package into Lambda).

KEY DESIGN: the trained model is NOT saved to a local file (Lambda's
filesystem is ephemeral and resets between invocations, and isn't
reachable by the dashboard anyway). Instead, the model is serialized to
bytes and stored directly as a MongoDB document. The dashboard reads the
currently "active" model from MongoDB — making MongoDB the single source
of truth end-to-end, with zero local files or manual steps required.
"""

import os
import re
import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
from pymongo import MongoClient
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

EARLY_CUTOFF = 6
TARGET_LOW, TARGET_HIGH = 18, 30
MIN_SAMPLES_PER_CATEGORY = 10
IMPLAUSIBLE_THRESHOLD = 50.0


def get_mongo_db():
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI environment variable not set.")
    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    return client["youtube_engagement"]


# ---------------------------------------------------------------------------
# Step 1: Pull + reconstruct raw data from MongoDB
# ---------------------------------------------------------------------------
def pull_and_reconstruct(db) -> pd.DataFrame:
    videos = pd.DataFrame(list(db["videos_master"].find({}, {"_id": 0})))
    snapshots = pd.DataFrame(list(db["snapshots"].find({}, {"_id": 0})))

    if videos.empty or snapshots.empty:
        raise ValueError("No data found in MongoDB — nothing to retrain on.")

    merged = snapshots.merge(videos, on="video_id", how="inner")
    merged["published_at"] = pd.to_datetime(merged["published_at"], utc=True)
    merged["snapshot_time"] = pd.to_datetime(merged["snapshot_time"], utc=True)
    merged["video_age_hours"] = (
        (merged["snapshot_time"] - merged["published_at"]).dt.total_seconds() / 3600
    )
    return merged


# ---------------------------------------------------------------------------
# Step 2: Clean (cummax fix for view-count recalibration)
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["video_id", "video_age_hours"]).copy()
    df = df.sort_values(["video_id", "snapshot_time"]).reset_index(drop=True)

    static_cols = ["title", "channel", "category_id", "duration", "published_at"]
    for col in static_cols:
        df[col] = df.groupby("video_id")[col].transform(lambda x: x.ffill().bfill())

    count_cols = ["views", "likes", "comments"]
    for col in count_cols:
        df[col] = df.groupby("video_id")[col].transform(lambda x: x.ffill().bfill())
    for col in count_cols:
        df[col] = df.groupby("video_id")[col].cummax()

    safe_views = df["views"].clip(lower=1)
    df["like_rate"] = (df["likes"] / safe_views) * 100
    df["comment_rate"] = (df["comments"] / safe_views) * 100
    df["engagement_rate"] = ((df["likes"] + df["comments"]) / safe_views) * 100

    df = df.dropna(subset=["views", "likes", "comments", "like_rate", "comment_rate", "engagement_rate"])

    def parse_duration(duration_str):
        if pd.isna(duration_str):
            return np.nan
        match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", str(duration_str))
        if not match:
            return np.nan
        h = int(match.group(1)) if match.group(1) else 0
        m = int(match.group(2)) if match.group(2) else 0
        s = int(match.group(3)) if match.group(3) else 0
        return h * 3600 + m * 60 + s

    df["duration_seconds"] = df["duration"].apply(parse_duration)

    has_early = set(df[df["video_age_hours"] <= EARLY_CUTOFF]["video_id"].unique())
    has_target = set(df[(df["video_age_hours"] >= TARGET_LOW) &
                         (df["video_age_hours"] <= TARGET_HIGH)]["video_id"].unique())
    usable_videos = has_early & has_target

    return df[df["video_id"].isin(usable_videos)].copy()


# ---------------------------------------------------------------------------
# Step 3: Feature engineering (early window + target + outlier exclusion)
# ---------------------------------------------------------------------------
def build_training_table(df: pd.DataFrame):
    df["upload_hour"] = df["published_at"].dt.hour
    df["upload_dayofweek"] = df["published_at"].dt.dayofweek
    df["is_weekend"] = df["upload_dayofweek"].isin([5, 6]).astype(int)

    early_df = df[df["video_age_hours"] <= EARLY_CUTOFF].copy()
    early_idx = early_df.groupby("video_id")["video_age_hours"].idxmax()
    early_features = early_df.loc[early_idx].copy()
    early_features = early_features.rename(columns={
        "views": "early_views", "likes": "early_likes", "comments": "early_comments",
        "views_per_hour": "early_views_per_hour", "like_rate": "early_like_rate",
        "comment_rate": "early_comment_rate", "engagement_rate": "early_engagement_rate",
        "video_age_hours": "early_snapshot_age_hours",
    })
    early_features["early_views_per_hour"] = (
        early_features["early_views"] / early_features["early_snapshot_age_hours"].clip(lower=0.01)
    )
    early_snapshot_counts = early_df.groupby("video_id").size().rename("early_snapshot_count")
    early_features = early_features.merge(early_snapshot_counts, on="video_id")

    early_cols = [
        "video_id", "category_id", "duration_seconds", "upload_hour", "upload_dayofweek",
        "is_weekend", "early_snapshot_age_hours", "early_snapshot_count", "early_views",
        "early_likes", "early_comments", "early_views_per_hour", "early_like_rate",
        "early_comment_rate", "early_engagement_rate",
    ]
    early_features = early_features[early_cols]

    target_window_df = df[(df["video_age_hours"] >= TARGET_LOW) & (df["video_age_hours"] <= TARGET_HIGH)].copy()
    target_window_df["distance_from_24h"] = (target_window_df["video_age_hours"] - 24).abs()
    target_idx = target_window_df.groupby("video_id")["distance_from_24h"].idxmin()
    target_rows = target_window_df.loc[target_idx].copy()
    target_rows = target_rows.rename(columns={
        "engagement_rate": "target_engagement_rate", "video_age_hours": "target_snapshot_age_hours",
    })
    target_rows = target_rows[["video_id", "target_snapshot_age_hours", "target_engagement_rate"]]

    implausible = target_rows[target_rows["target_engagement_rate"] > IMPLAUSIBLE_THRESHOLD]["video_id"].tolist()
    target_rows = target_rows[~target_rows["video_id"].isin(implausible)]

    training_table = early_features.merge(target_rows, on="video_id", how="inner")
    return training_table, implausible


# ---------------------------------------------------------------------------
# Step 4: Train + evaluate
# ---------------------------------------------------------------------------
def train_and_evaluate(training_table: pd.DataFrame):
    category_counts = training_table["category_id"].value_counts()
    rare_categories = category_counts[category_counts < MIN_SAMPLES_PER_CATEGORY].index.tolist()
    training_table = training_table.copy()
    training_table["category_grouped"] = training_table["category_id"].apply(
        lambda c: "Other" if c in rare_categories else str(c)
    )

    feature_cols_raw = [
        "category_grouped", "duration_seconds", "upload_hour", "upload_dayofweek",
        "is_weekend", "early_snapshot_age_hours", "early_snapshot_count",
        "early_views", "early_likes", "early_comments", "early_views_per_hour",
        "early_like_rate", "early_comment_rate", "early_engagement_rate",
    ]
    X = training_table[feature_cols_raw].copy()
    X = pd.get_dummies(X, columns=["category_grouped"], prefix="cat")
    y = training_table["target_engagement_rate"].copy()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # Smaller grid + n_jobs=1: Lambda has limited, shared CPU — a large grid
    # with n_jobs=-1 can be slower/unreliable in this environment than on a
    # full local machine, so keep this modest to stay well within the
    # Lambda timeout.
    param_grid = {"n_estimators": [200, 300], "max_depth": [6, 8, None]}
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=kf, scoring="r2", n_jobs=1)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_

    model = RandomForestRegressor(random_state=42, **best_params)
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=1)
    rmse_scores = -cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=1)
    mae_scores = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=1)

    final_model = RandomForestRegressor(random_state=42, **best_params)
    final_model.fit(X, y)

    metrics = {
        "r2_mean": float(r2_scores.mean()), "r2_std": float(r2_scores.std()),
        "rmse_mean": float(rmse_scores.mean()), "mae_mean": float(mae_scores.mean()),
        "n_videos": int(len(training_table)), "n_categories": int(training_table["category_id"].nunique()),
        "best_params": best_params,
    }
    return final_model, list(X.columns), rare_categories, metrics


# ---------------------------------------------------------------------------
# Step 5: MongoDB-based model storage (replaces local .pkl files entirely)
# ---------------------------------------------------------------------------
def serialize_to_bytes(obj) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    return buffer.getvalue()


def get_current_champion_r2(db):
    doc = db["model_store"].find_one({"is_active": True})
    if doc is None:
        return None
    return doc["metrics"]["r2_mean"]


def deploy_new_model(db, model, feature_cols, rare_categories, metrics):
    """Stores the new model as the active model in MongoDB, deactivating the previous one."""
    db["model_store"].update_many({"is_active": True}, {"$set": {"is_active": False}})

    new_doc = {
        "trained_at": datetime.now(timezone.utc),
        "model_bytes": serialize_to_bytes(model),
        "feature_columns_bytes": serialize_to_bytes(feature_cols),
        "rare_categories_bytes": serialize_to_bytes(rare_categories),
        "metrics": metrics,
        "is_active": True,
    }
    db["model_store"].insert_one(new_doc)


def run_retrain() -> dict:
    db = get_mongo_db()
    run_record = {"timestamp": datetime.now(timezone.utc), "deployed": False}

    try:
        raw = pull_and_reconstruct(db)
        cleaned = clean_data(raw)
        training_table, excluded = build_training_table(cleaned)
        run_record["excluded_outlier_videos"] = excluded

        if len(training_table) < 50:
            run_record["status"] = f"skipped — too few usable videos ({len(training_table)})"
            db["retrain_log"].insert_one(run_record)
            return run_record

        challenger_model, feature_cols, rare_cats, metrics = train_and_evaluate(training_table)
        run_record["metrics"] = metrics

        champion_r2 = get_current_champion_r2(db)
        run_record["champion_r2_before"] = champion_r2

        should_deploy = champion_r2 is None or metrics["r2_mean"] >= champion_r2

        if should_deploy:
            deploy_new_model(db, challenger_model, feature_cols, rare_cats, metrics)
            run_record["deployed"] = True
            run_record["status"] = "deployed — new model is equal or better"
        else:
            run_record["status"] = "not deployed — challenger did not beat current champion"

    except Exception as e:
        run_record["status"] = f"failed — {str(e)}"

    db["retrain_log"].insert_one(run_record)
    return run_record


def lambda_handler(event, context):
    result = run_retrain()
    print(result)

    # Build an EXPLICITLY JSON-safe response rather than reusing `result`
    # directly. pymongo's insert_one() mutates the dict it's given, adding
    # an "_id" field (a MongoDB ObjectId) — Lambda's response marshaling
    # can't serialize that, or the datetime "timestamp" field, so we pick
    # out only the plain, safe values explicitly.
    safe_response = {
        "status": result.get("status"),
        "deployed": result.get("deployed"),
    }
    if "metrics" in result:
        safe_response["metrics"] = {
            k: (v if not isinstance(v, dict) else v)  # best_params is already plain types (ints/strings)
            for k, v in result["metrics"].items()
        }
    if "excluded_outlier_videos" in result:
        safe_response["excluded_outlier_videos"] = result["excluded_outlier_videos"]
    if "champion_r2_before" in result:
        safe_response["champion_r2_before"] = result["champion_r2_before"]

    return safe_response


if __name__ == "__main__":
    print(run_retrain())
