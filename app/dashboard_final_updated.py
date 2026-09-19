"""
app/dashboard.py

Day 5 — Streamlit Dashboard (Mode 1: Analyze Live Video)

Run with:
    streamlit run app/dashboard.py
(run this command from the PROJECT ROOT folder)
"""

import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import streamlit as st

# Make sure Python can find src/predict_live_v3.py regardless of where
# streamlit is launched from
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, "..", "src")
sys.path.insert(0, _SRC_DIR)

from predict_live_v3_final import get_live_prediction, _model, _feature_columns  # noqa: E402
from mongo_setup import get_database  # noqa: E402

EARLY_CUTOFF = 6
TARGET_LOW, TARGET_HIGH = 18, 30


# ---------------------------------------------------------------------------
# Reference data for Virality scoring, Trend Insights, and Radar comparisons
# — pulled LIVE from MongoDB instead of static local CSV files. This
# eliminates the staleness problem entirely: no separate refresh script or
# scheduler needed. @st.cache_data(ttl=3600) means this only re-queries
# MongoDB once per hour per user session, not on every click, so it stays
# fast while never drifting far out of date.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_reference_data():
    db = get_database()
    videos = pd.DataFrame(list(db["videos_master"].find({}, {"_id": 0})))
    snapshots = pd.DataFrame(list(db["snapshots"].find({}, {"_id": 0})))

    merged = snapshots.merge(videos, on="video_id", how="inner")
    merged["published_at"] = pd.to_datetime(merged["published_at"], utc=True)
    merged["snapshot_time"] = pd.to_datetime(merged["snapshot_time"], utc=True)
    merged["video_age_hours"] = (
        (merged["snapshot_time"] - merged["published_at"]).dt.total_seconds() / 3600
    )

    df = merged.dropna(subset=["video_id", "video_age_hours"]).sort_values(["video_id", "snapshot_time"])
    for col in ["views", "likes", "comments"]:
        df[col] = df.groupby("video_id")[col].transform(lambda x: x.ffill().bfill())
        df[col] = df.groupby("video_id")[col].cummax()  # same view-count recalibration fix as training

    safe_views = df["views"].clip(lower=1)
    df["engagement_rate"] = ((df["likes"] + df["comments"]) / safe_views) * 100

    # "Latest snapshot per video" — used for category/upload-hour trend charts
    latest_idx = df.groupby("video_id")["video_age_hours"].idxmax()
    latest_snapshots = df.loc[latest_idx].copy()
    latest_snapshots["upload_hour"] = latest_snapshots["published_at"].dt.hour

    # Rebuild a training-table-equivalent (early + target) for the virality
    # percentile distribution and radar chart category benchmarks
    early_df = df[df["video_age_hours"] <= EARLY_CUTOFF]
    early_idx = early_df.groupby("video_id")["video_age_hours"].idxmax()
    early_features = early_df.loc[early_idx].copy()
    early_features["early_views_per_hour"] = early_features["views"] / early_features["video_age_hours"].clip(lower=0.01)
    early_features = early_features.rename(columns={
        "views": "early_views", "likes": "early_likes", "comments": "early_comments",
    })
    early_features["early_like_rate"] = (early_features["early_likes"] / early_features["early_views"].clip(lower=1)) * 100
    early_features["early_comment_rate"] = (early_features["early_comments"] / early_features["early_views"].clip(lower=1)) * 100
    early_features["early_engagement_rate"] = early_features["engagement_rate"]

    target_df = df[(df["video_age_hours"] >= TARGET_LOW) & (df["video_age_hours"] <= TARGET_HIGH)].copy()
    target_df["distance_from_24h"] = (target_df["video_age_hours"] - 24).abs()
    target_idx = target_df.groupby("video_id")["distance_from_24h"].idxmin()
    target_rows = target_df.loc[target_idx][["video_id", "engagement_rate"]].rename(
        columns={"engagement_rate": "target_engagement_rate"}
    )

    training_table = early_features[[
        "video_id", "category_id", "early_views_per_hour", "early_like_rate",
        "early_comment_rate", "early_engagement_rate"
    ]].merge(target_rows, on="video_id", how="inner")

    # Same outlier exclusion as the real training pipeline, for consistency
    training_table = training_table[training_table["target_engagement_rate"] <= 50.0]

    return training_table, latest_snapshots


_training_table, _latest_snapshots = load_reference_data()
_engagement_distribution = _training_table["target_engagement_rate"].values

# Full YouTube category ID -> name mapping, covering all 14 categories
# present in the expanded dataset (a few of these are grouped into "Other"
# by the model itself if they had too few training samples — see
# rare_categories_v2.pkl — but we still display their real name here for
# anything shown outside the model's own feature encoding, e.g. charts).
CATEGORY_NAMES = {
    1.0: "Film & Animation", 2.0: "Autos & Vehicles", 10.0: "Music",
    15.0: "Pets & Animals", 17.0: "Sports", 19.0: "Travel & Events",
    20.0: "Gaming", 22.0: "People & Blogs", 23.0: "Comedy",
    24.0: "Entertainment", 25.0: "News & Politics", 26.0: "Howto & Style",
    27.0: "Education", 28.0: "Science & Technology",
}

VIRALITY_PERCENTILE_THRESHOLD = 85  # top 15% = "Viral Potential"

# ---------------------------------------------------------------------------
# Thresholds for the Low / Medium / High badge — computed DYNAMICALLY from
# whatever training table is currently loaded (33rd / 66th percentile of
# its target distribution), rather than hardcoded numbers. This means the
# thresholds automatically stay correct after any future retrain, without
# needing to manually update this file again.
# ---------------------------------------------------------------------------
LOW_HIGH_CUTOFF = float(_training_table["target_engagement_rate"].quantile(0.33))
MEDIUM_HIGH_CUTOFF = float(_training_table["target_engagement_rate"].quantile(0.66))


def get_engagement_badge(value: float) -> tuple[str, str]:
    """Returns (label, color) for the predicted engagement rate."""
    if value < LOW_HIGH_CUTOFF:
        return "LOW", "#e74c3c"
    elif value < MEDIUM_HIGH_CUTOFF:
        return "MEDIUM", "#f39c12"
    else:
        return "HIGH", "#27ae60"


def get_virality_score(predicted_value: float) -> tuple[float, str, str]:
    """
    Returns (percentile, label, color) — how this prediction compares to
    the full training distribution. No new model needed; this is purely
    descriptive statistics on the existing model's output.
    """
    percentile = stats.percentileofscore(_engagement_distribution, predicted_value)
    if percentile >= VIRALITY_PERCENTILE_THRESHOLD:
        return percentile, "🔥 Viral Potential", "#e74c3c"
    elif percentile >= 60:
        return percentile, "📈 Above Average", "#f39c12"
    else:
        return percentile, "📊 Typical", "#3498db"


def get_category_trend_data() -> pd.DataFrame:
    """Category-wise average engagement, from actual historical snapshots."""
    cat_avg = _latest_snapshots.groupby("category_id")["engagement_rate"].mean().sort_values(ascending=False)
    cat_avg.index = [CATEGORY_NAMES.get(c, f"Category {c}") for c in cat_avg.index]
    return cat_avg


def get_upload_hour_trend_data() -> pd.DataFrame:
    """Average engagement by upload hour, from actual historical snapshots."""
    return _latest_snapshots.groupby("upload_hour")["engagement_rate"].mean()


def get_category_benchmark(category_id: float) -> dict:
    """Average early-signal stats for a category, for radar chart comparison."""
    cat_data = _training_table[_training_table["category_id"] == category_id]
    if cat_data.empty:
        return None
    return {
        "early_views_per_hour": cat_data["early_views_per_hour"].mean(),
        "early_like_rate": cat_data["early_like_rate"].mean(),
        "early_comment_rate": cat_data["early_comment_rate"].mean(),
        "early_engagement_rate": cat_data["early_engagement_rate"].mean(),
    }


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="YouTube Engagement Predictor", page_icon="📊", layout="wide")

st.title("📊 Real-Time YouTube Engagement Predictor")
st.caption(
    "Paste a link to a live YouTube video to predict its engagement rate "
    "at ~24 hours after publish, based on its early performance."
)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
video_url = st.text_input(
    "YouTube video URL",
    placeholder="https://www.youtube.com/watch?v=..."
)
analyze_clicked = st.button("Analyze", type="primary")

# ---------------------------------------------------------------------------
# Run prediction
# ---------------------------------------------------------------------------
if analyze_clicked and video_url:
    with st.spinner("Fetching live data and running prediction..."):
        try:
            result = get_live_prediction(video_url)
        except Exception as e:
            st.error(f"Couldn't analyze this video: {e}")
            result = None

    if result:
        # -- Warnings first, so the user sees caveats immediately
        for w in result["warnings"]:
            st.warning(w)

        # -- Video info header
        st.subheader(result["title"])
        st.caption(f"Channel: {result['channel']}  |  Category ID: {result['category_id']}  |  "
                   f"Video age: {result['video_age_hours']:.1f} hours")

        # -- Top row: current stats + prediction badge
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Views", f"{int(result['current_views']):,}")
        col2.metric("Current Likes", f"{int(result['current_likes']):,}")
        col3.metric("Current Comments", f"{int(result['current_comments']):,}")
        col4.metric("Current Engagement Rate", f"{result['current_engagement_rate']}%")

        st.divider()

        pred_col1, pred_col2 = st.columns([1, 2])

        with pred_col1:
            badge_label, badge_color = get_engagement_badge(result["predicted_engagement_rate_24h"])
            st.markdown(
                f"""
                <div style="padding: 20px; border-radius: 10px; background-color: {badge_color}22;
                            border: 2px solid {badge_color}; text-align: center;">
                    <div style="font-size: 14px; color: gray;">Predicted Engagement Rate (~24h)</div>
                    <div style="font-size: 36px; font-weight: bold; color: {badge_color};">
                        {result['predicted_engagement_rate_24h']}%
                    </div>
                    <div style="font-size: 18px; font-weight: bold; color: {badge_color};">
                        {badge_label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with pred_col2:
            # -- Current vs Predicted comparison chart
            fig = go.Figure(data=[
                go.Bar(
                    x=["Current", "Predicted (~24h)"],
                    y=[result["current_engagement_rate"], result["predicted_engagement_rate_24h"]],
                    marker_color=["#3498db", badge_color],
                    text=[f"{result['current_engagement_rate']}%", f"{result['predicted_engagement_rate_24h']}%"],
                    textposition="outside",
                )
            ])
            fig.update_layout(
                title="Current vs Predicted Engagement Rate",
                yaxis_title="Engagement Rate (%)",
                height=320,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # -- Feature importance chart
        st.subheader("What drives this prediction?")
        importances = pd.Series(
            _model.feature_importances_, index=_feature_columns
        ).sort_values(ascending=False).head(8)

        fig2 = go.Figure(data=[
            go.Bar(
                x=importances.values,
                y=importances.index,
                orientation="h",
                marker_color="#8e44ad",
            )
        ])
        fig2.update_layout(
            title="Top Feature Importances (model-wide)",
            xaxis_title="Importance",
            height=350,
            margin=dict(t=40, b=20),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.caption(
            "Note: feature importance reflects the model's overall learned behavior "
            "across all training videos, not just this specific prediction."
        )

        st.divider()

        # -- Virality Score (Phase 6 addition)
        st.subheader("Virality Potential")
        percentile, virality_label, virality_color = get_virality_score(
            result["predicted_engagement_rate_24h"]
        )

        gauge_col, text_col = st.columns([1, 1])
        with gauge_col:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=percentile,
                number={"suffix": "%"},
                title={"text": "Percentile vs. all tracked videos"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": virality_color},
                    "steps": [
                        {"range": [0, 60], "color": "#1a3a5c"},
                        {"range": [60, 85], "color": "#5c4a1a"},
                        {"range": [85, 100], "color": "#5c1a1a"},
                    ],
                },
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=50, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with text_col:
            st.markdown(
                f"""
                <div style="padding: 20px; border-radius: 10px; background-color: {virality_color}22;
                            border: 2px solid {virality_color}; text-align: center; margin-top: 40px;">
                    <div style="font-size: 22px; font-weight: bold; color: {virality_color};">
                        {virality_label}
                    </div>
                    <div style="font-size: 14px; color: gray; margin-top: 8px;">
                        This prediction ranks higher than {percentile:.0f}% of videos
                        in the tracked dataset.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # -- Radar chart: this video vs its category average (Phase 6 addition)
        category_benchmark = get_category_benchmark(result["category_id"])
        if category_benchmark:
            st.subheader("This Video vs. Category Average")

            this_video_stats = {
                "early_views_per_hour": result["current_views"] / max(result["video_age_hours"], 0.01),
                "early_like_rate": (result["current_likes"] / max(result["current_views"], 1)) * 100,
                "early_comment_rate": (result["current_comments"] / max(result["current_views"], 1)) * 100,
                "early_engagement_rate": result["current_engagement_rate"],
            }

            dims = ["Views/Hour", "Like Rate", "Comment Rate", "Engagement Rate"]
            this_vals, cat_vals = [], []
            for key in ["early_views_per_hour", "early_like_rate", "early_comment_rate", "early_engagement_rate"]:
                max_val = max(this_video_stats[key], category_benchmark[key], 0.01)
                this_vals.append((this_video_stats[key] / max_val) * 100)
                cat_vals.append((category_benchmark[key] / max_val) * 100)

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=this_vals, theta=dims, fill="toself", name="This Video"))
            fig_radar.add_trace(go.Scatterpolar(r=cat_vals, theta=dims, fill="toself", name="Category Average"))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                height=400,
                title="Normalized comparison (100 = higher of the two values)",
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.caption("No category benchmark available for comparison (category not in training data).")

elif analyze_clicked and not video_url:
    st.error("Please paste a YouTube video URL first.")

# ---------------------------------------------------------------------------
# Trend Insights Panel (Phase 6 addition) — descriptive analytics from the
# historical tracked dataset, shown regardless of whether a video has been
# analyzed. Not a prediction model — just aggregate stats from real
# collected snapshots, presented as expandable sections.
# ---------------------------------------------------------------------------
st.divider()
with st.expander("📊 Trend Insights (from tracked dataset)"):
    trend_col1, trend_col2 = st.columns(2)

    with trend_col1:
        st.markdown("**Average engagement by category**")
        cat_trend = get_category_trend_data()
        fig_cat = go.Figure(data=[
            go.Bar(x=cat_trend.values, y=cat_trend.index, orientation="h", marker_color="#2ecc71")
        ])
        fig_cat.update_layout(height=300, margin=dict(t=10, b=10), xaxis_title="Avg Engagement Rate (%)")
        st.plotly_chart(fig_cat, use_container_width=True)

    with trend_col2:
        st.markdown("**Average engagement by upload hour**")
        hour_trend = get_upload_hour_trend_data()
        fig_hour = go.Figure(data=[
            go.Bar(x=hour_trend.index, y=hour_trend.values, marker_color="#3498db")
        ])
        fig_hour.update_layout(
            height=300, margin=dict(t=10, b=10),
            xaxis_title="Upload Hour (24h)", yaxis_title="Avg Engagement Rate (%)"
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    st.caption(
        "Based on the latest tracked snapshot of each video in the historical dataset. "
        "Descriptive only — not a prediction."
    )

# ---------------------------------------------------------------------------
# Sidebar — project info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About this project")
    st.write(
        "This dashboard predicts a YouTube video's engagement rate at ~24 hours "
        "after publish, using only its first few hours of performance data."
    )
    st.write("**Model**: Random Forest Regressor (v2, hyperparameter-tuned)")
    st.write(f"**Training data**: {len(_training_table)} videos, continuously "
             f"collected via automated real-time pipeline (AWS Lambda + MongoDB)")
    known_cats = sorted(_training_table["category_id"].unique())
    known_cat_names = ", ".join(CATEGORY_NAMES.get(c, f"Category {c}") for c in known_cats)
    st.write(f"**Known categories ({len(known_cats)})**: {known_cat_names}")
    st.write(
        "**Limitation**: categories with very few training examples are "
        "grouped into 'Other' internally; predictions for videos outside "
        "known categories, or older than 6 hours, are shown with a "
        "reliability warning."
    )
    st.divider()
    st.caption("Data collection runs continuously and automatically via "
               "AWS Lambda + EventBridge + MongoDB Atlas (zero-cost tier). "
               "Future scope: additional platforms, scheduled model "
               "retraining, real-time streaming pipeline.")
