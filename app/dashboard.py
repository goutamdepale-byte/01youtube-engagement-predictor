"""
app/dashboard.py

Day 5 — Streamlit Dashboard (Mode 1: Analyze Live Video)

Run with:
    streamlit run app/dashboard.py
(run this command from the PROJECT ROOT folder)
"""

import os
import sys

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make sure Python can find src/predict_live.py regardless of where
# streamlit is launched from
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, "..", "src")
sys.path.insert(0, _SRC_DIR)

from predict_live import get_live_prediction, _model, _feature_columns  # noqa: E402

# ---------------------------------------------------------------------------
# Thresholds for the Low / Medium / High badge — derived from the actual
# training data's target distribution (33rd / 66th percentile), not
# arbitrary guesses.
# ---------------------------------------------------------------------------
LOW_HIGH_CUTOFF = 1.70
MEDIUM_HIGH_CUTOFF = 4.31


def get_engagement_badge(value: float) -> tuple[str, str]:
    """Returns (label, color) for the predicted engagement rate."""
    if value < LOW_HIGH_CUTOFF:
        return "LOW", "#e74c3c"
    elif value < MEDIUM_HIGH_CUTOFF:
        return "MEDIUM", "#f39c12"
    else:
        return "HIGH", "#27ae60"


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

elif analyze_clicked and not video_url:
    st.error("Please paste a YouTube video URL first.")

# ---------------------------------------------------------------------------
# Sidebar — project info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About this project")
    st.write(
        "This dashboard predicts a YouTube video's engagement rate at ~24 hours "
        "after publish, using only its first few hours of performance data."
    )
    st.write("**Model**: Random Forest Regressor")
    st.write("**Training data**: 162 videos, Aug 8–21, 2026")
    st.write("**Known categories**: Film & Animation, Music, Gaming, "
             "People & Blogs, Comedy, Entertainment")
    st.write(
        "**Limitation**: predictions for videos outside these categories, "
        "or older than 6 hours, are shown with a reliability warning."
    )
    st.divider()
    st.caption("Future scope: additional platforms, MongoDB storage, "
               "real-time streaming pipeline (Kafka/Redis).")
