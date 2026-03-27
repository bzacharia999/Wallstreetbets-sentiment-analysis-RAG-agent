"""
Dashboard page — KPI metrics and Plotly visualizations.
"""

import streamlit as st

from src.nlp.sentiment import generate_visualizations


def render():
    st.markdown(
        "<h1 style='text-align:center;'>Sentiment Dashboard</h1>",
        unsafe_allow_html=True,
    )

    if "df" not in st.session_state or "topic_model" not in st.session_state:
        st.warning("No data loaded. Go to **Home** to load posts first.")
        return

    df = st.session_state["df"]
    topic_model = st.session_state["topic_model"]

    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Posts", len(df))
    col2.metric("Bullish", len(df[df["sentiment_label"] == "positive"]))
    col3.metric("Bearish", len(df[df["sentiment_label"] == "negative"]))
    col4.metric("Neutral", len(df[df["sentiment_label"] == "neutral"]))

    st.divider()

    # Generate all visualizations
    figs = generate_visualizations(topic_model, df)

    # Row 1: Sentiment distribution + Ticker mentions
    c1, c2 = st.columns(2)
    with c1:
        if figs.get("sentiment_dist") is not None:
            st.plotly_chart(figs["sentiment_dist"], use_container_width=True)
    with c2:
        if figs.get("ticker_freq") is not None:
            st.plotly_chart(figs["ticker_freq"], use_container_width=True)
        else:
            st.info("No ticker symbols ($TICKER) found in posts.")

    # Row 2: Topic barchart (full width)
    if figs.get("topic_barchart") is not None:
        st.plotly_chart(figs["topic_barchart"], use_container_width=True)

    # Row 3: Expandable deep-dive charts
    with st.expander("Topic Map (Intertopic Distance)", expanded=False):
        if figs.get("topic_map") is not None:
            st.plotly_chart(figs["topic_map"], use_container_width=True)
        else:
            st.info("Not enough topics to generate a map.")

    with st.expander("Topic Hierarchy", expanded=False):
        if figs.get("topic_hierarchy") is not None:
            st.plotly_chart(figs["topic_hierarchy"], use_container_width=True)
        else:
            st.info("Not enough topics to generate a hierarchy.")

    with st.expander("Sentiment by Topic", expanded=False):
        if figs.get("topic_sentiment") is not None:
            st.plotly_chart(figs["topic_sentiment"], use_container_width=True)

    # Average sentiment score
    st.divider()
    avg_score = df["sentiment_score"].mean()
    dominant = df["sentiment_label"].value_counts().idxmax()
    emoji_map = {"positive": "Bullish", "negative": "Bearish", "neutral": "Neutral"}
    st.markdown(
        f"**Overall mood:** {emoji_map.get(dominant, dominant)} &nbsp;|&nbsp; "
        f"**Avg confidence:** {avg_score:.1%}"
    )
