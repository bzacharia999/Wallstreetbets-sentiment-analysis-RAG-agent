"""
Explorer page — searchable, filterable table of scraped posts.
"""

import streamlit as st
import pandas as pd


def render():
    st.markdown(
        "<h1 style='text-align:center;'>Post Explorer</h1>",
        unsafe_allow_html=True,
    )

    if "df" not in st.session_state:
        st.warning("No data loaded. Go to **Home** to load posts first.")
        return

    df: pd.DataFrame = st.session_state["df"]

    # ── Filters ─────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        sentiment_options = sorted(df["sentiment_label"].dropna().unique().tolist())
        sentiment_filter = st.multiselect(
            "Sentiment",
            options=sentiment_options,
            default=sentiment_options,
        )

    with col2:
        flair_options = sorted(df["flair"].dropna().unique().tolist())
        flair_filter = st.multiselect(
            "Flair",
            options=flair_options if flair_options else ["(none)"],
            default=flair_options if flair_options else [],
        )

    with col3:
        search_term = st.text_input("Search titles", placeholder="e.g. GME, TSLA")

    # ── Apply filters ───────────────────────────────────────────────────
    mask = df["sentiment_label"].isin(sentiment_filter)

    if flair_options:
        mask &= df["flair"].isin(flair_filter)

    if search_term:
        mask &= df["title"].str.contains(search_term, case=False, na=False)

    filtered = df.loc[mask]

    # ── Summary bar ─────────────────────────────────────────────────────
    st.caption(f"Showing **{len(filtered)}** of {len(df)} posts")

    # ── Table ───────────────────────────────────────────────────────────
    display_cols = [
        "title", "score", "num_comments",
        "sentiment_label", "sentiment_score",
        "topic_label", "tickers", "flair",
    ]
    # Keep only columns that actually exist
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=550,
        column_config={
            "title": st.column_config.TextColumn("Title", width="large"),
            "score": st.column_config.NumberColumn("Score", format="%d"),
            "num_comments": st.column_config.NumberColumn("Comments", format="%d"),
            "sentiment_label": st.column_config.TextColumn("Sentiment"),
            "sentiment_score": st.column_config.ProgressColumn(
                "Confidence", min_value=0, max_value=1, format="%.0%%"
            ),
            "topic_label": st.column_config.TextColumn("Topic"),
            "tickers": st.column_config.ListColumn("Tickers"),
            "flair": st.column_config.TextColumn("Flair"),
        },
    )

    # ── Expandable post detail ──────────────────────────────────────────
    st.divider()
    if not filtered.empty:
        with st.expander("View full text of selected post"):
            idx = st.selectbox(
                "Select post",
                options=filtered.index.tolist(),
                format_func=lambda i: filtered.loc[i, "title"][:80],
            )
            if idx is not None:
                row = filtered.loc[idx]
                st.markdown(f"### {row['title']}")
                st.caption(
                    f"by u/{row.get('author', '?')} | "
                    f"Score: {row.get('score', 0)} | "
                    f"Comments: {row.get('num_comments', 0)} | "
                    f"Sentiment: {row.get('sentiment_label', '?')}"
                )
                st.markdown(row.get("selftext", "") or "*No body text*")
