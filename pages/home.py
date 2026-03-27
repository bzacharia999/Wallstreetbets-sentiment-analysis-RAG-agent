"""
Home page — data loading (CSV upload / cached data) and full analysis pipeline.
"""

import streamlit as st

from src.scraper.reddit_scraper import (
    load_from_uploaded_file,
    save_posts,
    load_latest_posts,
)
from src.nlp.preprocessing import preprocess_posts
from src.nlp.sentiment import analyze
from src.rag.vector_store import build_index
from src.rag.agent import create_rag_chain


def render():
    # Hero section
    st.markdown(
        """
        <div style="text-align:center; padding: 2rem 0 1rem;">
            <h1 style="font-size:2.8rem; margin:0;">
                WallStreetBets Sentiment Analysis and Agent
            </h1>
            <p style="color:#9CA3AF; font-size:1.1rem; margin-top:0.5rem;">
                Load, analyze, and ask questions about data from r/wallstreetbets
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Data source selection
    source_tab = st.radio(
        "Choose data source",
        ["Upload CSV / JSON / Parquet", "Load previous session"],
        horizontal=True,
    )

    df = None

    if source_tab == "Upload CSV / JSON / Parquet":
        df = _upload_section()
    elif source_tab == "Load previous session":
        df = _load_previous_section()

    # Pipeline execution
    if df is not None:
        _run_pipeline(df)

    # Show status if data already loaded
    elif "df" in st.session_state:
        loaded_df = st.session_state["df"]
        st.info(
            f"**{len(loaded_df)}** posts already loaded in this session. "
            f"Navigate to other tabs or load new data."
        )


def _upload_section() -> "pd.DataFrame | None":
    """File upload UI."""
    st.markdown("#### Upload a WSB dataset")
    st.caption(
        "Upload a CSV, JSON, JSONL, or Parquet file containing WSB posts. "
        "Must have at least a **title** column. Optional: selftext, score, "
        "num_comments, author, flair."
    )

    uploaded = st.file_uploader(
        "Drop your file here",
        type=["csv", "json", "jsonl", "parquet"],
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        max_rows = st.slider(
            "Max rows to load",
            min_value=50,
            max_value=5000,
            value=500,
            step=50,
            help="Limit rows for faster processing. BERTopic works best with 50-500 posts.",
        )
    with col2:
        st.write("")
        st.write("")
        run = st.button("Load & Analyze", type="primary", use_container_width=True)

    if run and uploaded:
        try:
            df = load_from_uploaded_file(uploaded)
            if len(df) > max_rows:
                df = df.head(max_rows)
            st.write(f"Loaded **{len(df)}** rows from `{uploaded.name}`")
            return df
        except Exception as exc:
            st.error(f"Failed to load file: {exc}")
    elif run and not uploaded:
        st.warning("Please upload a file first.")

    return None


def _load_previous_section() -> "pd.DataFrame | None":
    """Load from previously saved parquet files."""
    st.markdown("#### Load from a previous session")

    prev_df = load_latest_posts()
    if prev_df is not None:
        st.write(f"Found saved data: **{len(prev_df)}** posts")
        if st.button("Load & Re-analyze", type="primary"):
            return prev_df
    else:
        st.info("No previously saved data found in `data/`.")

    return None


def _run_pipeline(df):
    """Run the full NLP + RAG pipeline on a loaded DataFrame."""
    with st.status("Running analysis pipeline…", expanded=True) as status:
        # Step 1 — Preprocess
        st.write("Cleaning & preprocessing text...")
        df = preprocess_posts(df)
        st.write(f"**{len(df)}** posts after filtering")

        if len(df) < 5:
            st.error("Too few posts after filtering. Need at least 5. Try loading more data.")
            return

        # Step 2 — Topic + Sentiment analysis
        st.write("Running BERTopic + FinBERT sentiment analysis...")
        df, topic_model = analyze(df)
        st.write("Topics & sentiment labels assigned")

        # Step 3 — Save
        path = save_posts(df)
        st.write(f"Saved to `{path.name}`")

        # Step 4 — Build RAG index
        st.write("Building vector store for RAG agent...")
        vector_store = build_index(df)
        rag_chain = create_rag_chain(vector_store)
        st.write("RAG index ready")

        # Persist to session state
        st.session_state["df"] = df
        st.session_state["topic_model"] = topic_model
        st.session_state["rag_chain"] = rag_chain

        status.update(label="Pipeline complete!", state="complete")

    # Summary
    st.success(f"Analyzed **{len(df)}** posts successfully!")

    col1, col2, col3 = st.columns(3)
    n_pos = len(df[df["sentiment_label"] == "positive"])
    n_neg = len(df[df["sentiment_label"] == "negative"])
    n_neu = len(df[df["sentiment_label"] == "neutral"])
    col1.metric("Bullish", n_pos)
    col2.metric("Bearish", n_neg)
    col3.metric("Neutral", n_neu)
