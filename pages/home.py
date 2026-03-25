"""
🏠 Home page — scrape controls and full analysis pipeline trigger.
"""

import streamlit as st

from src.scraper.reddit_scraper import scrape_wsb, save_posts
from src.nlp.preprocessing import preprocess_posts
from src.nlp.sentiment import analyze
from src.rag.vector_store import build_index
from src.rag.agent import create_rag_chain


def render():
    # ── Hero section ────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding: 2rem 0 1rem;">
            <h1 style="font-size:2.8rem; margin:0;">
                🚀 WSB Sentiment Analyzer
            </h1>
            <p style="color:#9CA3AF; font-size:1.1rem; margin-top:0.5rem;">
                Scrape · Analyze · Explore · Ask — all from r/wallstreetbets
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Controls ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        n_posts = st.slider(
            "Number of posts to scrape",
            min_value=10,
            max_value=500,
            value=50,
            step=10,
            help="Reddit API supports up to ~1 000 per listing. "
                 "BERTopic runs best with ≥30 posts.",
        )
    with col2:
        sort_method = st.selectbox(
            "Sort by",
            options=["hot", "top", "new", "rising"],
            index=0,
        )
    with col3:
        st.write("")  # vertical spacer
        st.write("")
        run = st.button("🔄 Scrape & Analyze", type="primary", use_container_width=True)

    # ── Pipeline execution ──────────────────────────────────────────────
    if run:
        with st.status("Running full pipeline…", expanded=True) as status:
            # Step 1 — Scrape
            st.write("📡  Fetching posts from r/wallstreetbets…")
            try:
                df = scrape_wsb(n_posts, sort_method)
            except ValueError as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                st.error(f"Scraping failed: {exc}")
                return
            st.write(f"✅  Retrieved **{len(df)}** posts")

            # Step 2 — Preprocess
            st.write("🧹  Cleaning & preprocessing text…")
            df = preprocess_posts(df)
            st.write(f"✅  **{len(df)}** posts after filtering")

            # Step 3 — Topic + Sentiment analysis
            st.write("🧠  Running BERTopic + FinBERT sentiment analysis…")
            df, topic_model = analyze(df)
            st.write("✅  Topics & sentiment labels assigned")

            # Step 4 — Save
            path = save_posts(df)
            st.write(f"💾  Saved to `{path.name}`")

            # Step 5 — Build RAG index
            st.write("📚  Building vector store for RAG agent…")
            vector_store = build_index(df)
            rag_chain = create_rag_chain(vector_store)
            st.write("✅  RAG index ready")

            # Persist to session state
            st.session_state["df"] = df
            st.session_state["topic_model"] = topic_model
            st.session_state["rag_chain"] = rag_chain

            status.update(label="✅ Pipeline complete!", state="complete")

        # ── Summary ─────────────────────────────────────────────────────
        st.success(f"Analyzed **{len(df)}** posts successfully!")

        col1, col2, col3 = st.columns(3)
        n_pos = len(df[df["sentiment_label"] == "positive"])
        n_neg = len(df[df["sentiment_label"] == "negative"])
        n_neu = len(df[df["sentiment_label"] == "neutral"])
        col1.metric("Bullish 🟢", n_pos)
        col2.metric("Bearish 🔴", n_neg)
        col3.metric("Neutral ⚪", n_neu)

    # ── Show status if data already loaded ──────────────────────────────
    elif "df" in st.session_state:
        df = st.session_state["df"]
        st.info(
            f"📦 **{len(df)}** posts already loaded in this session. "
            f"Navigate to other tabs or scrape again."
        )
    else:
        st.info("👆 Configure and click **Scrape & Analyze** to get started.")
