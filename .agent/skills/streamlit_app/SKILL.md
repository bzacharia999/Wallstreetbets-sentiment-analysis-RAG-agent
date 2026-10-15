---
name: Streamlit App
description: How to build the multi-page Streamlit interface that ties together scraping, sentiment analysis, and the RAG agent
---

# Streamlit App Skill

## Overview
Build a polished, multi-tab Streamlit app that serves as the unified interface for:
- Triggering Reddit scraping
- Viewing sentiment analysis dashboards
- Exploring individual posts
- Chatting with the RAG agent

---

## Prerequisites

```bash
pip install streamlit plotly pandas
```

---

## Implementation Steps

### 1. App Entry Point (`app.py`)

```python
import streamlit as st

st.set_page_config(
    page_title="WSB Sentiment & RAG Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar navigation
tab = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Dashboard", "🔍 Explorer", "🤖 Ask WSB"],
)

if tab == "🏠 Home":
    from pages import home
    home.render()
elif tab == "📊 Dashboard":
    from pages import dashboard
    dashboard.render()
elif tab == "🔍 Explorer":
    from pages import explorer
    explorer.render()
elif tab == "🤖 Ask WSB":
    from pages import ask_wsb
    ask_wsb.render()
```

### 2. Home Page — Scrape Controls

```python
# pages/home.py
import streamlit as st
from src.scraper.reddit_scraper import scrape_wsb, save_posts
from src.nlp.preprocessing import preprocess_posts
from src.nlp.sentiment import analyze

def render():
    st.title("🚀 WSB Sentiment Analyzer")
    st.markdown("Scrape, analyze, and explore r/wallstreetbets.")

    col1, col2 = st.columns(2)
    with col1:
        n_posts = st.slider("Number of posts", 10, 500, 100)
    with col2:
        sort_method = st.selectbox("Sort by", ["hot", "top", "new", "rising"])

    if st.button("🔄 Scrape & Analyze", type="primary"):
        with st.status("Working...", expanded=True) as status:
            st.write("Scraping r/wallstreetbets...")
            df = scrape_wsb(n_posts, sort_method)

            st.write("Preprocessing text...")
            df = preprocess_posts(df)

            st.write("Running sentiment analysis...")
            df, topic_model = analyze(df)

            save_posts(df)
            st.session_state["df"] = df
            st.session_state["topic_model"] = topic_model

            status.update(label="✅ Done!", state="complete")

        st.success(f"Analyzed {len(df)} posts successfully!")
```

### 3. Dashboard Page — Visualizations

```python
# pages/dashboard.py
import streamlit as st
from src.nlp.sentiment import generate_visualizations

def render():
    st.title("📊 Sentiment Dashboard")

    if "df" not in st.session_state:
        st.warning("No data loaded. Go to 🏠 Home to scrape posts first.")
        return

    df = st.session_state["df"]
    topic_model = st.session_state["topic_model"]
    figs = generate_visualizations(topic_model, df)

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Posts", len(df))
    col2.metric("Bullish 🟢", len(df[df["sentiment_label"] == "positive"]))
    col3.metric("Bearish 🔴", len(df[df["sentiment_label"] == "negative"]))
    col4.metric("Neutral ⚪", len(df[df["sentiment_label"] == "neutral"]))

    # Charts
    st.plotly_chart(figs["sentiment_dist"], use_container_width=True)
    st.plotly_chart(figs["topic_barchart"], use_container_width=True)

    with st.expander("Topic Map"):
        st.plotly_chart(figs["topic_map"], use_container_width=True)
    with st.expander("Topic Hierarchy"):
        st.plotly_chart(figs["topic_hierarchy"], use_container_width=True)
    with st.expander("Sentiment by Topic"):
        st.plotly_chart(figs["topic_sentiment"], use_container_width=True)
```

### 4. Explorer Page — Post Table

```python
# pages/explorer.py
import streamlit as st

def render():
    st.title("🔍 Post Explorer")

    if "df" not in st.session_state:
        st.warning("No data loaded.")
        return

    df = st.session_state["df"]

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        sentiment_filter = st.multiselect(
            "Sentiment", ["positive", "negative", "neutral"],
            default=["positive", "negative", "neutral"],
        )
    with col2:
        flair_options = df["flair"].dropna().unique().tolist()
        flair_filter = st.multiselect("Flair", flair_options, default=flair_options)

    filtered = df[
        df["sentiment_label"].isin(sentiment_filter)
        & df["flair"].isin(flair_filter)
    ]

    st.dataframe(
        filtered[["title", "score", "num_comments", "sentiment_label",
                   "sentiment_score", "topic_label", "tickers", "flair"]],
        use_container_width=True,
        height=500,
    )
```

### 5. Ask WSB Page — RAG Chat

```python
# pages/ask_wsb.py
import streamlit as st
from src.rag.agent import ask

def render():
    st.title("🤖 Ask WSB")
    st.caption("Ask anything about the scraped r/wallstreetbets posts.")

    if "rag_chain" not in st.session_state:
        st.warning("No RAG index loaded. Scrape posts first.")
        return

    # Chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("What do you want to know about WSB?"):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching posts..."):
                result = ask(st.session_state["rag_chain"], prompt)
            st.markdown(result["answer"])

            # Show sources
            with st.expander("📄 Sources"):
                for src in result["sources"]:
                    st.markdown(
                        f"- **{src['author']}** (score: {src['score']}, "
                        f"sentiment: {src['sentiment']})\n  _{src['content']}_"
                    )

        st.session_state["messages"].append(
            {"role": "assistant", "content": result["answer"]}
        )
```

---

## Key Considerations

| Topic | Guidance |
|-------|----------|
| **Session state** | All shared data (`df`, `topic_model`, `rag_chain`) lives in `st.session_state`. |
| **Caching** | Use `@st.cache_resource` for model loading, `@st.cache_data` for data transformations. |
| **Performance** | BERTopic + sentiment can take 1-3 min on CPU for 500 posts. Use `st.status` to show progress. |
| **Error handling** | Wrap API calls in `try/except` and show `st.error()` messages for missing credentials, network errors, etc. |
| **Layout** | Use `st.columns`, `st.expander`, `st.tabs` for a clean, non-cluttered layout. |
| **Theme** | Set a custom theme in `.streamlit/config.toml` for consistent branding. |

---

## Verification

1. `streamlit run app.py` starts without errors.
2. All four tabs render and show appropriate warnings when no data is loaded.
3. After scraping, the dashboard shows charts and the explorer shows a filterable table.
4. The RAG chat returns answers grounded in scraped post content.
5. Rerunning the app preserves session state (no re-scraping needed).
