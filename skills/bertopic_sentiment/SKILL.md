---
name: BERTopic Sentiment Analysis
description: How to perform topic modeling with BERTopic and sentiment classification on r/wallstreetbets posts
---

# BERTopic Sentiment Analysis Skill

## Overview
Use **BERTopic** to discover latent discussion topics in WSB posts, then layer a **transformer-based sentiment classifier** on top to produce per-post sentiment labels and per-topic sentiment distributions.

---

## Prerequisites

```bash
pip install bertopic sentence-transformers transformers torch plotly
```

> **GPU note:** BERTopic and transformers run significantly faster on a CUDA GPU. On CPU, limit dataset size ≤ 500 posts for reasonable runtimes.

---

## Implementation Steps

### 1. Topic Modeling with BERTopic

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

def create_topic_model(docs: list[str]) -> tuple[BERTopic, list[int]]:
    """
    Fit BERTopic on a list of cleaned text documents.
    Returns the model and per-doc topic assignments.
    """
    # Use a lightweight embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        nr_topics="auto",          # auto-reduce similar topics
        top_n_words=10,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics
```

#### Customizing BERTopic Components (Optional)

```python
from bertopic.representation import KeyBERTInspired
from umap import UMAP
from hdbscan import HDBSCAN

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine"),
    hdbscan_model=HDBSCAN(min_cluster_size=15, metric="euclidean", prediction_data=True),
    representation_model=KeyBERTInspired(),
    verbose=True,
)
```

### 2. Sentiment Classification

```python
from transformers import pipeline

def create_sentiment_pipeline():
    """
    Load a pre-trained sentiment classifier.
    Options:
      - 'finiteautomata/bertweet-base-sentiment-analysis' (general)
      - 'ProsusAI/finbert' (financial-domain)
    """
    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        max_length=512,
        truncation=True,
    )

def classify_sentiment(texts: list[str], sentiment_pipe) -> list[dict]:
    """Return list of {'label': ..., 'score': ...} dicts."""
    results = sentiment_pipe(texts, batch_size=32)
    return results
```

> **Model choice:** Use `ProsusAI/finbert` for financial text — it classifies as `positive`, `negative`, `neutral` with finance-aware understanding.

### 3. Combine Topics + Sentiment

```python
import pandas as pd

def analyze(df: pd.DataFrame) -> pd.DataFrame:
    """Run full topic + sentiment pipeline on preprocessed DataFrame."""
    docs = df["clean_text"].tolist()

    # Topic modeling
    topic_model, topics = create_topic_model(docs)
    df["topic_id"] = topics

    # Map topic IDs to labels
    topic_info = topic_model.get_topic_info()
    topic_map = dict(zip(topic_info["Topic"], topic_info["Name"]))
    df["topic_label"] = df["topic_id"].map(topic_map)

    # Sentiment classification
    sentiment_pipe = create_sentiment_pipeline()
    sentiments = classify_sentiment(docs, sentiment_pipe)
    df["sentiment_label"] = [s["label"] for s in sentiments]
    df["sentiment_score"] = [s["score"] for s in sentiments]

    return df, topic_model
```

### 4. Visualizations

```python
def generate_visualizations(topic_model: BERTopic, df: pd.DataFrame):
    """Generate Plotly figures for use in Streamlit."""
    figs = {}

    # BERTopic built-in visualizations
    figs["topic_map"] = topic_model.visualize_topics()
    figs["topic_barchart"] = topic_model.visualize_barchart(top_n_topics=10)
    figs["topic_hierarchy"] = topic_model.visualize_hierarchy()

    # Custom: sentiment distribution
    import plotly.express as px

    figs["sentiment_dist"] = px.histogram(
        df, x="sentiment_label", color="sentiment_label",
        title="Sentiment Distribution",
        color_discrete_map={"positive": "#22c55e", "negative": "#ef4444", "neutral": "#6b7280"},
    )

    # Custom: sentiment by topic
    topic_sentiment = (
        df.groupby(["topic_label", "sentiment_label"]).size()
        .reset_index(name="count")
    )
    figs["topic_sentiment"] = px.bar(
        topic_sentiment, x="topic_label", y="count", color="sentiment_label",
        title="Sentiment by Topic", barmode="group",
    )

    return figs
```

---

## Key Considerations

| Topic | Guidance |
|-------|----------|
| **Input text** | Pass `clean_text` (NOT lemmatized) — BERTopic and FinBERT use their own tokenizers. |
| **Topic -1** | BERTopic assigns `-1` to outlier documents that don't fit any cluster. Expect 10-30% outliers. |
| **Batch size** | Transformers sentiment pipeline supports `batch_size` — use 16-32 for efficiency. |
| **Token limit** | FinBERT has a 512-token limit. Long posts are auto-truncated. For better accuracy, pass just the first paragraph or title. |
| **Caching models** | Both `SentenceTransformer` and `transformers` cache models in `~/.cache/`. First run downloads ~400MB. |
| **Streamlit caching** | Wrap model loading in `@st.cache_resource` to avoid reloading on every rerun. |

---

## Verification

1. Run on a dataset of ≥50 posts and confirm topics are generated (non-trivial clusters beyond just `-1`).
2. Confirm sentiment labels are one of `positive`, `negative`, `neutral`.
3. Verify all visualizations render without error.
4. Check that the combined DataFrame has all expected columns: `topic_id`, `topic_label`, `sentiment_label`, `sentiment_score`.
