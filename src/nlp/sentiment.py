"""
Topic modeling (BERTopic) and sentiment classification (FinBERT).
"""

import pandas as pd
import plotly.express as px
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline

from src.utils.config import EMBEDDING_MODEL, SENTIMENT_MODEL


# ── Topic modeling ──────────────────────────────────────────────────────────

def create_topic_model(
    docs: list[str],
    embedding_model_name: str = EMBEDDING_MODEL,
) -> tuple[BERTopic, list[int]]:
    """
    Fit BERTopic on a list of cleaned text documents.

    Returns
    -------
    topic_model : BERTopic
    topics : list[int]
        Per-document topic assignments (-1 = outlier).
    """
    embedding_model = SentenceTransformer(embedding_model_name)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        nr_topics="auto",
        top_n_words=10,
        verbose=True,
    )

    topics, _probs = topic_model.fit_transform(docs)
    return topic_model, topics


# ── Sentiment classification ───────────────────────────────────────────────

def create_sentiment_pipeline(model_name: str = SENTIMENT_MODEL):
    """Load a HuggingFace sentiment-analysis pipeline (default: FinBERT)."""
    return hf_pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        truncation=True,
        max_length=512,
    )


def classify_sentiment(
    texts: list[str], sentiment_pipe, batch_size: int = 32
) -> list[dict]:
    """Classify a list of texts and return [{label, score}, ...]."""
    return sentiment_pipe(texts, batch_size=batch_size)


# ── Combined pipeline ──────────────────────────────────────────────────────

def analyze(df: pd.DataFrame) -> tuple[pd.DataFrame, BERTopic]:
    """
    Run the full topic + sentiment pipeline on a preprocessed DataFrame.
    Expects a 'clean_text' column.

    Returns the enriched DataFrame and the fitted BERTopic model.
    """
    df = df.copy()
    docs = df["clean_text"].tolist()

    # ── Topics ──
    topic_model, topics = create_topic_model(docs)
    df["topic_id"] = topics

    topic_info = topic_model.get_topic_info()
    topic_map = dict(zip(topic_info["Topic"], topic_info["Name"]))
    df["topic_label"] = df["topic_id"].map(topic_map)

    # ── Sentiment ──
    sentiment_pipe = create_sentiment_pipeline()
    sentiments = classify_sentiment(docs, sentiment_pipe)
    df["sentiment_label"] = [s["label"] for s in sentiments]
    df["sentiment_score"] = [s["score"] for s in sentiments]

    return df, topic_model


# ── Visualizations ─────────────────────────────────────────────────────────

_SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral": "#6b7280",
}


def generate_visualizations(
    topic_model: BERTopic, df: pd.DataFrame
) -> dict:
    """
    Generate a dict of Plotly figures for the Streamlit dashboard.

    Keys: sentiment_dist, topic_barchart, topic_map, topic_hierarchy,
          topic_sentiment, ticker_freq.
    """
    figs: dict = {}

    # ── BERTopic built-ins ──
    try:
        figs["topic_barchart"] = topic_model.visualize_barchart(top_n_topics=10)
    except Exception:
        figs["topic_barchart"] = None

    try:
        figs["topic_map"] = topic_model.visualize_topics()
    except Exception:
        figs["topic_map"] = None

    try:
        figs["topic_hierarchy"] = topic_model.visualize_hierarchy()
    except Exception:
        figs["topic_hierarchy"] = None

    # ── Custom: sentiment distribution ──
    figs["sentiment_dist"] = px.histogram(
        df,
        x="sentiment_label",
        color="sentiment_label",
        title="Sentiment Distribution",
        color_discrete_map=_SENTIMENT_COLORS,
    )
    figs["sentiment_dist"].update_layout(
        template="plotly_dark",
        showlegend=False,
        xaxis_title="Sentiment",
        yaxis_title="Count",
    )

    # ── Custom: sentiment by topic ──
    topic_sentiment = (
        df.groupby(["topic_label", "sentiment_label"])
        .size()
        .reset_index(name="count")
    )
    figs["topic_sentiment"] = px.bar(
        topic_sentiment,
        x="topic_label",
        y="count",
        color="sentiment_label",
        title="Sentiment by Topic",
        barmode="group",
        color_discrete_map=_SENTIMENT_COLORS,
    )
    figs["topic_sentiment"].update_layout(
        template="plotly_dark",
        xaxis_title="Topic",
        yaxis_title="Count",
    )

    # ── Custom: top tickers ──
    all_tickers = df["tickers"].explode().dropna()
    if not all_tickers.empty:
        ticker_counts = all_tickers.value_counts().head(15).reset_index()
        ticker_counts.columns = ["ticker", "mentions"]
        figs["ticker_freq"] = px.bar(
            ticker_counts,
            x="ticker",
            y="mentions",
            title="Most Mentioned Tickers",
            color="mentions",
            color_continuous_scale="Oranges",
        )
        figs["ticker_freq"].update_layout(template="plotly_dark")
    else:
        figs["ticker_freq"] = None

    return figs
