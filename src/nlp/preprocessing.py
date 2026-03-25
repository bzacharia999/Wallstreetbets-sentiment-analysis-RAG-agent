"""
NLP preprocessing for r/wallstreetbets posts.
Handles text cleaning, ticker extraction, and lemmatization.
"""

import re

import pandas as pd
import spacy
import emoji

from src.utils.config import SPACY_MODEL

# Load spaCy model once at module level (disable unused components for speed)
try:
    _nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
except OSError:
    raise OSError(
        f"spaCy model '{SPACY_MODEL}' not found. "
        f"Run: python -m spacy download {SPACY_MODEL}"
    )

# ── Ticker extraction ──────────────────────────────────────────────────────

_TICKER_PATTERN = re.compile(r"\$([A-Z]{1,5})\b")
_COMMON_WORDS = frozenset({
    "I", "A", "CEO", "IPO", "DD", "YOLO", "FD", "OTM", "ITM", "ATM",
    "EPS", "PE", "GDP", "SEC", "ETF", "FBI", "USA", "WSB", "IMO", "TL",
    "DR", "TLDR", "OP", "RIP", "LOL", "LMAO", "NFT", "API",
})


def extract_tickers(text: str) -> list[str]:
    """Extract stock ticker symbols prefixed with $ (e.g. $GME, $AMC)."""
    if not text:
        return []
    matches = _TICKER_PATTERN.findall(text)
    return list(dict.fromkeys(t for t in matches if t not in _COMMON_WORDS))


# ── Text cleaning ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean raw Reddit post text:
    - Handle [removed] / [deleted]
    - Decode emojis to text
    - Strip URLs, markdown, HTML entities
    - Normalize whitespace
    """
    if not text or text.strip() in ("[removed]", "[deleted]", ""):
        return ""

    # Decode emojis to descriptive text
    text = emoji.demojize(text, delimiters=(" ", " "))

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove Reddit markdown (bold, italic, headers, links)
    text = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", text)
    text = re.sub(r"#{1,6}\s?", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # Remove HTML entities
    text = re.sub(r"&[a-z]+;", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ── Lemmatization ──────────────────────────────────────────────────────────

def lemmatize(text: str) -> str:
    """Lemmatize text with stop-word and punctuation removal (spaCy)."""
    if not text:
        return ""
    doc = _nlp(text)
    return " ".join(
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and len(token.text) > 1
    )


# ── Full pipeline ──────────────────────────────────────────────────────────

def preprocess_posts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline on scraped posts.

    Adds columns: combined_text, clean_text, tickers, lemmatized_text.
    Drops rows where clean_text has fewer than 10 characters.
    """
    df = df.copy()

    # Merge title + selftext
    df["combined_text"] = (
        df["title"].fillna("") + " " + df["selftext"].fillna("")
    ).str.strip()

    # Clean text (for transformer-based models — keep raw tokens)
    df["clean_text"] = df["combined_text"].apply(clean_text)

    # Extract tickers before any text normalization
    df["tickers"] = df["combined_text"].apply(extract_tickers)

    # Lemmatize (for BERTopic keyword display, NOT used as model input)
    df["lemmatized_text"] = df["clean_text"].apply(lemmatize)

    # Filter out very short / empty posts
    df = df[df["clean_text"].str.len() > 10].reset_index(drop=True)

    return df
