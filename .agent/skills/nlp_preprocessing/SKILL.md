---
name: NLP Preprocessing
description: How to clean and preprocess r/wallstreetbets text data for downstream NLP models
---

# NLP Preprocessing Skill

## Overview
Clean raw Reddit post text into a normalized format suitable for BERTopic and sentiment analysis. WSB text is notoriously noisy — heavy use of emojis, slang, ticker symbols, and meme language.

---

## Prerequisites

```bash
pip install spacy emoji regex pandas
python -m spacy download en_core_web_sm
```

---

## Implementation Steps

### 1. Text Cleaning Pipeline

```python
import re
import emoji
import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_text(text: str) -> str:
    if not text or text == "[removed]" or text == "[deleted]":
        return ""

    # Decode emojis to text descriptions
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
```

### 2. Ticker Extraction

```python
TICKER_PATTERN = re.compile(r"\$([A-Z]{1,5})\b")
COMMON_WORDS = {"I", "A", "CEO", "IPO", "DD", "YOLO", "FD", "OTM", "ITM", "ATM", "EPS", "PE", "GDP", "SEC", "ETF", "FBI", "USA", "WSB"}

def extract_tickers(text: str) -> list[str]:
    """Extract stock ticker symbols prefixed with $."""
    matches = TICKER_PATTERN.findall(text)
    return [t for t in matches if t not in COMMON_WORDS]
```

> **Tip:** Also look for non-prefixed tickers (e.g., `GME`, `AMC`) by cross-referencing a known ticker list, but be careful of false positives.

### 3. Lemmatization & Stop-Word Removal

```python
def lemmatize(text: str) -> str:
    doc = nlp(text)
    return " ".join(
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and len(token.text) > 1
    )
```

### 4. Full Preprocessing Function

```python
import pandas as pd

def preprocess_posts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Combine title + selftext
    df["combined_text"] = (
        df["title"].fillna("") + " " + df["selftext"].fillna("")
    ).str.strip()

    # Clean
    df["clean_text"] = df["combined_text"].apply(clean_text)

    # Extract tickers before lemmatization
    df["tickers"] = df["combined_text"].apply(extract_tickers)

    # Lemmatize (for topic modeling — NOT for transformer-based sentiment)
    df["lemmatized_text"] = df["clean_text"].apply(lemmatize)

    # Drop empty rows
    df = df[df["clean_text"].str.len() > 10].reset_index(drop=True)

    return df
```

---

## Key Considerations

| Topic | Guidance |
|-------|----------|
| **Two text columns** | Use `clean_text` for transformer-based models (BERT, BERTopic) and `lemmatized_text` for traditional ML / topic keywords. Transformers have their own tokenizers — don't lemmatize their input. |
| **WSB slang** | Terms like `diamond hands 💎🙌`, `tendies`, `ape`, `moon` carry sentiment. Preserve them in `clean_text`. |
| **Short posts** | Title-only posts are common. The `combined_text` field handles this. |
| **Deleted/removed** | Reddit replaces deleted content with `[removed]` or `[deleted]`. Filter these early. |
| **Batch processing** | For large datasets, use `nlp.pipe()` instead of per-row `.apply()` for spaCy. |

---

## Verification

1. Process a sample DataFrame with 5 posts (including one `[removed]` post).
2. Confirm `clean_text` has no URLs, markdown, or HTML entities.
3. Confirm `tickers` correctly extracts `$GME`, `$AMC` etc.
4. Confirm `lemmatized_text` has no stop words or punctuation.
5. Confirm the `[removed]` post is filtered out.
