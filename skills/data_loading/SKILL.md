---
name: Data Loading
description: How to load r/wallstreetbets data from CSV upload, Kaggle datasets, or local files (no Reddit API needed)
---

# Data Loading Skill

## Overview
Load WSB post data without needing Reddit API access. Three supported sources:
1. **CSV/JSON/Parquet file upload** — drag and drop into Streamlit
2. **Kaggle datasets** — download popular WSB datasets via `kagglehub`
3. **Previously cached data** — reload from saved parquet files

> [!TIP]
> The CSV upload path requires **zero credentials** — just download a dataset from Kaggle manually, then upload it into the app.

---

## Recommended Kaggle Datasets

| Dataset | Slug | Description |
|---------|------|-------------|
| [WSB Posts](https://www.kaggle.com/datasets/gpreda/reddit-wallstreetbets-posts) | `gpreda/reddit-wallstreetbets-posts` | 100k+ posts, regularly updated |
| [WSB Posts & Comments](https://www.kaggle.com/datasets/unanimad/reddit-rwallstreetbets) | `unanimad/reddit-rwallstreetbets` | Posts + comments dataset |
| [WSB 2021 (GME era)](https://www.kaggle.com/datasets/mattop/wallstreetbets-reddit-posts-2021) | `mattop/wallstreetbets-reddit-posts-2021` | Historical 2021 posts |

---

## Prerequisites

For **file upload**: no additional setup needed.

For **Kaggle download** (programmatic):
```bash
pip install kagglehub
```

Then either:
- Set env vars `KAGGLE_USERNAME` and `KAGGLE_KEY`, **or**
- Place `kaggle.json` in `~/.kaggle/` (download from [kaggle.com/settings](https://www.kaggle.com/settings) → "Create New Token")

---

## Implementation Steps

### 1. Column Normalization

WSB datasets have inconsistent column names. The data loader auto-maps common variants:

| Expected Column | Also Accepts |
|-----------------|-------------|
| `title` | `Title`, `post_title`, `headline` |
| `selftext` | `body`, `self_text`, `text`, `content`, `post_text` |
| `score` | `upvotes`, `ups` |
| `num_comments` | `comments`, `comment_count` |
| `author` | `user`, `username`, `poster` |
| `flair` | `link_flair_text`, `post_flair` |

Only `title` is required. All other columns get sensible defaults if missing.

### 2. File Upload (Streamlit)

```python
uploaded = st.file_uploader("Upload WSB data", type=["csv", "json", "jsonl", "parquet"])
if uploaded:
    df = load_from_uploaded_file(uploaded)
```

### 3. Kaggle Download

```python
df = load_from_kaggle("gpreda/reddit-wallstreetbets-posts", max_rows=500)
```

### 4. Load Cached Data

```python
df = load_latest_posts()  # Loads most recent parquet from data/
```

---

## Key Considerations

| Topic | Guidance |
|-------|----------|
| **Row limits** | BERTopic + FinBERT on CPU works best with 50-500 posts. Use `max_rows` to cap. |
| **File size** | Streamlit's default upload limit is 200MB. Override in `.streamlit/config.toml` if needed. |
| **Column flexibility** | The normalizer handles most WSB dataset formats. If a dataset has unusual column names, rename `title` manually before uploading. |
| **Kaggle auth** | `kagglehub` reads from `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars or `~/.kaggle/kaggle.json`. |

---

## Verification

1. Upload a CSV with `title` and `body` columns → confirm `body` is mapped to `selftext`.
2. Upload a file with only a `title` column → confirm defaults are filled for missing columns.
3. Download from Kaggle with `max_rows=50` → confirm only 50 rows are returned.
4. Run pipeline → save → reload → confirm `load_latest_posts()` returns the same data.
