"""
Data loader — CSV file upload + parquet caching.
Replaces the Reddit API scraper (no API credentials needed).
"""

from datetime import datetime
from pathlib import Path
from io import BytesIO

import pandas as pd

from src.utils.config import DATA_DIR


# ── Required columns (minimum for the pipeline to work) ────────────────────
REQUIRED_COLS = {"title"}
OPTIONAL_COLS = {
    "selftext", "score", "num_comments", "created_utc",
    "author", "url", "flair", "id", "upvote_ratio", "is_self",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has all expected columns, filling missing ones."""
    df = df.copy()

    # Must have at least 'title'
    if "title" not in df.columns:
        # Try common alternative names
        for alt in ["Title", "post_title", "headline"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "title"})
                break
        else:
            raise ValueError(
                "Dataset must contain a 'title' column. "
                f"Found columns: {list(df.columns)}"
            )

    # Fill optional columns with defaults
    defaults = {
        "selftext": "",
        "score": 0,
        "num_comments": 0,
        "created_utc": "",
        "author": "unknown",
        "url": "",
        "flair": "",
        "id": "",
        "upvote_ratio": 0.0,
        "is_self": True,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            # Try common alternatives
            alt_map = {
                "selftext": ["body", "self_text", "text", "content", "post_text"],
                "score": ["upvotes", "ups"],
                "num_comments": ["comments", "comment_count"],
                "author": ["user", "username", "poster"],
                "flair": ["link_flair_text", "post_flair"],
            }
            found = False
            for alt in alt_map.get(col, []):
                if alt in df.columns:
                    df = df.rename(columns={alt: col})
                    found = True
                    break
            if not found:
                df[col] = default

    # Generate IDs if missing
    if df["id"].eq("").all() or df["id"].isna().all():
        df["id"] = [f"post_{i}" for i in range(len(df))]

    # Ensure string types
    for col in ["title", "selftext", "author", "url", "flair", "id"]:
        df[col] = df[col].astype(str).fillna("")

    return df


# ── CSV / file upload ──────────────────────────────────────────────────────

def load_from_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Load data from a Streamlit UploadedFile object (CSV or JSON).

    Parameters
    ----------
    uploaded_file : streamlit.runtime.uploaded_file_manager.UploadedFile

    Returns
    -------
    pd.DataFrame
    """
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".json") or name.endswith(".jsonl"):
        df = pd.read_json(uploaded_file, lines=name.endswith(".jsonl"))
    elif name.endswith(".parquet"):
        df = pd.read_parquet(BytesIO(uploaded_file.read()))
    else:
        raise ValueError(f"Unsupported file format: {name}. Use CSV, JSON, JSONL, or Parquet.")

    return _normalize_columns(df)


def load_from_csv_path(path: str | Path) -> pd.DataFrame:
    """Load from a local CSV/JSON/Parquet file path."""
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in (".json", ".jsonl"):
        df = pd.read_json(path, lines=path.suffix == ".jsonl")
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    return _normalize_columns(df)


# Caching helpers

def save_posts(df: pd.DataFrame) -> Path:
    """Save DataFrame to a timestamped parquet file in data/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = DATA_DIR / f"wsb_posts_{ts}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_latest_posts() -> pd.DataFrame | None:
    """Load the most recent parquet file, or return None if none exist."""
    files = sorted(DATA_DIR.glob("wsb_posts_*.parquet"))
    if not files:
        return None
    return pd.read_parquet(files[-1])
