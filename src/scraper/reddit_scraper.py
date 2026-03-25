"""
Reddit scraper using PRAW (Reddit's official API).
Fetches top N posts from r/wallstreetbets with caching to parquet.
"""

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import praw

from src.utils.config import (
    DATA_DIR,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
)


def _get_reddit_client() -> praw.Reddit:
    """Create a read-only Reddit client via official OAuth2 API."""
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        raise ValueError(
            "Reddit API credentials not set. "
            "Copy .env.example to .env and fill in REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET."
        )
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )


def scrape_wsb(n: int = 100, sort: str = "hot") -> pd.DataFrame:
    """
    Fetch the top *n* posts from r/wallstreetbets.

    Parameters
    ----------
    n : int
        Number of posts to retrieve (max ~1000 per Reddit API).
    sort : str
        One of "hot", "top", "new", "rising".

    Returns
    -------
    pd.DataFrame
        Columns: id, title, selftext, score, upvote_ratio, num_comments,
        created_utc, author, url, flair, is_self.
    """
    reddit = _get_reddit_client()
    subreddit = reddit.subreddit("wallstreetbets")

    sort_methods = {
        "hot": subreddit.hot,
        "top": subreddit.top,
        "new": subreddit.new,
        "rising": subreddit.rising,
    }
    fetcher = sort_methods.get(sort, subreddit.hot)

    posts: list[dict] = []
    for post in fetcher(limit=n):
        if post.stickied:
            continue  # Skip mod announcements
        posts.append(
            {
                "id": post.id,
                "title": post.title,
                "selftext": post.selftext or "",
                "score": post.score,
                "upvote_ratio": post.upvote_ratio,
                "num_comments": post.num_comments,
                "created_utc": datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                "author": str(post.author) if post.author else "[deleted]",
                "url": post.url,
                "flair": post.link_flair_text or "",
                "is_self": post.is_self,
            }
        )

    return pd.DataFrame(posts)


def scrape_with_backoff(
    n: int, sort: str = "hot", max_retries: int = 3
) -> pd.DataFrame:
    """Scrape with exponential backoff on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            return scrape_wsb(n, sort)
        except praw.exceptions.RedditAPIException as exc:
            wait = 2**attempt * 10
            print(f"Rate limited, retrying in {wait}s… ({exc})")
            time.sleep(wait)
    raise RuntimeError(f"Reddit API: max retries ({max_retries}) exceeded")


# ── Caching helpers ─────────────────────────────────────────────────────────

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
