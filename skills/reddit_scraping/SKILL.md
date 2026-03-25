---
name: Reddit Scraping
description: How to scrape top posts from r/wallstreetbets using PRAW with caching and rate-limit handling
---

# Reddit Scraping Skill

## Overview
Fetch the top N posts from r/wallstreetbets using **Reddit's official API** via **PRAW** (Python Reddit API Wrapper), with built-in caching and error handling.

> [!CAUTION]
> **Do NOT build a custom HTTP scraper.** Raw scraping Reddit will get you rate-limited and IP-banned. PRAW authenticates through Reddit's official OAuth2 API, automatically respects rate limits (~60 requests/minute), and handles token refresh — use it exclusively.

---

## Prerequisites

1. **Reddit API credentials** — Create an app at https://www.reddit.com/prefs/apps/:
   - Choose **"script"** type
   - Note the `client_id` (under the app name) and `client_secret`
2. Install dependencies:
   ```bash
   pip install praw pandas python-dotenv
   ```
3. Create a `.env` file:
   ```
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_client_secret
   REDDIT_USER_AGENT=wsb-sentiment-agent/1.0
   ```

---

## Implementation Steps

### 1. Initialize the Reddit Client

```python
import praw
from dotenv import load_dotenv
import os

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)
```

- Use **read-only mode** (no username/password needed).
- PRAW automatically handles OAuth2 token refresh.

### 2. Fetch Posts

```python
import pandas as pd
from datetime import datetime

def scrape_wsb(n: int = 100, sort: str = "hot") -> pd.DataFrame:
    subreddit = reddit.subreddit("wallstreetbets")

    sort_methods = {
        "hot": subreddit.hot,
        "top": subreddit.top,
        "new": subreddit.new,
        "rising": subreddit.rising,
    }
    fetcher = sort_methods.get(sort, subreddit.hot)

    posts = []
    for post in fetcher(limit=n):
        posts.append({
            "id": post.id,
            "title": post.title,
            "selftext": post.selftext,
            "score": post.score,
            "upvote_ratio": post.upvote_ratio,
            "num_comments": post.num_comments,
            "created_utc": datetime.utcfromtimestamp(post.created_utc),
            "author": str(post.author),
            "url": post.url,
            "flair": post.link_flair_text,
            "is_self": post.is_self,
        })

    return pd.DataFrame(posts)
```

### 3. Caching

```python
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def save_posts(df: pd.DataFrame) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = DATA_DIR / f"wsb_posts_{ts}.parquet"
    df.to_parquet(path, index=False)
    return path

def load_latest_posts() -> pd.DataFrame | None:
    files = sorted(DATA_DIR.glob("wsb_posts_*.parquet"))
    return pd.read_parquet(files[-1]) if files else None
```

- Use **Parquet** format — compact, typed, fast reads.
- Timestamps in filenames prevent overwrites and provide history.

### 4. Rate-Limit Handling

PRAW respects Reddit's rate limits automatically (≈60 requests/minute). If you need extra safety:

```python
import time

def scrape_with_backoff(n: int, sort: str, max_retries: int = 3) -> pd.DataFrame:
    for attempt in range(max_retries):
        try:
            return scrape_wsb(n, sort)
        except praw.exceptions.RedditAPIException as e:
            wait = 2 ** attempt * 10
            print(f"Rate limited, waiting {wait}s... ({e})")
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded")
```

---

## Key Considerations

| Topic | Guidance |
|-------|----------|
| **Post limit** | Reddit API caps at ~1000 posts per listing. For larger datasets consider Pushshift/PSAW. |
| **Selftext** | Many WSB posts are link-only (`is_self=False`). Filter or handle missing `selftext`. |
| **Comments** | Not fetched by default. Use `post.comments.replace_more(limit=0)` if needed — much slower. |
| **Flairs** | WSB uses flairs like `DD`, `YOLO`, `Meme`, `Discussion` — great for stratified analysis. |
| **Stickied posts** | Filter out stickied posts (`post.stickied`) — they're usually mod announcements. |

---

## Verification

1. Run the scraper with `n=10` and `sort="hot"` and confirm a DataFrame with 10 rows is returned.
2. Verify all expected columns are present and non-null where appropriate.
3. Confirm the parquet file is saved in `data/`.
4. Run again — confirm caching/loading from file works.
