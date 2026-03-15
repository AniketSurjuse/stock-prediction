"""
ingest_news.py — Scrape RSS feeds from Indian financial news sources.

Usage:
    python ingest_news.py
"""
import re
import traceback
from datetime import datetime
from email.utils import parsedate_to_datetime

import feedparser

from db import get_conn, init_db

# ---------------------------------------------------------------------------
# RSS feed registry
# ---------------------------------------------------------------------------
FEEDS = [
    # Moneycontrol
    {"source": "Moneycontrol", "url": "https://www.moneycontrol.com/rss/marketreports.xml"},
    {"source": "Moneycontrol", "url": "https://www.moneycontrol.com/rss/business.xml"},
    # Economic Times
    {"source": "Economic Times", "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"},
    {"source": "Economic Times", "url": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"},
    # Livemint
    {"source": "Livemint", "url": "https://www.livemint.com/rss/markets"},
    # Business Standard
    {"source": "Business Standard", "url": "https://www.business-standard.com/rss/markets-106.rss"},
]

# Map each tracked symbol to keywords we'll scan headlines for
SYMBOL_KEYWORDS = {
    "RELIANCE":     ["reliance", "ril", "mukesh ambani", "jio"],
    "TCS":          ["tcs", "tata consultancy"],
    "HDFCBANK":     ["hdfc bank", "hdfcbank"],
    "INFY":         ["infosys", "infy"],
    "ICICIBANK":    ["icici bank", "icicibank"],
    "HINDUNILVR":   ["hindustan unilever", "hul", "hindunilvr"],
    "ITC":          ["itc ltd", "itc limited", " itc "],
    "LT":           ["larsen", "l&t", " l&t"],
    "SBIN":         ["sbi", "state bank"],
    "BAJFINANCE":   ["bajaj finance", "bajfinance"],
}

# Rudimentary sentiment keywords (Phase 3 will replace with FinBERT)
_BULL = {"surge", "gain", "rally", "beat", "upgrade", "buy", "positive",
         "growth", "profit", "record", "strong", "wins", "rise", "rises"}
_BEAR = {"fall", "drop", "loss", "miss", "downgrade", "sell", "negative",
         "decline", "weak", "cut", "crash", "slump", "concern", "worry"}


def _keyword_sentiment(headline: str) -> str:
    words = set(re.findall(r'\w+', headline.lower()))
    bull  = len(words & _BULL)
    bear  = len(words & _BEAR)
    if bull > bear:
        return "bullish"
    if bear > bull:
        return "bearish"
    return "neutral"


def _match_symbol(headline: str) -> str | None:
    h = headline.lower()
    for sym, keywords in SYMBOL_KEYWORDS.items():
        if any(kw in h for kw in keywords):
            return sym
    return None


def _parse_date(entry) -> str:
    """Return ISO-8601 string from feed entry, fallback to now."""
    for attr in ("published", "updated"):
        val = getattr(entry, attr, None)
        if val:
            try:
                return parsedate_to_datetime(val).isoformat()
            except Exception:
                pass
    return datetime.now().isoformat()


def scrape_feed(feed_cfg: dict) -> int:
    source = feed_cfg["source"]
    url    = feed_cfg["url"]
    parsed = feedparser.parse(url)

    if parsed.bozo and not parsed.entries:
        print(f"  [{source}] failed to parse feed: {url}")
        return 0

    rows = []
    for entry in parsed.entries:
        headline = entry.get("title", "").strip()
        if not headline:
            continue

        link      = entry.get("link", "")
        symbol    = _match_symbol(headline)
        sentiment = _keyword_sentiment(headline)
        pub_at    = _parse_date(entry)

        rows.append((symbol, headline, source, link, pub_at, sentiment))

    if not rows:
        return 0

    with get_conn() as conn:
        conn.executemany("""
            INSERT OR IGNORE INTO news_articles
                (symbol, headline, source, url, published_at, raw_sentiment)
            VALUES (?,?,?,?,?,?)
        """, rows)

    inserted = len(rows)
    print(f"  [{source}] stored up to {inserted} articles")
    return inserted


def run_ingestion():
    """Scrape all feeds. Called by scheduler or CLI."""
    init_db()
    total   = 0
    errors  = []
    started = datetime.now().isoformat()

    print(f"[news] Starting ingestion — {len(FEEDS)} feeds")
    for feed in FEEDS:
        try:
            total += scrape_feed(feed)
        except Exception as e:
            errors.append(feed["source"])
            print(f"  [{feed['source']}] ERROR: {e}")
            traceback.print_exc()

    status = "ok" if not errors else f"error:{','.join(errors)}"
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO ingestion_log (run_at, job, status, detail) VALUES (?,?,?,?)",
            (started, "news", status, f"articles={total}")
        )

    print(f"[news] Done — {total} articles, errors={errors or 'none'}")
    return total


if __name__ == "__main__":
    run_ingestion()