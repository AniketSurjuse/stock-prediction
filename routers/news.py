from fastapi import APIRouter
from db import get_conn
from dummy_data import get_news

router = APIRouter()


def _real_news_count() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) as n FROM news_articles").fetchone()["n"]


@router.get("/")
def news_feed(limit: int = 30, symbol: str = None, sentiment: str = None):
    if _real_news_count() == 0:
        items = get_news(limit=50)
        if symbol:
            items = [n for n in items if n["symbol"] == symbol]
        if sentiment:
            items = [n for n in items if n["sentiment"] == sentiment]
        return items[:limit]

    query  = "SELECT * FROM news_articles WHERE 1=1"
    params = []
    if symbol:
        query  += " AND symbol=?"
        params.append(symbol)
    if sentiment:
        query  += " AND raw_sentiment=?"
        params.append(sentiment)
    query += " ORDER BY published_at DESC LIMIT ?"
    params.append(limit)

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()

    results = []
    for r in rows:
        results.append({
            "id":           r["id"],
            "symbol":       r["symbol"] or "MARKET",
            "company":      r["symbol"] or "General",
            "headline":     r["headline"],
            "sentiment":    r["raw_sentiment"] or "neutral",
            "source":       r["source"] or "Unknown",
            "url":          r["url"] or None,
            "published_at": r["published_at"] or "",
            "minutes_ago":  0,
        })
    return results