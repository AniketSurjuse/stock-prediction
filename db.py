"""
db.py — SQLite connection + schema setup.
Call init_db() once on startup; use get_conn() everywhere else.
"""
import sqlite3
import os

DB_PATH = os.environ.get("DB_PATH", "trading.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row          # rows behave like dicts
    conn.execute("PRAGMA journal_mode=WAL") # safe for concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables if they don't exist yet."""
    with get_conn() as conn:
        conn.executescript("""
        -- Master list of tracked stocks
        CREATE TABLE IF NOT EXISTS stocks (
            symbol      TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            base_price  REAL NOT NULL
        );

        -- Daily / intraday OHLCV candles
        CREATE TABLE IF NOT EXISTS price_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT NOT NULL REFERENCES stocks(symbol),
            date        TEXT NOT NULL,          -- YYYY-MM-DD
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL NOT NULL,
            volume      INTEGER,
            UNIQUE(symbol, date)               -- dedupe on upsert
        );

        -- Scraped news articles
        CREATE TABLE IF NOT EXISTS news_articles (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT,              -- NULL = market-wide news
            headline        TEXT NOT NULL,
            source          TEXT,
            url             TEXT UNIQUE,       -- dedupe by URL
            published_at    TEXT,              -- ISO-8601
            raw_sentiment   TEXT               -- bullish / bearish / neutral (filled in P3)
        );

        -- Track every ingestion run
        CREATE TABLE IF NOT EXISTS ingestion_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at      TEXT NOT NULL,         -- ISO-8601
            job         TEXT NOT NULL,         -- 'prices' | 'news'
            status      TEXT NOT NULL,         -- 'ok' | 'error'
            detail      TEXT
        );
        """)

        # Seed the 10 stocks if table is empty
        conn.executemany(
            "INSERT OR IGNORE INTO stocks (symbol, name, base_price) VALUES (?,?,?)",
            [
                ("RELIANCE",    "Reliance Industries",          2850),
                ("TCS",         "Tata Consultancy Services",    3920),
                ("HDFCBANK",    "HDFC Bank",                    1680),
                ("INFY",        "Infosys",                      1545),
                ("ICICIBANK",   "ICICI Bank",                   1120),
                ("HINDUNILVR",  "Hindustan Unilever",           2380),
                ("ITC",         "ITC Ltd",                       458),
                ("LT",          "Larsen & Toubro",              3560),
                ("SBIN",        "State Bank of India",           795),
                ("BAJFINANCE",  "Bajaj Finance",                7240),
            ]
        )
    print(f"[db] Initialised — {DB_PATH}")