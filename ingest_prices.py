"""
ingest_prices.py — Fetch OHLCV via Yahoo Finance REST API directly.

Bypasses yfinance entirely to avoid the crumb/cookie fetch that fails
on corporate proxies. Uses Yahoo's v8/finance/chart endpoint directly.

Usage:
    python ingest_prices.py            # last 1 day
    python ingest_prices.py --days 1825 # 5-year backfill
"""
import argparse
import time
import traceback
from datetime import datetime, timedelta

import requests
import pandas as pd

from db import get_conn, init_db

# Yahoo Finance chart endpoint — same one that worked in the diagnostic
YF_URL    = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
YF_URL2   = "https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
CHUNK_DAYS = 365   # max days per request to avoid timezone errors

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finance.yahoo.com",
}


def _ns(symbol: str) -> str:
    return f"{symbol}.NS"


def fetch_symbol_chunk(symbol: str, start_ts: int, end_ts: int,
                       session: requests.Session) -> pd.DataFrame:
    """
    Fetch OHLCV for one symbol and one date chunk via Yahoo REST API.
    Returns a DataFrame with DatetimeIndex and OHLCV columns.
    """
    ticker = _ns(symbol)
    params = {
        "interval":    "1d",
        "period1":     start_ts,
        "period2":     end_ts,
        "events":      "div,splits",
        "includeAdjustedClose": "true",
    }

    for base_url in [YF_URL, YF_URL2]:
        try:
            url = base_url.format(ticker=ticker)
            r   = session.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()

            result = data.get("chart", {}).get("result")
            if not result:
                continue

            result   = result[0]
            timestamps = result.get("timestamp", [])
            if not timestamps:
                return pd.DataFrame()

            indicators = result["indicators"]
            quote      = indicators["quote"][0]
            adjclose   = indicators.get("adjclose", [{}])[0].get("adjclose", quote["close"])

            df = pd.DataFrame({
                "open":   quote["open"],
                "high":   quote["high"],
                "low":    quote["low"],
                "close":  adjclose,
                "volume": quote["volume"],
            }, index=pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Asia/Kolkata"))

            df.index = df.index.normalize().tz_localize(None)  # strip tz, keep date
            df = df.dropna(subset=["close"])
            return df

        except Exception as e:
            print(f"  [{symbol}] {base_url.split('/')[2]} error: {e}")
            continue

    return pd.DataFrame()


def fetch_and_store(symbol: str, days: int, session: requests.Session) -> int:
    """
    Fetch `days` of history for `symbol` in CHUNK_DAYS slices and store.
    Returns number of rows upserted.
    """
    end_dt   = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days)

    all_rows = []
    cursor   = start_dt

    while cursor < end_dt:
        chunk_end  = min(cursor + timedelta(days=CHUNK_DAYS), end_dt)
        start_ts   = int(cursor.timestamp())
        end_ts     = int(chunk_end.timestamp())

        df = fetch_symbol_chunk(symbol, start_ts, end_ts, session)
        if not df.empty:
            all_rows.append(df)

        cursor = chunk_end
        time.sleep(0.5)

    if not all_rows:
        print(f"  [{symbol}] no data returned")
        return 0

    combined = pd.concat(all_rows)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    rows = []
    for ts, row in combined.iterrows():
        rows.append((
            symbol,
            ts.strftime("%Y-%m-%d"),
            round(float(row["open"]),  2) if pd.notna(row["open"])   else None,
            round(float(row["high"]),  2) if pd.notna(row["high"])   else None,
            round(float(row["low"]),   2) if pd.notna(row["low"])    else None,
            round(float(row["close"]), 2),
            int(row["volume"])             if pd.notna(row["volume"]) else 0,
        ))

    with get_conn() as conn:
        conn.executemany("""
            INSERT INTO price_history (symbol, date, open, high, low, close, volume)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(symbol, date) DO UPDATE SET
                open=excluded.open, high=excluded.high,
                low=excluded.low,   close=excluded.close,
                volume=excluded.volume
        """, rows)

    print(f"  [{symbol}] upserted {len(rows)} rows")
    return len(rows)


def run_ingestion(days: int = 1):
    init_db()

    with get_conn() as conn:
        symbols = [r["symbol"] for r in conn.execute("SELECT symbol FROM stocks").fetchall()]

    started = datetime.now().isoformat()
    total   = 0
    errors  = []

    session = requests.Session()
    session.headers.update(HEADERS)

    print(f"[prices] Ingesting {len(symbols)} stocks, {days}d window (direct Yahoo API)")
    for sym in symbols:
        try:
            total += fetch_and_store(sym, days, session)
        except Exception as e:
            errors.append(sym)
            print(f"  [{sym}] ERROR: {e}")
            traceback.print_exc()
        time.sleep(1)

    status = "ok" if not errors else f"error:{','.join(errors)}"
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO ingestion_log (run_at, job, status, detail) VALUES (?,?,?,?)",
            (started, "prices", status, f"rows={total}")
        )

    print(f"[prices] Done — {total} rows, errors={errors or 'none'}")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=1)
    args = parser.parse_args()
    run_ingestion(days=args.days)