"""
Run this to diagnose what's actually failing.
    python test_connection.py
"""
import sys

print("=== Network & yfinance diagnostics ===\n")

# Test 1: Basic internet
print("[1] Testing basic internet connectivity...")
try:
    import urllib.request
    urllib.request.urlopen("https://www.google.com", timeout=5)
    print("    Google: OK")
except Exception as e:
    print(f"    Google: FAILED — {e}")

# Test 2: Yahoo Finance directly
print("\n[2] Testing Yahoo Finance API directly...")
try:
    import urllib.request
    req = urllib.request.Request(
        "https://query1.finance.yahoo.com/v8/finance/chart/RELIANCE.NS?interval=1d&range=5d",
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        body = r.read(200)
        print(f"    Status: {r.status}")
        print(f"    Body preview: {body[:100]}")
except Exception as e:
    print(f"    Yahoo Finance: FAILED — {e}")

# Test 3: yfinance with single ticker, 5 days
print("\n[3] Testing yfinance — 5 days, single ticker...")
try:
    import yfinance as yf
    df = yf.download("RELIANCE.NS", period="5d", interval="1d",
                     auto_adjust=True, progress=False)
    if df.empty:
        print("    Result: EMPTY dataframe")
    else:
        print(f"    Result: OK — {len(df)} rows")
        print(f"    Columns: {list(df.columns)}")
        print(f"    Last row:\n{df.tail(1)}")
except Exception as e:
    print(f"    yfinance: FAILED — {e}")

# Test 4: yfinance with period= instead of start/end
print("\n[4] Testing yfinance — 1y period (no start/end dates)...")
try:
    import yfinance as yf
    df = yf.download("TCS.NS", period="1y", interval="1d",
                     auto_adjust=True, progress=False)
    if df.empty:
        print("    Result: EMPTY dataframe")
    else:
        print(f"    Result: OK — {len(df)} rows")
except Exception as e:
    print(f"    yfinance 1y: FAILED — {e}")

print("\n=== Done ===")