import random
import math
from datetime import datetime, timedelta

STOCKS = {
    "RELIANCE": {"name": "Reliance Industries", "base": 2850},
    "TCS":      {"name": "Tata Consultancy Services", "base": 3920},
    "HDFCBANK": {"name": "HDFC Bank", "base": 1680},
    "INFY":     {"name": "Infosys", "base": 1545},
    "ICICIBANK":{"name": "ICICI Bank", "base": 1120},
    "HINDUNILVR":{"name":"Hindustan Unilever", "base": 2380},
    "ITC":      {"name": "ITC Ltd", "base": 458},
    "LT":       {"name": "Larsen & Toubro", "base": 3560},
    "SBIN":     {"name": "State Bank of India", "base": 795},
    "BAJFINANCE":{"name":"Bajaj Finance", "base": 7240},
}

SENTIMENTS = ["bullish", "bearish", "neutral"]
SIGNALS    = ["BUY", "HOLD", "SELL"]

def _seeded_walk(symbol: str, days: int, base: float):
    """Reproducible random walk seeded by symbol so values are consistent."""
    rng = random.Random(hash(symbol) % (2**32))
    prices = []
    price = base
    for i in range(days):
        drift = rng.gauss(0.0002, 0.012)
        price = max(price * (1 + drift), base * 0.6)
        prices.append(round(price, 2))
    return prices


def get_price_series(symbol: str, days: int = 90):
    base  = STOCKS[symbol]["base"]
    actual = _seeded_walk(symbol, days, base)

    # Predicted series: actual + small normally-distributed error
    rng = random.Random(hash(symbol + "pred") % (2**32))
    predicted = [round(p * (1 + rng.gauss(0, 0.008)), 2) for p in actual]

    dates = [
        (datetime.today() - timedelta(days=days - i)).strftime("%Y-%m-%d")
        for i in range(days)
    ]
    return {"dates": dates, "actual": actual, "predicted": predicted}


def get_prediction(symbol: str):
    rng    = random.Random(hash(symbol + "sig") % (2**32))
    base   = STOCKS[symbol]["base"]
    series = get_price_series(symbol, 90)
    last   = series["actual"][-1]
    pred   = series["predicted"][-1]
    pct    = round((pred - last) / last * 100, 2)

    confidence = round(rng.uniform(0.62, 0.94), 2)
    sentiment_score = round(rng.uniform(-1, 1), 3)
    if sentiment_score > 0.2:
        sentiment = "bullish"
    elif sentiment_score < -0.2:
        sentiment = "bearish"
    else:
        sentiment = "neutral"

    if pct > 0.5 and sentiment != "bearish":
        signal = "BUY"
    elif pct < -0.5 or sentiment == "bearish":
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "symbol": symbol,
        "name": STOCKS[symbol]["name"],
        "current_price": last,
        "predicted_price": pred,
        "predicted_change_pct": pct,
        "signal": signal,
        "confidence": confidence,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "as_of": datetime.today().strftime("%Y-%m-%d"),
    }


NEWS_TEMPLATES = [
    ("{name} Q4 results beat street estimates by 8%, PAT up 22% YoY", "bullish"),
    ("{name} wins ₹4,200 cr government contract for infrastructure", "bullish"),
    ("Analysts upgrade {name} to 'Buy'; raise target price to ₹{tp}", "bullish"),
    ("{name} announces ₹3,000 cr share buyback programme", "bullish"),
    ("{name} management guides for 15–18% revenue growth in FY26", "bullish"),
    ("{name} Q3 profit misses estimates; management cautious on margins", "bearish"),
    ("FII selling pressure weighs on {name}; stock down 3.4% this week", "bearish"),
    ("{name} faces regulatory scrutiny over related-party transactions", "bearish"),
    ("Rising input costs likely to compress {name} margins in H2", "bearish"),
    ("{name} subsidiary reports ₹780 cr one-time write-off", "bearish"),
    ("{name} in talks to acquire mid-size player; deal details awaited", "neutral"),
    ("{name} CFO resigns; board to announce successor next quarter", "neutral"),
    ("Sector rotation: funds trim {name} exposure, rotate to defensives", "neutral"),
    ("{name} board meeting on December 18 to consider interim dividend", "neutral"),
    ("Brokerages mixed on {name} after management commentary", "neutral"),
]

def get_news(limit: int = 30):
    rng = random.Random(42)
    items = []
    symbols = list(STOCKS.keys())
    for i in range(limit):
        sym   = rng.choice(symbols)
        tmpl, sentiment = rng.choice(NEWS_TEMPLATES)
        tp    = rng.randint(100, 9000)
        headline = tmpl.format(name=STOCKS[sym]["name"], tp=tp)
        mins_ago = rng.randint(5, 480)
        items.append({
            "id": i + 1,
            "symbol": sym,
            "company": STOCKS[sym]["name"],
            "headline": headline,
            "sentiment": sentiment,
            "source": rng.choice(["Moneycontrol", "Economic Times", "Livemint", "Business Standard"]),
            "published_at": (datetime.now() - timedelta(minutes=mins_ago)).isoformat(),
            "minutes_ago": mins_ago,
        })
    items.sort(key=lambda x: x["minutes_ago"])
    return items