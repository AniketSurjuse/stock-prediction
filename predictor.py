"""
predictor.py — Load the trained TFT checkpoint and serve predictions.

Usage (standalone test):
    python predictor.py RELIANCE
"""
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer

from db import get_conn
from prepare_dataset import engineer_features, ENCODER_LENGTH, PREDICTION_LENGTH

MODELS_DIR    = Path("models")
CHECKPOINT    = MODELS_DIR / "tft_best-v2.ckpt"
METADATA_FILE = MODELS_DIR / "tft_metadata.pkl"

_model    = None
_metadata = None


def _model_available() -> bool:
    return CHECKPOINT.exists() and METADATA_FILE.exists()


def _load_model():
    global _model, _metadata
    if _model is not None:
        return _model, _metadata
    if not _model_available():
        return None, None
    with open(METADATA_FILE, "rb") as f:
        _metadata = pickle.load(f)
    _model = TemporalFusionTransformer.load_from_checkpoint(str(CHECKPOINT))
    _model.eval()
    if torch.cuda.is_available():
        _model = _model.cuda()
    print(f"[predictor] Model loaded — {CHECKPOINT}")
    return _model, _metadata


def reload_model():
    global _model, _metadata
    _model = None
    _metadata = None
    return _load_model()


def _load_recent_prices(symbol: str, days: int = 150) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql("""
            SELECT p.symbol, p.date, p.open, p.high, p.low, p.close, p.volume, s.name
            FROM price_history p
            JOIN stocks s ON s.symbol = p.symbol
            WHERE p.symbol = ?
            ORDER BY p.date DESC LIMIT ?
        """, conn, params=(symbol, days))
    if df.empty:
        raise ValueError(f"No price data for {symbol}.")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _dummy_fallback(symbol: str, reason: str = "") -> dict:
    from dummy_data import get_prediction
    result = get_prediction(symbol)
    result["source"] = f"dummy:{reason}" if reason else "dummy"
    return result


def _next_trading_days(from_date: pd.Timestamp, n: int) -> list:
    """Return next n trading days (Mon–Fri) after from_date as date strings."""
    days = []
    cursor = from_date
    while len(days) < n:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:   # Mon=0 … Fri=4
            days.append(cursor.strftime("%Y-%m-%d"))
    return days


def _run_forward_inference(model, meta, df: pd.DataFrame, n_steps: int = 2):
    """
    Run model on the tail of df to produce n_steps future predictions.
    Returns list of (date_str, predicted_price) for the next n_steps trading days.
    """
    from pytorch_forecasting import TimeSeriesDataSet

    df = df.copy()
    df["time_idx"] = range(len(df))
    last_idx = int(df["time_idx"].max())

    inference_ds = TimeSeriesDataSet.from_dataset(
        meta["train_dataset"], df,
        predict=True,
        stop_randomization=True,
        min_prediction_idx=last_idx - PREDICTION_LENGTH + 1,
    )
    loader = inference_ds.to_dataloader(
        train=False, batch_size=1, num_workers=0, shuffle=False
    )

    with torch.no_grad():
        # mode="quantiles" → shape (n, pred_len, 7)
        # pytorch-forecasting inverse-normalises automatically
        preds = model.predict(loader, mode="quantiles")

    preds = preds if isinstance(preds, torch.Tensor) else preds.output
    # preds[-1] = the sample anchored at the last available date
    last_pred = preds[-1]   # shape (pred_len, 7)

    last_date = df["date"].iloc[-1]
    future_dates = _next_trading_days(last_date, n_steps)

    results = []
    for i, date_str in enumerate(future_dates):
        if i >= last_pred.shape[0]:
            break
        p10 = float(last_pred[i, 1])
        p50 = float(last_pred[i, 3])
        p90 = float(last_pred[i, 5])
        results.append({"date": date_str, "p10": round(p10, 2),
                         "p50": round(p50, 2), "p90": round(p90, 2)})
    return results


def predict(symbol: str) -> dict:
    """Single next-day prediction for the watchlist card."""
    model, meta = _load_model()
    if model is None:
        return _dummy_fallback(symbol, "no_checkpoint")

    try:
        df = _load_recent_prices(symbol)
        df = engineer_features(df)
        if len(df) < ENCODER_LENGTH:
            return _dummy_fallback(symbol, "insufficient_data")

        forecasts = _run_forward_inference(model, meta, df, n_steps=2)
        if not forecasts:
            return _dummy_fallback(symbol, "no_forecast")

        current_price        = float(df["close"].iloc[-1])
        p50                  = forecasts[0]["p50"]
        p10                  = forecasts[0]["p10"]
        p90                  = forecasts[0]["p90"]
        predicted_price      = round(p50, 2)
        predicted_change_pct = round((predicted_price - current_price) / current_price * 100, 2)
        ci_width             = (p90 - p10) / max(current_price, 1)
        confidence           = round(float(np.clip(1 - ci_width * 5, 0.3, 0.97)), 2)
        signal               = ("BUY" if predicted_change_pct > 0.5
                                else "SELL" if predicted_change_pct < -0.5 else "HOLD")

        with get_conn() as conn:
            name = conn.execute(
                "SELECT name FROM stocks WHERE symbol=?", (symbol,)
            ).fetchone()["name"]

        return {
            "symbol":               symbol,
            "name":                 name,
            "current_price":        round(current_price, 2),
            "predicted_price":      predicted_price,
            "predicted_change_pct": predicted_change_pct,
            "signal":               signal,
            "confidence":           confidence,
            "sentiment":            "neutral",
            "sentiment_score":      0.0,
            "p10":                  round(p10, 2),
            "p90":                  round(p90, 2),
            "forecast_dates":       [f["date"] for f in forecasts],
            "forecast_prices":      [f["p50"] for f in forecasts],
            "as_of":                datetime.today().strftime("%Y-%m-%d"),
            "source":               "tft",
        }

    except Exception as e:
        print(f"[predictor] predict() error for {symbol}: {e}")
        return _dummy_fallback(symbol, str(e)[:60])


def predict_series(symbol: str, days: int = 90) -> dict:
    """
    Return actual close prices + 2-day forward forecast appended at the end.
    The chart will show:
      - Historical actual line for `days` days
      - 2 future predicted points connected to the last actual price
    """
    from dummy_data import get_price_series

    # Pull real actual prices
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT date, close FROM price_history
            WHERE symbol=? ORDER BY date DESC LIMIT ?
        """, (symbol, days)).fetchall()

    if not rows:
        data = get_price_series(symbol, days)
        data["source"] = "dummy"
        return data

    rows   = list(reversed(rows))
    dates  = [r["date"]  for r in rows]
    actual = [r["close"] for r in rows]

    model, meta = _load_model()
    if model is None:
        dummy     = get_price_series(symbol, len(dates))
        predicted = dummy["predicted"][:len(dates)]
        return {"dates": dates, "actual": actual, "predicted": predicted,
                "source": "real_dummy_pred", "forecast_start_idx": None}

    try:
        df = _load_recent_prices(symbol, days=days + ENCODER_LENGTH)
        df = engineer_features(df)
        if len(df) < ENCODER_LENGTH:
            raise ValueError("insufficient data for encoder")

        forecasts = _run_forward_inference(model, meta, df, n_steps=2)
        if not forecasts:
            raise ValueError("no forecasts returned")

        # Append 2 future dates and their predicted prices
        future_dates  = [f["date"] for f in forecasts]
        future_prices = [f["p50"]  for f in forecasts]

        # Build combined series:
        # - actual points: real close prices (predicted=None)
        # - bridge point: last actual date appears in both lines (connects them)
        # - forecast points: predicted prices only (actual=None)
        combined_dates     = dates + future_dates
        combined_actual    = actual + [None] * len(future_dates)
        # Predicted line: None for history, then bridge + 2 forecast points
        combined_predicted = [None] * len(dates) + future_prices

        # Bridge: last historical point shared by both lines so they connect
        bridge_idx                    = len(dates) - 1
        combined_predicted[bridge_idx] = actual[-1]

        return {
            "dates":            combined_dates,
            "actual":           combined_actual,
            "predicted":        combined_predicted,
            "source":           "tft",
            "forecast_start_idx": bridge_idx,   # frontend uses this to draw the divider
        }

    except Exception as e:
        print(f"[predictor] predict_series() error for {symbol}: {e}")
        dummy     = get_price_series(symbol, len(dates))
        predicted = dummy["predicted"][:len(dates)]
        return {"dates": dates, "actual": actual, "predicted": predicted,
                "source": "real_dummy_pred", "forecast_start_idx": None}


if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    print(f"\nPredicting for {symbol}...")
    result = predict(symbol)
    for k, v in result.items():
        print(f"  {k:28s}: {v}")