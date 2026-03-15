from fastapi import APIRouter, HTTPException
from db import get_conn

router = APIRouter()


def _get_prediction(symbol: str) -> dict:
    try:
        from predictor import predict
        return predict(symbol)
    except Exception as e:
        print(f"[stocks] prediction fallback for {symbol}: {e}")
        from dummy_data import get_prediction
        return get_prediction(symbol)


def _get_series(symbol: str, days: int) -> dict:
    try:
        from predictor import predict_series
        return predict_series(symbol, days)
    except Exception as e:
        print(f"[stocks] series fallback for {symbol}: {e}")
        from dummy_data import get_price_series
        data = get_price_series(symbol, days)
        data["source"] = "dummy"
        return data


@router.get("/")
def list_stocks():
    with get_conn() as conn:
        rows = conn.execute("SELECT symbol, name FROM stocks").fetchall()
    return [{"symbol": r["symbol"], "name": r["name"]} for r in rows]


@router.get("/predictions/all")
def all_predictions():
    with get_conn() as conn:
        symbols = [r["symbol"] for r in conn.execute("SELECT symbol FROM stocks").fetchall()]
    return [_get_prediction(sym) for sym in symbols]


@router.get("/{symbol}/series")
def price_series(symbol: str, days: int = 90):
    with get_conn() as conn:
        row = conn.execute("SELECT symbol FROM stocks WHERE symbol=?", (symbol,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Symbol not found")
    return _get_series(symbol, days)


@router.get("/{symbol}/predict")
def predict_stock(symbol: str):
    with get_conn() as conn:
        row = conn.execute("SELECT symbol FROM stocks WHERE symbol=?", (symbol,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Symbol not found")
    return _get_prediction(symbol)


@router.post("/reload-model")
def reload_model():
    """Force the predictor to reload the TFT checkpoint — call after retraining."""
    from predictor import reload_model as _reload
    model, _ = _reload()
    return {"status": "ok", "model_loaded": model is not None}