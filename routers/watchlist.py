from fastapi import APIRouter
from db import get_conn

router = APIRouter()


def _get_prediction(symbol: str) -> dict:
    try:
        from predictor import predict
        return predict(symbol)
    except Exception:
        from dummy_data import get_prediction
        return get_prediction(symbol)


@router.get("/")
def watchlist():
    with get_conn() as conn:
        symbols = [r["symbol"] for r in conn.execute("SELECT symbol FROM stocks").fetchall()]
    return [_get_prediction(sym) for sym in symbols]