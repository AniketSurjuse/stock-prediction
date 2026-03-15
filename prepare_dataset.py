"""
prepare_dataset.py — Load price history from SQLite, engineer features,
and return pytorch-forecasting TimeSeriesDataSet objects ready for TFT.

Can be imported by train.py or run standalone for a quick sanity check:
    python prepare_dataset.py
"""
import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder

from db import get_conn, init_db

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_ROWS_PER_STOCK = 60
ENCODER_LENGTH     = 60
PREDICTION_LENGTH  = 5
MAX_ENCODER_LENGTH = 90


def load_price_df() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql("""
            SELECT p.symbol, p.date, p.open, p.high, p.low, p.close, p.volume,
                   s.name
            FROM price_history p
            JOIN stocks s ON s.symbol = p.symbol
            ORDER BY p.symbol, p.date
        """, conn)

    if df.empty:
        raise ValueError(
            "price_history table is empty. Run `python ingest_prices.py --days 365` first."
        )

    df["date"] = pd.to_datetime(df["date"])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for symbol, grp in df.groupby("symbol"):
        grp = grp.sort_values("date").copy()

        if len(grp) < MIN_ROWS_PER_STOCK:
            print(f"  [prepare] Skipping {symbol} — only {len(grp)} rows")
            continue

        # Calendar features — cast to str so TFT treats them as categoricals
        grp["day_of_week"]  = grp["date"].dt.dayofweek.astype(str)
        grp["month"]        = grp["date"].dt.month.astype(str)
        grp["is_month_end"] = grp["date"].dt.is_month_end.astype(int).astype(str)

        # Rolling stats
        grp["rolling_7d_mean"]  = grp["close"].rolling(7,  min_periods=1).mean()
        grp["rolling_7d_std"]   = grp["close"].rolling(7,  min_periods=1).std().fillna(0)
        grp["rolling_30d_mean"] = grp["close"].rolling(30, min_periods=1).mean()

        # Returns
        grp["return_1d"] = grp["close"].pct_change().fillna(0)
        grp["return_5d"] = grp["close"].pct_change(5).fillna(0)

        # Log volume
        grp["log_volume"] = np.log1p(grp["volume"].clip(lower=1))

        grp["time_idx"] = range(len(grp))
        groups.append(grp)

    if not groups:
        raise ValueError("No stocks had enough data after filtering.")

    result = pd.concat(groups, ignore_index=True)
    print(f"  [prepare] {len(result)} rows across {result['symbol'].nunique()} stocks")
    return result


def build_datasets(df: pd.DataFrame):
    max_train_idx = int(df["time_idx"].max() * 0.8)
    train_df = df[df["time_idx"] <= max_train_idx].copy()
    val_df   = df.copy()

    all_symbols = sorted(df["symbol"].unique().tolist())

    # Pre-declare ALL possible category values so val set never hits unknown categories
    categorical_encoders = {
        "symbol":       NaNLabelEncoder(add_nan=True).fit(pd.Series(all_symbols)),
        "day_of_week":  NaNLabelEncoder(add_nan=True).fit(pd.Series([str(i) for i in range(7)])),
        "month":        NaNLabelEncoder(add_nan=True).fit(pd.Series([str(i) for i in range(1, 13)])),
        "is_month_end": NaNLabelEncoder(add_nan=True).fit(pd.Series(["0", "1"])),
    }

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="close",
        group_ids=["symbol"],
        min_encoder_length=ENCODER_LENGTH // 2,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=PREDICTION_LENGTH,
        static_categoricals=["symbol"],
        time_varying_known_categoricals=["day_of_week", "month", "is_month_end"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=[
            "close", "open", "high", "low",
            "rolling_7d_mean", "rolling_7d_std", "rolling_30d_mean",
            "return_1d", "return_5d", "log_volume",
        ],
        categorical_encoders=categorical_encoders,
        target_normalizer=GroupNormalizer(
            groups=["symbol"],
            transformation="softplus",
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        val_df,
        predict=True,
        stop_randomization=True,
        min_prediction_idx=max_train_idx + 1,
    )

    return training, validation, df


def get_datasets():
    init_db()
    print("[prepare] Loading price history...")
    df = load_price_df()
    print("[prepare] Engineering features...")
    df = engineer_features(df)
    print("[prepare] Building TimeSeriesDataSets...")
    train_ds, val_ds, df = build_datasets(df)
    print(f"[prepare] Train size: {len(train_ds)} | Val size: {len(val_ds)}")
    return train_ds, val_ds, df


if __name__ == "__main__":
    train_ds, val_ds, df = get_datasets()
    print("\nSample batch:")
    x, y = next(iter(train_ds.to_dataloader(batch_size=4, num_workers=0)))
    print(f"  encoder_cont shape : {x['encoder_cont'].shape}")
    print(f"  target shape       : {y[0].shape}")
    print("Dataset preparation OK.")