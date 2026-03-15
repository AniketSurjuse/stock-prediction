"""
train.py — Train a Temporal Fusion Transformer on Nifty top-10 price history.

Usage:
    python train.py               # train with defaults
    python train.py --epochs 30   # override epochs
    python train.py --fast        # quick smoke-test (2 epochs, small batch)
"""
import argparse
import pickle
from pathlib import Path

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from prepare_dataset import get_datasets, PREDICTION_LENGTH

MODELS_DIR    = Path("models")
CHECKPOINT    = MODELS_DIR / "tft_best.ckpt"
METADATA_FILE = MODELS_DIR / "tft_metadata.pkl"
LOGS_DIR      = Path("logs")


def train(epochs: int = 50, batch_size: int = 64, fast: bool = False):
    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    if fast:
        epochs     = 2
        batch_size = 16
        print("[train] Fast mode — 2 epochs, batch 16")

    # ------------------------------------------------------------------ data
    train_ds, val_ds, df = get_datasets()

    train_loader = train_ds.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
    )
    val_loader = val_ds.to_dataloader(
        train=False,
        batch_size=batch_size * 2,
        num_workers=0,
        shuffle=False,
    )

    # ----------------------------------------------------------------- model
    tft = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=3e-3,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    print(f"[train] Model params: {tft.size() / 1e3:.1f}k")

    # -------------------------------------------------------------- callbacks
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True,
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(MODELS_DIR),
        filename="tft_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # ---------------------------------------------------------------- trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"[train] Using accelerator: {accelerator.upper()}")
    if accelerator == "gpu":
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}")

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[early_stop, checkpoint_cb, lr_monitor],
        logger=CSVLogger(str(LOGS_DIR), name="tft"),
        enable_progress_bar=True,
        log_every_n_steps=5,
    )

    # ------------------------------------------------------------------- fit
    print(f"[train] Starting training — {epochs} max epochs")
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_ckpt = checkpoint_cb.best_model_path
    print(f"[train] Best checkpoint : {best_ckpt}")
    print(f"[train] Best val_loss   : {checkpoint_cb.best_model_score:.4f}")

    # --------------------------------------------------------- save metadata
    metadata = {
        "prediction_length": PREDICTION_LENGTH,
        "train_dataset":     train_ds,
        "feature_cols": [
            "close", "open", "high", "low",
            "rolling_7d_mean", "rolling_7d_std", "rolling_30d_mean",
            "return_1d", "return_5d", "log_volume",
        ],
    }
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)
    print(f"[train] Metadata saved → {METADATA_FILE}")

    return best_ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,  default=50)
    parser.add_argument("--batch-size", type=int,  default=64)
    parser.add_argument("--fast",       action="store_true",
                        help="Smoke-test: 2 epochs, small batch")
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, fast=args.fast)