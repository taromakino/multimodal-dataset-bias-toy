import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


def make_trainer(dpath, seed, n_steps):
    return pl.Trainer(
        logger=CSVLogger(dpath, name="", version=seed),
        callbacks=[
            ModelCheckpoint(monitor="val_loss", filename="best")],
        max_steps=n_steps,
        accelerator="gpu" if torch.cuda.is_available() else "cpu")