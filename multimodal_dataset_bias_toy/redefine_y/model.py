import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class Model(pl.LightningModule):
    def __init__(self, seed, dpath, lr):
        super().__init__()
        self.save_hyperparameters()
        self.seed = seed
        self.dpath = dpath
        self.lr = lr
        self.alpha = nn.Parameter(torch.tensor(1.))


    def loss(self, u, x, y):
        # x0, x1 = torch.chunk(x, 2, 1)
        raise NotImplementedError


    def training_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)


    def validation_epoch_end(self, outputs):
        self.log("alpha", self.alpha)


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)