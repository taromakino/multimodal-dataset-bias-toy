import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class Model(pl.LightningModule):
    def __init__(self, seed, dpath, input_dim, temperature, lr):
        super().__init__()
        self.save_hyperparameters()
        self.seed = seed
        self.dpath = dpath
        self.temperature = temperature
        self.lr = lr
        self.alpha = nn.Parameter(torch.ones(2 * input_dim))


    def loss(self, u, x, y):
        x0, x1 = torch.chunk(x, 2, 1)
        logits = self.temperature * ((x0 * x1).sum(dim=1) + (self.alpha * u).sum(dim=1))
        loss = F.binary_cross_entropy_with_logits(logits, y.squeeze())
        return loss


    def training_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)


    def validation_epoch_end(self, outputs):
        self.log("alpha_norm", torch.linalg.vector_norm(self.alpha))


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)