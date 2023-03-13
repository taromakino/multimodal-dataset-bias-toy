import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from utils.stats import diag_gaussian_log_prob


class Model(pl.LightningModule):
    def __init__(self, seed, dpath, y_sd, lr):
        super().__init__()
        self.save_hyperparameters()
        self.seed = seed
        self.dpath = dpath
        self.y_sd = y_sd
        self.lr = lr
        self.alpha = nn.Parameter(torch.tensor(1.))


    def loss(self, u, x, y):
        x0, x1 = torch.chunk(x, 2, 1)
        mu_y_ux = (x0.mean(dim=1) + x1.mean(dim=1))[:, None] + self.alpha * u
        var_y_ux = self.y_sd ** 2 * torch.ones_like(mu_y_ux)
        log_p_y_ux = diag_gaussian_log_prob(y, mu_y_ux, var_y_ux, self.device).mean()
        return -log_p_y_ux


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