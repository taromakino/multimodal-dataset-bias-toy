import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.nn_utils import MLP

class UnimodalEnsemble(pl.LightningModule):
    def __init__(self, seed, dpath, data_dim, hidden_dims, lr):
        super().__init__()
        self.save_hyperparameters()
        self.seed = seed
        self.dpath = dpath
        self.lr = lr
        self.model0 = MLP(data_dim, hidden_dims, 1)
        self.model1 = MLP(data_dim, hidden_dims, 1)

    def forward(self, x, y):
        x0, x1 = torch.chunk(x, 2, 1)
        pred0 = self.model0(x0)
        pred1 = self.model1(x1)
        pred = (pred0 + pred1) / 2
        return F.mse_loss(pred, y)

    def training_step(self, batch, batch_idx):
        loss = self.forward(*batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(*batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss = self.forward(*batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

class Multimodal(pl.LightningModule):
    def __init__(self, seed, dpath, data_dim, hidden_dims, lr):
        super().__init__()
        self.save_hyperparameters()
        self.seed = seed
        self.dpath = dpath
        self.lr = lr
        self.model = MLP(2 * data_dim, hidden_dims, 1)

    def forward(self, x, y):
        pred = self.model(x)
        return F.mse_loss(pred, y)

    def training_step(self, batch, batch_idx):
        loss = self.forward(*batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(*batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss = self.forward(*batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)