import math
import pytorch_lightning as pl
import torch
from torch.optim import Adam
from utils.nn_utils import MLP


class Mine(pl.LightningModule):
    def __init__(self, seed, data_dim, hidden_dims, include_y, lr):
        super().__init__()
        self.save_hyperparameters()
        self.seed = seed
        self.include_y = include_y
        self.lr = lr
        self.net = MLP(3 * data_dim + (1 if include_y else 0), hidden_dims, 1)


    def loss(self, u, x, y):
        u_shuffled = u[torch.randperm(u.shape[0])]
        if self.include_y:
            joint_term = self.net(u, x, y)
            product_term = self.net(u_shuffled, x, y)
        else:
            joint_term = self.net(u, x)
            product_term = self.net(u_shuffled, x)
        product_term = -math.log(len(product_term)) + torch.logsumexp(product_term, 0)
        # product_term = torch.exp(product_term - 1).mean()
        return -joint_term.mean() + product_term


    def training_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)