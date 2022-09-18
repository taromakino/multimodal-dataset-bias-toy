import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

class Swish(nn.Module):
    def forward(self, x):
        return swish(x)

def swish(x):
    return x * torch.sigmoid(x)

def gaussian_nll(x, mu, logvar):
    if x.shape[1] == 1:
        dist = Normal(loc=mu, scale=torch.sqrt(torch.exp(logvar)))
    else:
        cov_mat = torch.diag_embed(torch.exp(logvar), offset=0, dim1=-2, dim2=-1)
        dist = MultivariateNormal(loc=mu, covariance_matrix=cov_mat)
    return -dist.log_prob(x)

def posterior_kld(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

def make_trainer(name, seed, n_epochs, patience):
    return pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=CSVLogger("log", name, version=seed),
        max_epochs=n_epochs,
        callbacks=[
            ModelCheckpoint(monitor="val_loss"),
            EarlyStopping(monitor="val_loss", patience=patience)])