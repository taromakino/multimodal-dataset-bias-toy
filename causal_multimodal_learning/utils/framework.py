import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.distributions
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

def gaussian_kld(mu_p, logvar_p, mu_q, logvar_q):
    if mu_p.shape[1] == 1:
        dist_p = Normal(loc=mu_p, scale=torch.sqrt(torch.exp(logvar_p)))
        dist_q = Normal(loc=mu_q, scale=torch.sqrt(torch.exp(logvar_q)))
    else:
        cov_mat_p = torch.diag_embed(torch.exp(logvar_p), offset=0, dim1=-2, dim2=-1)
        cov_mat_q = torch.diag_embed(torch.exp(logvar_q), offset=0, dim1=-2, dim2=-1)
        dist_p = MultivariateNormal(loc=mu_p, covariance_matrix=cov_mat_p)
        dist_q = MultivariateNormal(loc=mu_q, covariance_matrix=cov_mat_q)
    return torch.distributions.kl_divergence(dist_p, dist_q)

def posterior_kld(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

def make_trainer(name, seed, n_epochs, patience):
    return pl.Trainer(
        logger=CSVLogger("log", name, version=seed),
        callbacks=[
            ModelCheckpoint(monitor="val_loss"),
            EarlyStopping(monitor="val_loss", patience=patience)],
        max_epochs=n_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu")