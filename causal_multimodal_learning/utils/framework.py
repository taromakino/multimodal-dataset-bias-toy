import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.distributions
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

def make_gaussian(mu, logvar):
    '''
    The inputs must have shape (batch_size, dim). If we were to pass in a 1D array, it's ambiguous whether to return a
    batch of univariate Gaussians, or a single multivariate Gaussian.
    '''
    batch_size, dim = mu.shape
    mu, logvar = mu.squeeze(), logvar.squeeze()
    if dim == 1:
        dist = Normal(loc=mu, scale=torch.exp(logvar / 2))
    else:
        cov_mat = torch.diag_embed(torch.exp(logvar), offset=0, dim1=-2, dim2=-1)
        dist = MultivariateNormal(loc=mu, covariance_matrix=cov_mat)
    return dist

def gaussian_nll(x, mu, logvar):
    dist = make_gaussian(mu, logvar)
    return -dist.log_prob(x.squeeze())

def gaussian_kld(mu_p, logvar_p, mu_q, logvar_q):
    p = make_gaussian(mu_p, logvar_p)
    q = make_gaussian(mu_q, logvar_q)
    return torch.distributions.kl_divergence(p, q)

def prior_kld(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

def make_trainer(dpath, seed, n_epochs, patience):
    return pl.Trainer(
        logger=CSVLogger(dpath, name=None, version=seed),
        callbacks=[
            ModelCheckpoint(monitor="val_loss"),
            EarlyStopping(monitor="val_loss", patience=patience)],
        max_epochs=n_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu")