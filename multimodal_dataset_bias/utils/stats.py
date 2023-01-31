import torch
import torch.distributions
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from utils.nn_utils import device


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


def make_standard_normal(latent_dim):
    mu = torch.zeros(1, latent_dim, device=device())
    logvar = torch.zeros(1, latent_dim, device=device())
    return make_gaussian(mu, logvar)


def gaussian_nll(x, mu, logvar):
    dist = make_gaussian(mu, logvar)
    return -dist.log_prob(x.squeeze())


def gaussian_kl(mu_p, logvar_p, mu_q, logvar_q):
    p = make_gaussian(mu_p, logvar_p)
    q = make_gaussian(mu_q, logvar_q)
    return torch.distributions.kl_divergence(p, q)


def prior_kl(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def log_avg_prob(x):
    return -torch.log(torch.tensor(len(x))) + torch.logsumexp(x, 0)


def row_mean(x):
    return x if len(x.shape) == 1 else x.mean(axis=1)