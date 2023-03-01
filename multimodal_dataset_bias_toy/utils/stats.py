import torch
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal


EPSILON = 1e-8


def make_gaussian(mu, var):
    '''
    The inputs must have shape (batch_size, dim). If we were to pass in a 1D array, it's ambiguous whether to return a
    batch of univariate Gaussians, or a single multivariate Gaussian.
    '''
    batch_size, dim = mu.shape
    mu, var = mu.squeeze(), var.squeeze()
    if dim == 1:
        dist = Normal(loc=mu, scale=var.sqrt())
    else:
        cov_mat = torch.diag_embed(var, offset=0, dim1=-2, dim2=-1)
        dist = MultivariateNormal(loc=mu, covariance_matrix=cov_mat)
    return dist


def row_mean(x):
    return x if len(x.shape) == 1 else x.mean(axis=1)