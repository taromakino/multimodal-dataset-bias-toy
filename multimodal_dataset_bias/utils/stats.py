import torch
import torch.distributions as distributions
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal


class MixtureSameFamily(distributions.Distribution):
    def __init__(self, mixture_distribution, components_distribution):
        self._mixture_distribution = mixture_distribution
        self._components_distribution = components_distribution

        if not isinstance(self._mixture_distribution, distributions.Categorical):
            raise ValueError

        if not isinstance(self._components_distribution, distributions.Distribution):
            raise ValueError

        # Check that batch size matches
        mdbs = self._mixture_distribution.batch_shape
        cdbs = self._components_distribution.batch_shape[:-1]
        if len(mdbs) != 0 and mdbs != cdbs:
            raise ValueError("Inconsistent batch shapes")

        # Check that the number of mixture components matches
        km = self._mixture_distribution.logits.shape[-1]
        kc = self._components_distribution.batch_shape[-1]
        if km is not None and kc is not None and km != kc:
            raise ValueError("Inconsistent number of mixture components")
        self._num_components = km

        event_shape = self._components_distribution.event_shape
        self._event_ndims = len(event_shape)
        super(MixtureSameFamily, self).__init__(batch_shape=cdbs, event_shape=event_shape)


    @property
    def mixture_distribution(self):
        return self._mixture_distribution


    @property
    def components_distribution(self):
        return self._components_distribution


    @property
    def mean(self):
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        return torch.sum(probs * self.components_distribution.mean, dim=-1-self._event_ndims)  # [B, E]


    @property
    def variance(self):
        # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        mean_cond_var = torch.sum(probs*self.components_distribution.variance, dim=-1-self._event_ndims)
        var_cond_mean = torch.sum(probs * (self.components_distribution.mean - self._pad(self.mean)).pow(2.0),
            dim=-1-self._event_ndims)
        return mean_cond_var + var_cond_mean


    def log_prob(self, x):
        x = self._pad(x)
        log_prob_x = self.components_distribution.log_prob(x)  # [S, B, k]
        log_mix_prob = torch.log_softmax(self.mixture_distribution.logits, dim=-1)  # [B, k]
        return torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)  # [S, B]


    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            # [n, B]
            mix_sample = self.mixture_distribution.sample(sample_shape)
            # [n, B, k, E]
            comp_sample = self.components_distribution.sample(sample_shape)
            # [n, B, k]
            mask = F.one_hot(mix_sample, self._num_components)
            # [n, B, k, [1]*E]
            mask = self._pad_mixture_dimensions(mask)
            return torch.sum(comp_sample * mask.float(), dim=-1-self._event_ndims)


    def _pad(self, x):
        d = len(x.shape) - self._event_ndims
        s = x.shape
        x = x.reshape(*s[:d], 1, *s[d:])
        return x


    def _pad_mixture_dimensions(self, x):
        dist_batch_ndims = self.batch_shape.numel()
        cat_batch_ndims = self.mixture_distribution.batch_shape.numel()
        pad_ndims = 0 if cat_batch_ndims == 1 else \
            dist_batch_ndims - cat_batch_ndims
        s = x.shape
        x = torch.reshape(x, shape=(*s[:-1], *(pad_ndims*[1]), *s[-1:], *(self._event_ndims*[1])))
        return x


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


def bernoulli_log_prob(x, mu):
    return (x * torch.log(mu) + (1 - x) * torch.log(1 - mu)).sum(-1)


def diag_gaussian_log_prob(x, mu, var, device):
    c = 2 * torch.pi * torch.ones(1).to(device)
    return (-0.5 * (torch.log(c) + var.log() + (x - mu).pow(2).div(var))).sum(-1)


def log_avg_prob(x):
    return -torch.log(torch.tensor(len(x))) + torch.logsumexp(x, 0)


def row_mean(x):
    return x if len(x.shape) == 1 else x.mean(axis=1)