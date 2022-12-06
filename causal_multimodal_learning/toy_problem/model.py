import torch
import torch.nn as nn
from utils.nn_utils import MLP, to_device
from utils.stats import gaussian_nll, log_avg_prob, make_gaussian, prior_kld

class GaussianNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        self.mu_net = MLP(in_dim, hidden_dims, out_dim)
        self.logvar_net = MLP(in_dim, hidden_dims, out_dim)

    def forward(self, *args):
        mu = self.mu_net(*args)
        logvar = self.logvar_net(*args)
        return mu, logvar

class GenerativeModel(nn.Module):
    def __init__(self, data_dim, hidden_dims, latent_dim, beta, n_samples):
        super().__init__()
        self.beta = beta
        self.n_samples = n_samples
        self.encoder_xy = GaussianNetwork(3 * data_dim, hidden_dims, latent_dim)
        self.encoder_x = GaussianNetwork(2 * data_dim, hidden_dims, latent_dim)
        self.decoder = GaussianNetwork(latent_dim + 2 * data_dim, hidden_dims, data_dim)

    def sample_z(self, mu, logvar):
        if self.training:
            sd = torch.exp(logvar / 2) # Same as sqrt(exp(logvar))
            eps = torch.randn_like(sd)
            return mu + eps * sd
        else:
            return mu

    def forward(self, x0, x1, y):
        # ELBO loss
        mu_xy, logvar_xy = self.encoder_xy(x0, x1, y)
        z = self.sample_z(mu_xy, logvar_xy)
        mu_reconst, logvar_reconst = self.decoder(x0, x1, z)
        reconst_loss = gaussian_nll(y, mu_reconst, logvar_reconst)
        # KLD loss
        mu_x, logvar_x = self.encoder_x(x0, x1)
        posterior_xy_dist = make_gaussian(mu_xy, logvar_xy)
        posterior_x_dist = make_gaussian(mu_x, logvar_x)
        kld_loss = torch.distributions.kl_divergence(posterior_xy_dist, posterior_x_dist)
        prior_kld_loss = prior_kld(mu_x, logvar_x)
        return (reconst_loss + kld_loss + self.beta * prior_kld_loss).mean()

class AggregatedPosterior:
    def __init__(self, data_test, encoder_x):
        super().__init__()
        self.posterior_dists = []
        for x0, x1, _ in data_test:
            x0, x1 = to_device(x0, x1)
            mu_x, logvar_x = encoder_x(x0, x1)
            self.posterior_dists.append(make_gaussian(mu_x, logvar_x))

    def log_prob(self, z):
        out = []
        for posterior_dist in self.posterior_dists:
            out.append(posterior_dist.log_prob(z))
        out = torch.stack(out)
        return log_avg_prob(out)