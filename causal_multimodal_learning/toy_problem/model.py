import pytorch_lightning as pl
import torch
import torch.nn as nn
from utils.nn_utils import MLP
from utils.stats import conditional_logpy_x, interventional_logpy_x, gaussian_nll, make_gaussian, make_standard_normal, prior_kld
from torch.optim import AdamW

class GaussianNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        self.mu_net = MLP(in_dim, hidden_dims, out_dim)
        self.logvar_net = MLP(in_dim, hidden_dims, out_dim)

    def forward(self, *args):
        mu = self.mu_net(*args)
        logvar = self.logvar_net(*args)
        return mu, logvar

class Model(pl.LightningModule):
    def __init__(self, data_dim, hidden_dims, latent_dim, beta, n_samples, lr, wd):
        super().__init__()
        self.save_hyperparameters()
        self.beta = beta
        self.n_samples = n_samples
        self.lr = lr
        self.wd = wd
        self.encoder_xy = GaussianNetwork(3 * data_dim, hidden_dims, latent_dim)
        self.encoder_x = GaussianNetwork(2 * data_dim, hidden_dims, latent_dim)
        self.decoder = GaussianNetwork(latent_dim + 2 * data_dim, hidden_dims, data_dim)
        self.prior = make_standard_normal(1, latent_dim)

    def sample_z(self, mu, logvar):
        if self.training:
            sd = torch.exp(logvar / 2) # Same as sqrt(exp(logvar))
            eps = torch.randn_like(sd)
            return mu + eps * sd
        else:
            return mu

    def loss(self, x0, x1, y):
        # ELBO loss
        mu_xy, logvar_xy = self.encoder_xy(x0, x1, y)
        z = self.sample_z(mu_xy, logvar_xy)
        mu_reconst, logvar_reconst = self.decoder(x0, x1, z)
        reconst_loss = gaussian_nll(y, mu_reconst, logvar_reconst)
        kld_loss = prior_kld(mu_xy, logvar_xy)
        # Posterior loss
        mu_x, logvar_x = self.encoder_x(x0, x1)
        posterior_xy_dist = make_gaussian(mu_xy.clone().detach(), logvar_xy.clone().detach())
        posterior_x_dist = make_gaussian(mu_x, logvar_x)
        posterior_loss = torch.distributions.kl_divergence(posterior_xy_dist, posterior_x_dist)
        return reconst_loss, kld_loss, posterior_loss

    def inference(self, x0, x1, y):
        assert len(x0) == 1  # Assumes batch_size=1
        x0_rep = torch.repeat_interleave(x0, repeats=self.n_samples, dim=0)
        x1_rep = torch.repeat_interleave(x1, repeats=self.n_samples, dim=0)
        mu_x, logvar_x = self.encoder_x(x0, x1)
        posterior_x_dist = make_gaussian(mu_x, logvar_x)
        z = posterior_x_dist.sample((self.n_samples,))
        mu_reconst, logvar_reconst = self.decoder(x0_rep, x1_rep, z[:, None] if len(z.shape) == 1 else z)
        decoder_dist = make_gaussian(mu_reconst, logvar_reconst)
        logp_y_xz = decoder_dist.log_prob(y.squeeze())
        conditional_logp = conditional_logpy_x(logp_y_xz)
        interventional_logp = interventional_logpy_x(self.prior.log_prob(z), posterior_x_dist.log_prob(z), logp_y_xz)
        return conditional_logp, interventional_logp

    def training_step(self, batch, batch_idx):
        reconst_loss, kld_loss, posterior_loss = self.loss(*batch)
        return (reconst_loss + self.beta * kld_loss + posterior_loss).mean()

    def validation_step(self, batch, batch_idx):
        reconst_loss, kld_loss, posterior_loss = self.loss(*batch)
        self.log("val_elbo_loss", (reconst_loss + kld_loss).mean(), on_step=False, on_epoch=True)
        self.log("val_kld_loss", kld_loss.mean(), on_step=False, on_epoch=True)
        self.log("val_posterior_loss", posterior_loss.mean(), on_step=False, on_epoch=True)
        conditional_logp, interventional_logp = self.inference(*batch)
        self.log("val_loss", -conditional_logp, on_step=False, on_epoch=True) # Minimize -log p
        self.log("val_interventional_logp", interventional_logp, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        conditional_logp, interventional_logp = self.inference(*batch)
        self.log("test_conditional_logp", conditional_logp, on_step=False, on_epoch=True)
        self.log("test_interventional_logp", interventional_logp, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)