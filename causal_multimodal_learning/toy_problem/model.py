import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
from utils.nn_utils import MLP, device, load_model
from utils.stats import gaussian_nll, make_gaussian, prior_kld
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

class PosteriorX(pl.LightningModule):
    def __init__(self, data_dim, hidden_dims, latent_dim, lr, wd, batch_size):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.wd = wd
        self.encoder_xy = GaussianNetwork(3 * data_dim, hidden_dims, latent_dim)
        self.encoder_x = GaussianNetwork(2 * data_dim, hidden_dims, latent_dim)
        self.prior = make_gaussian(torch.zeros((batch_size, latent_dim), device=device()), torch.zeros((batch_size,
            latent_dim), device=device()))

    def loss(self, x0, x1, y):
        mu_xy, logvar_xy = self.encoder_xy(x0, x1, y)
        mu_x, logvar_x = self.encoder_x(x0, x1)
        posterior_xy_dist = make_gaussian(mu_xy.clone().detach(), logvar_xy.clone().detach())
        posterior_x_dist = make_gaussian(mu_x, logvar_x)
        loss = torch.distributions.kl_divergence(posterior_xy_dist, posterior_x_dist).mean()
        kld = prior_kld(mu_x, logvar_x).mean()
        return loss, kld

    def training_step(self, batch, batch_idx):
        loss, kld = self.loss(*batch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, kld = self.loss(*batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_kld", kld, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, kld = self.loss(*batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_kld", kld, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return AdamW(self.encoder_x.parameters(), lr=self.lr, weight_decay=self.wd)

class SemiSupervisedVae(pl.LightningModule):
    def __init__(self, data_dim, hidden_dims, latent_dim, lr, wd):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.wd = wd
        self.encoder = GaussianNetwork(3 * data_dim, hidden_dims, latent_dim)
        self.decoder = GaussianNetwork(latent_dim + 2 * data_dim, hidden_dims, data_dim)

    def sample_z(self, mu, logvar):
        if self.training:
            sd = torch.exp(logvar / 2) # Same as sqrt(exp(logvar))
            eps = torch.randn_like(sd)
            return mu + eps * sd
        else:
            return mu

    def loss(self, x0, x1, y):
        mu_z, logvar_z = self.encoder(x0, x1, y)
        kld_loss = prior_kld(mu_z, logvar_z)
        z = self.sample_z(mu_z, logvar_z)
        mu_y, logvar_y = self.decoder(x0, x1, z)
        reconst_loss = gaussian_nll(y, mu_y, logvar_y)
        return kld_loss, reconst_loss

    def training_step(self, batch, batch_idx):
        kld_loss, reconst_loss = self.loss(*batch)
        return (kld_loss + reconst_loss).mean()

    def validation_step(self, batch, batch_idx):
        kld_loss, reconst_loss = self.loss(*batch)
        self.log("val_loss", (kld_loss + reconst_loss).mean(), on_step=False, on_epoch=True)
        self.log("val_kld_loss", kld_loss.mean(), on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        kld_loss, reconst_loss = self.loss(*batch)
        self.log("test_loss", (kld_loss + reconst_loss).mean(), on_step=False, on_epoch=True)
        self.log("test_kld_loss", kld_loss.mean(), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
    
class InferenceNetwork(pl.LightningModule):
    def __init__(self, dpath, seed, n_samples, latent_dim):
        super().__init__()
        self.save_hyperparameters()
        self.n_samples = n_samples
        self.vae = load_model(SemiSupervisedVae, os.path.join(dpath, "vae", f"version_{seed}", "checkpoints"))
        self.posterior_x = load_model(PosteriorX, os.path.join(dpath, "posterior_x", f"version_{seed}", "checkpoints"))
        self.prior = make_gaussian(torch.zeros(latent_dim, device=device())[None], torch.zeros(latent_dim,
            device=device())[None])

    def test_step(self, batch, batch_idx):
        x0, x1, y = batch
        assert len(x0) == 1  # Assumes batch_size=1
        mu_x, logvar_x = self.posterior_x.encoder_x(x0, x1)
        posterior_x_dist = make_gaussian(mu_x, logvar_x)
        x0_rep = torch.repeat_interleave(x0, repeats=self.n_samples, dim=0)
        x1_rep = torch.repeat_interleave(x1, repeats=self.n_samples, dim=0)
        z = posterior_x_dist.sample((self.n_samples,))
        y_mu, y_logvar = self.vae.decoder(x0_rep, x1_rep, z[:, None] if len(z.shape) == 1 else z)
        decoder_dist = make_gaussian(y_mu, y_logvar)
        y_logp = decoder_dist.log_prob(y.squeeze())
        conditional_logp = -torch.log(torch.tensor(self.n_samples)) + torch.logsumexp(y_logp, 0).item()
        interventional_logp = -torch.log(torch.tensor(self.n_samples)) + torch.logsumexp(
            self.prior.log_prob(z) - posterior_x_dist.log_prob(z) + y_logp, 0).item()
        self.log("conditional_logp", conditional_logp, on_step=True, on_epoch=True)
        self.log("interventional_logp", interventional_logp, on_step=True, on_epoch=True)