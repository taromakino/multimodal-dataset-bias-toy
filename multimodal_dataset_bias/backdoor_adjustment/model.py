import numpy as np
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from utils.nn_utils import MLP
from utils.stats import gaussian_nll, log_avg_prob, make_gaussian, make_standard_normal, prior_kld

class GaussianMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.mu_net = MLP(input_dim, hidden_dims, output_dim)
        self.logvar_net = MLP(input_dim, hidden_dims, output_dim)

    def forward(self, *args):
        return self.mu_net(*args), self.logvar_net(*args)

class Model(pl.LightningModule):
    def __init__(self, seed, dpath, task, data_dim, hidden_dims, latent_dim, lr, n_samples, n_posteriors,
            checkpoint_fpath, posterior_params_fpath):
        super().__init__()
        self.save_hyperparameters()
        self.seed = seed
        self.dpath = dpath
        self.task = task
        self.lr = lr
        self.n_samples = n_samples
        self.n_posteriors = n_posteriors
        self.encoder_xy = GaussianMLP(2 * data_dim + 1, hidden_dims, latent_dim)
        self.encoder_x = GaussianMLP(2 * data_dim, hidden_dims, latent_dim)
        self.decoder = GaussianMLP(2 * data_dim + latent_dim, [], 1)
        if checkpoint_fpath:
            self.load_state_dict(torch.load(checkpoint_fpath)["state_dict"])
        if task == "posterior_kld":
            self.freeze()
            self.encoder_x.requires_grad_(True)
            self.test_mu_x, self.test_logvar_x = [], []
        elif task == "log_marginal_likelihood":
            self.test_mu_x, self.test_logvar_x = torch.load(posterior_params_fpath)

    def sample_z(self, mu, logvar):
        if self.training:
            sd = torch.exp(logvar / 2) # Same as sqrt(exp(logvar))
            eps = torch.randn_like(sd)
            return mu + eps * sd
        else:
            return mu

    def forward(self, x, y):
        if self.task == "vae":
            return self.elbo(x, y)
        elif self.task == "posterior_kld":
            return self.posterior_kld(x, y)
        elif self.task == "log_marginal_likelihood":
            return self.log_marginal_likelihood(x, y)

    def elbo(self, x, y):
        mu_xy, logvar_xy = self.encoder_xy(x, y)
        z = self.sample_z(mu_xy, logvar_xy)
        mu_reconst, logvar_reconst = self.decoder(x, z)
        reconst_loss = gaussian_nll(y, mu_reconst, logvar_reconst)
        kld_loss = prior_kld(mu_xy, logvar_xy)
        return {
            "loss": (reconst_loss + kld_loss).mean(),
            "kld": kld_loss.mean()
        }

    def posterior_kld(self, x, y):
        mu_x, logvar_x = self.encoder_x(x)
        mu_xy, logvar_xy = self.encoder_xy(x, y)
        posterior_xy = make_gaussian(mu_xy, logvar_xy)
        posterior_x = make_gaussian(mu_x, logvar_x)
        return {
            "loss": torch.distributions.kl_divergence(posterior_xy, posterior_x).mean(),
            "mu_x": mu_x.detach().cpu(),
            "logvar_x": logvar_x.detach().cpu()}

    def log_marginal_likelihood(self, x, y):
        x_rep = torch.repeat_interleave(x, repeats=self.n_samples, dim=0)
        mu_x, logvar_x = self.encoder_x(x)
        posterior_x = make_gaussian(mu_x, logvar_x)
        z = posterior_x.sample((self.n_samples,))
        mu_reconst, logvar_reconst = self.decoder(x_rep, z)
        decoder_dist = make_gaussian(mu_reconst, logvar_reconst)
        logp_y_xz = decoder_dist.log_prob(y.squeeze())
        assert logp_y_xz.shape == torch.Size([self.n_samples])  # (n_samples,)

        n_test = len(self.test_mu_x)
        if self.n_posteriors < n_test:
            idxs = np.random.choice(n_test, self.n_posteriors, replace=False)
            test_mu_x = self.test_mu_x[idxs]
            test_logvar_x = self.test_logvar_x[idxs]
        else:
            test_mu_x = self.test_mu_x
            test_logvar_x = self.test_logvar_x
        agg_posterior = make_gaussian(test_mu_x, test_logvar_x)
        logp_adjust = log_avg_prob(agg_posterior.log_prob(z[:, None, :]).T)
        assert logp_adjust.shape == torch.Size([self.n_samples])  # (n_samples,)

        return {
            "conditional_lml": log_avg_prob(logp_y_xz),
            "interventional_lml": log_avg_prob(logp_adjust - posterior_x.log_prob(z) + logp_y_xz)}

    def training_step(self, batch, batch_idx):
        out = self.forward(*batch)
        self.log("train_loss", out["loss"], on_step=False, on_epoch=True)
        if self.task == "vae":
            self.log("train_kld", out["kld"], on_step=False, on_epoch=True)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self.forward(*batch)
        self.log("val_loss", out["loss"], on_step=False, on_epoch=True)
        if self.task == "vae":
            self.log("val_kld", out["kld"], on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        out = self.forward(*batch)
        if "loss" in out:
            self.log("test_loss", out["loss"], on_step=False, on_epoch=True)
        if self.task == "posterior_kld":
            self.test_mu_x.append(out["mu_x"])
            self.test_logvar_x.append(out["logvar_x"])
        elif self.task == "log_marginal_likelihood":
            self.log("conditional_lml", out["conditional_lml"], on_step=False, on_epoch=True)
            self.log("interventional_lml", out["interventional_lml"], on_step=False, on_epoch=True)

    def test_epoch_end(self, outputs):
        if self.task == "posterior_kld":
            self.test_mu_x = torch.vstack(self.test_mu_x)
            self.test_logvar_x = torch.vstack(self.test_logvar_x)
            torch.save((self.test_mu_x, self.test_logvar_x), os.path.join(self.dpath, f"version_{self.seed}", "posterior_params.pt"))

    def configure_optimizers(self):
        if self.task == "posterior_kld":
            return Adam(self.encoder_x.parameters(), lr=self.lr)
        else:
            return Adam(self.parameters(), lr=self.lr)