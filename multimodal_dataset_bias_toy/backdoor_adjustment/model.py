import numpy as np
import os
import pytorch_lightning as pl
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils.stats import MixtureSameFamily, diag_gaussian_log_prob, log_avg_prob, make_gaussian


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.shared_trunk = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Linear(128, 64),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Linear(64, 64),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
        )
        self.mu_head = nn.Linear(64, output_dim)
        self.logvar_head = nn.Linear(64, output_dim)

    def forward(self, x):
        output = x.view(x.shape[0], -1)
        output = self.shared_trunk(output)
        return self.mu_head(output), self.logvar_head(output)


def make_decoder(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.LeakyReLU(inplace=True, negative_slope=0.1),
        nn.Linear(64, 128),
        nn.Dropout(p=0.1),
        nn.LeakyReLU(inplace=True, negative_slope=0.1),
        nn.Linear(128, 128),
        nn.LeakyReLU(inplace=True, negative_slope=0.1),
        nn.Linear(128, output_dim),
    )


class Model(pl.LightningModule):
    def __init__(self, seed, dpath, task, data_dim, latent_dim, n_components, lr, n_samples, n_posteriors,
            checkpoint_fpath, posterior_params_fpath):
        super().__init__()
        self.save_hyperparameters()
        self.seed = seed
        self.dpath = dpath
        self.task = task
        self.lr = lr
        self.n_samples = n_samples
        self.n_posteriors = n_posteriors
        self.q_z_xy_net = Encoder(2 * data_dim + 1, latent_dim)
        self.q_z_x_net = Encoder(2 * data_dim, latent_dim)
        self.p_y_xz_net = make_decoder(2 * data_dim + latent_dim, 1)
        self.logits_c = nn.Parameter(torch.ones(n_components) / n_components)
        self.mu_z_c = nn.Parameter(torch.zeros(n_components, latent_dim))
        self.logvar_z_c = nn.Parameter(torch.zeros(n_components, latent_dim))
        nn.init.xavier_normal_(self.mu_z_c)
        nn.init.xavier_normal_(self.logvar_z_c)

        if checkpoint_fpath:
            self.load_state_dict(torch.load(checkpoint_fpath)["state_dict"])
        if task == "posterior_kl":
            self.freeze()
            self.q_z_x_net.requires_grad_(True)
            self.test_mu_x, self.test_logvar_x = [], []
        elif task == "log_marginal_likelihood":
            self.test_mu_x, self.test_logvar_x = torch.load(posterior_params_fpath)


    def sample_z(self, mu, var):
        if self.training:
            sd = var.sqrt()
            eps = torch.randn_like(sd)
            return mu + eps * sd
        else:
            return mu


    def forward(self, x, y):
        if self.task == "vae":
            return self.loss(x, y)
        elif self.task == "posterior_kl":
            return self.posterior_kl(x, y)
        elif self.task == "log_marginal_likelihood":
            return self.log_marginal_likelihood(x, y)


    def loss(self, x, y):
        # z ~ q(z|x,y)
        mu_tilde, logvar_tilde = self.q_z_xy_net(torch.hstack((x, y)))
        var_tilde = F.softplus(logvar_tilde)
        z = self.sample_z(mu_tilde, var_tilde)
        # E_q(c,z|x,y)[log p(y|x,z)]
        mu_y = self.p_y_xz_net(torch.hstack((x, z)))
        log_p_y_xz = diag_gaussian_log_prob(y, mu_y, torch.ones_like(mu_y), self.device).mean()
        # KL(q(z|x,y) || p(z))
        p_c = distributions.Categorical(logits=self.logits_c)
        p_z_c = distributions.Independent(distributions.Normal(loc=self.mu_z_c, scale=torch.exp(0.5 * self.logvar_z_c)), 1)
        p_z = MixtureSameFamily(p_c, p_z_c)
        log_q_z_xy = diag_gaussian_log_prob(z, mu_tilde, var_tilde, self.device)
        kl = (log_q_z_xy - p_z.log_prob(z)).mean()
        elbo = log_p_y_xz - kl
        return {
            "loss": -elbo,
            "kl": kl
        }


    def posterior_kl(self, x, y):
        mu_x, logvar_x = self.q_z_x_net(x)
        mu_xy, logvar_xy = self.q_z_xy_net(torch.hstack((x, y)))
        posterior_xy = make_gaussian(mu_xy, F.softplus(logvar_xy))
        posterior_x = make_gaussian(mu_x, F.softplus(logvar_x))
        return {
            "loss": distributions.kl_divergence(posterior_xy, posterior_x).mean(),
            "mu_x": mu_x.detach().cpu(),
            "logvar_x": logvar_x.detach().cpu()}


    def log_marginal_likelihood(self, x, y):
        x_rep = torch.repeat_interleave(x, repeats=self.n_samples, dim=0)
        mu_x, logvar_x = self.q_z_x_net(x)
        posterior_x = make_gaussian(mu_x, F.softplus(logvar_x))
        z = posterior_x.sample((self.n_samples,))
        mu_reconst = self.p_y_xz_net(torch.hstack((x_rep, z)))
        logvar_reconst = torch.zeros_like(mu_reconst) # Temporarily hard-coded for Var=1, make this configurable
        decoder_dist = make_gaussian(mu_reconst, F.softplus(logvar_reconst))
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
        agg_posterior = make_gaussian(test_mu_x, F.softplus(test_logvar_x))
        logp_adjust = log_avg_prob(agg_posterior.log_prob(z[:, None, :]).T)
        assert logp_adjust.shape == torch.Size([self.n_samples])  # (n_samples,)

        return {
            "conditional_lml": log_avg_prob(logp_y_xz),
            "interventional_lml": log_avg_prob(logp_adjust - posterior_x.log_prob(z) + logp_y_xz)}


    def training_step(self, batch, batch_idx):
        out = self.forward(*batch)
        self.log("train_loss", out["loss"], on_step=False, on_epoch=True)
        if self.task == "vae":
            self.log("train_kl", out["kl"], on_step=False, on_epoch=True)
        return out["loss"]


    def validation_step(self, batch, batch_idx):
        out = self.forward(*batch)
        self.log("val_loss", out["loss"], on_step=False, on_epoch=True)
        if self.task == "vae":
            self.log("val_kl", out["kl"], on_step=False, on_epoch=True)


    def test_step(self, batch, batch_idx):
        out = self.forward(*batch)
        if "loss" in out:
            self.log("test_loss", out["loss"], on_step=False, on_epoch=True)
        if self.task == "posterior_kl":
            self.test_mu_x.append(out["mu_x"])
            self.test_logvar_x.append(out["logvar_x"])
        elif self.task == "log_marginal_likelihood":
            self.log("conditional_lml", out["conditional_lml"], on_step=False, on_epoch=True)
            self.log("interventional_lml", out["interventional_lml"], on_step=False, on_epoch=True)


    def test_epoch_end(self, outputs):
        if self.task == "posterior_kl":
            self.test_mu_x = torch.vstack(self.test_mu_x)
            self.test_logvar_x = torch.vstack(self.test_logvar_x)
            torch.save((self.test_mu_x, self.test_logvar_x), os.path.join(self.dpath, f"version_{self.seed}", "posterior_params.pt"))


    def configure_optimizers(self):
        if self.task == "posterior_kl":
            return Adam(self.q_z_x_net.parameters(), lr=self.lr)
        else:
            return Adam(self.parameters(), lr=self.lr)