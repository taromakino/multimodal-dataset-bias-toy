import numpy as np
import os
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils.nn_utils import MLP
from utils.stats import log_avg_prob, make_gaussian


class GaussianMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.mu_net = MLP(input_dim, hidden_dims, output_dim)
        self.logvar_net = MLP(input_dim, hidden_dims, output_dim)

    def forward(self, *args):
        return self.mu_net(*args), F.softplus(self.logvar_net(*args))


class GaussianMixtureMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, n_components):
        super().__init__()
        self.mu_net = MLP(input_dim, hidden_dims, n_components * latent_dim)
        self.logvar_net = MLP(input_dim, hidden_dims, n_components * latent_dim)
        self.output_shape = (-1, n_components, latent_dim)

    def forward(self, *args):
        mu = self.mu_net(*args).reshape(self.output_shape)
        var = F.softplus(self.logvar_net(*args).reshape(self.output_shape))
        return mu, var


class Model(pl.LightningModule):
    def __init__(self, seed, dpath, task, data_dim, hidden_dims, latent_dim, n_components, lr, n_samples, n_posteriors,
            checkpoint_fpath, posterior_params_fpath):
        super().__init__()
        self.save_hyperparameters()
        self.seed = seed
        self.dpath = dpath
        self.task = task
        self.lr = lr
        self.n_samples = n_samples
        self.n_posteriors = n_posteriors
        self.q_z_xy_net = GaussianMLP(2 * data_dim + 1, hidden_dims, latent_dim)
        self.q_z_x_net = GaussianMLP(2 * data_dim, hidden_dims, latent_dim)
        self.p_y_xz_net = GaussianMLP(2 * data_dim + latent_dim, hidden_dims, 1)
        self.p_c_x_net = MLP(2 * data_dim, hidden_dims, n_components)
        self.p_z_cx_net = GaussianMixtureMLP(2 * data_dim, hidden_dims, latent_dim, n_components)

        if checkpoint_fpath:
            self.load_state_dict(torch.load(checkpoint_fpath)["state_dict"])
        if task == "posterior_kl":
            self.freeze()
            self.q_z_x_net.requires_grad_(True)
            self.test_mu_x, self.test_var_x = [], []
        elif task == "log_marginal_likelihood":
            self.test_mu_x, self.test_var_x = torch.load(posterior_params_fpath)


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
        mu_z_xy, var_z_xy = self.q_z_xy_net(torch.hstack((x, y)))
        dist_z_xy = make_gaussian(mu_z_xy, var_z_xy)
        z = self.sample_z(mu_z_xy, var_z_xy)
        # E_q(c,z|x,y)[log p(y|x,z)]
        mu_y_xz, var_y_xz = self.p_y_xz_net(torch.hstack((x, z)))
        dist_y_xz = make_gaussian(mu_y_xz[:, None], var_y_xz[:, None])
        log_p_y_xz = dist_y_xz.log_prob(y).mean()
        # KL(q(z|x,y) || p(z))
        dist_c_x = D.Categorical(logits=self.p_c_x_net(x))
        mu_z_cx, var_z_cx = self.p_z_cx_net(x)
        dist_z_cx = D.Independent(D.Normal(mu_z_cx, var_z_cx.sqrt()), 1)
        dist_z_x = D.MixtureSameFamily(dist_c_x, dist_z_cx)
        log_q_z_xy = dist_z_xy.log_prob(z)
        kl = (log_q_z_xy - dist_z_x.log_prob(z)).mean()
        elbo = log_p_y_xz - kl
        return {
            "loss": -elbo,
            "kl": kl
        }


    def posterior_kl(self, x, y):
        mu_x, var_x = self.q_z_x_net(x)
        mu_xy, var_xy = self.q_z_xy_net(torch.hstack((x, y)))
        posterior_xy = make_gaussian(mu_xy, var_xy)
        posterior_x = make_gaussian(mu_x, var_x)
        return {
            "loss": D.kl_divergence(posterior_xy, posterior_x).mean(),
            "mu_x": mu_x.detach().cpu(),
            "var_x": var_x.detach().cpu()}


    def log_marginal_likelihood(self, x, y):
        x_rep = torch.repeat_interleave(x, repeats=self.n_samples, dim=0)
        mu_z_x, var_z_x = self.q_z_x_net(x)
        dist_z_x = make_gaussian(mu_z_x, var_z_x)
        z = dist_z_x.sample((self.n_samples,))
        mu_y_xz = self.p_y_xz_net(torch.hstack((x_rep, z)))
        dist_y_xz = make_gaussian(mu_y_xz, torch.ones_like(mu_y_xz))
        log_prob_y_xz = dist_y_xz.log_prob(y.squeeze())
        assert log_prob_y_xz.shape == torch.Size([self.n_samples])  # (n_samples,)

        n_test = len(self.test_mu_x)
        if self.n_posteriors < n_test:
            idxs = np.random.choice(n_test, self.n_posteriors, replace=False)
            test_mu_x = self.test_mu_x[idxs]
            test_var_x = self.test_var_x[idxs]
        else:
            test_mu_x = self.test_mu_x
            test_var_x = self.test_var_x
        agg_posterior = make_gaussian(test_mu_x, test_var_x)
        logp_adjust = log_avg_prob(agg_posterior.log_prob(z[:, None, :]).T)
        assert logp_adjust.shape == torch.Size([self.n_samples])  # (n_samples,)

        return {
            "conditional_lml": log_avg_prob(log_prob_y_xz),
            "interventional_lml": log_avg_prob(logp_adjust - dist_z_x.log_prob(z) + log_prob_y_xz)}


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
            self.test_var_x.append(out["var_x"])
        elif self.task == "log_marginal_likelihood":
            self.log("conditional_lml", out["conditional_lml"], on_step=False, on_epoch=True)
            self.log("interventional_lml", out["interventional_lml"], on_step=False, on_epoch=True)


    def test_epoch_end(self, outputs):
        if self.task == "posterior_kl":
            self.test_mu_x = torch.vstack(self.test_mu_x)
            self.test_var_x = torch.vstack(self.test_var_x)
            torch.save((self.test_mu_x, self.test_var_x), os.path.join(self.dpath, f"version_{self.seed}", "posterior_params.pt"))


    def configure_optimizers(self):
        if self.task == "posterior_kl":
            return Adam(self.q_z_x_net.parameters(), lr=self.lr)
        else:
            return Adam(self.parameters(), lr=self.lr)