import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils.nn_utils import MLP
from utils.stats import diag_gaussian_log_prob


class GaussianMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_class):
        super().__init__()
        self.mu_net = MLP(input_dim, hidden_dims, output_dim, activation_class)
        self.logvar_net = MLP(input_dim, hidden_dims, output_dim, activation_class)

    def forward(self, *args):
        return self.mu_net(*args), F.softplus(self.logvar_net(*args))


class IdentifiableVAE(pl.LightningModule):
    def __init__(self, input_dim, y_sd, hidden_dims, latent_dim, n_components, n_samples, lr):
        super().__init__()
        self.save_hyperparameters()
        self.y_sd = y_sd
        self.n_samples = n_samples
        self.lr = lr
        self.q_z_xy_net = GaussianMLP(2 * input_dim + 1, hidden_dims, latent_dim, nn.ReLU)
        self.p_y_xz_net = MLP(2 * input_dim + latent_dim, hidden_dims, 1, nn.ReLU)
        self.logits_c = nn.Parameter(torch.ones(n_components))
        self.mu_z_c = nn.Parameter(torch.zeros(n_components, latent_dim))
        self.logvar_z_c = nn.Parameter(torch.zeros(n_components, latent_dim))
        nn.init.xavier_normal_(self.mu_z_c)
        nn.init.xavier_normal_(self.logvar_z_c)


    def sample_z(self, mu, var):
        sd = var.sqrt()
        eps = torch.randn(self.n_samples, *sd.shape).to(self.device)
        return mu + eps * sd


    def loss(self, x, y):
        batch_size = len(x) # For assertions
        # z ~ q(z|x,y)
        mu_z_xy, var_z_xy = self.q_z_xy_net(x, y)
        z = self.sample_z(mu_z_xy, var_z_xy)
        log_q_z_xy = diag_gaussian_log_prob(z, mu_z_xy, var_z_xy, self.device).view(-1)
        assert log_q_z_xy.shape == (self.n_samples * batch_size,)
        # E_q(z|x,y)[log p(y|x,z)]
        x = torch.repeat_interleave(x[None], repeats=self.n_samples, dim=0)
        y = torch.repeat_interleave(y[None], repeats=self.n_samples, dim=0)
        x, y, z = x.view(-1, x.shape[-1]), y.view(-1, y.shape[-1]), z.view(-1, z.shape[-1])
        mu_y_xz = self.p_y_xz_net(x, z)[:, None]
        var_y_xz = self.y_sd ** 2 * torch.ones_like(mu_y_xz)
        log_p_y_xz = diag_gaussian_log_prob(y, mu_y_xz, var_y_xz, self.device)
        assert log_p_y_xz.shape == (self.n_samples * batch_size,)
        # KL(q(z|x,y) || p(z))
        dist_c = D.Categorical(logits=self.logits_c)
        var_z_c = F.softplus(self.logvar_z_c)
        dist_z_c = D.Independent(D.Normal(self.mu_z_c, var_z_c.sqrt()), 1)
        dist_z = D.MixtureSameFamily(dist_c, dist_z_c)
        log_p_z = dist_z.log_prob(z)
        assert log_p_z.shape == (self.n_samples * batch_size,)
        kl = (log_q_z_xy - log_p_z).mean()
        elbo = log_p_y_xz.mean() - kl
        return {
            "loss": -elbo,
            "kl": kl
        }


    def training_step(self, batch, batch_idx):
        out = self.loss(*batch)
        self.log("train_loss", out["loss"], on_step=False, on_epoch=True)
        self.log("train_kl", out["kl"], on_step=False, on_epoch=True)
        return out["loss"]


    def validation_step(self, batch, batch_idx):
        out = self.loss(*batch)
        self.log("val_loss", out["loss"], on_step=False, on_epoch=True)
        self.log("val_kl", out["kl"], on_step=False, on_epoch=True)


    def test_step(self, batch, batch_idx):
        out = self.loss(*batch)
        self.log("test_loss", out["loss"], on_step=False, on_epoch=True)


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


class VanillaVAE(pl.LightningModule):
    def __init__(self, input_dim, y_sd, hidden_dims, latent_dim, n_samples, lr):
        super().__init__()
        self.save_hyperparameters()
        self.y_sd = y_sd
        self.n_samples = n_samples
        self.lr = lr
        self.q_z_xy_net = GaussianMLP(2 * input_dim + 1, hidden_dims, latent_dim, nn.Sigmoid)
        self.p_y_xz_net = MLP(2 * input_dim + latent_dim, hidden_dims, 1, nn.Sigmoid)


    def sample_z(self, mu, var):
        sd = var.sqrt()
        eps = torch.randn(self.n_samples, *sd.shape).to(self.device)
        return mu + eps * sd


    def loss(self, x, y):
        batch_size = len(x)  # For assertions
        # z ~ q(z|x,y)
        mu_z_xy, var_z_xy = self.q_z_xy_net(x, y)
        z = self.sample_z(mu_z_xy, var_z_xy)
        log_q_z_xy = diag_gaussian_log_prob(z, mu_z_xy, var_z_xy, self.device).view(-1)
        assert log_q_z_xy.shape == (self.n_samples * batch_size,)
        # E_q(z|x,y)[log p(y|x,z)]
        x = torch.repeat_interleave(x[None], repeats=self.n_samples, dim=0)
        y = torch.repeat_interleave(y[None], repeats=self.n_samples, dim=0)
        x, y, z = x.view(-1, x.shape[-1]), y.view(-1, y.shape[-1]), z.view(-1, z.shape[-1])
        mu_y_xz = self.p_y_xz_net(x, z)[:, None]
        var_y_xz = self.y_sd ** 2 * torch.ones_like(mu_y_xz)
        log_p_y_xz = diag_gaussian_log_prob(y, mu_y_xz, var_y_xz, self.device)
        assert log_p_y_xz.shape == (self.n_samples * batch_size,)
        # KL(q(z|x,y) || p(z))
        kl = (-0.5 * torch.sum(1 + var_z_xy.log() - mu_z_xy.pow(2) - var_z_xy, dim=1)).mean()
        elbo = log_p_y_xz.mean() - kl
        return {
            "loss": -elbo,
            "kl": kl
        }


    def training_step(self, batch, batch_idx):
        out = self.loss(*batch)
        self.log("train_loss", out["loss"], on_step=False, on_epoch=True)
        self.log("train_kl", out["kl"], on_step=False, on_epoch=True)
        return out["loss"]


    def validation_step(self, batch, batch_idx):
        out = self.loss(*batch)
        self.log("val_loss", out["loss"], on_step=False, on_epoch=True)
        self.log("val_kl", out["kl"], on_step=False, on_epoch=True)


    def test_step(self, batch, batch_idx):
        out = self.loss(*batch)
        self.log("test_loss", out["loss"], on_step=False, on_epoch=True)


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)