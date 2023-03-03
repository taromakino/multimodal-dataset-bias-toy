import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils.nn_utils import MLP
from utils.stats import diag_gaussian_log_prob


class GaussianMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.mu_net = MLP(input_dim, hidden_dims, output_dim)
        self.logvar_net = MLP(input_dim, hidden_dims, output_dim)

    def forward(self, *args):
        return self.mu_net(*args), F.softplus(self.logvar_net(*args))


class Model(pl.LightningModule):
    def __init__(self, seed, dpath, input_dim, y_sd, hidden_dims, latent_dim, n_components, n_samples, lr):
        super().__init__()
        self.save_hyperparameters()
        self.seed = seed
        self.dpath = dpath
        self.y_sd = y_sd
        self.n_samples = n_samples
        self.lr = lr
        self.q_z_xy_net = GaussianMLP(3 * input_dim, hidden_dims, latent_dim)
        self.q_z_x_net = GaussianMLP(2 * input_dim, hidden_dims, latent_dim)
        self.p_y_xz_net = MLP(2 * input_dim + latent_dim, hidden_dims, input_dim)
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
        # z ~ q(z|x,y)
        mu_z_xy, var_z_xy = self.q_z_xy_net(x, y)
        mu_z_xy, var_z_xy = mu_z_xy, var_z_xy
        if len(mu_z_xy.shape) == 1:
            mu_z_xy, var_z_xy = mu_z_xy[:, None], var_z_xy[:, None]
        z = self.sample_z(mu_z_xy, var_z_xy)
        log_q_z_xy = diag_gaussian_log_prob(z, mu_z_xy, var_z_xy, self.device).view(-1)
        # E_q(z|x,y)[log p(y|x,z)]
        x = torch.repeat_interleave(x[None], repeats=self.n_samples, dim=0)
        y = torch.repeat_interleave(y[None], repeats=self.n_samples, dim=0)
        x, y, z = x.view(-1, x.shape[-1]), y.view(-1, y.shape[-1]), z.view(-1, z.shape[-1])
        mu_y_xz = self.p_y_xz_net(x, z)
        var_y_xz = self.y_sd ** 2 * torch.ones_like(mu_y_xz)
        log_p_y_xz = diag_gaussian_log_prob(y, mu_y_xz, var_y_xz, self.device).mean()
        # KL(q(z|x,y) || p(z))
        dist_c = D.Categorical(logits=self.logits_c)
        var_z_c = F.softplus(self.logvar_z_c)
        dist_z_c = D.Independent(D.Normal(self.mu_z_c, var_z_c.sqrt()), 1)
        dist_z = D.MixtureSameFamily(dist_c, dist_z_c)
        kl = (log_q_z_xy - dist_z.log_prob(z)).mean()
        elbo = log_p_y_xz - kl
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