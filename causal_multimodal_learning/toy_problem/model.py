import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.framework import gaussian_nll, make_gaussian, prior_kld
from torch.optim import Adam

class EncoderX(nn.Module):
    def __init__(self, data_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(2 * data_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, latent_dim)
        self.fc32 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x0, x1):
        out = F.silu(self.fc1(torch.hstack((x0, x1))))
        out = F.silu(self.fc2(out))
        return self.fc31(out), self.fc32(out)

class EncoderXy(nn.Module):
    def __init__(self, data_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(3 * data_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, latent_dim)
        self.fc32 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x0, x1, y):
        out = F.silu(self.fc1(torch.hstack((x0, x1, y))))
        out = F.silu(self.fc2(out))
        return self.fc31(out), self.fc32(out)

class Decoder(nn.Module):
    def __init__(self, data_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + 2 * data_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, data_dim)

    def forward(self, x0, x1, z):
        out = F.silu(self.fc1(torch.hstack((x0, x1, z))))
        out = F.silu(self.fc2(out))
        return self.fc3(out)

class PosteriorX(pl.LightningModule):
    def __init__(self, lr, posterior_xy, data_dim, hidden_dim, latent_dim):
        super().__init__()
        self.save_hyperparameters(ignore=["posterior_xy"])
        self.lr = lr
        self.posterior_xy = posterior_xy
        self.posterior_x = EncoderX(data_dim, hidden_dim, latent_dim)

    def set_posterior_xy(self, posterior_xy):
        self.posterior_xy = posterior_xy

    def loss(self, x0, x1, y):
        mu_xy, logvar_xy = self.posterior_xy(x0, x1, y)
        posterior_xy_dist = make_gaussian(mu_xy.clone().detach(), logvar_xy.clone().detach())
        mu_x, logvar_x = self.posterior_x(x0, x1)
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
        return Adam(self.posterior_x.parameters(), lr=self.lr)

class SemiSupervisedVae(pl.LightningModule):
    def __init__(self, lr, data_dim, hidden_dim, latent_dim):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.encoder = EncoderXy(data_dim, hidden_dim, latent_dim)
        self.decoder_mu = Decoder(data_dim, hidden_dim, latent_dim)
        self.decoder_logvar = Decoder(data_dim, hidden_dim, latent_dim)

    def sample_z(self, mu, logvar):
        if self.training:
            sd = torch.exp(logvar / 2) # Same as sqrt(exp(logvar))
            eps = torch.randn_like(sd)
            return mu + eps * sd
        else:
            return mu

    def loss(self, x0, x1, y):
        mu, logvar = self.encoder(x0, x1, y)
        kld_loss = prior_kld(mu, logvar)
        z = self.sample_z(mu, logvar)
        y_mu = self.decoder_mu(x0, x1, z)
        y_logvar = self.decoder_logvar(x0, x1, z)
        reconst_loss = gaussian_nll(y, y_mu, y_logvar)
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
        return Adam(self.parameters(), lr=self.lr)