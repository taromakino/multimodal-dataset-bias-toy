import pytorch_lightning as pl
import torch
import torch.nn as nn
from utils.framework import gaussian_kld, gaussian_nll, posterior_kld, swish
from torch.optim import Adam

class InputEncoder(nn.Module):
    def __init__(self, data_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(2 * data_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, latent_dim)
        self.fc32 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x0, x1):
        h = swish(self.fc1(torch.hstack((x0, x1))))
        h = swish(self.fc2(h))
        return self.fc31(h), self.fc32(h)

class InputTargetEncoder(nn.Module):
    def __init__(self, data_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(3 * data_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, latent_dim)
        self.fc32 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x0, x1, y):
        h = swish(self.fc1(torch.hstack((x0, x1, y))))
        h = swish(self.fc2(h))
        return self.fc31(h), self.fc32(h)

class Decoder(nn.Module):
    def __init__(self, data_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + 2 * data_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, data_dim)

    def forward(self, x0, x1, z):
        h = swish(self.fc1(torch.hstack((x0, x1, z))))
        h = swish(self.fc2(h))
        return self.fc3(h)

class SemiSupervisedVae(pl.LightningModule):
    def __init__(self, lr, data_dim, hidden_dim, latent_dim, alpha):
        super().__init__()
        self.lr = lr
        self.alpha = alpha
        self.input_target_encoder = InputTargetEncoder(data_dim, hidden_dim, latent_dim)
        self.input_encoder = InputEncoder(data_dim, hidden_dim, latent_dim)
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
        # Vanilla VAE loss
        mu_input_target, logvar_input_target = self.input_target_encoder(x0, x1, y)
        z = self.sample_z(mu_input_target, logvar_input_target)
        y_mu = self.decoder_mu(x0, x1, z)
        y_logvar = self.decoder_logvar(x0, x1, z)
        reconst_loss = gaussian_nll(y, y_mu, y_logvar)
        posterior_kld_loss = posterior_kld(mu_input_target, logvar_input_target)
        # KL(q(z | x, x', y) || q(z | x, x')), don't backprop through q(z | x, x', y)
        mu_input, logvar_input = self.input_encoder(x0, x1)
        gaussian_kld_loss = gaussian_kld(mu_input_target.clone().detach(), logvar_input_target.clone().detach(),
            mu_input, logvar_input)
        return reconst_loss, posterior_kld_loss, gaussian_kld_loss

    def training_step(self, batch, batch_idx):
        reconst_loss, posterior_kld_loss, gaussian_kld_loss = self.loss(*batch)
        return (reconst_loss + posterior_kld_loss + self.alpha * gaussian_kld_loss).mean()

    def validation_step(self, batch, batch_idx):
        reconst_loss, posterior_kld_loss, gaussian_kld_loss = self.loss(*batch)
        self.log("val_posterior_kld_loss", posterior_kld_loss.mean(), on_step=False, on_epoch=True)
        self.log("val_gaussian_kld_loss", gaussian_kld_loss.mean(), on_step=False, on_epoch=True)
        self.log("val_loss", (reconst_loss + posterior_kld_loss).mean(), on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        reconst_loss, posterior_kld_loss, gaussian_kld_loss = self.loss(*batch)
        self.log("test_posterior_kld_loss", posterior_kld_loss.mean(), on_step=False, on_epoch=True)
        self.log("test_gaussian_kld_loss", gaussian_kld_loss.mean(), on_step=False, on_epoch=True)
        self.log("test_loss", (reconst_loss + posterior_kld_loss).mean(), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), self.lr)