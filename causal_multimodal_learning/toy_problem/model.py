import pytorch_lightning as pl
import torch
import torch.nn as nn
from utils.framework import gaussian_nll, posterior_kld, swish
from torch.optim import Adam

class ScalarEncoder(nn.Module):
    def __init__(self, data_dim, hidden_dim, latent_dim):
        super(ScalarEncoder, self).__init__()
        self.fc1 = nn.Linear(3 * data_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, latent_dim)
        self.fc32 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = swish(self.fc1(x))
        h = swish(self.fc2(h))
        return self.fc31(h), self.fc32(h)

class ScalarDecoder(nn.Module):
    def __init__(self, data_dim, hidden_dim, latent_dim):
        super(ScalarDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + 2 * data_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, data_dim)

    def forward(self, x):
        h = swish(self.fc1(x))
        h = swish(self.fc2(h))
        return self.fc3(h)

class SemiSupervisedVae(pl.LightningModule):
    def __init__(self, lr, data_dim, hidden_dim, latent_dim):
        super(SemiSupervisedVae, self).__init__()
        self.lr = lr
        self.encoder = ScalarEncoder(data_dim, hidden_dim, latent_dim)
        self.decoder_mu = ScalarDecoder(data_dim, hidden_dim, latent_dim)
        self.decoder_logvar = ScalarDecoder(data_dim, hidden_dim, latent_dim)

    def sample_z(self, mu, logvar):
        if self.training:
            sd = torch.exp(logvar / 2) # Same as sqrt(exp(logvar))
            eps = torch.randn_like(sd)
            return mu + eps * sd
        else:
            return mu

    def loss(self, x0, x1, y):
        z_mu, z_logvar = self.encoder(torch.hstack((x0, x1, y)))
        z = self.sample_z(z_mu, z_logvar)
        y_mu = self.decoder_mu(torch.hstack((z, x0, x1)))
        y_logvar = self.decoder_logvar(torch.hstack((z, x0, x1)))
        reconst_loss = gaussian_nll(y, y_mu, y_logvar)
        kld_loss = posterior_kld(z_mu, z_logvar)
        return (reconst_loss + kld_loss).mean()

    def training_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), self.lr)