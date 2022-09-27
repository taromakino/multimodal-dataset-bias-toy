import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.nn_utils import MLP, device
from utils.stats import gaussian_nll, make_gaussian, prior_kld
from torch.optim import Adam

class MixtureDensityNetwork(nn.Module):
    def __init__(self, data_dim, hidden_dims, latent_dim, n_components):
        super().__init__()
        self.prior = MLP(2 * data_dim, hidden_dims, n_components)
        self.encoders = nn.ModuleList()
        for _ in range(n_components):
            self.encoders.append(MLP(2 * data_dim, hidden_dims, [latent_dim] * 2))

    def forward(self, x0, x1, z):
        # Log-density
        gaussian_logp = []
        for encoder in self.encoders:
            mu, logvar = encoder(x0, x1)
            gaussian_dist = make_gaussian(mu, logvar)
            gaussian_logp.append(gaussian_dist.log_prob(z))
        gaussian_logp = torch.stack(gaussian_logp, dim=-1)
        prior_logp = F.log_softmax(self.prior(x0, x1), dim=-1)
        return torch.logsumexp(gaussian_logp + prior_logp, -1)

    def sample(self, x0, x1, n_samples):
        z = []
        for x0_elem, x1_elem in zip(x0, x1):
            z_batch = []
            component_counts = (F.softmax(self.prior(x0_elem, x1_elem), dim=0) * n_samples).round().int()
            for component_idx in range(self.n_components):
                mu, logvar = self.encoders[component_idx](x0_elem, x1_elem)
                dist = make_gaussian(mu[None], logvar[None])
                z_batch.append(dist.sample((component_counts[component_idx],)))
            z_batch = torch.cat(z_batch, dim=0) # (sample dim, latent_dim)
            z.append(z_batch[:n_samples]) # Adjust for rounding error
        return torch.stack(z, dim=1) # (sample dim, batch dim, latent dim)

class PosteriorX(pl.LightningModule):
    def __init__(self, data_dim, hidden_dims, latent_dim, lr, batch_size, n_components, n_samples):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.n_samples = n_samples
        self.posterior_xy = MLP(3 * data_dim, hidden_dims, [latent_dim] * 2)
        self.posterior_x = MixtureDensityNetwork(data_dim, hidden_dims, latent_dim, n_components)
        self.prior = make_gaussian(torch.zeros((batch_size, n_components), device=device()), torch.zeros((batch_size,
            n_components), device=device()))

    def loss(self, x0, x1, y):
        mu_xy, logvar_xy = self.posterior_xy(x0, x1, y)
        posterior_xy_dist = make_gaussian(mu_xy.clone().detach(), logvar_xy.clone().detach())
        z = posterior_xy_dist.sample((self.n_samples,))
        posterior_xy_logp = posterior_xy_dist.log_prob(z)
        posterior_x_logp = self.posterior_x(x0, x1, z)
        prior_logp = self.prior.log_prob(z)
        loss = (posterior_xy_logp - posterior_x_logp).mean()
        kld = (posterior_x_logp - prior_logp).mean()
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
    def __init__(self, data_dim, hidden_dims, latent_dim, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.encoder = MLP(3 * data_dim, hidden_dims, [latent_dim] * 2)
        self.decoder_mu = MLP(latent_dim + 2 * data_dim, hidden_dims, data_dim)
        self.decoder_logvar = MLP(latent_dim + 2 * data_dim, hidden_dims, data_dim)

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