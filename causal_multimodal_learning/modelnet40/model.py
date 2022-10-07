import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.nn_utils import MLP, device, load_model
from utils.stats import make_gaussian, prior_kld
from torch.optim import AdamW
from torchvision.models import resnet18, ResNet18_Weights

N_CLASSES = 40

def make_resnet_embedder():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    embed_dim = model.fc.in_features
    model.fc = nn.Identity(embed_dim)
    return model, embed_dim

class GaussianNetwork(nn.Module):
    def __init__(self, hidden_dims, output_dim, is_y_input):
        super().__init__()
        self.is_y_input = is_y_input
        self.img_embedder0, embed_dim0 = make_resnet_embedder()
        self.img_embedder1, embed_dim1 = make_resnet_embedder()
        in_dim = embed_dim0 + embed_dim1 + (40 if is_y_input else 0)
        self.mu_net = MLP(in_dim, hidden_dims, output_dim)
        self.logvar_net = MLP(in_dim, hidden_dims, output_dim)

    def forward(self, *args):
        if self.is_y_input:
            x0, x1, y = args
            x0_embed = self.img_embedder0(x0)
            x1_embed = self.img_embedder1(x1)
            mu = self.mu_net(x0_embed, x1_embed, y)
            logvar = self.logvar_net(x0_embed, x1_embed, y)
        else:
            x0, x1 = args
            x0_embed = self.img_embedder0(x0)
            x1_embed = self.img_embedder1(x1)
            mu = self.mu_net(x0_embed, x1_embed)
            logvar = self.logvar_net(x0_embed, x1_embed)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, hidden_dims, latent_dim):
        super().__init__()
        self.img_embedder0, embed_dim0 = make_resnet_embedder()
        self.img_embedder1, embed_dim1 = make_resnet_embedder()
        in_dim = embed_dim0 + embed_dim1 + latent_dim
        self.y_net = MLP(in_dim, hidden_dims, N_CLASSES)

    def forward(self, x0, x1, z):
        x0_embed = self.img_embedder0(x0)
        x1_embed = self.img_embedder1(x1)
        return self.y_net(x0_embed, x1_embed, z)

class PosteriorX(pl.LightningModule):
    def __init__(self, hidden_dims, latent_dim, lr, wd, batch_size):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.wd = wd
        self.encoder_xy = GaussianNetwork(hidden_dims, latent_dim, N_CLASSES)
        self.encoder_x = GaussianNetwork(hidden_dims, latent_dim, 0)
        self.prior = make_gaussian(torch.zeros((batch_size, latent_dim), device=device()), torch.zeros((batch_size,
            latent_dim), device=device()))

    def loss(self, x0, x1, y):
        mu_xy, logvar_xy = self.encoder_xy(x0, x1, F.one_hot(y, N_CLASSES))
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
    def __init__(self, hidden_dims, latent_dim, lr, wd):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.wd = wd
        self.encoder = GaussianNetwork(hidden_dims, latent_dim, N_CLASSES)
        self.decoder = Decoder(hidden_dims, latent_dim)

    def sample_z(self, mu, logvar):
        if self.training:
            sd = torch.exp(logvar / 2) # Same as sqrt(exp(logvar))
            eps = torch.randn_like(sd)
            return mu + eps * sd
        else:
            return mu

    def loss(self, x0, x1, y):
        mu_z, logvar_z = self.encoder(x0, x1, F.one_hot(y, N_CLASSES))
        kld_loss = prior_kld(mu_z, logvar_z)
        z = self.sample_z(mu_z, logvar_z)
        y_reconst = self.decoder(x0, x1, z)
        reconst_loss = F.cross_entropy(y_reconst, y, reduction="none")
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
        y_rep = torch.repeat_interleave(y, repeats=self.n_samples, dim=0)
        z = posterior_x_dist.sample((self.n_samples,))
        y_logp = -F.cross_entropy(self.vae.decoder(x0_rep, x1_rep, z), y_rep, reduction="none")
        conditional_logp = -torch.log(torch.tensor(self.n_samples)) + torch.logsumexp(y_logp, 0).item()
        interventional_logp = -torch.log(torch.tensor(self.n_samples)) + torch.logsumexp(
            self.prior.log_prob(z) - posterior_x_dist.log_prob(z) + y_logp, 0).item()
        self.log("conditional_logp", conditional_logp, on_step=True, on_epoch=True)
        self.log("interventional_logp", interventional_logp, on_step=True, on_epoch=True)