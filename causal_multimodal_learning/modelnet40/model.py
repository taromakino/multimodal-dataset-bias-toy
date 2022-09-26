import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
from utils.stats import gaussian_kld, prior_kld
from torch import nn
from torch.optim import Adam

def make_resnet_encoder(): # Replace with pl version
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Identity(model.fc.out_features)
    return model

class InputEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder_x0 = make_resnet_encoder()
        self.encoder_x1 = make_resnet_encoder()
        self.fc_mu = nn.Linear(2 * 2048, latent_dim)
        self.fc_var = nn.Linear(2 * 2048, latent_dim)

    def forward(self, x0, x1):
        x0 = self.encoder_x0(x0)
        x1 = self.encoder_x1(x1)
        merged = torch.cat((x0, x1), dim=1)
        return self.fc_mu(merged), self.fc_var(merged)

class InputTargetEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder_x0 = make_resnet_encoder()
        self.encoder_x1 = make_resnet_encoder()
        self.fc_mu = nn.Linear(2 * 2048 + 1, latent_dim)
        self.fc_var = nn.Linear(2 * 2048 + 1, latent_dim)

    def forward(self, x0, x1, y):
        x0 = self.encoder_x0(x0)
        x1 = self.encoder_x1(x1)
        merged = torch.cat((x0, x1, y[:, None]), dim=1)
        return self.fc_mu(merged), self.fc_var(merged)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder_x0 = make_resnet_encoder()
        self.encoder_x1 = make_resnet_encoder()
        self.fc_y = nn.Linear(2 * 2048 + latent_dim, 40)

    def forward(self, x0, x1, z):
        x0 = self.encoder_x0(x0)
        x1 = self.encoder_x1(x1)
        return self.fc_y(torch.cat((x0, x1, z), dim=1))

class SemiSupervisedVae(pl.LightningModule):
    def __init__(self, lr, latent_dim, alpha):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.alpha = alpha
        self.input_target_encoder = InputTargetEncoder(latent_dim)
        self.input_encoder = InputEncoder(latent_dim)
        self.decoder = Decoder(latent_dim)

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
        y_reconst = self.decoder(x0, x1, z)
        reconst_loss = F.cross_entropy(y_reconst, y, reduction="none")
        posterior_kld_loss = prior_kld(mu_input_target, logvar_input_target)
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
        self.log("test_loss", (reconst_loss + posterior_kld_loss).mean(), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), self.lr)