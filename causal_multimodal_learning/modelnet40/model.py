import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
from utils.framework import posterior_kld
from torch import nn
from torch.optim import Adam

def make_resnet_encoder():
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Identity(model.fc.out_features)
    return model

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.x0_encoder = make_resnet_encoder()
        self.x1_encoder = make_resnet_encoder()
        self.fc_mu = nn.Linear(2 * 512 + 1, latent_dim)
        self.fc_var = nn.Linear(2 * 512 + 1, latent_dim)

    def forward(self, x0, x1, y):
        x0 = self.x0_encoder(x0)
        x1 = self.x1_encoder(x1)
        merged = torch.cat((x0, x1, y[:, None]), dim=1)
        return self.fc_mu(merged), self.fc_var(merged)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.x0_encoder = make_resnet_encoder()
        self.x1_encoder = make_resnet_encoder()
        self.fc_y = nn.Linear(2 * 512 + latent_dim, 40)

    def forward(self, x0, x1, z):
        x0 = self.x0_encoder(x0)
        x1 = self.x1_encoder(x1)
        return self.fc_y(torch.cat((x0, x1, z), dim=1))

class SemiSupervisedVae(pl.LightningModule):
    def __init__(self, lr, latent_dim):
        super().__init__()
        self.lr = lr
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def sample_z(self, mu, logvar):
        if self.training:
            sd = torch.exp(logvar / 2) # Same as sqrt(exp(logvar))
            eps = torch.randn_like(sd)
            return mu + eps * sd
        else:
            return mu

    def loss(self, x0, x1, y):
        mu, logvar = self.encoder(x0, x1, y)
        z = self.sample_z(mu, logvar)
        y_reconst = self.decoder(x0, x1, z)
        reconst_loss = F.cross_entropy(y_reconst, y, reduction="none")
        kld_loss = posterior_kld(mu, logvar)
        return (reconst_loss + kld_loss).mean()

    def training_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), self.lr)