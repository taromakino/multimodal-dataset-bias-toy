import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dims):
        super().__init__()
        module_list = []
        last_in_dim = input_dim
        for hidden_dim in hidden_dims:
            module_list.append(nn.Linear(last_in_dim, hidden_dim))
            module_list.append(nn.ReLU())
            module_list.append(nn.Dropout())
            last_in_dim = hidden_dim
        self.module_list = nn.Sequential(*module_list)
        if isinstance(output_dims, list):
            self.output_layers = nn.ModuleList()
            for output_dim in output_dims:
                self.output_layers.append(nn.Linear(hidden_dims[-1], output_dim))
        else:
            self.output_layers = nn.Linear(hidden_dims[-1], output_dims)

    def forward(self, *args):
        x = torch.hstack((args)) if len(args) > 1 else args
        x = self.module_list(x)
        if isinstance(self.output_layers, nn.ModuleList):
            return [output_layer(x) for output_layer in self.output_layers]
        else:
            return self.output_layers(x)

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(*args):
    return [arg.to(device()) for arg in args]

def make_trainer(dpath, seed, n_epochs, patience):
    return pl.Trainer(
        logger=CSVLogger(dpath, name=None, version=seed),
        callbacks=[
            ModelCheckpoint(monitor="val_loss"),
            EarlyStopping(monitor="val_loss", patience=patience)],
        max_epochs=n_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu")

def make_tester(dpath, seed):
    return pl.Trainer(
        logger=CSVLogger(dpath, name=None, version=seed),
        max_epochs=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu")

def load_model(model_class, dpath):
    ckpt_fpath = os.path.join(dpath, os.listdir(dpath)[0])
    return model_class.load_from_checkpoint(ckpt_fpath)