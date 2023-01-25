import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        module_list = []
        last_in_dim = input_dim
        for hidden_dim in hidden_dims:
            module_list.append(nn.Linear(last_in_dim, hidden_dim))
            module_list.append(nn.ReLU())
            last_in_dim = hidden_dim
        module_list.append(nn.Linear(last_in_dim, output_dim))
        self.module_list = nn.Sequential(*module_list)

    def forward(self, *args):
        return self.module_list(torch.hstack(args))

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(*args):
    return [arg.to(device()) for arg in args]

def make_trainer(dpath, seed, patience):
    return pl.Trainer(
        logger=CSVLogger(dpath, name="", version=seed),
        callbacks=[
            ModelCheckpoint(monitor="val_loss", filename="best"),
            EarlyStopping(monitor="val_loss", patience=patience)])