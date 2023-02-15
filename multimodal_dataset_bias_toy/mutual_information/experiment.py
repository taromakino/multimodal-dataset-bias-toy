import os
import pandas as pd
import pytorch_lightning as pl
from argparse import ArgumentParser
from data import make_data
from model import Mine
from utils.file import save_file
from utils.nn_utils import make_trainer


def train_mi(fpath):
    df = pd.read_csv(fpath)
    return -df.train_loss.min()


def main(config):
    seed = config.__dict__.pop("seed")
    save_file(config, os.path.join(config.dpath, "args.pkl"))
    os.makedirs(config.dpath, exist_ok=True)
    pl.seed_everything(seed)
    data_train, data_val, data_test = make_data(seed, config.data_dim, config.n_train, config.n_val, config.n_test,
        config.u_sd, config.x_sd, config.y_sd, config.s_shift, config.batch_size, True, config.n_workers)
    model_uxy = Mine(seed, config.data_dim, config.hidden_dims, True, config.lr)
    trainer_uxy = make_trainer(os.path.join(config.dpath, "uxy"), seed, config.n_epochs)
    trainer_uxy.fit(model_uxy, data_train)
    model_ux = Mine(seed, config.data_dim, config.hidden_dims, False, config.lr)
    trainer_ux = make_trainer(os.path.join(config.dpath, "ux"), seed, config.n_epochs)
    trainer_ux.fit(model_ux, data_train)
    mi_uxy = train_mi(os.path.join(config.dpath, "uxy", f"version_{seed}", "metrics.csv"))
    mi_ux = train_mi(os.path.join(config.dpath, "ux", f"version_{seed}", "metrics.csv"))
    print(mi_uxy - mi_ux)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_dim", type=int, default=1)
    parser.add_argument("--n_train", type=int, default=5000)
    parser.add_argument("--n_val", type=int, default=1000)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--u_sd", type=float, default=1)
    parser.add_argument("--x_sd", type=float, default=1)
    parser.add_argument("--y_sd", type=float, default=1)
    parser.add_argument("--s_shift", type=float, default=None)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128, 128])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--n_workers", type=int, default=20)
    main(parser.parse_args())