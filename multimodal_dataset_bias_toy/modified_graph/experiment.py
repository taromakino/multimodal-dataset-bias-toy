import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from data import make_data
from model import Model
from pytorch_lightning.loggers import CSVLogger
from utils.file import save_file


def make_trainer(dpath, seed, n_epochs, n_gpus):
    return pl.Trainer(
        logger=CSVLogger(dpath, name="", version=seed),
        max_epochs=n_epochs,
        gpus=n_gpus)


def main(config):
    seed = config.__dict__.pop("seed")
    save_file(config, os.path.join(config.dpath, "args.pkl"))
    os.makedirs(config.dpath, exist_ok=True)
    pl.seed_everything(seed)
    data_train, data_val, data_test = make_data(seed, config.data_dim, config.n_examples, config.u_sd, config.x_sd,
        config.y_sd, config.s_shift, False, True, config.batch_size, config.n_workers)
    model = Model(seed, config.dpath, config.y_sd, config.lr)
    trainer = make_trainer(config.dpath, seed, config.n_epochs, config.n_gpus)
    trainer.fit(model, data_train)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_dim", type=int, default=1)
    parser.add_argument("--n_examples", nargs="+", type=int, default=[10000, 1000, 1000])
    parser.add_argument("--u_sd", type=float, default=0.1)
    parser.add_argument("--x_sd", type=float, default=0.1)
    parser.add_argument("--y_sd", type=float, default=0.1)
    parser.add_argument("--s_shift", type=float, default=None)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--n_workers", type=int, default=20)
    parser.add_argument("--is_test", action="store_true")
    main(parser.parse_args())