import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from data import make_data
from model import VAE
from utils.file import save_file
from utils.nn_utils import make_trainer


def main(config):
    seed = config.__dict__.pop("seed")
    save_file(config, os.path.join(config.dpath, "args.pkl"))
    os.makedirs(config.dpath, exist_ok=True)
    pl.seed_everything(seed)
    data_train, data_val, data_test = make_data(seed, config.input_dim, config.sample_size, config.split_ratios,
        config.u_sd, config.x_sd, config.y_sd, True, False, config.batch_size, config.n_workers)
    model = VAE(config.input_dim, config.y_sd, config.hidden_dims, config.latent_dim, config.n_components,
        config.n_samples, config.lr)
    trainer = make_trainer(config.dpath, seed, config.n_epochs, config.n_early_stop, config.n_gpus)
    trainer.fit(model, data_train, data_val)
    trainer.test(model, data_test, ckpt_path="best")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input_dim", type=int, default=16)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--split_ratios", nargs="+", type=float, default=[0.6, 0.2, 0.2])
    parser.add_argument("--u_sd", type=float, default=1)
    parser.add_argument("--x_sd", type=float, default=1)
    parser.add_argument("--y_sd", type=float, default=1)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128, 128])
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--n_components", type=int, default=16)
    parser.add_argument("--n_samples", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--n_early_stop", type=int, default=50)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--n_workers", type=int, default=20)
    args = parser.parse_args()
    assert sum(args.split_ratios) == 1
    main(args)