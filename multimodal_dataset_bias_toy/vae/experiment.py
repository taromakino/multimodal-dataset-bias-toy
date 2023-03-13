import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from data import make_data
from model import IdentifiableVAE, VanillaVAE
from utils.file import save_file
from utils.nn_utils import make_trainer


def main(config):
    seed = config.__dict__.pop("seed")
    save_file(config, os.path.join(config.dpath, "args.pkl"))
    os.makedirs(config.dpath, exist_ok=True)
    pl.seed_everything(seed)
    data_train, data_val = make_data(seed, config.input_dim, config.n_trainval, config.train_ratio, config.u_sd,
        config.x_sd, config.y_sd, True, False, config.batch_size, config.n_workers)
    if config.is_vanilla:
        model = VanillaVAE(config.input_dim, config.y_sd, config.hidden_dims, config.latent_dim, config.n_samples,
            config.lr)
    else:
        model = IdentifiableVAE(config.input_dim, config.y_sd, config.hidden_dims, config.latent_dim, config.n_components,
            config.n_samples, config.lr)

    trainer = make_trainer(config.dpath, seed, config.n_accumulate, config.n_epochs, config.n_gpus)
    trainer.fit(model, data_train, data_val)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input_dim", type=int, default=16)
    parser.add_argument("--n_trainval", type=int, default=10000)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--u_sd", type=float, default=1)
    parser.add_argument("--x_sd", type=float, default=1)
    parser.add_argument("--y_sd", type=float, default=1)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128, 128])
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--n_components", type=int, default=16)
    parser.add_argument("--n_samples", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_accumulate", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--n_workers", type=int, default=20)
    parser.add_argument("--is_vanilla", action="store_true")
    main(parser.parse_args())