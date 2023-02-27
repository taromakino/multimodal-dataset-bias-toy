import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from data import make_data
from model import Model
from utils.file import save_file
from utils.nn_utils import make_trainer


def main(config):
    seed = config.__dict__.pop("seed")
    dpath = os.path.join(config.dpath, str(config.task))
    save_file(config, os.path.join(dpath, "args.pkl"))
    os.makedirs(dpath, exist_ok=True)
    pl.seed_everything(seed)
    test_batch_size = 1 if config.task == "log_marginal_likelihood" else config.batch_size
    data_train, data_val, data_test = make_data(seed, config.data_dim, config.n_train, config.n_val, config.n_test,
        config.u_sd, config.x_sd, config.y_sd, config.s_shift, config.batch_size, config.batch_size, test_batch_size,
        False, config.n_workers)
    model = Model(seed, dpath, config.task, config.data_dim, config.hidden_dims, config.latent_dim, config.n_components,
        config.lr, config.n_samples, config.n_posteriors, config.checkpoint_fpath, config.posterior_params_fpath)
    trainer = make_trainer(dpath, seed, config.n_epochs, config.n_gpus)
    if config.is_test:
        trainer.test(model, data_test)
    else:
        trainer.fit(model, data_train, data_val)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_fpath", type=str, default=None)
    parser.add_argument("--posterior_params_fpath", type=str, default=None)
    parser.add_argument("--data_dim", type=int, default=1)
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--n_val", type=int, default=10000)
    parser.add_argument("--n_test", type=int, default=10000)
    parser.add_argument("--u_sd", type=float, default=0.1)
    parser.add_argument("--x_sd", type=float, default=0.1)
    parser.add_argument("--y_sd", type=float, default=0.1)
    parser.add_argument("--s_shift", type=float, default=None)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128, 128])
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--n_components", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=512)
    parser.add_argument("--n_posteriors", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--n_workers", type=int, default=20)
    parser.add_argument("--is_test", action="store_true")
    main(parser.parse_args())