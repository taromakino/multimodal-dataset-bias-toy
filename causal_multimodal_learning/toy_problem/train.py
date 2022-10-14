import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from utils.file import save_file
from utils.nn_utils import make_trainer
from toy_problem.data import make_data
from toy_problem.model import Model

def main(args):
    seed = args.__dict__.pop("seed")
    save_file(args, os.path.join(args.dpath, "args.pkl"))
    pl.seed_everything(seed)
    data_train, data_val, data_test = make_data(seed, args.n_examples, args.data_dim, args.u_mult, args.batch_size, args.n_workers)
    model = Model(args.data_dim, args.hidden_dims, args.latent_dim, args.beta, args.n_samples, args.lr, args.wd)
    trainer = make_trainer(args.dpath, seed, args.n_epochs, args.patience)
    trainer.fit(model, data_train, data_val)
    trainer.test(model, data_test, ckpt_path="best")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_examples", nargs="+", type=int, default=[10000, 1000, 1000])
    parser.add_argument("--data_dim", type=int, default=1)
    parser.add_argument("--u_mult", type=float, default=0)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[16, 16])
    parser.add_argument("--latent_dim", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--n_epochs", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_workers", type=int, default=20)
    main(parser.parse_args())