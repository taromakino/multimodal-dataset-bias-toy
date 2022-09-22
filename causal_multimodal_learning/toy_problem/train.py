import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from utils.file import save_file
from utils.framework import make_trainer
from toy_problem.data import make_data
from toy_problem.model import SemiSupervisedVae

def main(args):
    save_file(args, os.path.join(args.dpath, f"version_{args.seed}", "args.pkl"))
    pl.seed_everything(args.seed)
    model = SemiSupervisedVae(args.lr, args.data_dim, args.hidden_dim, args.latent_dim, args.alpha)
    data_train, data_val, data_test = make_data(args.seed, args.n_examples, args.data_dim, args.u_mult,
        args.trainval_ratios, args.batch_size, args.n_workers)
    trainer = make_trainer(args.dpath, args.seed, args.n_epochs, args.patience)
    trainer.fit(model, data_train, data_val)
    trainer.test(model, data_test)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_examples", type=int, default=10000)
    parser.add_argument("--data_dim", type=int, default=1)
    parser.add_argument("--u_mult", type=float, default=1)
    parser.add_argument("--trainval_ratios", nargs="+", type=float, default=[0.8, 0.1])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--n_workers", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0)
    main(parser.parse_args())