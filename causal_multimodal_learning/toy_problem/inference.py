import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from toy_problem.data import make_data
from toy_problem.model import InferenceNetwork
from utils.file import load_file
from utils.nn_utils import make_tester

def main(args):
    pl.seed_everything(args.seed)
    hparams = load_file(os.path.join(args.dpath, "args.pkl"))
    _, _, data_test = make_data(args.seed, hparams.n_examples, hparams.data_dim, args.u_mult, hparams.trainval_ratios,
        1, args.n_workers)
    net = InferenceNetwork(args.dpath, args.seed, args.n_samples, hparams.latent_dim)
    tester = make_tester(os.path.join(args.dpath, "inference", f"u={args.u_mult}"), args.seed)
    tester.test(net, data_test)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--u_mult", type=float, default=0)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_workers", type=int, default=20)
    main(parser.parse_args())