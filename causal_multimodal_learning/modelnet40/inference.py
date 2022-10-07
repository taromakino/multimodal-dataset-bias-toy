import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from modelnet40.data import make_data
from modelnet40.model import InferenceNetwork
from utils.file import load_file
from utils.nn_utils import make_tester

def main(args):
    pl.seed_everything(args.seed)
    hparams = load_file(os.path.join(args.dpath, "args.pkl"))
    _, _, data_test = make_data(args.seed, 1, 1, args.n_workers)
    net = InferenceNetwork(args.dpath, args.seed, args.n_samples, args.n_samples_per_batch, hparams.latent_dim)
    tester = make_tester(os.path.join(args.dpath, "inference"), args.seed)
    tester.test(net, data_test)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_samples_per_batch", type=int, default=2500)
    parser.add_argument("--subset_ratio_range", nargs="+", type=float, default=["1", "0.75", "0.5", "0.25"])
    parser.add_argument("--n_workers", type=int, default=20)
    main(parser.parse_args())