import os
from argparse import ArgumentParser
from toy_problem.data import make_data
from toy_problem.model import Model
from utils.file import load_file
from utils.nn_utils import load_model, make_tester

def main(args):
    model = load_model(Model, os.path.join(args.dpath, f"version_{args.seed}", "checkpoints"))
    hparams = load_file(os.path.join(args.dpath, "args.pkl"))
    _, _, data_test = make_data(args.seed, hparams.n_examples, hparams.data_dim, args.u_mult, 1, hparams.n_workers)
    tester = make_tester(os.path.join(args.dpath, "inference", f"u={args.u_mult}"), args.seed)
    tester.test(model, data_test)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--u_mult", type=float, default=0)
    main(parser.parse_args())