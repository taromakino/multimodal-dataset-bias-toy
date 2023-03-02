import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from utils.plot import *


def alpha(fpath):
    df = pd.read_csv(fpath)
    return df.alpha.iloc[df.loss.argmin()]


def main(args):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.5))
    # s_shift
    vars = []
    for s_shift in args.s_shift_range:
        values = []
        for seed in range(args.n_seeds):
            values.append(alpha(os.path.join(args.dpath, f"s_shift={s_shift}", f"version_{seed}", "metrics.csv")))
        vars.append(np.var(values))
    axes[0].plot(range(len(vars)), vars)
    axes[0].set_xticks(range(len(args.s_shift_range)))
    axes[0].set_xticklabels(args.s_shift_range)
    axes[0].set_xlabel(r"$c$")
    axes[0].set_ylabel(r"$Var(\alpha)$")
    # n_train
    vars = []
    for n_train in args.n_train_range:
        values = []
        for seed in range(args.n_seeds):
            values.append(alpha(os.path.join(args.dpath, f"n_train={n_train}", f"version_{seed}", "metrics.csv")))
        vars.append(np.std(values))
    axes[1].plot(range(len(vars)), vars)
    axes[1].set_xticks(range(len(args.n_train_range)))
    axes[1].set_xticklabels(args.n_train_range)
    axes[1].set_xlabel("Training set size")
    axes[1].set_ylabel(r"$Var(\alpha)$")
    fig.tight_layout()
    os.makedirs("fig", exist_ok=True)
    plt.savefig(os.path.join("fig", "modified_graph.png"), bbox_inches="tight")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results/modified_graph")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--s_shift_range", nargs="+", type=float, default=[-4, -2, 0, 2, 4])
    parser.add_argument("--n_train_range", nargs="+", type=int, default=[25600, 6400, 1600, 400, 100])
    main(parser.parse_args())