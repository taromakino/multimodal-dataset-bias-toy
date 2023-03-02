import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from utils.plot import *


def mutual_info(fpath):
    df = pd.read_csv(fpath)
    return -df.val_loss.min()


def main(args):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.5))
    # alpha
    means, sds = [], []
    for alpha in args.alpha_range:
        values = []
        for seed in range(args.n_seeds):
            mi_uxy = mutual_info(os.path.join(args.dpath, f"alpha={alpha}", "uxy", f"version_{seed}", "metrics.csv"))
            mi_ux = mutual_info(os.path.join(args.dpath, f"alpha={alpha}", "ux", f"version_{seed}", "metrics.csv"))
            values.append(mi_uxy - mi_ux)
        means.append(np.mean(values))
        sds.append(np.std(values))
    axes[0].errorbar(range(len(means)), means, sds)
    axes[0].set_xticks(range(len(args.alpha_range)))
    axes[0].set_xticklabels(args.alpha_range)
    axes[0].set_xlabel(r"$c$")
    axes[0].set_ylabel(r"$I(U; Y \mid X, X')$")
    # n_train
    means, sds = [], []
    for n_train in args.n_train_range:
        values = []
        for seed in range(args.n_seeds):
            mi_uxy = mutual_info(os.path.join(args.dpath, f"n_train={n_train}", "uxy", f"version_{seed}", "metrics.csv"))
            mi_ux = mutual_info(os.path.join(args.dpath, f"n_train={n_train}", "ux", f"version_{seed}", "metrics.csv"))
            values.append(mi_uxy - mi_ux)
        means.append(np.mean(values))
        sds.append(np.std(values))
    axes[1].errorbar(range(len(means)), means, sds)
    axes[1].set_xticks(range(len(args.n_train_range)))
    axes[1].set_xticklabels(args.n_train_range)
    axes[1].set_xlabel("Training set size")
    axes[1].set_ylabel(r"$I(U; Y \mid X, X')$")
    fig.tight_layout()
    os.makedirs("fig", exist_ok=True)
    plt.savefig(os.path.join("fig", "mutual_information.png"), bbox_inches="tight")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results/mutual_information")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--alpha_range", nargs="+", type=float, default=[-4, -2, 0, 2, 4])
    parser.add_argument("--n_train_range", nargs="+", type=int, default=[8000, 4000, 2000, 1000, 500])
    main(parser.parse_args())