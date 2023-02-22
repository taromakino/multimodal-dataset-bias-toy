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
    # s_shift
    means, sds = [], []
    for s_shift in args.s_shift_range:
        values = []
        for seed in range(args.n_seeds):
            mi_uxy = mutual_info(os.path.join(args.dpath, f"s_shift={s_shift}", "uxy", f"version_{seed}", "metrics.csv"))
            mi_ux = mutual_info(os.path.join(args.dpath, f"s_shift={s_shift}", "ux", f"version_{seed}", "metrics.csv"))
            values.append(mi_uxy - mi_ux)
        means.append(np.mean(values))
        sds.append(np.std(values))
    axes[0].errorbar(range(len(means)), means, sds)
    axes[0].set_xticks(range(len(args.s_shift_range)))
    axes[0].set_xticklabels(args.s_shift_range)
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
    plt.savefig(os.path.join("fig", "mutual_information.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results/mutual_information")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--s_shift_range", nargs="+", type=float, default=[-4, -2, 0, 2, 4])
    parser.add_argument("--n_train_range", nargs="+", type=int, default=[3200, 1600, 800, 400, 200])
    main(parser.parse_args())