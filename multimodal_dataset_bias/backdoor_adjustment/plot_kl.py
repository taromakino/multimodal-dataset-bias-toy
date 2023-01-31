import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from utils.plot_settings import *


def kld(fpath):
    df = pd.read_csv(fpath)
    idx = df.val_loss.argmin()
    return df.val_kld[idx]


def main(args):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.5))
    # s_shift
    means, sds = [], []
    for swap_ratio in args.swap_ratio_range:
        values = []
        for seed in range(args.n_seeds):
            fpath = os.path.join(args.dpath, f"swap_ratio={swap_ratio}", "vae", f"version_{seed}", "metrics.csv")
            values.append(kld(fpath))
        means.append(np.mean(values))
        sds.append(np.std(values))
    axes[0].errorbar(range(len(means)), means, sds)
    axes[0].set_xticks(range(len(args.swap_ratio_range)))
    axes[0].set_xticklabels(args.swap_ratio_range)
    axes[0].set_xlabel("Swap ratio")
    axes[0].set_ylabel("KL")
    # n_trainval
    means, sds = [], []
    for n_trainval in args.n_trainval_range:
        values = []
        for seed in range(args.n_seeds):
            fpath = os.path.join(args.dpath, f"n_trainval={n_trainval}", "vae", f"version_{seed}", "metrics.csv")
            values.append(kld(fpath))
        means.append(np.mean(values))
        sds.append(np.std(values))
    axes[1].errorbar(range(len(means)), means, sds)
    axes[1].set_xticks(range(len(args.n_trainval_range)))
    axes[1].set_xticklabels(args.n_trainval_range)
    axes[1].set_xlabel("Training set size")
    axes[1].set_ylabel("KL")
    fig.tight_layout()
    os.makedirs("fig", exist_ok=True)
    plt.savefig(os.path.join("fig", "kl.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results/backdoor_adjustment")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--swap_ratio_range", nargs="+", type=int, default=[0.0, 0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--n_trainval_range", nargs="+", type=int, default=[1600, 800, 400, 200, 100])
    main(parser.parse_args())