import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from utils.plot_settings import *

def kl_loss(fpath):
    df = pd.read_csv(fpath)
    idx = df.val_loss.argmin()
    return df.val_kld[idx]

def main(args):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    # s_shift
    means, sds = [], []
    for s_shift in args.s_shift_range:
        dpath = os.path.join(args.dpath, f"s_shift={s_shift}")
        values = []
        for seed in range(args.n_seeds):
            fpath = os.path.join(dpath, "vae", f"version_{seed}", "metrics.csv")
            values.append(kl_loss(fpath))
        means.append(np.mean(values))
        sds.append(np.std(values))
    axes[0].errorbar(range(len(means)), means, sds)
    axes[0].set_xticks(range(len(args.s_shift_range)))
    axes[0].set_xticklabels(args.s_shift_range)
    axes[0].set_xlabel(r"$c$")
    axes[0].set_ylabel("KL")
    # n_train
    means, sds = [], []
    for n_train in args.n_train_range:
        dpath = os.path.join(args.dpath, f"n_train={n_train}")
        values = []
        for seed in range(args.n_seeds):
            fpath = os.path.join(dpath, "vae", f"version_{seed}", "metrics.csv")
            values.append(kl_loss(fpath))
        means.append(np.mean(values))
        sds.append(np.std(values))
    axes[1].errorbar(range(len(means)), means, sds)
    axes[1].set_xticks(range(len(args.n_train_range)))
    axes[1].set_xticklabels(args.n_train_range)
    axes[1].set_xlabel("Training set size")
    axes[1].set_ylabel("KL")
    fig.tight_layout()
    os.makedirs("fig", exist_ok=True)
    plt.savefig(os.path.join("fig", "vae.pdf"), bbox_inches="tight")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--s_shift_range", nargs="+", type=int, default=[-4, -2, 0, 2, 4])
    parser.add_argument("--n_train_range", nargs="+", type=int, default=[1600, 800, 400, 200, 100])
    main(parser.parse_args())