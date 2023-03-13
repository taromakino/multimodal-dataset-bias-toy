import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from utils.plot import *


def kl(fpath):
    df = pd.read_csv(fpath)
    idx = df.val_loss.argmin()
    return df.val_kl[idx]


def main(args):
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    means, sds = [], []
    for n_train in args.n_train_range:
        values = []
        for seed in range(args.n_seeds):
            fpath = os.path.join(args.dpath, f"n_train={n_train}", f"version_{seed}", "metrics.csv")
            values.append(kl(fpath))
        means.append(np.mean(values))
        sds.append(np.std(values))
    ax.errorbar(range(len(means)), means, sds)
    ax.set_xticks(range(len(args.n_train_range)))
    ax.set_xticklabels(args.n_train_range)
    ax.set_xlabel("Training set size")
    ax.set_ylabel("KL")
    fig.tight_layout()
    os.makedirs("fig", exist_ok=True)
    plt.savefig(os.path.join("fig", "fig.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results/vae")
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--n_train_range", nargs="+", type=int, default=[25600, 6400, 1600, 400, 100])
    main(parser.parse_args())