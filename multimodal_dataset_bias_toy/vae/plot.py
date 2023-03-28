import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from utils.plot import *


def kl(fpath):
    df = pd.read_csv(fpath)
    return df.test_kl.iloc[-1]


def main(args):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    means, sds = [], []
    for sample_size in args.sample_size_range:
        values = []
        for seed in range(args.n_seeds):
            fpath = os.path.join(args.dpath, f"n={sample_size}", f"version_{seed}", "metrics.csv")
            values.append(kl(fpath))
        means.append(np.mean(values))
        sds.append(np.std(values))
    ax.errorbar(range(len(means)), means, sds)
    ax.set_xticks(range(len(args.sample_size_range)))
    ax.set_xticklabels(args.sample_size_range)
    ax.set_xlabel("Sample size")
    ax.set_ylabel("KL")
    fig.tight_layout()
    os.makedirs("fig", exist_ok=True)
    plt.savefig(os.path.join("fig", "fig.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, required=True)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--sample_size_range", nargs="+", type=int, default=[25600, 6400, 1600, 400, 100])
    main(parser.parse_args())