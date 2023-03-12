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
    for subset_ratio in args.subset_ratio_range:
        values = []
        for seed in range(args.n_seeds):
            fpath = os.path.join(args.dpath, f"data_seed=0,subset_ratio={subset_ratio},train_ratio=0.8",
                f"version_{seed}", "metrics.csv")
            values.append(kl(fpath))
        means.append(np.mean(values))
        sds.append(np.std(values))
    ax.errorbar(range(len(means)), means, sds)
    ax.set_xticks(range(len(args.subset_ratio_range)))
    ax.set_xticklabels(args.subset_ratio_range)
    ax.set_xlabel("Subset ratio")
    ax.set_ylabel("KL")
    fig.tight_layout()
    os.makedirs("fig", exist_ok=True)
    plt.savefig(os.path.join("fig", "fig.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results/vqa")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--subset_ratio_range", nargs="+", type=float, default=[0.32, 0.16, 0.08, 0.01, 0.005])
    main(parser.parse_args())