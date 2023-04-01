import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from utils.plot import *


def log_prob(fpath):
    df = pd.read_csv(fpath)
    return -df.test_loss.iloc[-1]


def main(args):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    means, sds = [], []
    for dataset_size in args.dataset_size_range:
        values = []
        for seed in range(args.n_seeds):
            multimodal_fpath = os.path.join(args.dpath, "multimodal", f"data_seed=0,n={dataset_size}", f"version_{seed}",
                "metrics.csv")
            unimodal_fpath = os.path.join(args.dpath, "unimodal", f"data_seed=0,n={dataset_size}", f"version_{seed}",
                "metrics.csv")
            log_prob_multimodal = log_prob(multimodal_fpath)
            log_prob_unimodal = log_prob(unimodal_fpath)
            values.append(log_prob_multimodal - log_prob_unimodal)
        means.append(np.mean(values))
        sds.append(np.std(values))
    ax.errorbar(range(len(means)), means, sds)
    ax.set_xticks(range(len(args.dataset_size_range)))
    ax.set_xticklabels(args.dataset_size_range)
    ax.set_xlabel("Dataset size")
    ax.set_ylabel(r"$\Delta \log p(y \mid x, x')$")
    fig.tight_layout()
    os.makedirs("fig", exist_ok=True)
    plt.savefig(os.path.join("fig", "vqa,comparison.pdf"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--dataset_size_range", nargs="+", type=int, default=[500, 2000, 8000, 32000, 128000])
    main(parser.parse_args())