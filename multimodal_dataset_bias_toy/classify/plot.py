import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from utils.plot import *


def log_prob(fpath):
    df = pd.read_csv(fpath)
    return -df.test_loss.iloc[-1]


def main(args):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    values = []
    for dataset_size in args.dataset_size_range:
        values_row = []
        for seed in range(args.n_seeds):
            multimodal_fpath = os.path.join(args.dpath, "classify", f"d={args.input_dim}", f"n={dataset_size}",
                "multimodal", f"version_{seed}", "metrics.csv")
            unimodal_fpath = os.path.join(args.dpath, "classify", f"d={args.input_dim}", f"n={dataset_size}",
                "unimodal", f"version_{seed}", "metrics.csv")
            log_prob_multimodal = log_prob(multimodal_fpath)
            log_prob_unimodal = log_prob(unimodal_fpath)
            values_row.append(log_prob_multimodal - log_prob_unimodal)
        values.append(values_row)
    values = pd.DataFrame(np.array(values).T).melt()
    sns.lineplot(data=values, x="variable", y="value", errorbar="sd", err_style="bars", err_kws={"capsize": 4}, ax=ax)
    ax.set_xticks(range(len(args.dataset_size_range)))
    ax.set_xticklabels(args.dataset_size_range)
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Generalization gap")
    ax.grid(alpha=0.5, linewidth=0.5)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35)
    os.makedirs("fig", exist_ok=True)
    plt.savefig(os.path.join("fig", f"toy_problem,comparison,d={args.input_dim}.pdf"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--dataset_size_range", nargs="+", type=int, default=[100, 400, 1600, 6400, 25600])
    main(parser.parse_args())