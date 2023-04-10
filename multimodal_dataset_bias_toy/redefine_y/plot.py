import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from utils.plot import *


def abs_alpha(fpath):
    df = pd.read_csv(fpath)
    return abs(df.alpha_norm.iloc[df.val_loss.argmin()])


def main(args):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    values = []
    for dataset_size in args.dataset_size_range:
        values_row = []
        for seed in range(args.n_seeds):
            values_row.append(abs_alpha(os.path.join(args.dpath, "redefine_y", f"d={args.input_dim}", f"n={dataset_size}",
                f"version_{seed}", "metrics.csv")))
        values.append(values_row)
    values = pd.DataFrame(np.array(values).T).melt()
    sns.lineplot(data=values, x="variable", y="value", errorbar="sd", err_style="bars", ax=ax)
    ax.set_xticks(range(len(args.dataset_size_range)))
    ax.set_xticklabels(args.dataset_size_range)
    ax.set_xlabel("Dataset size")
    ax.set_ylabel(r"$||\alpha||$")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35)
    os.makedirs("fig", exist_ok=True)
    plt.savefig(os.path.join("fig", f"toy_problem,redefine_y,d={args.input_dim}.pdf"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--dataset_size_range", nargs="+", type=int, default=[100, 400, 1600, 6400, 25600])
    main(parser.parse_args())