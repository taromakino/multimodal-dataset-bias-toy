import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from utils.plot_settings import *


def log_marginal_likelihood(fpath):
    df = pd.read_csv(fpath)
    return df.conditional_lml[0], df.interventional_lml[0]


def plot(ax, dpath, n_seeds, arg_name, arg_range):
    conditional_means, conditional_sds = [], []
    interventional_means, interventional_sds = [], []
    for s_shift in arg_range:
        conditional_values, interventional_values = [], []
        for seed in range(n_seeds):
            fpath = os.path.join(dpath, f"{arg_name}={s_shift}", "log_marginal_likelihood", f"version_{seed}", "metrics.csv")
            conditional_value, interventional_value = log_marginal_likelihood(fpath)
            conditional_values.append(conditional_value)
            interventional_values.append(interventional_value)
        conditional_means.append(np.mean(conditional_values))
        conditional_sds.append(np.std(conditional_values))
        interventional_means.append(np.mean(interventional_values))
        interventional_sds.append(np.std(interventional_values))
    ax.errorbar(range(len(conditional_means)), conditional_means, conditional_sds, label=r"$\log p(y \mid x, x')$")
    ax.errorbar(range(len(interventional_means)), interventional_means, interventional_sds,
        label=r"$\log p(y \mid do(x), do(x'))$")
    ax.set_xticks(range(len(arg_range)))
    ax.set_xticklabels(arg_range)


def main(args):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.5))
    # s_shift
    plot(axes[0], args.dpath, args.n_seeds, "s_shift", args.s_shift_range)
    axes[0].set_xlabel(r"$c$")
    # n_train
    plot(axes[1], args.dpath, args.n_seeds, "n_train", args.n_train_range)
    axes[1].set_xlabel("Training set size")
    for ax in axes:
        ax.set_ylabel("KL")
    fig.tight_layout()
    os.makedirs("fig", exist_ok=True)
    plt.savefig(os.path.join("fig", "log_marginal_likelihood.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results/backdoor_adjustment")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--s_shift_range", nargs="+", type=int, default=[-4, -2, 0, 2, 4])
    parser.add_argument("--n_train_range", nargs="+", type=int, default=[3200, 1600, 800, 400, 200])
    main(parser.parse_args())