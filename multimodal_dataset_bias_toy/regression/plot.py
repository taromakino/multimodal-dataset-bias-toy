import os
import pandas as pd
from argparse import ArgumentParser
from utils.plot import *
from scipy.stats import ttest_rel


def loss(fpath):
    df = pd.read_csv(fpath)
    return df.test_loss.iloc[-1]


def p_value(dpath, n_seeds):
    unimodal_values, multimodal_values = [], []
    for seed in range(n_seeds):
        unimodal_values.append(loss(os.path.join(dpath, "unimodal", f"version_{seed}", "metrics.csv")))
        multimodal_values.append(loss(os.path.join(dpath, "multimodal", f"version_{seed}", "metrics.csv")))
    return ttest_rel(unimodal_values, multimodal_values, alternative="less")[1]


def main(args):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.5))
    # s_shift
    p_values = []
    for s_shift in args.s_shift_range:
        p_values.append(p_value(os.path.join(args.dpath, f"s_shift={s_shift}"), args.n_seeds))
    axes[0].plot(p_values)
    axes[0].set_xticks(range(len(args.s_shift_range)))
    axes[0].set_xticklabels(args.s_shift_range)
    axes[0].set_xlabel(r"$c$")
    axes[0].set_ylabel("p-value")
    # n_train
    p_values = []
    for n_train in args.n_train_range:
        p_values.append(p_value(os.path.join(args.dpath, f"n_train={n_train}"), args.n_seeds))
    axes[1].plot(p_values)
    axes[1].set_xticks(range(len(args.n_train_range)))
    axes[1].set_xticklabels(args.n_train_range)
    axes[1].set_xlabel("Training set size")
    axes[1].set_ylabel("p-value")
    fig.tight_layout()
    os.makedirs("fig", exist_ok=True)
    plt.savefig(os.path.join("fig", "p_value.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results/regression")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--s_shift_range", nargs="+", type=int, default=[-4, -2, 0, 2, 4])
    parser.add_argument("--n_train_range", nargs="+", type=int, default=[3200, 1600, 800, 400, 200])
    main(parser.parse_args())