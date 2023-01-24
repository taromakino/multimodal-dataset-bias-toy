import os
import pandas as pd
from argparse import ArgumentParser
from utils.plot_settings import *
from scipy.stats import ttest_ind

def log_marginal_likelihood(fpath):
    df = pd.read_csv(fpath)
    return df.conditional_lml[0], df.interventional_lml[0]

def p_value(dpath, n_seeds):
    conditional_lml_values, interventional_lml_values = [], []
    for seed in range(n_seeds):
        fpath = os.path.join(dpath, "log_marginal_likelihood", f"version_{seed}", "metrics.csv")
        conditional_lml, interventional_lml = log_marginal_likelihood(fpath)
        conditional_lml_values.append(conditional_lml)
        interventional_lml_values.append(interventional_lml)
    return ttest_ind(conditional_lml_values, interventional_lml_values, alternative="less")[1]

def main(args):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.5))
    # s_shift
    p_values = []
    for s_shift in args.s_shift_range:
        dpath = os.path.join(args.dpath, f"s_shift={s_shift}")
        p_values.append(p_value(dpath, args.n_seeds))
    axes[0].plot(p_values)
    axes[0].set_xticks(range(len(args.s_shift_range)))
    axes[0].set_xticklabels(args.s_shift_range)
    axes[0].set_xlabel(r"$c$")
    axes[0].set_ylabel("p-value")
    # n_train
    p_values = []
    for n_train in args.n_train_range:
        dpath = os.path.join(args.dpath, f"n_train={n_train}")
        p_values.append(p_value(dpath, args.n_seeds))
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
    parser.add_argument("--dpath", type=str, default="results/backdoor_adjustment")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--s_shift_range", nargs="+", type=int, default=[-4, -2, 0, 2, 4])
    parser.add_argument("--n_train_range", nargs="+", type=int, default=[1600, 800, 400, 200, 100])
    main(parser.parse_args())