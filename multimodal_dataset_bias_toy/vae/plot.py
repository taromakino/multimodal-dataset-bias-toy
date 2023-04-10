import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from utils.plot import *


def kl(fpath):
    df = pd.read_csv(fpath)
    return df.test_kl.iloc[-1]


def main(args):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for i, input_dim in enumerate(args.input_dims):
        values = []
        for dataset_size in args.dataset_size_range:
            values_row = []
            for seed in range(args.n_seeds):
                fpath = os.path.join(args.dpath, "vae", f"d={input_dim}", f"n={dataset_size}", f"version_{seed}", "metrics.csv")
                values_row.append(kl(fpath))
            values.append(values_row)
        values = pd.DataFrame(np.array(values).T).melt()
        sns.lineplot(data=values, x="variable", y="value", errorbar="sd", ax=ax, legend=False, label=rf"$D={input_dim}$")
    ax.set_xticks(range(len(args.dataset_size_range)))
    ax.set_xticklabels(args.dataset_size_range)
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("KL")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=[0.5, 0])
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35)
    os.makedirs("fig", exist_ok=True)
    plt.savefig(os.path.join("fig", "toy_problem,detection.pdf"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--input_dims", nargs="+", type=int, default=[1, 16])
    parser.add_argument("--dataset_size_range", nargs="+", type=int, default=[100, 400, 1600, 6400, 25600])
    main(parser.parse_args())