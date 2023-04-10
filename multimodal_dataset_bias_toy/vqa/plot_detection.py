import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from utils.plot import *


def kl(fpath):
    df = pd.read_csv(fpath)
    idx = df.val_loss.argmin()
    return df.val_kl[idx]


def plot(ax, args, is_identifiable, x_offset):
    values = []
    for dataset_size in args.dataset_size_range:
        values_row = []
        for seed in range(args.n_seeds):
            suffix = "" if is_identifiable else ",vanilla"
            fpath = os.path.join(args.dpath, "vae" + suffix, f"data_seed=0,n={dataset_size}", f"version_{seed}",
                "metrics.csv")
            values_row.append(kl(fpath))
        values.append(values_row)
    values = pd.DataFrame(np.array(values).T).melt()
    values.variable += x_offset
    label = "Identifiable" if is_identifiable else "Unidentifiable"
    sns.lineplot(data=values, x="variable", y="value", errorbar="sd", err_style="bars", ax=ax, label=label, legend=False)
    ax.set_xticks(range(len(args.dataset_size_range)))
    ax.set_xticklabels(args.dataset_size_range)
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("KL")


def main(args):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plot(ax, args, True, 0)
    plot(ax, args, False, 0.05)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=[0.5, 0])
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35)
    os.makedirs("fig", exist_ok=True)
    plt.savefig(os.path.join("fig", "vqa,detection.pdf"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--dataset_size_range", nargs="+", type=int, default=[500, 2000, 8000, 32000, 128000])
    main(parser.parse_args())