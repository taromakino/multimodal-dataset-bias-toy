import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from utils.plot_settings import *

def kl_loss(fpath):
    df = pd.read_csv(fpath)
    idx = df.val_loss.argmin()
    return df.val_kld[idx]

def main(args):
    means, sds = [], []
    for s_shift in args.s_shift_range:
        dpath = os.path.join(args.dpath, f"s_shift={s_shift}")
        values = []
        for seed in range(args.n_seeds):
            vae_fpath = os.path.join(dpath, "vae", f"version_{seed}", "metrics.csv")
            values.append(kl_loss(vae_fpath))
        means.append(np.mean(values))
        sds.append(np.std(values))
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    ax.errorbar(range(len(means)), means, sds)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(args.s_shift_range)
    ax.set_xlabel(r"$c$")
    ax.set_ylabel(r"$D_{KL}(q(z \mid x, x', y) \mid\mid p(z))$")
    fig.tight_layout()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--s_shift_range", nargs="+", type=int, default=[-4, -2, 0, 2, 4])
    parser.add_argument("--dpath", type=str, required=True)
    main(parser.parse_args())