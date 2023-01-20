import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from utils.plot_settings import *

def test_loss(fpath):
    df = pd.read_csv(fpath)
    return df.test_loss.iloc[-1]

def kl_ratio(fpath):
    df = pd.read_csv(fpath)
    idx = df.val_loss.argmin()
    # return df.val_kld[idx] / df.val_loss[idx]
    return df.val_kld[idx]

def main(args):
    regression_means, regression_sds = [], []
    vae_means, vae_sds = [], []
    for s_shift in args.s_shift_range:
        dpath = os.path.join(args.dpath, f"s_shift={s_shift}")
        regression_inner = []
        vae_inner = []
        for seed in range(args.n_seeds):
            unimodal_fpath = os.path.join(dpath, "unimodal_ensemble", f"version_{seed}", "metrics.csv")
            mutimodal_fpath = os.path.join(dpath, "multimodal", f"version_{seed}", "metrics.csv")
            vae_fpath = os.path.join(dpath, "vae", f"version_{seed}", "metrics.csv")
            unimodal_loss = test_loss(unimodal_fpath)
            multimodal_loss = test_loss(mutimodal_fpath)
            regression_inner.append(multimodal_loss / unimodal_loss)
            vae_inner.append(kl_ratio(vae_fpath))
        regression_means.append(np.mean(regression_inner))
        regression_sds.append(np.std(regression_inner))
        vae_means.append(np.mean(vae_inner))
        vae_sds.append(np.std(vae_inner))
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    ax.errorbar(range(len(vae_means)), vae_means, vae_sds)
    ax.set_xticks(range(len(vae_means)))
    ax.set_xticklabels(args.s_shift_range)
    ax.set_xlabel("Spuriousness")
    ax.set_ylabel("KL")
    fig.tight_layout()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--s_shift_range", nargs="+", type=int, default=[-4, -2, 0, 2, 4])
    parser.add_argument("--dpath", type=str, required=True)
    main(parser.parse_args())