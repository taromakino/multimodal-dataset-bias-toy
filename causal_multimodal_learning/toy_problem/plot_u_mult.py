import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from utils.plot_settings import *

def main(args):
    conditional_means, conditional_sds = [], []
    interventional_means, interventional_sds = [], []
    for u_mult in args.u_mult_range:
        conditional, interventional = [], []
        for seed in range(args.n_seeds):
            fpath = os.path.join(args.dpath, "inference", f"u={u_mult}", f"version_{seed}", "metrics.csv")
            df = pd.read_csv(fpath)
            conditional.append(df.test_conditional_logp_epoch.iloc[-1])
            interventional.append(df.test_interventional_logp_epoch.iloc[-1])
        conditional_means.append(np.mean(conditional))
        conditional_sds.append(np.std(conditional))
        interventional_means.append(np.mean(interventional))
        interventional_sds.append(np.std(interventional))
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.errorbar(np.arange(len(args.u_mult_range)), conditional_means, conditional_sds, label=r"$\log p(y \mid x, x')$")
    ax.errorbar(np.arange(len(args.u_mult_range)) + 0.05, interventional_means, interventional_sds,
        label=r"$\log p(y \mid do(x), do(x'))$")
    plt.savefig(os.path.join(args.dpath, "fig.pdf"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--u_mult_range", nargs="+", type=str, default=["4.0", "2.0", "1.0", "0.0"])
    main(parser.parse_args())