import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from utils.plot_settings import *

def main(args):
    conditional_means, conditional_sds = [], []
    interventional_means, interventional_sds = [], []
    for n_train in args.n_train_range:
        conditional, interventional = [], []
        for seed in range(args.n_seeds):
            fpath = os.path.join(args.dpath, f"n={n_train}", f"version_{seed}", "metrics.csv")
            df = pd.read_csv(fpath)
            conditional.append(df.test_conditional_logp.iloc[-1])
            interventional.append(df.test_interventional_logp.iloc[-1])
        conditional_means.append(np.mean(conditional))
        conditional_sds.append(np.std(conditional))
        interventional_means.append(np.mean(interventional))
        interventional_sds.append(np.std(interventional))
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.errorbar(np.arange(len(args.n_train_range)), conditional_means, conditional_sds, label=r"$\log p(y \mid x, x')$")
    ax.errorbar(np.arange(len(args.n_train_range)) + 0.05, interventional_means, interventional_sds,
        label=r"$\log p(y \mid do(x), do(x'))$")
    plt.savefig(os.path.join(args.dpath, "fig.pdf"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--n_train_range", nargs="+", type=str, default=["100", "200", "400", "800"])
    parser.add_argument("--data_dim", type=int, default=1)
    main(parser.parse_args())