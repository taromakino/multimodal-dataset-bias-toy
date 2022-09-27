import os
import numpy as np
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from toy_problem.data import make_data
from toy_problem.model import PosteriorX, SemiSupervisedVae
from utils.file import load_file
from utils.nn_utils import load_model
from utils.stats import make_gaussian
from utils.plot_settings import *

def main(args):
    confounded_means, confounded_sds = [], []
    deconfounded_means, deconfounded_sds = [], []
    for u_mult in args.u_mult_range:
        confounded_logps, deconfounded_logps = [], []
        for seed in range(args.n_seeds):
            pl.seed_everything(seed)
            args = load_file(os.path.join(args.dpath, "args.pkl"))
            _, _, data_test = make_data(seed, args.n_examples, args.data_dim, u_mult, args.trainval_ratios, 1)

            vae = load_model(SemiSupervisedVae, os.path.join("results", "vae", f"version_{seed}", "checkpoints"))
            posterior_x = load_model(PosteriorX, os.path.join("results", "posterior_x", f"version_{seed}", "checkpoints"))
            prior = make_gaussian(torch.zeros(args.latent_dim)[None], torch.zeros(args.latent_dim)[None])

            confounded_logp = deconfounded_logp = 0
            for x0, x1, y in data_test:
                z = posterior_x.posterior_x.sample(x0, x1, args.n_samples).squeeze()
                x0_rep, x1_rep = x0.repeat(args.n_samples, 1), x1.repeat(args.n_samples, 1)

                y_mu = vae.decoder_mu(x0_rep, x1_rep, z)
                y_logvar = vae.decoder_logvar(x0_rep, x1_rep, z)
                decoder_dist = make_gaussian(y_mu, y_logvar)
                y_logp = decoder_dist.log_prob(y.squeeze())

                confounded_logp += -torch.log(torch.tensor(args.n_samples)) + torch.logsumexp(y_logp, 0).item()
                deconfounded_logp += -torch.log(torch.tensor(args.n_samples)) + torch.logsumexp(prior.log_prob(z) -
                    posterior_x.posterior_x(x0_rep, x1_rep, z) + y_logp, 0).item()
            confounded_logps.append(confounded_logp)
            deconfounded_logps.append(deconfounded_logp)
        confounded_means.append(np.mean(confounded_logps))
        confounded_sds.append(np.std(confounded_logps))
        deconfounded_means.append(np.mean(deconfounded_logps))
        deconfounded_sds.append(np.std(deconfounded_logps))
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.errorbar(np.arange(len(args.u_mult_range)), confounded_means, confounded_sds, label=r"$\log p(y \mid x, x')$")
    ax.errorbar(np.arange(len(args.u_mult_range)) + 0.05, deconfounded_means, deconfounded_sds,
        label=r"$\log p(y \mid do(x), do(x'))$")
    ax.set_xticks(range(len(args.u_mult_range)), args.u_mult_range)
    ax.set_xlabel(r"$\beta_U$")
    ax.set_ylabel("Log-likelihood")
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(args.dpath, "fig.png"), bbox_inches="tight")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--u_mult_range", nargs="+", type=float, default=[1, 0.75, 0.5, 0.25, 0])
    main(parser.parse_args())
