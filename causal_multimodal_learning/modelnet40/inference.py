import os
import numpy as np
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from modelnet40.data import make_data
from modelnet40.model import PosteriorX, SemiSupervisedVae
from utils.file import load_file, save_file
from utils.nn_utils import load_model
from utils.stats import make_gaussian
from utils.plot_settings import *

def main(args):
    confounded_means, confounded_sds = [], []
    deconfounded_means, deconfounded_sds = [], []
    for subset_ratio in args.subset_ratio_range:
        confounded_logps, deconfounded_logps = [], []
        for seed in range(args.n_seeds):
            pl.seed_everything(seed)
            hparams = load_file(os.path.join(args.dpath, f"r={subset_ratio}", "args.pkl"))
            _, _, data_test = make_data(seed, 1, 1, args.n_workers)

            vae = load_model(SemiSupervisedVae, os.path.join(args.dpath, f"r={subset_ratio}", "vae", f"version_{seed}",
                "checkpoints"))
            posterior_x = load_model(PosteriorX, os.path.join(args.dpath, f"r={subset_ratio}", "posterior_x",
                f"version_{seed}", "checkpoints"))
            prior = make_gaussian(torch.zeros(hparams.latent_dim)[None], torch.zeros(hparams.latent_dim)[None])

            confounded_logp = deconfounded_logp = 0
            for x0, x1, y in data_test:
                mu_x, logvar_x = posterior_x.encoder_x(x0, x1)
                posterior_x_dist = make_gaussian(mu_x, logvar_x)
                z = posterior_x_dist.sample((args.n_samples,))
                x0_rep, x1_rep = x0.repeat(args.n_samples, 1), x1.repeat(args.n_samples, 1)

                y_mu, y_logvar = vae.decoder(x0_rep, x1_rep, z)
                decoder_dist = make_gaussian(y_mu, y_logvar)
                y_logp = decoder_dist.log_prob(y.squeeze())

                confounded_logp += -torch.log(torch.tensor(args.n_samples)) + torch.logsumexp(y_logp, 0).item()
                deconfounded_logp += -torch.log(torch.tensor(args.n_samples)) + torch.logsumexp(prior.log_prob(z) -
                    posterior_x_dist.log_prob(z) + y_logp, 0).item()
            n_examples = len(data_test.dataset)
            confounded_logp, deconfounded_logp = confounded_logp / n_examples, deconfounded_logp / n_examples
            confounded_logps.append(confounded_logp)
            deconfounded_logps.append(deconfounded_logp)
        confounded_means.append(np.mean(confounded_logps))
        confounded_sds.append(np.std(confounded_logps))
        deconfounded_means.append(np.mean(deconfounded_logps))
        deconfounded_sds.append(np.std(deconfounded_logps))
    save_file((confounded_means, confounded_sds, deconfounded_means, deconfounded_sds), os.path.join(args.dpath,
        "inference.pkl"))
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.errorbar(np.arange(len(args.u_mult_range)), confounded_means, confounded_sds, label=r"$\log P(Y \mid X, X')$")
    ax.errorbar(np.arange(len(args.u_mult_range)) + 0.05, deconfounded_means, deconfounded_sds,
        label=r"$\log P(Y \mid do(X), do(X'))$")
    ax.set_xticks(range(len(args.u_mult_range)), args.u_mult_range)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("Log-density")
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(args.dpath, "fig.pdf"), bbox_inches="tight")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--subset_ratio_range", nargs="+", type=float, default=[1, 0.75, 0.5, 0.25])
    parser.add_argument("--n_workers", type=int, default=20)
    main(parser.parse_args())