import os
import numpy as np
import pytorch_lightning as pl
import torch
from toy_problem.data import make_data
from utils.file import load_file
from utils.stats import make_gaussian
from utils.plot_settings import *

n_seeds = 5
n_samples = 1000

confounded_means, confounded_sds = [], []
deconfounded_means, deconfounded_sds = [], []

u_mult_range = [1, 0.75, 0.5, 0.25, 0]
for u_mult in u_mult_range:
    confounded_logps, deconfounded_logps = [], []
    for seed in range(n_seeds):
        pl.seed_everything(seed)
        args = load_file(os.path.join("results", "args.pkl"))
        _, _, data_test = make_data(seed, args.n_examples, args.data_dim, u_mult, args.trainval_ratios, 1)

        decoder_mu = Decoder(args.data_dim, args.hidden_dim, args.latent_dim)
        decoder_logvar = Decoder(args.data_dim, args.hidden_dim, args.latent_dim)
        posterior_x = EncoderX(args.data_dim, args.hidden_dim, args.latent_dim)
        decoder_mu.load_state_dict(torch.load(os.path.join("results", "vae", f"version_{seed}", "decoder_mu.pt")))
        decoder_logvar.load_state_dict(torch.load(os.path.join("results", "vae", f"version_{seed}", "decoder_logvar.pt")))
        posterior_x.load_state_dict(torch.load(os.path.join("results", "posterior_x", f"version_{seed}", "posterior_x.pt")))
        prior = make_gaussian(torch.zeros(args.latent_dim)[None], torch.zeros(args.latent_dim)[None])

        confounded_logp = deconfounded_logp = 0
        for x0, x1, y in data_test:
            mu_x, logvar_x = posterior_x(x0, x1)
            posterior_x_dist = make_gaussian(mu_x, logvar_x)

            x0_rep, x1_rep = x0.repeat(n_samples, 1), x1.repeat(n_samples, 1)
            z = posterior_x_dist.sample((n_samples,))

            y_mu = decoder_mu(x0_rep, x1_rep, z)
            y_logvar = decoder_logvar(x0_rep, x1_rep, z)
            decoder_dist = make_gaussian(y_mu, y_logvar)
            y_logp = decoder_dist.log_prob(y.squeeze())

            confounded_logp += -torch.log(torch.tensor(n_samples)) + torch.logsumexp(y_logp, 0).item()
            deconfounded_logp += -torch.log(torch.tensor(n_samples)) + torch.logsumexp(prior.log_prob(z) -
                posterior_x_dist.log_prob(z) + y_logp, 0).item()
        confounded_logps.append(confounded_logp)
        deconfounded_logps.append(deconfounded_logp)
    confounded_means.append(np.mean(confounded_logps))
    confounded_sds.append(np.std(confounded_logps))
    deconfounded_means.append(np.mean(deconfounded_logps))
    deconfounded_sds.append(np.std(deconfounded_logps))
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.errorbar(np.arange(len(u_mult_range)), confounded_means, confounded_sds, label=r"$\log p(y \mid x, x')$")
ax.errorbar(np.arange(len(u_mult_range)) + 0.05, deconfounded_means, deconfounded_sds, label=r"$\log p(y \mid do(x), do(x'))$")
ax.set_xticks(range(len(u_mult_range)), u_mult_range)
ax.set_xlabel(r"$\beta_U$")
ax.set_ylabel("Log-likelihood")
ax.grid(True)
ax.legend()
fig.tight_layout()