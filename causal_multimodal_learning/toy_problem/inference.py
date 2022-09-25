import os
import numpy as np
import pytorch_lightning as pl
import torch
from toy_problem.data import make_data
from toy_problem.model import Decoder, EncoderX
from utils.file import load_file
from utils.framework import make_gaussian

n_seeds = 1
n_samples = 10000
is_deconfounding = False

log_likelihoods = []
for seed in range(n_seeds):
    pl.seed_everything(seed)
    args = load_file(os.path.join("results", "args.pkl"))
    _, _, data_test = make_data(seed, args.n_examples, args.data_dim, 0, args.trainval_ratios, 1)

    decoder_mu = Decoder(args.data_dim, args.hidden_dim, args.latent_dim)
    decoder_logvar = Decoder(args.data_dim, args.hidden_dim, args.latent_dim)
    posterior_x = EncoderX(args.data_dim, args.hidden_dim, args.latent_dim)
    decoder_mu.load_state_dict(torch.load(os.path.join("results", "vae", f"version_{seed}", "decoder_mu.pt")))
    decoder_logvar.load_state_dict(torch.load(os.path.join("results", "vae", f"version_{seed}", "decoder_logvar.pt")))
    posterior_x.load_state_dict(torch.load(os.path.join("results", "posterior_x", f"version_{seed}", "posterior_x.pt")))
    prior = make_gaussian(torch.zeros(args.latent_dim)[None], torch.zeros(args.latent_dim)[None])

    log_likelihood = 0
    for x0, x1, y in data_test:
        mu_x, logvar_x = posterior_x(x0, x1)
        posterior_x_dist = make_gaussian(mu_x, logvar_x)

        x0_rep, x1_rep = x0.repeat(n_samples, 1), x1.repeat(n_samples, 1)
        z = posterior_x_dist.sample((n_samples,))

        y_mu = decoder_mu(x0_rep, x1_rep, z)
        y_logvar = decoder_logvar(x0_rep, x1_rep, z)
        decoder_dist = make_gaussian(y_mu, y_logvar)
        y_logp = decoder_dist.log_prob(y.squeeze())

        if is_deconfounding:
            pred = -torch.log(torch.tensor(n_samples)) + torch.logsumexp(prior.log_prob(z) - posterior_x_dist.log_prob(z) + y_logp, 0)
        else:
            pred = -torch.log(torch.tensor(n_samples)) + torch.logsumexp(y_logp, 0)
        log_likelihood += pred.item()
    log_likelihoods.append(log_likelihood)
print(f"{np.mean(log_likelihoods):.3f} +/- {np.std(log_likelihoods):.3f}")