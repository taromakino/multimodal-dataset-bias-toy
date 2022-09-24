import os
import numpy as np
import pytorch_lightning as pl
import torch
from toy_problem.data import make_data
from toy_problem.model import SemiSupervisedVae
from utils.file import load_file
from utils.framework import make_gaussian

alpha = 1
u_mult = 1
n_seeds = 5
n_samples = 1000
is_deconfounding = True

from utils.framework import prior_kld

log_likelihoods = []
for seed in range(n_seeds):
    dpath = f"results/u={u_mult},a={alpha}/version_{seed}"
    args = load_file(os.path.join(dpath, "args.pkl"))
    ckpt_fpath = os.path.join(dpath, "checkpoints", os.listdir(os.path.join(dpath, "checkpoints"))[0])

    pl.seed_everything(args.seed)

    model = SemiSupervisedVae.load_from_checkpoint(ckpt_fpath, hparams_file=os.path.join(dpath, "hparams.yaml"))
    _, _, data_test = make_data(seed, args.n_examples, args.data_dim, args.u_mult, args.trainval_ratios, 1)

    prior = torch.distributions.MultivariateNormal(loc=torch.zeros(args.latent_dim), covariance_matrix=torch.eye(args.latent_dim))

    log_likelihood = 0
    posterior_klds = []
    for x0, x1, y in data_test:
        mu, logvar = model.input_encoder(x0, x1)
        posterior = make_gaussian(mu, logvar)
        posterior_klds.append(prior_kld(mu, logvar).item())

        x0_rep, x1_rep = x0.repeat(n_samples, 1), x1.repeat(n_samples, 1)
        z = posterior.sample((n_samples,))

        y_mu = model.decoder_mu(x0_rep, x1_rep, z)
        y_logvar = model.decoder_mu(x0_rep, x1_rep, z)
        decoder_dist = make_gaussian(y_mu, y_logvar)
        y_logp = decoder_dist.log_prob(y.squeeze())

        if is_deconfounding:
            pred = -torch.log(torch.tensor(n_samples)) + torch.logsumexp(prior.log_prob(z) - posterior.log_prob(z) + y_logp, 0)
        else:
            pred = -torch.log(torch.tensor(n_samples)) + torch.logsumexp(y_logp, 0)
        log_likelihood += pred.item()
    log_likelihoods.append(log_likelihood)
print(f"{np.mean(log_likelihoods):.3f} +/- {np.std(log_likelihoods):.3f}")