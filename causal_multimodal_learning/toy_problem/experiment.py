import numpy as np
import os
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from torch.optim import Adam
from utils.file import save_file, write
from utils.nn_utils import device, to_device
from utils.stats import log_avg_prob, make_gaussian, make_standard_normal
from toy_problem.data import make_data
from toy_problem.model import GenerativeModel, EncoderX, AggregatedPosterior

def train(data_train, data_val, model, lr, wd, patience):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
    optimal_val_epoch = 0
    optimal_val_loss = float("inf")
    optimal_weights_fpath = os.path.join(model.dpath, "optimal_weights.pt")
    epoch = 0
    while True:
        model.train()
        for x0, x1, y in data_train:
            x0, x1, y = to_device(x0, x1, y)
            loss_train = model(x0, x1, y)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
        model.eval()
        loss_val = []
        for x0, x1, y in data_val:
            x0, x1, y = to_device(x0, x1, y)
            loss_val.append(model(x0, x1, y).item())
        loss_val = np.mean(loss_val)
        write(os.path.join(model.dpath, "summary_val.txt"), str(loss_val))
        if loss_val < optimal_val_loss:
            optimal_val_loss = loss_val
            optimal_val_epoch = epoch
            torch.save(model.state_dict(), optimal_weights_fpath)
        if epoch - optimal_val_epoch == patience:
            break
        epoch += 1
    model.load_state_dict(torch.load(optimal_weights_fpath))

def main(args):
    seed = args.__dict__.pop("seed")
    save_file(args, os.path.join(args.dpath, "args.pkl"))
    dpath = os.path.join(args.dpath, f"s={seed}")
    os.makedirs(dpath, exist_ok=True)
    pl.seed_everything(seed)
    data_train, data_val, data_test = make_data(seed, args.n_examples, args.train_ratio, args.data_dim,
        args.batch_size, args.n_workers)
    model = GenerativeModel(dpath, args.data_dim, args.hidden_dims, args.latent_dim)
    model.to(device())
    train(data_train, data_val, model, args.lr, args.wd, args.patience)
    encoder_x = EncoderX(dpath, model.encoder_xy, args.data_dim, args.hidden_dims, args.latent_dim)
    encoder_x.to(device())
    train(data_train, data_val, encoder_x, args.lr, args.wd, args.patience)
    encoder_x = encoder_x.encoder_x # Clean this up
    model.eval()
    encoder_x.eval()
    replacement_dist = AggregatedPosterior(data_test, encoder_x) if args.is_aggregated_posterior else \
        make_standard_normal(1, args.latent_dim)
    conditional_logp = interventional_logp = 0
    for x0, x1, y in data_test:
        x0, x1, y = to_device(x0, x1, y)
        x0_rep = torch.repeat_interleave(x0, repeats=args.n_samples, dim=0)
        x1_rep = torch.repeat_interleave(x1, repeats=args.n_samples, dim=0)
        mu_x, logvar_x = encoder_x(x0, x1)
        posterior_x_dist = make_gaussian(mu_x, logvar_x)
        z = posterior_x_dist.sample((args.n_samples,))
        mu_reconst, logvar_reconst = model.decoder(x0_rep, x1_rep, z[:, None] if len(z.shape) == 1 else z)
        decoder_dist = make_gaussian(mu_reconst, logvar_reconst)
        logp_y_xz = decoder_dist.log_prob(y.squeeze())
        conditional_logp += log_avg_prob(logp_y_xz).item()
        interventional_logp += log_avg_prob(replacement_dist.log_prob(z) - posterior_x_dist.log_prob(z) + logp_y_xz).item()
    write(os.path.join(dpath, "summary_test.txt"), f"{conditional_logp},{interventional_logp}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_examples", nargs="+", type=int, default=[1000, 1000])
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--data_dim", type=int, default=1)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[20, 20])
    parser.add_argument("--latent_dim", type=int, default=10)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--n_workers", type=int, default=20)
    parser.add_argument("--is_aggregated_posterior", action="store_true")
    main(parser.parse_args())