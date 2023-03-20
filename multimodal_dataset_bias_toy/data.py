import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import expit as sigmoid
from utils.stats import row_mean


def make_isotropic_cov(dim, sd):
    return np.diag(np.repeat(sd ** 2, dim))


def normalize(x_train, x_val, x_test):
    x_mean, x_sd = x_train.mean(axis=0), x_train.std(axis=0)
    x_train = (x_train - x_mean) / x_sd
    x_val = (x_val - x_mean) / x_sd
    x_test = (x_test - x_mean) / x_sd
    return x_train, x_val, x_test


def to_torch(*arrs):
    out = [torch.tensor(arr)[:, None] if len(arr.shape)== 1 else torch.tensor(arr) for arr in arrs]
    if len(out) == 1:
        return out[0]
    else:
        return out


def make_dataloader(data_tuple, batch_size, n_workers, is_train):
    return DataLoader(TensorDataset(*data_tuple), shuffle=is_train, batch_size=batch_size, num_workers=n_workers,
        pin_memory=True, persistent_workers=True)


def make_raw_data(rng, input_dim, n_examples, u_sd, x_sd, y_sd):
    u = rng.normal(loc=0, scale=u_sd, size=n_examples)
    eps_x0 = rng.multivariate_normal(mean=np.zeros(input_dim), cov=make_isotropic_cov(input_dim, x_sd), size=n_examples)
    eps_x1 = rng.multivariate_normal(mean=np.zeros(input_dim), cov=make_isotropic_cov(input_dim, x_sd), size=n_examples)
    x0 = u[:, None] * np.ones_like(eps_x0) + eps_x0
    x1 = -u[:, None] * np.ones_like(eps_x1) + eps_x1
    x = np.c_[x0, x1]
    y = np.ones(n_examples)
    y[np.where(row_mean(x0) * row_mean(x1) > 0)] = 0
    return u.astype("float32"), x.astype("float32"), y.astype("float32")


def make_data(seed, input_dim, sample_size, split_ratios, u_sd, x_sd, y_sd, is_normalizing, is_including_u, batch_size, n_workers):
    rng = np.random.RandomState(seed)
    u, x, y = make_raw_data(rng, input_dim, sample_size, u_sd, x_sd, y_sd)
    train_ratio, val_ratio, test_ratio = split_ratios
    n_train, n_val = int(train_ratio * sample_size), int(val_ratio * sample_size)
    u_train, x_train, y_train = u[:n_train], x[:n_train], y[:n_train]
    u_val, x_val, y_val = u[n_train:n_train+n_val], x[n_train:n_train+n_val], y[n_train:n_train+n_val]
    u_test, x_test, y_test = u[n_train+n_val:], x[n_train+n_val:], y[n_train+n_val:]

    if is_normalizing:
        u_train, u_val, u_test = normalize(u_train, u_val, u_test)
        x_train, x_val, x_test = normalize(x_train, x_val, x_test)

    u_train, u_val, u_test = to_torch(u_train, u_val, u_test)
    x_train, x_val, x_test = to_torch(x_train, x_val, x_test)
    y_train, y_val, y_test = to_torch(y_train, y_val, y_test)

    if is_including_u:
        data_train = make_dataloader((u_train, x_train, y_train), batch_size, n_workers, True)
        data_val = make_dataloader((u_val, x_val, y_val), batch_size, n_workers, False)
        data_test = make_dataloader((u_test, x_test, y_test), batch_size, n_workers, False)
    else:
        data_train = make_dataloader((x_train, y_train), batch_size, n_workers, True)
        data_val = make_dataloader((x_val, y_val), batch_size, n_workers, False)
        data_test = make_dataloader((x_test, y_test), batch_size, n_workers, False)
    return data_train, data_val, data_test