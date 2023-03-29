import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import expit as sigmoid


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


def make_raw_data(rng, sample_size, input_dim, temperature):
    u_dim = 2 * input_dim
    u = rng.multivariate_normal(mean=np.zeros(u_dim), cov=np.eye(u_dim), size=sample_size)
    eps_x0 = rng.multivariate_normal(mean=np.zeros(input_dim), cov=make_isotropic_cov(input_dim, 0.1), size=sample_size)
    eps_x1 = rng.multivariate_normal(mean=np.zeros(input_dim), cov=make_isotropic_cov(input_dim, 0.1), size=sample_size)
    x0_idxs = rng.choice(u_dim, input_dim, replace=False)
    x1_idxs = np.setdiff1d(np.arange(u_dim), x0_idxs)
    origin_offset = abs(rng.multivariate_normal(mean=1.5 * np.ones(input_dim), cov=make_isotropic_cov(input_dim, 0.1),
        size=sample_size))
    x0 = u[:, x0_idxs] + origin_offset + eps_x0
    x1 = u[:, x1_idxs] + eps_x1
    x = np.c_[x0, x1]
    y = rng.binomial(1, sigmoid(temperature * (x0 * x1).sum(axis=1)))
    return u.astype("float32"), x.astype("float32"), y.astype("float32")


def make_data(seed, n_examples, input_dim, temperature, is_normalize, is_include_u, batch_size, n_workers):
    rng = np.random.RandomState(seed)
    n_train, n_val, _ = n_examples
    u, x, y = make_raw_data(rng, sum(n_examples), input_dim, temperature)
    u_train, x_train, y_train = u[:n_train], x[:n_train], y[:n_train]
    u_val, x_val, y_val = u[n_train:n_train+n_val], x[n_train:n_train+n_val], y[n_train:n_train+n_val]
    u_test, x_test, y_test = u[n_train+n_val:], x[n_train+n_val:], y[n_train+n_val:]

    if is_normalize:
        u_train, u_val, u_test = normalize(u_train, u_val, u_test)
        x_train, x_val, x_test = normalize(x_train, x_val, x_test)

    u_train, u_val, u_test = to_torch(u_train, u_val, u_test)
    x_train, x_val, x_test = to_torch(x_train, x_val, x_test)
    y_train, y_val, y_test = to_torch(y_train, y_val, y_test)

    if is_include_u:
        data_train = make_dataloader((u_train, x_train, y_train), batch_size, n_workers, True)
        data_val = make_dataloader((u_val, x_val, y_val), batch_size, n_workers, False)
        data_test = make_dataloader((u_test, x_test, y_test), batch_size, n_workers, False)
    else:
        data_train = make_dataloader((x_train, y_train), batch_size, n_workers, True)
        data_val = make_dataloader((x_val, y_val), batch_size, n_workers, False)
        data_test = make_dataloader((x_test, y_test), batch_size, n_workers, False)
    return data_train, data_val, data_test