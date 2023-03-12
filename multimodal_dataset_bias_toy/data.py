import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.stats import row_mean


def make_isotropic_cov(dim, sd):
    return np.diag(np.repeat(sd ** 2, dim))


def normalize(x_train, x_val):
    x_mean, x_sd = x_train.mean(axis=0), x_train.std(axis=0)
    x_train = (x_train - x_mean) / x_sd
    x_val = (x_val - x_mean) / x_sd
    return x_train, x_val


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
    eps_y = rng.normal(loc=0, scale=y_sd, size=n_examples)
    x0 = u[:, None] * np.ones_like(eps_x0) + eps_x0
    x1 = u[:, None] ** 2 * np.ones_like(eps_x1) + eps_x1
    x = np.c_[x0, x1]
    y = row_mean(x0 + x1) + eps_y
    return u.astype("float32"), x.astype("float32"), y.astype("float32"), eps_y.astype("float32")


def make_data(seed, input_dim, n_train, n_val, u_sd, x_sd, y_sd, is_normalizing, is_including_u, batch_size, n_workers):
    n_trainval = n_train + n_val
    rng = np.random.RandomState(seed)
    u_trainval, x_trainval, y_trainval, _ = make_raw_data(rng, input_dim, n_trainval, u_sd, x_sd, y_sd)

    u_train, x_train, y_train = u_trainval[:n_train], x_trainval[:n_train], y_trainval[:n_train]
    u_val, x_val, y_val = u_trainval[n_train:], x_trainval[n_train:], y_trainval[n_train:]

    if is_normalizing:
        u_train, u_val = normalize(u_train, u_val)
        x_train, x_val = normalize(x_train, x_val)

    u_train, u_val = to_torch(u_train, u_val)
    x_train, x_val = to_torch(x_train, x_val)
    y_train, y_val = to_torch(y_train, y_val)

    if is_including_u:
        data_train = make_dataloader((u_train, x_train, y_train), batch_size, n_workers, True)
        data_val = make_dataloader((u_val, x_val, y_val), batch_size, n_workers, False)
    else:
        data_train = make_dataloader((x_train, y_train), batch_size, n_workers, True)
        data_val = make_dataloader((x_val, y_val), batch_size, n_workers, False)
    return data_train, data_val