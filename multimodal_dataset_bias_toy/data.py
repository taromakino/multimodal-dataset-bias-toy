import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.stats import row_mean


def sigmoid(x, shift):
    return 1 / (1 + np.exp(-(x - shift)))


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


def make_standard_data(rng, data_dim, n_examples, u_sd, x_sd, y_sd):
    u = rng.normal(loc=0, scale=u_sd, size=n_examples)
    x0_noise = rng.multivariate_normal(mean=np.zeros(data_dim), cov=make_isotropic_cov(data_dim, x_sd), size=n_examples)
    x1_noise = rng.multivariate_normal(mean=np.zeros(data_dim), cov=make_isotropic_cov(data_dim, x_sd), size=n_examples)
    y_noise = rng.normal(loc=0, scale=y_sd, size=n_examples)
    x0 = u[:, None] * np.ones_like(x0_noise) + x0_noise
    x1 = u[:, None] ** 2 * np.ones_like(x1_noise) + x1_noise
    x = np.c_[x0, x1]
    y = row_mean(x0 + x1) + y_noise
    return u.astype("float32"), x.astype("float32"), y.astype("float32"), y_noise.astype("float32")


def make_selection_biased_data(rng, data_dim, n_examples, u_sd, x_sd, y_sd, s_shift):
    u_all, x_all, y_all = [], [], []
    count = 0
    while count < n_examples:
        u, x, y, y_noise = make_standard_data(rng, data_dim, n_examples, u_sd, x_sd, y_sd)
        collider = u * y_noise
        collider = (collider - collider.mean()) / collider.std()
        prob = sigmoid(collider, s_shift)
        s = rng.binomial(1, prob)
        idxs = np.where(s == 1)[0]
        u_all.append(u[idxs])
        x_all.append(x[idxs])
        y_all.append(y[idxs])
        count += len(idxs)
    u_all = np.concatenate(u_all)[:n_examples]
    x_all = np.concatenate(x_all)[:n_examples]
    y_all = np.concatenate(y_all)[:n_examples]
    return u_all, x_all, y_all


def make_data(seed, data_dim, n_examples, u_sd, x_sd, y_sd, s_shift, normalize, include_u, batch_size, n_workers):
    n_train, n_val, n_test = n_examples
    n_trainval = n_train + n_val
    rng = np.random.RandomState(seed)
    if s_shift is None:
        u_trainval, x_trainval, y_trainval, _ = make_standard_data(rng, data_dim, n_trainval, u_sd, x_sd, y_sd)
    else:
        u_trainval, x_trainval, y_trainval = make_selection_biased_data(rng, data_dim, n_trainval, u_sd, x_sd, y_sd, s_shift)
    u_test, x_test, y_test, _ = make_standard_data(np.random.RandomState(2 ** 32 - 1), data_dim, n_test, u_sd, x_sd, y_sd)

    u_train, x_train, y_train = u_trainval[:n_train], x_trainval[:n_train], y_trainval[:n_train]
    u_val, x_val, y_val = u_trainval[n_train:], x_trainval[n_train:], y_trainval[n_train:]

    if normalize:
        u_train, u_val, u_test = normalize(u_train, u_val, u_test)
        x_train, x_val, x_test = normalize(x_train, x_val, x_test)

    u_train, u_val, u_test = to_torch(u_train, u_val, u_test)
    x_train, x_val, x_test = to_torch(x_train, x_val, x_test)
    y_train, y_val, y_test = to_torch(y_train, y_val, y_test)

    if include_u:
        data_train = make_dataloader((u_train, x_train, y_train), batch_size, n_workers, True)
        data_val = make_dataloader((u_val, x_val, y_val), batch_size, n_workers, False)
        data_test = make_dataloader((u_test, x_test, y_test), batch_size, n_workers, False)
    else:
        data_train = make_dataloader((x_train, y_train), batch_size, n_workers, True)
        data_val = make_dataloader((x_val, y_val), batch_size, n_workers, False)
        data_test = make_dataloader((x_test, y_test), batch_size, n_workers, False)
    return data_train, data_val, data_test