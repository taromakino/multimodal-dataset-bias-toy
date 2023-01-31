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


def make_standard_data(rng, data_dim, n_examples):
    u = rng.multivariate_normal(mean=np.zeros(data_dim), cov=make_isotropic_cov(data_dim, 0.1), size=n_examples)
    x0_noise = rng.multivariate_normal(mean=np.zeros(data_dim), cov=make_isotropic_cov(data_dim, 0.01), size=n_examples)
    x1_noise = rng.multivariate_normal(mean=np.zeros(data_dim), cov=make_isotropic_cov(data_dim, 0.01), size=n_examples)
    y_noise = rng.normal(loc=0, scale=1, size=n_examples)
    x0 = u + x0_noise
    x1 = u ** 2 + x1_noise
    x = np.c_[x0, x1]
    y = row_mean(x0 + x1) + y_noise
    return u.astype("float32"), x.astype("float32"), y.astype("float32")


def make_selection_biased_data(rng, data_dim, n_examples, s_shift):
    x_all, y_all = [], []
    count = 0
    while count < n_examples:
        u, x, y = make_standard_data(rng, data_dim, n_examples)
        collider = row_mean(u) * y
        collider = (collider - collider.mean()) / collider.std()
        prob = sigmoid(collider, s_shift)
        s = rng.binomial(1, prob)
        idxs = np.where(s == 1)[0]
        x_all.append(x[idxs])
        y_all.append(y[idxs])
        count += len(idxs)
    x_all = np.concatenate(x_all)[:n_examples]
    y_all = np.concatenate(y_all)[:n_examples]
    return x_all, y_all


def make_data(seed, data_dim, n_trainval, n_test, train_ratio, s_shift, batch_size, n_workers):
    if s_shift is None:
        _, x_trainval, y_trainval = make_standard_data(np.random.RandomState(seed), data_dim, n_trainval)
    else:
        x_trainval, y_trainval = make_selection_biased_data(np.random.RandomState(seed), data_dim, n_trainval, s_shift)
    _, x_test, y_test = make_standard_data(np.random.RandomState(2 ** 32 - 1), data_dim, n_test)

    n_train = int(len(x_trainval) * train_ratio)
    x_train, y_train = x_trainval[:n_train], y_trainval[:n_train]
    x_val, y_val = x_trainval[n_train:], y_trainval[n_train:]

    x_train, x_val, x_test = to_torch(*normalize(x_train, x_val, x_test))
    y_train, y_val, y_test = to_torch(y_train, y_val, y_test)

    data_train = make_dataloader((x_train, y_train), batch_size, n_workers, True)
    data_val = make_dataloader((x_val, y_val), batch_size, n_workers, False)
    data_test = make_dataloader((x_test, y_test), 1, n_workers, False)
    return data_train, data_val, data_test