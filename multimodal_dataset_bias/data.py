import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.stats import row_mean


def sigmoid(x, shift):
    return 1 / (1 + np.exp(-(x - shift)))


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


def make_raw_data(seed, data_dim, n_trainval, n_test, train_ratio, swap_ratio):
    n_total = n_trainval + n_test
    rng = np.random.RandomState(seed)
    if data_dim == 1:
        u = rng.normal(loc=0, scale=1, size=n_total).astype("float32")
        x0_noise = rng.normal(loc=0, scale=0.01, size=n_total).astype("float32")
        x1_noise = rng.normal(loc=0, scale=0.01, size=n_total).astype("float32")
    else:
        u = rng.multivariate_normal(mean=np.zeros(data_dim), cov=np.diag(np.repeat(1 ** 2, data_dim)),
            size=n_total).astype("float32")
        x0_noise = rng.multivariate_normal(mean=np.zeros(data_dim), cov=np.diag(np.repeat(0.01 ** 2, data_dim)),
            size=n_total).astype("float32")
        x1_noise = rng.multivariate_normal(mean=np.zeros(data_dim), cov=np.diag(np.repeat(0.01 ** 2, data_dim)),
            size=n_total).astype("float32")
    y_noise = rng.normal(loc=0, scale=5, size=n_total).astype("float32")
    x0 = u + x0_noise
    x1 = u ** 2 + x1_noise
    x = np.c_[x0, x1]
    y = row_mean(x0 + x1) + y_noise

    if swap_ratio is None:
        trainval_idxs = rng.choice(n_total, n_trainval, replace=False)
        test_idxs = np.setdiff1d(np.arange(n_total), trainval_idxs)
    else:
        collider = row_mean(u) * y
        sorted_idxs = np.argsort(collider)
        trainval_idxs = sorted_idxs[:n_trainval]
        test_idxs = sorted_idxs[n_trainval:]

        n_swap = min(int(swap_ratio * n_trainval), n_test)
        trainval_swap_idxs = rng.choice(n_trainval, n_swap, replace=False)
        test_swap_idxs = rng.choice(n_test, n_swap, replace=False)
        trainval_swap_copy = trainval_idxs[trainval_swap_idxs].copy()
        trainval_idxs[trainval_swap_idxs] = test_idxs[test_swap_idxs].copy()
        test_idxs[test_swap_idxs] = trainval_swap_copy
    train_idxs = rng.choice(trainval_idxs, int(train_ratio * n_trainval), replace=False)
    val_idxs = np.setdiff1d(trainval_idxs, train_idxs)
    x_train, y_train = x[train_idxs], y[train_idxs]
    x_val, y_val = x[val_idxs], y[val_idxs]
    x_test, y_test = x[test_idxs], y[test_idxs]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def make_data(seed, data_dim, n_trainval, n_test, train_ratio, swap_ratio, batch_size, n_workers):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = make_raw_data(seed, data_dim, n_trainval, n_test,
        train_ratio, swap_ratio)

    x_train, x_val, x_test = to_torch(*normalize(x_train, x_val, x_test))
    y_train, y_val, y_test = to_torch(y_train, y_val, y_test)

    data_train = make_dataloader((x_train, y_train), batch_size, n_workers, True)
    data_val = make_dataloader((x_val, y_val), batch_size, n_workers, False)
    data_test = make_dataloader((x_test, y_test), 1, n_workers, False)
    return data_train, data_val, data_test