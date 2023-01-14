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

def make_raw_data(seed, n_examples, data_dim, is_spurious, s_shift):
    rng = np.random.RandomState(seed)
    if data_dim == 1:
        u = rng.normal(loc=0, scale=1, size=n_examples).astype("float32")
        x0_noise = rng.normal(loc=0, scale=0.01, size=n_examples).astype("float32")
        x1_noise = rng.normal(loc=0, scale=0.01, size=n_examples).astype("float32")
    else:
        u = rng.multivariate_normal(mean=np.zeros(data_dim), cov=np.diag(np.repeat(1 ** 2, data_dim)),
            size=n_examples).astype("float32")
        x0_noise = rng.multivariate_normal(mean=np.zeros(data_dim), cov=np.diag(np.repeat(0.01 ** 2, data_dim)),
            size=n_examples).astype("float32")
        x1_noise = rng.multivariate_normal(mean=np.zeros(data_dim), cov=np.diag(np.repeat(0.01 ** 2, data_dim)),
            size=n_examples).astype("float32")
    y_noise = rng.normal(loc=0, scale=5, size=n_examples).astype("float32")
    x0 = u + x0_noise
    x1 = u**2 + x1_noise
    y = row_mean(x0 + x1) + y_noise
    if is_spurious:
        prob = sigmoid(row_mean(u) * y, s_shift) # This doesn't work with + instead of *, why?
        s = rng.binomial(1, prob)
        idxs = np.where(s == 1)[0]
        x0, x1, y, u = x0[idxs], x1[idxs], y[idxs], u[idxs]
    return np.c_[x0, x1], y

def make_data(seed, n_examples, train_ratio, s_shift, data_dim, batch_size, n_workers):
    n_trainval, n_test = n_examples
    x_trainval, y_trainval = make_raw_data(seed, n_trainval, data_dim, True, s_shift)
    n_train = int(len(x_trainval) * train_ratio)
    x_train, y_train = x_trainval[:n_train], y_trainval[:n_train]
    x_val, y_val = x_trainval[n_train:], y_trainval[n_train:]
    x_test, y_test = make_raw_data(2 ** 32 - 1, n_test, data_dim, False, s_shift)

    x_train, x_val, x_test = to_torch(*normalize(x_train, x_val, x_test))
    y_train, y_val, y_test = to_torch(y_train, y_val, y_test)

    data_train = make_dataloader((x_train, y_train), batch_size, n_workers, True)
    data_val = make_dataloader((x_val, y_val), batch_size, n_workers, False)
    data_test = make_dataloader((x_test, y_test), 1, n_workers, False)
    return data_train, data_val, data_test