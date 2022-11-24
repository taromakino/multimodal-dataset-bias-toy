import numpy as np
import torch
from scipy.special import expit as sigmoid
from torch.utils.data import DataLoader, TensorDataset

def normalize(x_train, x_val, x_test):
    x_mean, x_sd = x_train.mean(axis=0), x_train.std(axis=0)
    x_train = (x_train - x_mean) / x_sd
    x_val = (x_val - x_mean) / x_sd
    x_test = (x_test - x_mean) / x_sd
    return x_train, x_val, x_test

def to_torch(*arrs):
    result = [torch.tensor(arr)[:, None] if len(arr.shape)== 1 else torch.tensor(arr) for arr in arrs]
    if len(result) == 1:
        return result[0]
    else:
        return result

def make_dataloader(data_tuple, batch_size, n_workers, is_train):
    return DataLoader(TensorDataset(*data_tuple), shuffle=is_train, batch_size=batch_size, num_workers=n_workers,
        pin_memory=True, persistent_workers=True)

def make_dataset(seed, n_examples, data_dim, is_spurious):
    rng = np.random.RandomState(seed)
    if data_dim == 1:
        u = rng.normal(size=n_examples).astype("float32")
        x0_noise = rng.normal(size=n_examples).astype("float32")
        x1_noise = rng.normal(size=n_examples).astype("float32")
        y_noise = rng.normal(size=n_examples).astype("float32")
    else:
        u = rng.multivariate_normal(mean=np.zeros(data_dim), cov=np.eye(data_dim), size=n_examples).astype("float32")
        x0_noise = rng.multivariate_normal(mean=np.zeros(data_dim), cov=np.eye(data_dim), size=n_examples).astype("float32")
        x1_noise = rng.multivariate_normal(mean=np.zeros(data_dim), cov=np.eye(data_dim), size=n_examples).astype("float32")
        y_noise = rng.multivariate_normal(mean=np.zeros(data_dim), cov=np.eye(data_dim), size=n_examples).astype("float32")
    x0 = u + x0_noise
    x1 = u**2 + x1_noise
    y = x0 + x1 + y_noise
    if is_spurious:
        p = (u * y)
        if len(p.shape) > 1:
            p = p.sum(axis=1)
        p = sigmoid(p)
        v = rng.binomial(1, p)
        idxs = np.where(v == 1)[0]
        x0, x1, y, u = x0[idxs], x1[idxs], y[idxs], u[idxs]
    return x0, x1, y, u

def make_data(seed, n_examples, train_ratio, data_dim, batch_size, n_workers):
    n_trainval, n_test = n_examples
    x0_trainval, x1_trainval, y_trainval, u_trainval = make_dataset(seed, n_trainval, data_dim, True)
    n_train = int(len(x0_trainval) * train_ratio)
    x0_train, x1_train, y_train, u_train = x0_trainval[:n_train], x1_trainval[:n_train], y_trainval[:n_train], \
        u_trainval[:n_train]
    x0_val, x1_val, y_val, u_val = x0_trainval[n_train:], x1_trainval[n_train:], y_trainval[n_train:], \
        u_trainval[n_train:]
    x0_test, x1_test, y_test, u_test = make_dataset(2**32 - 1, n_test, data_dim, False)

    x0_train, x0_val, x0_test = normalize(x0_train, x0_val, x0_test)
    x1_train, x1_val, x1_test = normalize(x1_train, x1_val, x1_test)

    x0_train, x1_train = to_torch(x0_train, x1_train)
    x0_val, x1_val = to_torch(x0_val, x1_val)
    x0_test, x1_test = to_torch(x0_test, x1_test)
    y_train, y_val, y_test = to_torch(y_train, y_val, y_test)

    data_train = make_dataloader((x0_train, x1_train, y_train), batch_size, n_workers, True)
    data_val = make_dataloader((x0_val, x1_val, y_val), 1, n_workers, False)
    data_test = make_dataloader((x0_test, x1_test, y_test), 1, n_workers, False)
    return data_train, data_val, data_test