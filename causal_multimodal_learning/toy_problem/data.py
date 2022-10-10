import numpy as np
import torch
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

def make_dataset(seed, n_examples, data_dim, u_mult):
    rng = np.random.RandomState(seed)
    if data_dim == 1:
        u = rng.normal(size=n_examples).astype("float32")
        x0_noise = rng.normal(size=n_examples).astype("float32")
        x1_noise = rng.normal(size=n_examples).astype("float32")
        y_noise = rng.normal(loc=0, scale=0.1, size=n_examples).astype("float32")
    else:
        u = rng.multivariate_normal(mean=np.zeros(data_dim), cov=np.eye(data_dim), size=n_examples).astype("float32")
        x0_noise = rng.multivariate_normal(mean=np.zeros(data_dim), cov=np.eye(data_dim), size=n_examples).astype("float32")
        x1_noise = rng.multivariate_normal(mean=np.zeros(data_dim), cov=np.eye(data_dim), size=n_examples).astype("float32")
        y_noise = rng.multivariate_normal(mean=np.zeros(data_dim), cov=np.diag(np.repeat(0.1**2, data_dim)),
            size=n_examples).astype("float32")
    x0 = u + x0_noise
    x1 = u**2 + x1_noise
    y = x0 + x1 + u_mult * u + y_noise
    return x0, x1, y

def make_data(seed, n_examples, data_dim, u_mult, batch_size, n_workers):
    n_train, n_val, n_test = n_examples
    x0, x1, y = make_dataset(seed, sum(n_examples), data_dim, u_mult)
    x0_train, x1_train, y_train = x0[:n_train], x1[:n_train], y[:n_train]
    x0_val, x1_val, y_val = x0[n_train:(n_train + n_val)], x1[n_train:(n_train + n_val)], y[n_train:(n_train + n_val)]
    x0_test, x1_test, y_test = x0[n_train + n_val:], x1[n_train + n_val:], y[n_train + n_val:]

    x0_train, x0_val, x0_test = normalize(x0_train, x0_val, x0_test)
    x1_train, x1_val, x1_test = normalize(x1_train, x1_val, x1_test)

    x0_train, x1_train = to_torch(x0_train, x1_train)
    x0_val, x1_val = to_torch(x0_val, x1_val)
    x0_test, x1_test = to_torch(x0_test, x1_test)
    y_train, y_val, y_test = to_torch(y_train, y_val, y_test)

    data_train = make_dataloader((x0_train, x1_train, y_train), batch_size, n_workers, True)
    data_val = make_dataloader((x0_val, x1_val, y_val), batch_size, n_workers, False)
    data_test = make_dataloader((x0_test, x1_test, y_test), batch_size, n_workers, False)
    return data_train, data_val, data_test