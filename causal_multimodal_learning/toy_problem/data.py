import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def split_data(trainval_ratios, *arrays):
    n_train, n_val = [int(len(arrays[0]) * split_ratio) for split_ratio in trainval_ratios]
    arrays_train = [array[:n_train] for array in arrays]
    arrays_val = [array[n_train:n_train + n_val] for array in arrays]
    if sum(trainval_ratios) == 1:
        return arrays_train, arrays_val
    else:
        arrays_test = [array[n_train + n_val:] for array in arrays]
        return arrays_train, arrays_val, arrays_test

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

def make_dataloader(data_tuple, batch_size, is_train):
    return DataLoader(TensorDataset(*data_tuple), batch_size=batch_size, shuffle=is_train)

def make_dataset(seed, n_examples, data_dim, u_mult):
    rng = np.random.RandomState(seed)
    if data_dim == 1:
        u = rng.normal(size=n_examples).astype("float32")
        x0_noise = rng.normal(size=n_examples).astype("float32")
        x1_noise = rng.normal(size=n_examples).astype("float32")
        y_noise = u_mult * u + rng.normal(loc=0, scale=0.1, size=n_examples).astype("float32")
    else:
        u = rng.multivariate_normal(mean=np.zeros(data_dim), cov=np.eye(data_dim), size=n_examples).astype("float32")
        x0_noise = rng.multivariate_normal(mean=np.zeros(data_dim), cov=np.eye(data_dim), size=n_examples).astype("float32")
        x1_noise = rng.multivariate_normal(mean=np.zeros(data_dim), cov=np.eye(data_dim), size=n_examples).astype("float32")
        y_noise = u_mult * u + rng.multivariate_normal(mean=np.zeros(data_dim), cov=np.diag(np.repeat(0.1, data_dim)),
            size=n_examples).astype("float32")
    x0 = u + x0_noise
    x1 = u**2 + x1_noise
    y = x0 + x1 + y_noise
    return x0, x1, y

def make_data(seed, n_examples, data_dim, u_mult, trainval_ratios, batch_size):
    x0, x1, y = make_dataset(seed, n_examples, data_dim, u_mult)
    (x0_train, x1_train, y_train), (x0_val, x1_val, y_val), (x0_test, x1_test, y_test) = \
        split_data(trainval_ratios, x0, x1, y)

    x0_train, x0_val, x0_test = normalize(x0_train, x0_val, x0_test)
    x1_train, x1_val, x1_test = normalize(x1_train, x1_val, x1_test)

    x0_train, x1_train = to_torch(x0_train, x1_train)
    x0_val, x1_val = to_torch(x0_val, x1_val)
    x0_test, x1_test = to_torch(x0_test, x1_test)
    y_train, y_val, y_test = to_torch(y_train, y_val, y_test)

    data_train = make_dataloader((x0_train, x1_train, y_train), batch_size, True)
    data_val = make_dataloader((x0_val, x1_val, y_val), batch_size, False)
    data_test = make_dataloader((x0_test, x1_test, y_test), batch_size, False)
    return data_train, data_val, data_test