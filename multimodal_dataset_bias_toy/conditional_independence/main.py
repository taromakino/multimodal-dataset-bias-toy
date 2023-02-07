import numpy as np
from conditional_independence import partial_correlation_suffstat, partial_correlation_test
from data import make_standard_data, make_selection_biased_data
from utils.plot_settings import *


def p_value(u, x, y):
    uyx = np.hstack((u, y[:, None], x))
    suffstat = partial_correlation_suffstat(uyx)
    return partial_correlation_test(suffstat, 0, 1, {2, 3})["p_value"]


data_dim = 1
n_seeds = 15
n_examples = 10000
s_shift_range = [-3, -2, -1, 0, 1]
n_examples_range = [1600, 800, 400, 200, 100]
u_sd = 1
x_sd = 1
y_sd = 1

fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.5))

means, sds = [], []
for s_shift in s_shift_range:
    values = []
    for seed in range(n_seeds):
        u, x, y = make_selection_biased_data(np.random.RandomState(seed), data_dim, n_examples, u_sd, x_sd, y_sd, s_shift)
        values.append(p_value(u, x, y))
    means.append(np.mean(values))
    sds.append(np.std(values))
axes[0].errorbar(range(len(means)), means, sds)
axes[0].set_xticks(range(len(s_shift_range)))
axes[0].set_xticklabels(s_shift_range)
axes[0].set_xlabel(r"$c$")
axes[0].set_ylabel("p-value")

means, sds = [], []
for n_examples in n_examples_range:
    values = []
    for seed in range(n_seeds):
        u, x, y = make_standard_data(np.random.RandomState(seed), data_dim, n_examples, u_sd, x_sd, y_sd)
        values.append(p_value(u, x, y))
    means.append(np.mean(values))
    sds.append(np.std(values))
axes[1].errorbar(range(len(means)), means, sds)
axes[1].set_xticks(range(len(n_examples_range)))
axes[1].set_xticklabels(n_examples_range)
axes[1].set_xlabel("Dataset size")
axes[1].set_ylabel("p-value")

fig.tight_layout()