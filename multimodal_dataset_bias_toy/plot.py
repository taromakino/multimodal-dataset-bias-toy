import numpy as np
import os
from data import make_raw_data
from utils.plot import *


for name, sample_size in zip(["small", "large"], [100, 1600]):
    rng = np.random.RandomState(1)
    input_dim = 1
    origin_offset = 1.5
    x_sd = 0.1
    temperature = 100

    u, x, y = make_raw_data(rng, sample_size, input_dim, origin_offset, x_sd, temperature)
    neg_idxs = np.where(y == 0)
    pos_idxs = np.where(y == 1)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(x[neg_idxs, 0], x[neg_idxs, 1], color="red", label=r"$Y = 0$")
    ax.scatter(x[pos_idxs, 0], x[pos_idxs, 1], color="blue", label=r"$Y = 1$")
    ax.axhline(0, color="black")
    ax.axvline(0, color="black")
    ax.set_xlabel(r"$X$")
    ax.set_ylabel(r"$X'$")
    ax.axis("equal")
    ax.grid(alpha=0.5, linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=[0.5, 0])
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35)
    os.makedirs("fig", exist_ok=True)
    plt.savefig(f"fig/toy_problem,{name}_dataset.pdf")