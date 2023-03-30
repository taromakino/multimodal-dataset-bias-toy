import numpy as np
from data import make_raw_data
from utils.plot import *


rng = np.random.RandomState(1)
input_dim = 1
sample_size = 1600
origin_offset = 1.5
temperature = 100

u, x, y = make_raw_data(rng, sample_size, input_dim, origin_offset, temperature)
neg_idxs = np.where(y == 0)
pos_idxs = np.where(y == 1)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.scatter(x[neg_idxs, 0], x[neg_idxs, 1], color="red", label=r"$Y = 0$")
ax.scatter(x[pos_idxs, 0], x[pos_idxs, 1], color="blue", label=r"$Y = 1$")
ax.axhline(0, color="gray", linestyle="--")
ax.axvline(0, color="gray", linestyle="--")
ax.set_xlabel(r"$X$")
ax.set_ylabel(r"$X'$")
ax.axis("equal")

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=[0.5, 0])
fig.tight_layout()
fig.subplots_adjust(bottom=0.3)