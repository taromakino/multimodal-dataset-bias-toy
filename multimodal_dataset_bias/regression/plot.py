import numpy as np
import os
import pandas as pd
from utils.plot_settings import *

def test_loss(fpath):
    df = pd.read_csv(fpath)
    return df.test_loss.iloc[-1]

n_seeds = 5

for s_shift in (-4, 4):
    ratios = []
    for seed in range(n_seeds):
        unimodal_fpath = os.path.join("results", str(s_shift), "unimodal_ensemble", f"version_{seed}", "metrics.csv")
        unimodal_loss = test_loss(unimodal_fpath)
        mutimodal_fpath = os.path.join("results", str(s_shift), "multimodal", f"version_{seed}", "metrics.csv")
        multimodal_loss = test_loss(mutimodal_fpath)
        ratios.append(multimodal_loss / unimodal_loss)
    print(s_shift, np.mean(ratios))