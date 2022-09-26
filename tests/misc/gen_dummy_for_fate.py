import numpy as np
import pandas as pd


n_samples = 10000
ids = np.arange(n_samples, dtype=int)
ys = np.zeros(n_samples, dtype=int)
active_train = np.stack([ids, ys, ys]).T
passive_train = np.stack([ids, ys]).T

active_train = pd.DataFrame(active_train, columns=["id", "y", "x0"])
passive_train = pd.DataFrame(passive_train, columns=["id", "x0"])
active_train.to_csv("dummy_active_full.csv", index=False)
passive_train.to_csv("dummy_passive_full.csv", index=False)
