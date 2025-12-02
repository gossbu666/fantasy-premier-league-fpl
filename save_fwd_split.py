#!/usr/bin/env python3
"""
Save a fixed temporal train/test split for FWD based on gameweek (round).
Use this split consistently for all FWD experiments.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_csv("data/processed/FWD_features_enhanced.csv")

X = df.drop(columns=["target"])
y = df["target"]

gss = GroupShuffleSplit(n_splits=1, test_size=0.366, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df["round"]))

np.save("analysis/FWD_train_idx.npy", train_idx)
np.save("analysis/FWD_test_idx.npy", test_idx)

print(f"Saved FWD indices: train={len(train_idx)}, test={len(test_idx)}")
