import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
labels_fp = r"[redacted]"
df = pd.read_csv(labels_fp)
labels_and_fp_cols = df[["derived_label","filepath"]]
print(labels_and_fp_cols)
train_df, valid_df = train_test_split(labels_and_fp_cols,
    test_size=0.2,
    stratify=labels_and_fp_cols["derived_label"],
    random_state=37)
labels_to_indexes = {}
for i in range(labels_and_fp_cols["derived_label"].nunique()):
    labels_to_indexes[labels_and_fp_cols["derived_label"].unique()[i]] = i
train_paths_list = train_df["filepath"].tolist()
valid_paths_list = valid_df["filepath"].tolist()
train_labels_list = []
for i in train_df["derived_label"]:
    train_labels_list.append(labels_to_indexes[i])
print(train_labels_list)