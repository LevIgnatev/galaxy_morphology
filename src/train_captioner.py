import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
from data_loader import load_data

labels_and_paths_csv_fp_PC_sample = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\labels\labels_manifest_1000.csv" # sample labels filepath (PC)
labels_and_paths_csv_fp_laptop_sample = r"C:\Users\79263\galaxy_morphology_ml_captioning\data\labels\labels_manifest_1000.csv" # sample labels filepath (laptop)
labels_and_paths_csv_fp_PC_full = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\processed\manifest_train_and_val.csv" # full labels filepath (PC)
labels_and_paths_csv_fp_laptop_full = r"C:\Users\79263\galaxy_morphology_ml_captioning\data\processed\manifest_train_and_val.csv" # full labels filepath (laptop)
full_dataset_present = False

np.random.seed(5629) # random seed for reproducibility
tf.random.set_seed(5629)

