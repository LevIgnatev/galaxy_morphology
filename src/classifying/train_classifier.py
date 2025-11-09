# imports
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
from keras.src.callbacks import EarlyStopping
from pathlib import Path
import os

from data_loader import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "labels"

labels_and_paths_csv_fp_sample = DATA_PATH / "labels_manifest_1000.csv" # sample labels filepath
labels_and_paths_csv_fp_full = PROJECT_ROOT / "data" / "processed" / "manifest_train_and_val.csv" # full labels filepath
full_dataset_present = True

np.random.seed(5629) # random seed for reproducibility
tf.random.set_seed(5629)

inputs = tf.keras.Input((224,224,3))
x = layers.RandomFlip(["horizontal", "vertical"])(inputs)
x = layers.RandomRotation(0.2)(x)
x = layers.RandomTranslation(0.1, 0.1)(x)
x = tf.keras.applications.resnet50.preprocess_input(x)

resnet = tf.keras.applications.ResNet50(
    include_top=False, weights="imagenet", pooling="avg"
)
resnet.trainable = False
x = resnet(x, training=False)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(5 if full_dataset_present else 4, activation="softmax")(x)
baseline_CNN_model = tf.keras.Model(inputs, outputs)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) # Adam optimizer as a superior for this task

baseline_CNN_model.compile( # compilation
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

train_sample, valid_sample = load_data(labels_and_paths_csv_fp_full) # data loading (see data_loader.py)

history = baseline_CNN_model.fit( # fitting
    train_sample,
    validation_data=valid_sample,
    epochs=1,
    #callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
)

SAVE_DIR = Path(os.getenv("OUT_DIR", Path.cwd() / "outputs"))
SAVE_DIR.mkdir(parents=True, exist_ok=True)

baseline_CNN_model.save_weights(SAVE_DIR / "ckpt_classifier_full_small.keras")

hist = history.history

# plotting time!
plt.plot(hist['loss'], label='Train Loss')
plt.plot(hist['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

plt.plot(hist['accuracy'], label='Train Accuracy')
plt.plot(hist['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
