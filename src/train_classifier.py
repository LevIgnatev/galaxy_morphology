# imports
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
from data_loader import load_data

labels_and_paths_csv_fp_PC_sample = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\labels\labels_manifest_1000.csv" #labels filepath (PC)
labels_and_paths_csv_fp_laptop_sample = r"C:\Users\79263\galaxy_morphology_ml_captioning\data\labels\labels_manifest_1000.csv" #labels filepath (laptop)
labels_and_paths_csv_fp_PC_full = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\processed\manifest_train_and_val.csv"
labels_and_paths_csv_fp_laptop_full = r"C:\Users\79263\galaxy_morphology_ml_captioning\data\processed\manifest_train_and_val.csv"

np.random.seed(5629) # random seed for reproducibility
tf.random.set_seed(5629)

baseline_CNN_model = tf.keras.models.Sequential([ # baseline CNN model (input -> resnet50 -> convolution and max pooling layers -> dense layers with relu and softmax
    layers.InputLayer(input_shape=(224,224,3)),

    tf.keras.applications.ResNet50(include_top=False,),

    layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'),
    layers.MaxPool2D(pool_size=2, strides=2, padding='same'),

    layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', padding='same'),
    layers.MaxPool2D(pool_size=2, strides=2, padding='same'),

    layers.Conv2D(filters=256, kernel_size=3, strides=1, activation='relu', padding='same'),
    layers.Conv2D(filters=256, kernel_size=3, strides=1, activation='relu', padding='same'),
    layers.MaxPool2D(pool_size=2, strides=2, padding='same'),

    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(4, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(epsilon=0.001) # Adam optimizer as a superior for this task

baseline_CNN_model.compile( # compilation
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

train_sample, valid_sample = load_data(labels_and_paths_csv_fp_PC_sample, "PC") # data loading (see data_loader.py)

history = baseline_CNN_model.fit( # fitting
    train_sample,
    validation_data=valid_sample,
    epochs=1
)

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
