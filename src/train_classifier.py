# imports
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
from keras.src.applications.convnext import preprocess_input
from keras.src.callbacks import EarlyStopping
from prompt_toolkit.key_binding.bindings.named_commands import self_insert

from data_loader import load_data

labels_and_paths_csv_fp_PC_sample = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\labels\labels_manifest_1000.csv" # sample labels filepath (PC)
labels_and_paths_csv_fp_laptop_sample = r"C:\Users\79263\galaxy_morphology_ml_captioning\data\labels\labels_manifest_1000.csv" # sample labels filepath (laptop)
labels_and_paths_csv_fp_PC_full = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\processed\manifest_train_and_val.csv" # full labels filepath (PC)
labels_and_paths_csv_fp_laptop_full = r"C:\Users\79263\galaxy_morphology_ml_captioning\data\processed\manifest_train_and_val.csv" # full labels filepath (laptop)
full_dataset_present = False

np.random.seed(5629) # random seed for reproducibility
tf.random.set_seed(5629)


baseline_CNN_model = tf.keras.models.Sequential([ # baseline CNN model (input -> resnet50 -> convolution and max pooling layers -> dense layers with relu and softmax
    layers.InputLayer(input_shape=(224,224,3)),

    layers.RandomFlip(["horizontal", "vertical"]),
    layers.RandomRotation(0.2),
    layers.RandomTranslation(0.1, 0.1),

    layers.Lambda(tf.keras.applications.resnet50.preprocess_input, name="resnet_preproc"),

    tf.keras.applications.ResNet50(include_top=False, weights='imagenet', pooling='avg'),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(5 if full_dataset_present else 4, activation='softmax')
])
baseline_CNN_model.layers[5].trainable = False

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) # Adam optimizer as a superior for this task

baseline_CNN_model.compile( # compilation
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

train_sample, valid_sample = load_data(labels_and_paths_csv_fp_PC_sample, "PC") # data loading (see data_loader.py)

history = baseline_CNN_model.fit( # fitting
    train_sample,
    validation_data=valid_sample,
    epochs=5,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
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
