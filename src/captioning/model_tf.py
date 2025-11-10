from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
import json
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "labels"
config_fp = DATA_PATH / "captions" / "config.json"
vocab_fp = DATA_PATH / "captions" / "vocab.json"
resnet_weights_fp = PROJECT_ROOT / "checkpoints" / "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

from build_dataset import dataset

np.random.seed(5629) # random seed for reproducibility
tf.random.set_seed(5629)

config = json.load(open(config_fp))
vocab = json.load(open(vocab_fp))
vocab_size = len(vocab)

# Encoder
inputs = tf.keras.Input((224,224,3))

x = layers.RandomFlip(["horizontal", "vertical"])(inputs)
x = layers.RandomRotation(0.2)(x)
x = layers.RandomTranslation(0.1, 0.1)(x)
x = tf.keras.applications.resnet50.preprocess_input(x)

resnet = tf.keras.applications.ResNet50(
    include_top=False, weights=resnet_weights_fp, pooling="avg"
)
resnet.trainable = False
x = resnet(x, training=False)
img_emb = layers.Dense(256, activation="relu")(x)

h0 = layers.Dense(units=256, activation='tanh')(img_emb)
c0 = layers.Dense(units=256, activation='tanh')(img_emb)

# Decoder
token_input = layers.Input((config['max_len'],), dtype='int32')

token_embedding = layers.Embedding(vocab_size, 256, mask_zero=True)(token_input)
LSTM_out = layers.LSTM(256, return_sequences=True)(token_embedding, initial_state=[h0, c0])
dropped = layers.Dropout(0.3)(LSTM_out)
logits = layers.TimeDistributed(layers.Dense(vocab_size))(dropped)


captioner_model = tf.keras.Model(inputs=[inputs, token_input], outputs=logits)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

captioner_model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

train_ds, valid_ds = dataset()

history = captioner_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=1,
    #callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
)
hist = history.history

SAVE_DIR = Path(os.getenv("OUT_DIR", Path.cwd() / "outputs"))
SAVE_DIR.mkdir(parents=True, exist_ok=True)

captioner_model.save_weights(SAVE_DIR / "captioner_model1.h5")

#Plotting time!
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
