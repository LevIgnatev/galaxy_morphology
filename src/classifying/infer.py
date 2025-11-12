import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import tensorflow.keras.layers as layers

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def predict_image(image_path):
    classifier_weights_fp = PROJECT_ROOT / "checkpoints" / "ckpt_classifier_full.h5"

    inputs = tf.keras.Input((224, 224, 3))
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
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
    outputs = layers.Dense(5, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.load_weights(classifier_weights_fp)

    labels = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "manifest_train_and_val.csv")["derived_label"].astype(str).dropna().unique().tolist()

    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, (224, 224))
    x =  tf.cast(img, tf.float32)[None, ...]

    probs = np.asarray(model.predict(x, verbose=0)[0])

    top_idx = np.argsort(-probs)

    for rank, i in enumerate(top_idx, 1):
        name = labels[i]
        print(f"{rank}. {name:20s} probability = {probs[i]:.4f}")

if __name__ == "__main__":
    predict_image(str(PROJECT_ROOT / "sample_images" / "107682.jpg"))
