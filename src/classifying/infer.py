import numpy as np
import pandas as pd
import tensorflow as tf

def predict_image(image_path):
    model = tf.keras.models.load_model(r"/checkpoints/ckpt_classifier_full.keras", compile=False)

    labels = pd.read_csv(r"/data/processed/manifest_train_and_val.csv")["derived_label"].astype(str).dropna().unique().tolist()

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
    predict_image(r"/data/raw/images/12510.jpg")
