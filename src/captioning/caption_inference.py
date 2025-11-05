from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "labels"
config_fp = DATA_PATH / "captions" / "config.json"
vocab_fp = DATA_PATH / "captions" / "vocab.json"
captioner_weights_fp = PROJECT_ROOT / "checkpoints" / "captioner_model.keras"
image_fp = PROJECT_ROOT / "sample_images" / "42.jpg"

config = json.load(open(config_fp))
vocab = json.load(open(vocab_fp))
vocab_size = len(vocab)

inferred_captioner = tf.keras.models.load_model(captioner_weights_fp)

image = tf.io.read_file(str(image_fp))
image = tf.image.decode_image(image, channels=3)
image.set_shape([None, None, 3])
image = tf.image.resize(image, [224, 224])
image = tf.cast(image, tf.float32)
image.set_shape((224, 224, 3))
image = tf.expand_dims(image, 0)

#Start sequence
seq = np.full((1, config["max_len"]), config["pad_id"], dtype=np.int32)
seq[0, 0] = config["bos_id"]

for i in range(config["max_len"] - 1):
    # Infer the next token
    seq_as_a_tensor = tf.convert_to_tensor(seq, dtype=tf.int32)
    output_logits = inferred_captioner([image, seq_as_a_tensor], training=False)
    token_id = int(np.argmax(output_logits[0, i, :]))
    seq[0, i + 1] = token_id
    if token_id == config["eos_id"]: break

tokens = [vocab[i] for i in seq[0, :]]
caption = " ".join(tokens)

print(caption)
