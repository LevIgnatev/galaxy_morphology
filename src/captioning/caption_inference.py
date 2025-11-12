from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "labels"
config_fp = DATA_PATH / "captions" / "config.json"
vocab_fp = DATA_PATH / "captions" / "vocab.json"
captioner_weights_fp = PROJECT_ROOT / "checkpoints" / "captioner_model.h5"
resnet_weights_fp = PROJECT_ROOT / "checkpoints" / "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"


def generate_caption(image_fp):
    config = json.load(open(config_fp))
    vocab = json.load(open(vocab_fp))

    #-----------
    vocab_size = len(vocab)

    inputs = tf.keras.Input((224, 224, 3))

    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
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
    #-----------

    captioner_model.load_weights(captioner_weights_fp)

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
        output_logits = captioner_model([image, seq_as_a_tensor], training=False)
        token_id = int(np.argmax(output_logits[0, i, :]))
        seq[0, i + 1] = token_id
        if token_id == config["eos_id"]: break

    tokens = [vocab[i] for i in seq[0, :]]
    caption = " ".join(tokens)

    return caption.replace("<bos>", "").replace("<eos>", "").replace("<pad>", "").replace("a a ", "a ").strip()

print(generate_caption(PROJECT_ROOT / "sample_images" / "42.jpg"))
