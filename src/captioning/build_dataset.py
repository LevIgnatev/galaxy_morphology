from pathlib import Path
import pandas as pd
import tensorflow as tf
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "labels"

all_captions_path = DATA_PATH / "captions" / "captions_sample.csv"
caption_train_path = DATA_PATH / "captions" /"train_captions_sample.txt"
caption_valid_path = DATA_PATH / "captions" /"val_captions_sample.txt"

config_fp = DATA_PATH / "captions" / "config.json"

config = json.load(open(config_fp))

from tokenizer import encode

def dataset():
    manifest = pd.read_csv(all_captions_path)
    train_ids = pd.read_csv(
        caption_train_path,
        header=None,
        names=["objid"],
        dtype=str,
        sep=r"\s+",
        engine="python",
    )
    valid_ids = pd.read_csv(
        caption_valid_path,
        header=None,
        names=["objid"],
        dtype=str,
        sep=r"\s+",
        engine="python",
    )
    df_train = manifest.merge(train_ids, on=["objid"], how="inner")
    df_valid = manifest.merge(valid_ids, on=["objid"], how="inner")

    train_paths_list = df_train["filepath"].tolist()
    train_captions_list = df_train["caption"].astype(str).tolist()
    valid_paths_list = df_valid["filepath"].tolist()
    valid_captions_list = df_valid["caption"].astype(str).tolist()

    def PREPROCESS(path, caption):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32)
        image.set_shape((224, 224, 3))

        cap_in, cap_out, mask = encode(caption)
        cap_in = tf.cast(cap_in, tf.int32)
        cap_out = tf.cast(cap_out, tf.int32)
        mask = tf.cast(mask, tf.float32)
        tf.ensure_shape(cap_in, (config['max_len'],))
        tf.ensure_shape(cap_out, (config['max_len'],))
        tf.ensure_shape(mask, (config['max_len'],))

        return image, cap_in, cap_out, mask

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths_list,
                                                   train_captions_list))
    train_ds = (
        train_ds.shuffle(buffer_size=len(train_paths_list), seed=37)
        .map(PREPROCESS, num_parallel_calls=AUTOTUNE)
        .map(lambda image, cap_in, cap_out, mask: ((image, cap_in), cap_out, mask))
        .batch(32)
        .prefetch(AUTOTUNE)
    )

    valid_ds = tf.data.Dataset.from_tensor_slices((valid_paths_list,
                                                   valid_captions_list))
    valid_ds = (
        valid_ds
        .map(PREPROCESS, num_parallel_calls=AUTOTUNE)
        .map(lambda image, cap_in, cap_out, mask: ((image, cap_in), cap_out, mask))
        .batch(32)
        .prefetch(AUTOTUNE)
    )

    return train_ds, valid_ds
