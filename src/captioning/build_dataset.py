from pathlib import Path
import pandas as pd
import tensorflow as tf
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "labels"

all_captions_path = DATA_PATH / "captions" / "captions_sample.csv"
caption_train_path = DATA_PATH / "captions" /"train_captions_sample.txt"
caption_valid_path = DATA_PATH / "captions" /"val_captions_sample.txt"

config_fp = DATA_PATH / "captions" / "config.json"

config = json.load(open(config_fp))

from tokenizer import encode

def dataset():
    
    def _to_posix_abs(p: str) -> str:
        p = str(p).replace("\\", "/")
        if not p.startswith("/"):
            p = (PROJECT_ROOT / p).as_posix()
        return p
    
    manifest = pd.read_csv(all_captions_path, dtype={"objid": str})
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
    fps = df_train["filepath"]
    train_paths_list = df_train["filepath"].astype(str).apply(lambda x: str(PROJECT_ROOT / x)).tolist()
    train_captions_list = df_train["caption"].astype(str).tolist()
    valid_paths_list = df_valid["filepath"].astype(str).apply(lambda x: str(PROJECT_ROOT / x)).tolist()
    valid_captions_list = df_valid["caption"].astype(str).tolist()

    train_captions_in = []
    train_captions_out = []
    train_captions_masks = []

    for cap in train_captions_list:
        cap_in, cap_out, msk = encode(cap)
        train_captions_in.append(cap_in)
        train_captions_out.append(cap_out)
        train_captions_masks.append(msk)

    train_captions_in = np.array(train_captions_in, dtype="int32")
    train_captions_out = np.array(train_captions_out, dtype="int32")
    train_captions_masks = np.array(train_captions_masks, dtype="float32")

    valid_captions_in = []
    valid_captions_out = []
    valid_captions_masks = []

    for cap in valid_captions_list:
        cap_in, cap_out, msk = encode(cap)
        valid_captions_in.append(cap_in)
        valid_captions_out.append(cap_out)
        valid_captions_masks.append(msk)

    valid_captions_in = np.array(valid_captions_in, dtype="int32")
    valid_captions_out = np.array(valid_captions_out, dtype="int32")
    valid_captions_masks = np.array(valid_captions_masks, dtype="float32")

    def PREPROCESS(path, cap_in, cap_out, mask):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32)
        image.set_shape((224, 224, 3))

        tf.ensure_shape(cap_in, (config['max_len'],))
        tf.ensure_shape(cap_out, (config['max_len'],))
        tf.ensure_shape(mask, (config['max_len'],))

        return image, cap_in, cap_out, mask

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths_list,
                                                   train_captions_in, train_captions_out,
                                                   train_captions_masks))
    train_ds = (
        train_ds.shuffle(buffer_size=len(train_paths_list), seed=37)
        .map(PREPROCESS, num_parallel_calls=AUTOTUNE)
        .map(lambda image, cap_in, cap_out, mask: ((image, cap_in), cap_out, mask))
        .batch(32)
        .prefetch(AUTOTUNE)
    )

    valid_ds = tf.data.Dataset.from_tensor_slices((valid_paths_list,
                                                   valid_captions_in, valid_captions_out,
                                                   valid_captions_masks))
    valid_ds = (
        valid_ds
        .map(PREPROCESS, num_parallel_calls=AUTOTUNE)
        .map(lambda image, cap_in, cap_out, mask: ((image, cap_in), cap_out, mask))
        .batch(32)
        .prefetch(AUTOTUNE)
    )

    return train_ds, valid_ds
