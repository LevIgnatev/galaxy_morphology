from pathlib import Path
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "labels"
all_captions_path = DATA_PATH / "captions" / "captions_sample.csv"
caption_train_path = DATA_PATH / "captions" /"train_captions_sample.txt"
caption_valid_path = DATA_PATH / "captions" /"val_captions_sample.txt"
images_dir = DATA_PATH / "thumbs"
labels_fp = DATA_PATH / "labels_manifest_1000.csv"
laptop_or_PC = "laptop"
manifest_fp = ""

from tokenizer import encode

def clean_ds_ffs():
    manifest = pd.read_csv(manifest_fp)
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
        cap_in, cap_out, mask = encode(caption)
        return image, cap_in, cap_out, mask

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths_list,
                                                   train_captions_list))
    train_ds = (
        train_ds.shuffle(buffer_size=len(train_paths_list), seed=37)
        .map(PREPROCESS, num_parallel_calls=AUTOTUNE)
        .batch(32)
        .prefetch(AUTOTUNE)
    )

    valid_ds = tf.data.Dataset.from_tensor_slices((valid_paths_list,
                                                   valid_captions_list))
    valid_ds = (
        valid_ds
        .map(PREPROCESS, num_parallel_calls=AUTOTUNE)
        .batch(32)
        .prefetch(AUTOTUNE)
    )

    return train_ds, valid_ds

def dataset(): # main function for loading the data from filepaths
    df1 = pd.read_csv(all_captions_path, dtype={"objid": str})
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
    df_train = df1.merge(train_ids, how="inner", on="objid").reset_index(drop=True)
    df_valid = df1.merge(valid_ids, how="inner", on="objid").reset_index(drop=True)

    captions_train = df_train["caption"].astype(str).tolist()
    captions_valid = df_valid["caption"].astype(str).tolist()

    captions_train_list = []
    for caption in captions_train:
        cap_in, cap_out, mask = encode(caption)
        captions_train_list.append([cap_in, cap_out, mask])

    captions_valid_list = []
    for caption in captions_valid:
        cap_in, cap_out, mask = encode(caption)
        captions_valid_list.append([cap_in, cap_out, mask])

    df = pd.read_csv(labels_fp) # read the labels csv file
    labels_and_fp_cols = df[["derived_label", f"filepath_{laptop_or_PC}"]] # select needed labels

    train_df, valid_df = train_test_split(labels_and_fp_cols, # split the data for training (80%) and validation (20%)
                                          test_size=0.2,
                                          stratify=labels_and_fp_cols["derived_label"],
                                          random_state=37)

    labels_to_indexes = {} # create a dictionary with labels and indexes (elliptical=0, spiral=1, etc.)
    unique_labels = labels_and_fp_cols["derived_label"].nunique()
    for i in range (unique_labels):
        labels_to_indexes[labels_and_fp_cols["derived_label"].unique()[i]] = i

    train_paths_list = train_df[f"filepath_{laptop_or_PC}"].tolist() # convert dataframes to lists
    valid_paths_list = valid_df[f"filepath_{laptop_or_PC}"].tolist()
    train_labels_list = []
    valid_labels_list = []
    for i in train_df["derived_label"]:
        train_labels_list.append(labels_to_indexes[i])
    for i in valid_df["derived_label"]:
        valid_labels_list.append(labels_to_indexes[i])

    def PREPROCESS(path, label): # function to flatten the images and one-hot encode the labels
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32) # cast the image to a tensor with numbers between 0 and 255
        labels = tf.one_hot(label, unique_labels) # one-hot encoding the labels
        return image, labels

    AUTOTUNE = tf.data.AUTOTUNE # AUTOTUNE variable for an optimized performance

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths_list, captions_train_list)) # main data pipeline (map, then the classic shuffle - batch - prefetch) for training data
    train_ds = (
        train_ds.shuffle(buffer_size=len(train_paths_list), seed=37)
        .map(PREPROCESS, num_parallel_calls=AUTOTUNE)
        .batch(32)
        .prefetch(AUTOTUNE)
    )

    valid_ds = tf.data.Dataset.from_tensor_slices((valid_paths_list, captions_valid_list)) # main data pipeline (map, then the shuffle - batch - prefetch) for validation data
    valid_ds = (
        valid_ds
        .map(PREPROCESS, num_parallel_calls=AUTOTUNE)
        .batch(32)
        .prefetch(AUTOTUNE)
    )

    return train_ds, valid_ds
