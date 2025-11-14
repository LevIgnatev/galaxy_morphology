# important imports
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_data(labels_fp): # main function for loading the data from filepaths

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    
    df = pd.read_csv(labels_fp) # read the labels csv file

    labels_and_fp_cols = df[["derived_label", "filepath"]] # select needed labels

    train_df, valid_df = train_test_split(labels_and_fp_cols, # split the data for training (80%) and validation (20%)
                                          test_size=0.2,
                                          stratify=labels_and_fp_cols["derived_label"],
                                          random_state=37)

    labels_to_indexes = {} # create a dictionary with labels and indexes (elliptical=0, spiral=1, etc.)
    unique_labels = labels_and_fp_cols["derived_label"].nunique()
    for i in range (unique_labels):
        labels_to_indexes[labels_and_fp_cols["derived_label"].unique()[i]] = i

    train_paths_list = train_df["filepath"].apply(lambda x: str(PROJECT_ROOT / x)).tolist() # convert dataframes to lists
    valid_paths_list = valid_df["filepath"].apply(lambda x: str(PROJECT_ROOT / x)).tolist()
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

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths_list, train_labels_list)) # main data pipeline (map, then the classic shuffle - batch - prefetch) for training data
    train_ds = (
        train_ds.shuffle(buffer_size=len(train_paths_list), seed=37)
        .map(PREPROCESS, num_parallel_calls=AUTOTUNE)
        .batch(32)
        .prefetch(AUTOTUNE)
    )

    valid_ds = tf.data.Dataset.from_tensor_slices((valid_paths_list, valid_labels_list)) # main data pipeline (map, then the shuffle - batch - prefetch) for validation data
    valid_ds = (
        valid_ds
        .map(PREPROCESS, num_parallel_calls=AUTOTUNE)
        .batch(32)
        .prefetch(AUTOTUNE)
    )
    return train_ds, valid_ds
