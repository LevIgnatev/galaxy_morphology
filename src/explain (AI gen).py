import numpy as np, tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

def gradcam(model, img, last_conv="conv5_block3_out"):
    grad_model = tf.keras.models.Model([model.inputs],
                                       [model.get_layer(last_conv).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(tf.expand_dims(img,0))
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0,1,2))
    cam = tf.reduce_sum(tf.multiply(weights, conv_out[0]), axis=-1).numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    return cam