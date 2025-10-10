# gradcam_visualizer_AI_gen.py

import os, cv2, numpy as np, tensorflow as tf

# ---- Paths (edit these) ----
MODEL_PATH = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\checkpoints\ckpt_classifier_full.keras"
IMG_PATH   = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\labels\thumbs\236175.jpg"
OUT_PATH   = r"/data/labels/gradcam_out/gradcam_example_6.jpg"

# ---- 1) Load your original model (augmentation+preprocess+ResNet(pooling='avg')+Dense head) ----
orig = tf.keras.models.load_model(MODEL_PATH, compile=False)
num_classes = orig.output_shape[-1]

# ---- 2) Build a "gradcam-ready" clone with ResNet50(pooling=None) so we can access conv maps ----
inp = tf.keras.Input(shape=(224, 224, 3), name="input")

# Use the SAME preprocessing you used in training (you put it inside the model then;
# we put it here explicitly so the pipeline is: raw -> preprocess -> resnet_no_pool)
x = tf.keras.applications.resnet50.preprocess_input(inp)
resnet_no_pool = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", pooling=None)
resnet_no_pool.trainable = False

# Last conv feature map (we'll use this for CAM)
conv_feat = resnet_no_pool(x)                      # (7,7,2048)
gap = tf.keras.layers.GlobalAveragePooling2D(name="gap")(conv_feat)   # identical to pooling='avg'

# Recreate your head (128 -> dropout -> softmax(num_classes))
dense1 = tf.keras.layers.Dense(128, activation="relu", name="dense_clone")
drop1  = tf.keras.layers.Dropout(0.2, name="dropout_clone")
logits = tf.keras.layers.Dense(num_classes, activation="softmax", name="head_clone")(drop1(dense1(gap)))

cam_model = tf.keras.Model(inp, [conv_feat, logits], name="gradcam_ready")

# ---- 3) Copy head weights from the original model ----
# Find original Dense layers (first is 128-ReLU, second is num_classes)
orig_denses = [l for l in orig.layers if isinstance(l, tf.keras.layers.Dense)]
if len(orig_denses) < 2:
    raise RuntimeError("Did not find two Dense layers in the original model.")

dense1.set_weights(orig_denses[0].get_weights())
cam_model.get_layer("head_clone").set_weights(orig_denses[1].get_weights())

# ---- 4) Load image (raw, uint8), DO NOT preprocess here (the model does it) ----
img_bgr = cv2.imread(IMG_PATH)
if img_bgr is None:
    raise FileNotFoundError(IMG_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
x_in = img_resized.astype("float32")[None, ...]   # (1,224,224,3)

# ---- 5) Forward + Grad-CAM ----
with tf.GradientTape() as tape:
    conv_out, preds = cam_model(x_in, training=False)     # conv_out: (1,7,7,2048)
    class_idx = tf.argmax(preds[0])                       # or set an int manually
    loss = preds[:, class_idx]

grads = tape.gradient(loss, conv_out)                     # (1,7,7,2048)
weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)  # (1,1,1,2048)
cam = tf.nn.relu(tf.reduce_sum(weights * conv_out, axis=-1))[0].numpy()  # (7,7)
cam -= cam.min()
cam /= (cam.max() + 1e-8)
cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)

# ---- 6) Overlay heatmap on the original image ----
heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
overlay = np.clip(0.5 * img_resized + 0.5 * heat, 0, 255).astype(np.uint8)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
cv2.imwrite(OUT_PATH, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print(f"Saved Grad-CAM to {OUT_PATH}")

def generate_gradcam_overlay_from_array(model_path: str, image_rgb_224_np):
    """
    Wrapper that uses the same Grad-CAM pipeline defined above,
    but takes a (224,224,3) RGB float/uint8 array and returns (overlay_uint8, softmax_probs).
    """
    import numpy as _np, cv2 as _cv2, tensorflow as _tf

    num_classes = _tf.keras.models.load_model(model_path, compile=False).output_shape[-1]

    inp = _tf.keras.Input(shape=(224, 224, 3), name="input")
    x = _tf.keras.applications.resnet50.preprocess_input(inp)
    resnet_no_pool = _tf.keras.applications.ResNet50(include_top=False, weights="imagenet", pooling=None)
    resnet_no_pool.trainable = False
    conv_feat = resnet_no_pool(x)
    gap = _tf.keras.layers.GlobalAveragePooling2D(name="gap")(conv_feat)
    dense1 = _tf.keras.layers.Dense(128, activation="relu", name="dense_clone")
    drop1  = _tf.keras.layers.Dropout(0.2, name="dropout_clone")
    head   = _tf.keras.layers.Dense(num_classes, activation="softmax", name="head_clone")(drop1(dense1(gap)))
    cam_model = _tf.keras.Model(inp, [conv_feat, head], name="gradcam_ready")

    orig = _tf.keras.models.load_model(model_path, compile=False)
    orig_denses = [l for l in orig.layers if isinstance(l, _tf.keras.layers.Dense)]
    dense1.set_weights(orig_denses[0].get_weights())
    cam_model.get_layer("head_clone").set_weights(orig_denses[1].get_weights())

    x_in = image_rgb_224_np.astype("float32")[None, ...]

    with _tf.GradientTape() as tape:
        conv_out, preds = cam_model(x_in, training=False)
        class_idx = _tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads   = tape.gradient(loss, conv_out)
    weights = _tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam     = _tf.nn.relu(_tf.reduce_sum(weights * conv_out, axis=-1))[0].numpy()
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)
    cam = _cv2.resize(cam, (224, 224), interpolation=_cv2.INTER_LINEAR)

    heat = _cv2.applyColorMap((cam * 255).astype(_np.uint8), _cv2.COLORMAP_JET)
    heat = _cv2.cvtColor(heat, _cv2.COLOR_BGR2RGB)
    overlay = _np.clip(0.5 * image_rgb_224_np + 0.5 * heat, 0, 255).astype(_np.uint8)
    return overlay, preds.numpy()[0]


if __name__ == "__main__":
    # keep your existing single-image script run here so CLI usage still works
    pass
