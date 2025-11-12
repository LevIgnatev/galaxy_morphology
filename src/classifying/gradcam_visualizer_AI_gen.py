def generate_gradcam_overlay_from_array(
    weights_fp: str,
    image_rgb_224_np,
    num_classes: int = 5,
):
    import numpy as _np, cv2 as _cv2, tensorflow as _tf

    # --- rebuild EXACT classifier (same as infer.py but with explicit GAP) ---
    inputs = _tf.keras.Input((224, 224, 3))
    x = _tf.keras.layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = _tf.keras.layers.RandomRotation(0.2)(x)
    x = _tf.keras.layers.RandomTranslation(0.1, 0.1)(x)
    x = _tf.keras.applications.resnet50.preprocess_input(x)

    # IMPORTANT: pooling=None so we can grab conv maps from the SAME call
    base = _tf.keras.applications.ResNet50(include_top=False, weights="imagenet", pooling=None)
    base.trainable = False
    conv_feat = base(x, training=False)                           # (None, 7, 7, 2048)
    gap = _tf.keras.layers.GlobalAveragePooling2D(name="gap")(conv_feat)

    dense1 = _tf.keras.layers.Dense(128, activation="relu", name="dense_128")(gap)
    drop1  = _tf.keras.layers.Dropout(0.2, name="dropout_02")(dense1)
    head   = _tf.keras.layers.Dense(num_classes, activation="softmax", name="head")(drop1)

    cam_model = _tf.keras.Model(inputs, [conv_feat, head], name="gradcam_model")
    cam_model.load_weights(str(weights_fp))

    x_in = _tf.convert_to_tensor(image_rgb_224_np.astype("float32")[None, ...])

    with _tf.GradientTape() as tape:
        conv_out, preds = cam_model(x_in, training=False)
        class_idx = _tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads   = tape.gradient(loss, conv_out)                        # (1,H,W,C)
    weights = _tf.reduce_mean(grads, axis=(1, 2), keepdims=True)   # (1,1,1,C)
    cam     = _tf.nn.relu(_tf.reduce_sum(weights * conv_out, axis=-1))[0].numpy()

    cam -= cam.min()
    cam /= (cam.max() + 1e-8)
    cam = _cv2.resize(cam, (224, 224), interpolation=_cv2.INTER_LINEAR)

    heat = _cv2.applyColorMap((cam * 255).astype(_np.uint8), _cv2.COLORMAP_JET)
    heat = _cv2.cvtColor(heat, _cv2.COLOR_BGR2RGB)
    overlay = _np.clip(0.5 * image_rgb_224_np + 0.5 * heat, 0, 255).astype(_np.uint8)
    return overlay, preds.numpy()[0]
